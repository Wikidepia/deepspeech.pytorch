import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import glu
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU,
    'cnn': None,
    'glu_small':None,
    'glu_large':None,
    'glu_flexible':None,
    'large_cnn':None,
    'cnn_residual':None,
    'cnn_jasper':None
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.ByteTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True, bnm=0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.bnm = bnm
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size, momentum=bnm)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        assert x.is_cuda
        max_seq_length = x.size(0)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
            # x = x._replace(data=self.batch_norm(x.data))
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths.data.cpu().numpy())
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=max_seq_length)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        # x = x.to('cuda')
        return x


class DeepBatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, num_layers=1,
                 batch_norm=True, sum_directions=True, **kwargs):
        super(DeepBatchRNN, self).__init__()
        self._bidirectional = bidirectional
        rnns = []
        rnn = BatchRNN(input_size=input_size, hidden_size=hidden_size, rnn_type=rnn_type, bidirectional=bidirectional,
                       batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(num_layers - 1):
            rnn = BatchRNN(input_size=hidden_size, hidden_size=hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, batch_norm=batch_norm)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.sum_directions = sum_directions

    def flatten_parameters(self):
        for x in range(len(self.rnns)):
            self.rnns[x].flatten_parameters()

    def forward(self, x, lengths):
        max_seq_length = x.size(0)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.data.squeeze(0).cpu().numpy())
        x = self.rnns(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=max_seq_length)
        return x, None


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(Lookahead, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input.size()[1:])).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


DEBUG = 0


class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=6,
                 audio_conf=None,
                 bidirectional=True, context=20, bnm=0.1,
                 dropout=0, cnn_width=256,
                 phoneme_count=0
                 ):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self._version = '0.0.1'
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = nb_layers
        self._rnn_type = rnn_type
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._bidirectional = bidirectional
        self._bnm = bnm
        self._dropout = dropout
        self._cnn_width = cnn_width
        if phoneme_count > 0:
            self._phoneme_count = phoneme_count

        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)
        num_classes = len(self._labels)

        self.dropout1 = nn.Dropout(p=0.1, inplace=True)
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32, momentum=bnm),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32, momentum=bnm),
            nn.Hardtanh(0, 20, inplace=True),
        ))

        if self._rnn_type == 'cnn': #  wav2letter with some features
            size = rnn_hidden_size
            modules = Wav2Letter(
                DotDict({
                    'size':size, # here it defines model epilog size
                    'bnorm':True,
                    'bnm':self._bnm,
                    'dropout':dropout,
                    'cnn_width':self._cnn_width, # cnn filters 
                    'not_glu':self._bidirectional, # glu or basic relu
                    'repeat_layers':self._hidden_layers, # depth, only middle part
                    'kernel_size':13
                })                
            )
            self.rnns = nn.Sequential(*modules)
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels=size, out_channels=num_classes, kernel_size=1)
            )
        elif self._rnn_type == 'cnn_residual': #  wav2letter with some features
            size = rnn_hidden_size
            self.rnns = ResidualWav2Letter(
                DotDict({
                    'size': rnn_hidden_size,  # here it defines model epilog size
                    'bnorm': True,
                    'bnm': self._bnm,
                    'dropout': dropout,
                    'cnn_width': self._cnn_width,  # cnn filters 
                    'not_glu': self._bidirectional,  # glu or basic relu
                    'repeat_layers': self._hidden_layers,  # depth, only middle part
                    'kernel_size': 7,
                    'se_ratio': 0.25,
                    'skip': True
                })
            )
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels=size, out_channels=num_classes, kernel_size=1)
            )
            # make checkpoints reverse compatible
            if hasattr(self, '_phoneme_count'):
                self.fc_phoneme = nn.Sequential(
                    nn.Conv1d(in_channels=size,
                              out_channels=self._phoneme_count, kernel_size=1)
                )
        elif self._rnn_type == 'cnn_jasper': #  http://arxiv.org/abs/1904.03288
            size = 1024
            self.rnns = JasperNet(
                DotDict({
                    'input_channels':161,
                    'blocks':5,          
                    'sub_blocks':3,
                    'channels':[256,384,512,640,768],
                    'kernels':[11,13,17,21,25],
                    'prolog_dropout':0.8,
                    'block_dropouts':0.2,
                    'epilog_dropout':0.3,
                    'nonlinearity':nn.ReLU(inplace=True),
                    'se_ratios':[0.25,0.25,0.25,0.25,0.25],
                })                
            )
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels=size, out_channels=num_classes, kernel_size=1)
            )                 
        elif self._rnn_type == 'large_cnn':
            self.rnns = LargeCNN(
                DotDict({
                    'input_channels':161,
                    'bnm':bnm,
                    'dropout':dropout,
                })
            )
            # last GLU layer size
            size = self.rnns.last_channels            
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels=size, out_channels=num_classes, kernel_size=1)
            )            
        elif self._rnn_type == 'glu_small':
            self.rnns = SmallGLU(
                DotDict({
                    'input_channels':161,
                    'layer_num':self._hidden_layers,
                    'bnm':bnm,
                    'dropout':dropout,
                })
            )
            # last GLU layer size
            size = self.rnns.last_channels            
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels=size, out_channels=num_classes, kernel_size=1)
            )
        elif self._rnn_type == 'glu_large':
            self.rnns = LargeGLU(
                DotDict({
                    'input_channels':161
                })
            )            
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels=size, out_channels=num_classes, kernel_size=1)
            )
        elif self._rnn_type == 'glu_flexible':
            raise NotImplementedError("Customizable GLU not yet implemented") 
        else:  # original ds2
            # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
            rnn_input_size = int(math.floor((sample_rate * window_size + 1e-2) / 2) + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41 + 1e-2) / 2 + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21 + 1e-2) / 2 + 1)
            rnn_input_size *= 32

            rnns = []
            rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=supported_rnns[rnn_type],
                           bidirectional=bidirectional, batch_norm=False)
            rnns.append(('0', rnn))
            for x in range(nb_layers - 1):
                rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size,
                               rnn_type=supported_rnns[rnn_type],
                               bidirectional=bidirectional, bnm=bnm)
                rnns.append(('%d' % (x + 1), rnn))
            self.rnns = nn.Sequential(OrderedDict(rnns))

            self.lookahead = nn.Sequential(
                # consider adding batch norm?
                Lookahead(rnn_hidden_size, context=context),
                nn.Hardtanh(0, 20, inplace=True)
            ) if not bidirectional else None

            fully_connected = nn.Sequential(
                nn.BatchNorm1d(rnn_hidden_size, momentum=bnm),
                nn.Linear(rnn_hidden_size, num_classes, bias=False)
            )
            self.fc = nn.Sequential(
                SequenceWise(fully_connected),
            )

    def forward(self, x, lengths):
        # assert x.is_cuda
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths).cuda()

        if self._rnn_type in ['cnn', 'glu_small', 'glu_large', 'large_cnn',
                              'cnn_residual']:
            x = x.squeeze(1)
            x = self.rnns(x)
            if hasattr(self, '_phoneme_count'):
                x_phoneme = self.fc_phoneme(x)
                x_phoneme = x_phoneme.transpose(1, 2).transpose(0, 1).contiguous()
            x = self.fc(x)
            x = x.transpose(1, 2).transpose(0, 1).contiguous()
        else:
            # x = self.dropout1(x)
            x, _ = self.conv(x, output_lengths)
            # x = self.dropout2(x)
            if DEBUG: assert x.is_cuda
            # x = x.to('cuda')
            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
            x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
            assert x.is_cuda

            for rnn in self.rnns:
                x = rnn(x, output_lengths)
                assert x.is_cuda

            if not self._bidirectional:  # no need for lookahead layer in bidirectional
                x = self.lookahead(x)
                assert x.is_cuda

            x = self.fc(x)
        if DEBUG: assert x.is_cuda
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        outs = F.softmax(x, dim=-1)
        if DEBUG: assert outs.is_cuda
        if DEBUG: assert output_lengths.is_cuda

        if hasattr(self, '_phoneme_count'):
            x_phoneme = x_phoneme.transpose(0, 1)
            outs_phoneme = F.softmax(x_phoneme, dim=-1)
            # phoneme outputs will have the same length
            return x, outs, output_lengths, x_phoneme, outs_phoneme
        else:
            return x, outs, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'],
                    nb_layers=package['hidden_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=package['rnn_type'],
                    bnm=package.get('bnm', 0.1),
                    bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        if package['rnn_type'] != 'cnn':
            for x in model.rnns:
                x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        kwargs = {
            'rnn_hidden_size': package['hidden_size'],
            'nb_layers': package['hidden_layers'],
            'labels': package['labels'],
            'audio_conf': package['audio_conf'],
            'rnn_type': package['rnn_type'],
            'bnm': package.get('bnm', 0.1),
            'bidirectional': package.get('bidirectional', True),
            'dropout': package.get('dropout', 0),
            'cnn_width': package.get('cnn_width', 0),
            'phoneme_count': package.get('phoneme_count', 0)
        }
        model = cls(**kwargs)
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def add_phonemes_to_model(model,
                              phoneme_count=0):
        '''Add phonemes to an already pre-trained model
        '''
        model._phoneme_count = phoneme_count
        model.fc_phoneme = nn.Sequential(
            nn.Conv1d(in_channels=model._hidden_size,
                      out_channels=model._phoneme_count,
                      kernel_size=1)
        )
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, checkpoint=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None,
                  checkpoint_cer_results=None, checkpoint_wer_results=None, checkpoint_loss_results=None,
                  trainval_checkpoint_loss_results=None, trainval_checkpoint_cer_results=None, trainval_checkpoint_wer_results=None):
        model = model.module if DeepSpeech.is_parallel(model) else model
        package = {
            'version': model._version,
            'hidden_size': model._hidden_size,
            'hidden_layers': model._hidden_layers,
            'rnn_type': model._rnn_type,
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict(),
            'bnm': model._bnm,
            'bidirectional': model._bidirectional,
            'dropout':model._dropout,
            'cnn_width':model._cnn_width
        }
        if hasattr(model, '_phoneme_count'):
            package['phoneme_count'] = model._phoneme_count
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        package['checkpoint'] = checkpoint
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
            package['checkpoint_cer_results'] = checkpoint_cer_results
            package['checkpoint_wer_results'] = checkpoint_wer_results
            package['checkpoint_loss_results'] = checkpoint_loss_results
            # only if the relevant flag passed to args in train.py
            # otherwise always None
            package['trainval_checkpoint_loss_results'] = trainval_checkpoint_loss_results
            package['trainval_checkpoint_cer_results'] = trainval_checkpoint_cer_results
            package['trainval_checkpoint_wer_results'] = trainval_checkpoint_wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        return model.module._labels if model.is_parallel(model) else model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        return model.module._audio_conf if DeepSpeech.is_parallel(model) else model._audio_conf

    @staticmethod
    def get_meta(model):
        m = model.module if DeepSpeech.is_parallel(model) else model
        meta = {
            "version": m._version,
            "hidden_size": m._hidden_size,
            "hidden_layers": m._hidden_layers,
            "rnn_type": m._rnn_type
        }
        return meta

    @staticmethod
    def is_parallel(model):
        return isinstance(model, torch.nn.parallel.DataParallel) or \
               isinstance(model, torch.nn.parallel.DistributedDataParallel)


# bit ugly, but we need to clean things up!
def Wav2Letter(config):
    assert type(config)==DotDict
    not_glu = config.not_glu
    bnm = config.bnm
    def _block(in_channels, out_channels, kernel_size,
               padding=0, stride=1, bnorm=False, bias=True,
               dropout=0):
        # use self._bidirectional flag as a flag for GLU usage in the CNN
        # the flag is True by default, so use False
        if not not_glu:
            out_channels = int(out_channels * 2)

        res = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, padding=padding,
                         stride=stride, bias=bias)]
        # for non GLU networks
        if not_glu:
            if bnorm:
                res.append(nn.BatchNorm1d(out_channels, momentum=bnm))
        # use self._bidirectional flag as a flag for GLU usage in the CNN                    
        if not_glu:
            res.append(nn.ReLU(inplace=True))
        else:
            res.append(GLUModule(dim=1))
        # for GLU networks
        if not not_glu:
            if bnorm:
                res.append(nn.BatchNorm1d(int(out_channels//2),
                                          momentum=bnm))                    
        if dropout>0:
            res.append(nn.Dropout(dropout))
        return res

    size = config.size
    cnn_width = config.cnn_width
    bnorm = config.bnorm
    dropout = config.dropout
    repeat_layers = config.repeat_layers
    kernel_size = config.kernel_size # wav2letter default - 7
    padding = kernel_size // 2

    # "prolog"
    modules = _block(in_channels=161, out_channels=cnn_width, kernel_size=kernel_size,
                     padding=padding, stride=2, bnorm=bnorm, bias=not bnorm, dropout=dropout)
    
    # main convs
    for _ in range(0,repeat_layers):
        modules.extend(
            [*_block(in_channels=cnn_width, out_channels=cnn_width, kernel_size=kernel_size,
                     padding=padding, bnorm=bnorm, bias=not bnorm, dropout=dropout)]
        )
    # "epilog"
    modules.extend([*_block(in_channels=cnn_width, out_channels=size, kernel_size=31,
                            padding=15, bnorm=bnorm, bias=not bnorm, dropout=dropout)])
    modules.extend([*_block(in_channels=size, out_channels=size, kernel_size=1,
                            bnorm=bnorm, bias=not bnorm, dropout=dropout)])
    return modules


class ResidualWav2Letter(nn.Module):
    def __init__(self,config):
        super(ResidualWav2Letter, self).__init__()
        
        size = config.size
        cnn_width = config.cnn_width
        bnorm = config.bnorm
        bnm = config.bnm
        dropout = config.dropout
        repeat_layers = config.repeat_layers
        kernel_size = config.kernel_size # wav2letter default - 7
        padding = kernel_size // 2
        se_ratio = config.se_ratio
        skip = config.skip
        
        # "prolog"
        modules = [ResCNNBlock(_in=161, out=cnn_width, kernel_size=kernel_size,
                               padding=padding, stride=2,bnm=bnm, bias=not bnorm, dropout=dropout,
                               nonlinearity=nn.ReLU(inplace=True),
                               se_ratio=0,skip=False)] # no skips and attention
        
        # main convs
        for _ in range(0,repeat_layers):
            modules.extend(
                [ResCNNBlock(_in=cnn_width, out=cnn_width, kernel_size=kernel_size,
                               padding=padding, stride=1,bnm=bnm, bias=not bnorm, dropout=dropout,
                               nonlinearity=nn.ReLU(inplace=True),
                               se_ratio=se_ratio,skip=skip)]
            )
        # "epilog"
        modules.extend([ResCNNBlock(_in=cnn_width, out=size, kernel_size=31,
                                    padding=15, stride=1,bnm=bnm, bias=not bnorm, dropout=dropout,
                                    nonlinearity=nn.ReLU(inplace=True),
                                    se_ratio=0,skip=False)]) # no skips and attention
        modules.extend([ResCNNBlock(_in=size, out=size, kernel_size=1,
                                    padding=0, stride=1,bnm=bnm, bias=not bnorm, dropout=dropout,
                                    nonlinearity=nn.ReLU(inplace=True),
                                    se_ratio=0,skip=False)]) # no skips and attention
        
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)  


class GLUBlock(nn.Module):
    def __init__(self,
                 _in=1,
                 out=400,
                 kernel_size=13,
                 stride=1,
                 padding=0,
                 dropout=0.2,
                 bnm=0.1
                 ):
        super(GLUBlock, self).__init__()       
        
        self.conv = nn.Conv1d(_in,
                              out,
                              kernel_size,
                              stride=stride,
                              padding=padding)
        # self.conv = weight_norm(self.conv, dim=1)
        # self.norm = nn.InstanceNorm1d(out)    
        self.norm = nn.BatchNorm1d(out//2,
                                   momentum=bnm)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = glu(x,dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return x
    
    
class CNNBlock(nn.Module):
    def __init__(self,
                 _in=1,
                 out=400,
                 kernel_size=13,
                 stride=1,
                 padding=0,
                 dropout=0.1,
                 bnm=0.1,
                 nonlinearity=nn.ReLU(inplace=True),
                 bias=True
                 ):
        super(CNNBlock, self).__init__()       
        
        self.conv = nn.Conv1d(_in,
                              out,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.norm = nn.BatchNorm1d(out,
                                   momentum=bnm)
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return x


class ResCNNBlock(nn.Module):
    def __init__(self,
                 _in=1,
                 out=400,
                 kernel_size=13,
                 stride=1,
                 padding=0,
                 dropout=0.1,
                 bnm=0.1,
                 nonlinearity=nn.ReLU(inplace=True),
                 bias=True,
                 se_ratio=0,
                 skip=False
                 ):
        super(ResCNNBlock, self).__init__()       
        
        self.conv = nn.Conv1d(_in,
                              out,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.norm = nn.BatchNorm1d(out,
                                   momentum=bnm)
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout(dropout)
        self.se_ratio = se_ratio
        self.skip = skip        
        self.has_se = (self.se_ratio is not None) and (0 < self.se_ratio <= 1)
        # Squeeze and Excitation layer, if required
        if self.has_se:
            num_squeezed_channels = max(1, int(_in * self.se_ratio))
            self._se_reduce = Conv1dSamePadding(in_channels=out, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv1dSamePadding(in_channels=num_squeezed_channels, out_channels=out, kernel_size=1)        
        
    def forward(self, x):
        # be a bit more memory efficient during ablations
        if self.skip:
            inputs = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool1d(x, 1) # channel dimension
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x                
        if self.skip:
            x = x + inputs
        return x       


# http://arxiv.org/abs/1904.03288
class JasperNet(nn.Module):
    def __init__(self,
                 config):
        self.prolog = JasperProlog(config)
        self.body = JasperBody(config)
        self.epilog = JasperEpilog(config)
        
    def forward(self, x):
        x = self.prolog(x)
        x = self.body(x)
        x = self.epilog(x)
        return x         


class JasperCNNBlock(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=400,
                 kernel_size=11,
                 stride=1,
                 padding=0,
                 dropout=0.1,
                 nonlinearity=nn.ReLU(inplace=True),
                 bias=True,
                 se_ratio=0,
                 skip=False
                 ):
        super(JasperCNNBlock, self).__init__()       
        
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.norm = nn.BatchNorm1d(out)
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout(dropout)
        self.se_ratio = se_ratio
        self.has_se = (self.se_ratio is not None) and (0 < self.se_ratio <= 1)
        self.skip = skip
        if self.skip:
            self.skip_block = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                nn.BatchNorm1d(out),
            )
        # Squeeze and Excitation layer, if required
        if self.has_se:
            num_squeezed_channels = max(1, int(_in * self.se_ratio))
            self._se_reduce = Conv1dSamePadding(in_channels=out, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv1dSamePadding(in_channels=num_squeezed_channels, out_channels=out, kernel_size=1)        
        
    def forward(self, x,
                residual=None):
        skip=None
        if self.skip:
            skip = self.skip_block(x)
        x = self.conv(x)
        x = self.norm(x)
        if residual:
            x = x + residual        
        x = self.nonlinearity(x)
        x = self.dropout(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool1d(x, 1) # channel dimension
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x                
        return x,skip         


class JasperBlock(nn.Module):
    def __init__(self,
                 sub_blocks=5,
                 in_channels=256,
                 out_channels=256,
                 kernel_size=11,
                 dropout=0.1,
                 nonlinearity=nn.ReLU(inplace=True),
                 se_ratio=0.25,
                 ):
        super(JasperProlog, self).__init__()
        # upsampling 
        self.sub_blocks = nn.ModuleList([
            JasperCNNBlock(
                in_channels=in_channels if i==0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                dropout=dropout,
                nonlinearity=nonlinearity,
                se_ratio=se_ratio,
                skip=True if i==0 else False
            )
            for i in range(0,sub_blocks)
        ]) 
    def forward(self, x):
        # can be easily extended into densenet
        for i, sub_block in enumerate(self.sub_blocks):
            if i==0:
                x, residual = sub_block(x,None)
            elif i==len(self.convs)-1:
                x, _ = sub_block(x,residual)
            else:
                x, _ = sub_block(x,None)
        return x    

  
class JasperBody(nn.Module):
    def __init__(self, config):
        super(JasperProlog, self).__init__()
        self.convs = nn.ModuleList([
            JasperBlock(
                sub_blocks=config.sub_blocks,
                in_channels=config.channels[i],
                out_channels=config.channels[i+1],
                kernel_size=config.kernels[i],
                dropout=config.block_dropouts[i],
                nonlinearity=config.nonlinearity,
                se_ratio=config.se_ratios[i],
            )
            for i in range(0,config.blocks)
        ])         
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x     


class JasperProlog(nn.Module):
    def __init__(self, config):
        super(JasperProlog, self).__init__()       
        self.blocks =  JasperCNNBlock(
            in_channels=config.input_channels,
            out_channels=config.channels[0],
            kernel_size=config.kernels[0],
            stride=2,
            padding=config.kernels[0]//2,
            dropout=config.prolog_dropout,
            nonlinearity=config.nonlinearity,
            bias=True,
            se_ratio=0,
            skip=False
        )
    def forward(self, x):
        return self.blocks(x)

    
class JasperEpilog(nn.Module):
    def __init__(self, config):
        super(JasperEpilog, self).__init__()       
        self.blocks =  nn.Sequential(
            JasperCNNBlock(
                in_channels=config.channels[-1],
                out_channels=896,
                kernel_size=29,
                stride=2,
                padding=29//2,
                dropout=config.epilog_dropout,
                nonlinearity=config.nonlinearity,
                bias=True,
                se_ratio=0,
                skip=False
            ),
            JasperCNNBlock(
                in_channels=896,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                dropout=config.epilog_dropout,
                nonlinearity=config.nonlinearity,
                bias=True,
                se_ratio=0,
                skip=False
            ),            
        )        
    def forward(self, x):
        return self.blocks(x)
    
    
class SmallGLU(nn.Module):
    def __init__(self,config):
        super(SmallGLU, self).__init__()   
        bnm = config.bnm
        dropout = config.dropout 
        layer_outputs = [100,100,100,125,125,150,175,200,
                         225,250,250,250,300,300,375]
        layer_list = [
            GLUBlock(config.input_channels,200,13,1,6,dropout, bnm), # 1          
            GLUBlock(100,200,3,1,(1),dropout, bnm), # 2
            GLUBlock(100,200,4,1,(2),dropout, bnm), # 3
            GLUBlock(100,250,5,1,(2),dropout, bnm), # 4
            GLUBlock(125,250,6,1,(3),dropout, bnm), # 5
            GLUBlock(125,300,7,1,(3),dropout, bnm), # 6
            GLUBlock(150,350,8,1,(4),dropout, bnm), # 7
            GLUBlock(175,400,9,1,(4),dropout, bnm), # 8
            GLUBlock(200,450,10,1,(5),dropout, bnm), # 9
            GLUBlock(225,500,11,1,(5),dropout, bnm), # 10
            GLUBlock(250,500,12,1,(6),dropout, bnm), # 11
            GLUBlock(250,500,13,1,(6),dropout, bnm), # 12
            GLUBlock(250,600,14,1,(7),dropout, bnm), # 13
            GLUBlock(300,600,15,1,(7),dropout, bnm), # 14
            GLUBlock(300,750,21,1,(10),dropout, bnm), # 15        
        ]
        self.layers = nn.Sequential(*layer_list[:config.layer_num])
        self.last_channels = layer_outputs[config.layer_num-1]

    def forward(self, x):
        return self.layers(x)     


class LargeGLU(nn.Module):
    def __init__(self,config):
        super(LargeGLU, self).__init__()       
        layer_outputs = [200,220,242,266,292,321,353,388,426,
                         468,514,565,621,683,751,826,908]
        # in out kw stride padding dropout
        self.layers = nn.Sequential(
            # whole padding in one place
            GLUBlock(config.input_channels,400,13,1,170,0.2), # 1          
            GLUBlock(200,440,14,1,0,0.214), # 2
            GLUBlock(220,484,15,1,0,0.228), # 3
            GLUBlock(242,532,16,1,0,0.245), # 4
            GLUBlock(266,584,17,1,0,0.262), # 5
            GLUBlock(292,642,18,1,0,0.280), # 6
            GLUBlock(321,706,19,1,0,0.300), # 7
            GLUBlock(353,776,20,1,0,0.321), # 8
            GLUBlock(388,852,21,1,0,0.347), # 9
            GLUBlock(426,936,22,1,0,0.368), # 10
            GLUBlock(468,1028,23,1,0,0.393), # 11
            GLUBlock(514,1130,24,1,0,0.421), # 12
            GLUBlock(565,1242,25,1,0,0.450), # 13
            GLUBlock(621,1366,26,1,0,0.482), # 14
            GLUBlock(683,1502,27,1,0,0.516), # 15
            GLUBlock(751,1652,28,1,0,0.552), # 16
            GLUBlock(826,1816,29,1,0,0.590), # 17
        )
        self.last_channels = layer_outputs[config.layer_num-1]        

    def forward(self, x):
        return self.layers(x)     


class LargeCNN(nn.Module):
    def __init__(self,config):
        super(LargeCNN, self).__init__()       
        bnm = config.bnm
        dropout = config.dropout         
        # in out kw stride padding dropout
        self.layers = nn.Sequential(
            # whole padding in one place
            CNNBlock(config.input_channels,200,13,2,6,dropout, bnm), # 1          
            CNNBlock(200,220,14,1,7, dropout, bnm), # 2
            CNNBlock(220,242,15,1,7, dropout, bnm), # 3
            CNNBlock(242,266,16,1,8, dropout, bnm), # 4
            CNNBlock(266,292,17,1,8, dropout, bnm), # 5
            CNNBlock(292,321,18,1,9, dropout, bnm), # 6
            CNNBlock(321,353,19,1,9, dropout, bnm), # 7
            CNNBlock(353,388,20,1,10, dropout, bnm), # 8
            CNNBlock(388,426,21,1,10, dropout, bnm), # 9
            CNNBlock(426,468,22,1,11, dropout, bnm), # 10
            CNNBlock(468,514,23,1,11, dropout, bnm), # 11
            CNNBlock(514,565,24,1,12, dropout, bnm), # 12
            CNNBlock(565,621,25,1,12, dropout, bnm), # 13
            CNNBlock(621,683,26,1,13, dropout, bnm), # 14
            CNNBlock(683,751,27,1,13, dropout, bnm), # 15
            CNNBlock(751,826,28,1,14, dropout, bnm), # 16
            CNNBlock(826,826,29,1,14, dropout, bnm), # 17
        )
        self.last_channels = 826        

    def forward(self, x):
        return self.layers(x)  
    
    
class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value    


# wrap in module to use in sequential
class GLUModule(nn.Module):
    def __init__(self, dim=1):
        super(GLUModule, self).__init__()
        self.dim = 1

    def forward(self, x):
        return glu(x,dim=self.dim)           


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class Conv1dSamePadding(nn.Conv1d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride[0] # just a scalar

    def forward(self, x):
        iw = int(x.size()[-1])
        kw = int(self.weight.size()[-1])
        sw = self.stride
        ow = math.ceil(iw / sw)
        pad_w = max((ow - 1) * self.stride + (kw - 1) * self.dilation[0] + 1 - iw, 0)
        if pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2])
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def main():
    import os.path
    import argparse
    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                        help='Path to model file created by training')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model(args.model_path)
    print("Model name:         ", os.path.basename(args.model_path))
    print("DeepSpeech version: ", model._version)
    print("")
    print("Recurrent Neural Network Properties")
    print("  RNN Type:         ", model._rnn_type)
    print("  RNN Layers:       ", model._hidden_layers)
    print("  RNN Size:         ", model._hidden_size)
    print("  Classes:          ", len(model._labels))
    print("")
    print("Model Features")
    print("  Labels:           ", model._labels)
    print("  Sample Rate:      ", model._audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model._audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model._audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model._audio_conf.get("window_stride", "n/a"))
    if package.get('loss_results', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))
    if package.get('meta', None) is not None:
        print("")
        print("Additional Metadata")
        for k, v in model._meta:
            print("  ", k, ": ", v)


if __name__ == '__main__':
    main()
