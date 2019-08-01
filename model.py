import math
from collections import OrderedDict

import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import glu
from torch.nn.parameter import Parameter


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU,
    'cnn': None,
    'glu_small': None,
    'glu_large': None,
    'glu_flexible': None,
    'large_cnn': None,
    'cnn_residual': None,
    'cnn_jasper': None,
    'cnn_jasper_2': None,
    'cnn_residual_repeat': None,
    'tds':None
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
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768,
                 nb_layers=6,
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

        if self._rnn_type not in ['tds']:
            self.dropout1 = nn.Dropout(p=0.1, inplace=True)
            self.conv = MaskConv(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32, momentum=bnm),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32, momentum=bnm),
                nn.Hardtanh(0, 20, inplace=True),
            ))

        if self._rnn_type == 'cnn':  #  wav2letter with some features
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
        elif self._rnn_type == 'cnn_residual':  #  wav2letter with some features
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
        elif self._rnn_type == 'cnn_residual_repeat':  # repeat middle convs
            size = rnn_hidden_size
            self.rnns = ResidualRepeatWav2Letter(
                DotDict({
                    'size': rnn_hidden_size,  # here it defines model epilog size
                    'bnorm': True,
                    'bnm': self._bnm,
                    'dropout': dropout,
                    'cnn_width': self._cnn_width,  # cnn filters
                    'not_glu': self._bidirectional,  # glu or basic relu
                    'repeat_layers': self._hidden_layers,  # depth, only middle part
                    'kernel_size': 7,
                    'se_ratio': 0.2,
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
        elif self._rnn_type == 'tds':  # repeat middle convs
            # TDS config
            size = rnn_hidden_size
            input_channels = 161
            h = 81
            kernel_size = 21
            blocks = 3
            strides = [2, 2, 1]
            repeats = [2, 3, 6]
            channels = [10, 14, 18]

            self.rnns = TDS(
                DotDict({
                    'dropout': dropout,
                    'h': h,
                    'kernel_size': kernel_size,
                    'blocks': blocks,
                    'strides': strides,
                    'repeats': repeats,
                    'channels': channels,
                    'output_size': size,
                    'input_channels': input_channels
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
        elif self._rnn_type == 'cnn_jasper':  #  http://arxiv.org/abs/1904.03288
            size = 1024
            big_block_repeat = self._cnn_width // 5
            jasper_config = {
                'dense_residual':False,
                'input_channels':161,
                'bn_momentum':0.1,
                'bn_eps':1e-05,
                'activation_fn':nn.ReLU,
                'repeats':[1] + [self._hidden_layers] * self._cnn_width + [1,1],
                'channels':[256] + sorted(big_block_repeat * [256,384,512,640,768]) + [896,1024],
                'kernel_sizes':[11] + sorted(big_block_repeat * [11,13,17,21,25]) + [29,1],
                'strides':[2] + [1] * self._cnn_width + [1,1],
                'dilations':[1] + [1] * self._cnn_width + [2,1],
                'dropouts':[0.2] + sorted(big_block_repeat * [0.2,0.2,0.2,0.3,0.3]) + [0.4,0.4],
                'residual':[0] + [1] * self._cnn_width + [0,0],
            }
            print(jasper_config)
            self.rnns = JasperNet(jasper_config)
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels=size, out_channels=num_classes, kernel_size=1)
            )
        elif self._rnn_type == 'cnn_jasper_2':  #  http://arxiv.org/abs/1904.03288
            size = 1024
            jasper_config = {
                'input_channels':161,
                'bn_momentum':0.1,
                'bn_eps':1e-05,
                'activation_fn':nn.ReLU,
                'repeats':self._hidden_layers,
                'num_modules':self._cnn_width

            }
            print(jasper_config)
            self.rnns = JasperNetEasy(jasper_config)
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
        if DEBUG: print(lengths)
        lengths = lengths.cpu().int()
        if DEBUG: print(lengths)
        if DEBUG:
            output_lengths = self.get_seq_lens(lengths)
            print('Projected output lengths {}'.format(output_lengths))
        else:
            output_lengths = self.get_seq_lens(lengths).cuda()

        if self._rnn_type in ['cnn', 'glu_small', 'glu_large', 'large_cnn',
                              'cnn_residual', 'cnn_jasper', 'cnn_jasper_2',
                              'cnn_residual_repeat', 'tds']:
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
        if not DEBUG: assert x.is_cuda
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        outs = F.softmax(x, dim=-1)
        if not DEBUG: assert outs.is_cuda
        if not DEBUG: assert output_lengths.is_cuda

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
        if self._rnn_type not in ['tds']:
            for m in self.conv.modules():
                if type(m) == nn.modules.conv.Conv2d:
                    seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        elif self._rnn_type == 'tds':
            # all convolutions are forced to be same
            seq_len = (seq_len + 0.01) // reduce(lambda x, y: x*y, self.rnns.strides)
        else:
            raise NotImplementedError()
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


class ResidualRepeatWav2Letter(nn.Module):
    def __init__(self,config):
        super(ResidualRepeatWav2Letter, self).__init__()

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
        modules = [ResCNNRepeatBlock(_in=161, out=cnn_width, kernel_size=kernel_size,
                                     padding=padding, stride=2,bnm=bnm, bias=not bnorm, dropout=dropout,
                                     nonlinearity=nn.ReLU(inplace=True),
                                     se_ratio=0, skip=False, repeat=1)] # no skips and attention
        # main convs
        # 3 blocks
        dilated_blocks = [0]
        dilation_level = 2
        dilated_subblocks = [1,2]

        repeat_start = 2
        repeat_mid = 2
        repeat_end = 1

        repeats = [repeat_start,
                   repeat_mid,
                   repeat_end]

        for j in range(0,3):
            for _ in range(0,repeat_layers//3):
                # 1221 dilation blocks
                dilation = 1 + (j in dilated_blocks) * (_ in dilated_subblocks) * (dilation_level - 1)
                modules.extend(
                    [ResCNNRepeatBlock(_in=cnn_width, out=cnn_width, kernel_size=kernel_size,
                                       padding=padding, dilation=dilation,
                                       stride=1,bnm=bnm, bias=not bnorm, dropout=dropout,
                                       nonlinearity=nn.ReLU(inplace=True),
                                       se_ratio=se_ratio, skip=skip, repeat=repeats[j])]
                )
        # "epilog"
        modules.extend([ResCNNRepeatBlock(_in=cnn_width, out=size, kernel_size=31,
                                          padding=15, stride=1,bnm=bnm, bias=not bnorm, dropout=dropout,
                                          nonlinearity=nn.ReLU(inplace=True),
                                          se_ratio=0, skip=False, repeat=1)]) # no skips and attention
        modules.extend([ResCNNRepeatBlock(_in=size, out=size, kernel_size=1,
                                          padding=0, stride=1,bnm=bnm, bias=not bnorm, dropout=dropout,
                                          nonlinearity=nn.ReLU(inplace=True),
                                          se_ratio=0, skip=False, repeat=1)]) # no skips and attention

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


class ResCNNRepeatBlock(nn.Module):
    def __init__(self,
                 _in=1,
                 out=400,
                 kernel_size=13,
                 stride=1,
                 padding=0,
                 dilation=1,
                 dropout=0.1,
                 bnm=0.1,
                 nonlinearity=nn.ReLU(inplace=True),
                 bias=True,
                 se_ratio=0,
                 skip=False,
                 repeat=1):
        super(ResCNNRepeatBlock, self).__init__()

        self.skip = skip
        has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        dropout = nn.Dropout(dropout)
        if has_se:
            # squeeze after each block
            scse = SCSE(out,
                        kernel_size=1, se_ratio=se_ratio)

        modules = []
        if dilation>1:
            padding = dilation*(kernel_size-1)//2

        # just stick all the modules together
        for i in range(0,repeat):
            if i==0:
                modules.extend([nn.Conv1d(_in, out, kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          bias=bias),
                                nn.BatchNorm1d(out,
                                               momentum=bnm),
                                nonlinearity,
                                dropout])
            else:
                modules.extend([nn.Conv1d(out, out, kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          bias=bias),
                                nn.BatchNorm1d(out,
                                               momentum=bnm),
                                nonlinearity,
                                dropout])
            if has_se:
                modules.extend([scse])

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        if self.skip:  # be a bit more memory efficient during ablations
            inputs = x
        x = self.layers(x)
        if self.skip:
            x = x + inputs
        return x


 # SCSE attention block https://arxiv.org/abs/1803.02579
class SCSE(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=1, se_ratio=0.1):
        super(SCSE, self).__init__()
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        self._se_reduce = Conv1dSamePadding(in_channels=in_channels, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv1dSamePadding(in_channels=num_squeezed_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        x_squeezed = F.adaptive_avg_pool1d(x, 1) # channel dimension
        x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
        x = torch.sigmoid(x_squeezed) * x
        return x

class TDS(nn.Module):
    def __init__(self, config):
        super(TDS, self).__init__()

        self.input_channels = config.input_channels
        h = config.h # stft or mel
        self.h = h
        dropout = config.dropout
        kernel_size = config.kernel_size

        blocks =  config.blocks
        strides = config.strides
        self.strides = strides
        repeats =  config.repeats
        channels = config.channels
        output_size = config.output_size

        self.channels = channels

        assert len(repeats) == blocks
        assert len(channels) == blocks
        assert len(strides) == blocks
        # https://github.com/facebookresearch/wav2letter/blob/master/recipes/librispeech/configs/seq2seq_tds/network.arch#L1

        """
        REMOVE LATER
        V -1 NFEAT 1 0
        C2 1 10 21 1 2 1 -1 -1
        R
        DO 0.2
        LN 3
        TDS 10 21 80 0.2
        TDS 10 21 80 0.2
        C2 10 14 21 1 2 1 -1 -1
        R
        DO 0.2
        LN 3
        TDS 14 21 80 0.2
        TDS 14 21 80 0.2
        TDS 14 21 80 0.2
        C2 14 18 21 1 2 1 -1 -1
        R
        DO 0.2
        LN 3
        TDS 18 21 80 0.2
        TDS 18 21 80 0.2
        TDS 18 21 80 0.2
        TDS 18 21 80 0.2
        TDS 18 21 80 0.2
        TDS 18 21 80 0.2
        V 0 1440 1 0
        RO 1 0 3 2
        L 1440 1024
        """

        channels = [1] + channels
        modules = []
        for i, (repeat, stride) in enumerate(zip(repeats,
                                                 strides)):
            modules.extend([
                Conv2dSamePadding(channels[i],channels[i+1],(kernel_size,1),
                                  stride=(stride, 2 if i==0 else 1)), # adhere to 80 channels in mel, 161 => 80
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                SeqLayerNormView(), # move the normalized channel to the last channel
                nn.LayerNorm(channels[i+1]),
                SeqLayerNormRestore() # revert
            ])
            modules.extend([TDSBlock(channels[i+1],
                                     kernel_size,
                                     h,
                                     dropout) for j in range(0,repeat)])

        self.linear = nn.Conv1d(in_channels=self.h * self.channels[-1],
                                out_channels=output_size, kernel_size=1)
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        if DEBUG: print('Input {}'.format(x.size()))
        batch_size = x.size(0)
        time = x.size(2)
        # mel input is 161
        # the inside of the model is 80
        assert x.size(1) == self.input_channels
        # V -1 NFEAT 1 0 (w2l++)
        # (batch, time, h, 1) (pytorch)
        # or should it be (batch, 1, time, h) (pytorch)
        x = x.view(batch_size, 1, time, self.input_channels)
        if DEBUG: print('First view {}'.format(x.size()))
        x = self.layers(x)
        if DEBUG: print('After layers {}'.format(x.size()))
        time_downsampled = x.size(2)
        time_ratio = (time + 10) // time_downsampled
        if DEBUG: print('Effective network downsampling is {}'.format(time_ratio))
        # if time_ratio not in [2, 4, 8]:
        #    print(time, time_downsampled, time_ratio)
        # assert time_ratio in [2, 4, 8]
        # V 0 1440 1 0
        # RO 1 0 3 2
        x = x.view(batch_size, self.h * self.channels[-1], time_downsampled)
        x = self.linear(x)
        if DEBUG: print('After first fc {}'.format(x.size()))
        return x

# http://arxiv.org/abs/1904.02619
class TDSBlock(nn.Module):
    def __init__(self,
                 channels,
                 kernel_width,
                 h,
                 dropout
                ):
        super(TDSBlock, self).__init__()

        # https://github.com/facebookresearch/wav2letter/blob/153d6665ab008835560854d5071c106400c1cc21/src/module/TDSBlock.cpp#L26-L29
        # here they have l and l2
        # though in all of the places l2 equals l
        self.h = h
        self.c = channels
        l = self.c * self.h

        self.conv = nn.Sequential(
            Conv2dSamePadding(channels,channels,(kernel_width,1),
                              stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.Linear(l, l),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(l, l),
            # this differs a bit from this
            # https://github.com/facebookresearch/wav2letter/blob/153d6665ab008835560854d5071c106400c1cc21/src/module/TDSBlock.cpp#L41
            # here dropout is applied after reorder + view
            nn.Dropout(dropout),
        )
        # careful here
        # https://pytorch.org/docs/stable/nn.html#layernorm
        # https://github.com/facebookresearch/wav2letter/blob/153d6665ab008835560854d5071c106400c1cc21/src/module/TDSBlock.cpp#L46
        # fc.add(View(af::dim4(-1, h, c, 0)));
        # If a single integer is used, it is treated as a singleton list
        # and this module will normalize over the last dimension
        # which is expected to be of that specific size.
        self.ln_conv = nn.LayerNorm(self.c)
        self.ln_fc = nn.LayerNorm(self.c)

    def forward(self, x):
        # I guess input should look like
        # fc.add(View(af::dim4(-1, h, c, 0)))
        # or (time, h, c, batch) # https://github.com/facebookresearch/wav2letter/blob/master/docs/arch.md#writing-architecture-files
        # https://github.com/facebookresearch/wav2letter/blob/153d6665ab008835560854d5071c106400c1cc21/src/module/TDSBlock.cpp#L40
        # in PyTorch it would be
        # (batch, time, h, c) or should it be (batch, c, time, h) ?
        out = x
        out = self.conv(out) + out
        # Given normalized_shape=[10], expected input with shape [*, 10], but got input of size[2, 10, 120, 81]
        out = self.ln_conv(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        # fc.add(View(af::dim4(-1, l, 1, 0)))
        # fc.add(Reorder(1, 0, 2, 3))
        # (time, h, c, batch) => (time, h*с, 1, batch) => (h*с, time, 1, batch)
        # in Pytorch terms
        # (batch, time, h, c) => (batch, time, h*с, 1) => (batch, h*с, time, 1)
        # or should it be
        # (batch, c, time, h) => (batch, time, h*с, 1) => (batch, h*с, time, 1)
        # or
        # (batch, c, time, h) => (batch, h*с, time, 1) => (batch, time, h*с, 1) ?
        out = out.view(out.size(0), # batch
                       out.size(2), # time
                       self.h * self.c, # h * c
                       ) # .permute(0, 2, 1, 3) do we need this permute?
        # if DEBUG: print(out.size())
        # if DEBUG: print(self.fc(out).size())
        out = self.fc(out) + out
        # fc.add(Reorder(1, 0, 2, 3));
        # fc.add(View(af::dim4(-1, h, c, 0)));
        # (batch, time, h, c) or (batch, c, time, h) ?
        # .permute(0, 2, 1, 3)
        out = out.view(out.size(0), # batch
                       self.c, # c
                       out.size(1), # time
                       self.h, # h
                       )
        # Given normalized_shape=[10], expected input with shape [*, 10], but got input of size[2, 10, 120, 81]
        out = self.ln_fc(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return out


# http://arxiv.org/abs/1904.03288
class Jasper_conv_block(nn.Module):
    def __init__(self,
                 repeat,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 dropout=0,
                 residual=True,
                 bn_momentum=0.1,
                 bn_eps=1e-05,
                 activation_fn=None):
        super(Jasper_conv_block, self).__init__()

        self.bn =nn.ModuleList([nn.BatchNorm1d(num_features=out_channels, eps=bn_eps, momentum=bn_momentum) for i in range(repeat)])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        module_list = []
        if stride==1:
            module_list.append(JasperConv1dSame(self.in_channels, self.out_channels, kernel_size, stride, dilation))
        else:
            module_list.append(nn.Conv1d(self.in_channels, self.out_channels,kernel_size,
                                         stride=stride, dilation=dilation, padding=kernel_size//2))
        for rep in range(repeat-1):
            module_list.append(JasperConv1dSame(self.out_channels, self.out_channels, kernel_size, stride, dilation))

        self.module_list = nn.ModuleList(module_list)

        self.activation_fn = activation_fn()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, res_input=None):
        for i, module in enumerate(self.module_list):
            x = module(x)
            x = self.bn[i](x)
            if (i == (len(self.module_list)-1)) & (res_input is not None):
                x = x + res_input
            x = self.activation_fn(x)
            x = self.dropout(x)
        return x


class JasperConv1dSame(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        out_len = x.shape[2]
        padding = math.ceil(((out_len - 1) * self.stride[0] + self.kernel_size[0] + \
                             (self.kernel_size[0] - 1) * (self.dilation[0] - 1) - out_len))
        if padding > 0:
            x = F.pad(x, (padding//2, padding-padding//2))
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class JasperNet(nn.Module):
    def __init__(self, config=None):
        super(JasperNet, self).__init__()

        self.dense_residual = config['dense_residual']
        block_list = []
        all_skip_convs = []
        all_skip_bns = []
        for i, repeat in enumerate(config['repeats']):

            in_channels = config['input_channels'] if i==0 else config['channels'][i-1]
            out_channels = config['channels'][i]
            kernel_size = config['kernel_sizes'][i]
            stride = config['strides'][i]
            dilation = config['dilations'][i]
            bn_momentum = config['bn_momentum']
            bn_eps = config['bn_eps']
            dropout = config.get('dropout', config['dropouts'][i])
            residual = bool(config['residual'][i])
            activation_fn = config['activation_fn']

            block_list.append(Jasper_conv_block(repeat=repeat, in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, dilation=dilation,
                                         dropout=dropout, residual=residual,
                                         bn_momentum=bn_momentum, bn_eps=bn_eps, activation_fn=activation_fn))

            skip_convs = []
            skip_bns = []
            if residual:
                if self.dense_residual:
                    skip_convs = [nn.Conv1d(i.in_channels, out_channels, 1, bias=False) for i in block_list if i.residual]
                    skip_bns = [nn.BatchNorm1d(out_channels, bn_eps, bn_momentum) for i in block_list if i.residual]
                else:
                    skip_convs.append(nn.Conv1d(in_channels, out_channels, 1, bias=False))
                    skip_bns.append(nn.BatchNorm1d(out_channels, bn_eps, bn_momentum))

            skip_convs = nn.ModuleList(skip_convs)
            skip_bns = nn.ModuleList(skip_bns)

            all_skip_convs.append(skip_convs)
            all_skip_bns.append(skip_bns)

        self.block_list = nn.ModuleList(block_list)
        self.all_skip_convs = nn.ModuleList(all_skip_convs)
        self.all_skip_bns = nn.ModuleList(all_skip_bns)
    def forward(self, input_, return_skips=False):
        residuals = []
        if return_skips:
            skips = []
        x = input_
        for i, block in enumerate(self.block_list):
            res = 0
            if block.residual:
                if self.dense_residual:
                    residuals.append(x)
                else:
                    residuals = [x]
                # assert len(self.all_skip_convs[i]) == len(residuals)
                for skip_conv, skip_bn, residual in zip(self.all_skip_convs[i],
                                                        self.all_skip_bns[i],
                                                        residuals):
                    res += skip_bn(skip_conv(residual))
                x = block(x, res)
                if return_skips:
                    skips.append(x)
            else:
                x = block(x)
        if return_skips:
            return x, skips
        return x


class Jasper_non_repeat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, dropout=0, bn_momentum=0.1, bn_eps=1e-05, activation_fn=None):
        super(Jasper_non_repeat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.BatchNorm1d(num_features=out_channels, eps=bn_eps, momentum=bn_momentum)
        if stride==1:
            self.conv = JasperConv1dSame(self.in_channels, self.out_channels, kernel_size, stride, dilation)
        else:
            self.conv = nn.Conv1d(self.in_channels, self.out_channels,kernel_size,
                                         stride=stride, dilation=dilation, padding=kernel_size//2)
        self.activation_fn = activation_fn()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x


class Jasper_repeat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 repeat_5=False, stride=1, dilation=1, dropout=0, bn_momentum=0.1, bn_eps=1e-05, activation_fn=None):
        super(Jasper_repeat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat_5 = repeat_5

        # main repeats inside block
        repeat_0 = JasperConv1dSame(self.in_channels, self.out_channels, kernel_size, stride, dilation)
        repeat_1 = JasperConv1dSame(self.out_channels, self.out_channels, kernel_size, stride, dilation)
        repeat_2 = JasperConv1dSame(self.out_channels, self.out_channels, kernel_size, stride, dilation)
        if repeat_5:
            repeat_3 = JasperConv1dSame(self.out_channels, self.out_channels, kernel_size, stride, dilation)
            repeat_4 = JasperConv1dSame(self.out_channels, self.out_channels, kernel_size, stride, dilation)

        # bns
        bn_0 = nn.BatchNorm1d(num_features=out_channels, eps=bn_eps, momentum=bn_momentum)
        bn_1 = nn.BatchNorm1d(num_features=out_channels, eps=bn_eps, momentum=bn_momentum)
        bn_2 = nn.BatchNorm1d(num_features=out_channels, eps=bn_eps, momentum=bn_momentum)
        if repeat_5:
            bn_3 = nn.BatchNorm1d(num_features=out_channels, eps=bn_eps, momentum=bn_momentum)
            bn_4 = nn.BatchNorm1d(num_features=out_channels, eps=bn_eps, momentum=bn_momentum)

        self.residual = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.res_bn = nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)

        self.activation_fn = activation_fn()
        self.dropout = nn.Dropout(p=dropout)

        self.rep_0 = nn.Sequential(repeat_0, bn_0, self.activation_fn, self.dropout)
        self.rep_1 = nn.Sequential(repeat_1, bn_1, self.activation_fn, self.dropout)
        self.rep_2 = nn.Sequential(repeat_2, bn_2, self.activation_fn, self.dropout)
        if self.repeat_5:
            self.rep_3 = nn.Sequential(repeat_3, bn_3, self.activation_fn, self.dropout)
            self.rep_4 = nn.Sequential(repeat_4, bn_4, self.activation_fn, self.dropout)

    def forward(self, input_, return_skips=False):
        x = self.rep_0(input_)
        x = self.rep_1(x)
        x = self.rep_2(x)
        if self.repeat_5:
            x = self.rep_3(x)
            x = self.rep_4(x)
        res = self.res_bn(self.residual(input_))
        x = x + res
        x = self.activation_fn(x)
        x = self.dropout(x)
        if return_skips:
            return x, res
        return x, None


class JasperNetEasy(nn.Module):
    def __init__(self, config=None):
        super(JasperNetEasy,self).__init__()
        self.in_channels = config['input_channels']
        self.bn_momentum = config['bn_momentum']
        self.bn_eps = config['bn_eps']
        self.activation_fn = config['activation_fn']
        self.repeats = config['repeats']
        self.num_modules = config['num_modules']

        assert self.num_modules in [5,10]

        assert self.repeats in [3,5]
        self.repeat_5 = False
        if self.repeats == 5:
            self.repeat_5 = True

        self.init = Jasper_non_repeat(self.in_channels, out_channels=256, kernel_size=11, stride=2, dilation=1,
                                      dropout=0.2, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)

        self.block_0 = Jasper_repeat(in_channels=256, out_channels=256, kernel_size=11, repeat_5=self.repeat_5, stride=1, dilation=1,
                                     dropout=0.2, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)
        self.block_1 = Jasper_repeat(in_channels=256, out_channels=384, kernel_size=13, repeat_5=self.repeat_5, stride=1, dilation=1,
                                     dropout=0.2, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)
        self.block_2 = Jasper_repeat(in_channels=384, out_channels=512, kernel_size=17, repeat_5=self.repeat_5, stride=1, dilation=1,
                                     dropout=0.2, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)
        self.block_3 = Jasper_repeat(in_channels=512, out_channels=640, kernel_size=21, repeat_5=self.repeat_5, stride=1, dilation=1,
                                     dropout=0.3, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)
        self.block_4 = Jasper_repeat(in_channels=640, out_channels=768, kernel_size=25, repeat_5=self.repeat_5, stride=1, dilation=1,
                                     dropout=0.3, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)

        if self.num_modules == 10:
            self.block_0_0= Jasper_repeat(in_channels=256, out_channels=256, kernel_size=11, repeat_5=self.repeat_5, stride=1,
                                          dilation=1, dropout=0.2, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
                                          activation_fn=self.activation_fn)
            self.block_1_0 = Jasper_repeat(in_channels=384, out_channels=384, kernel_size=13, repeat_5=self.repeat_5, stride=1,
                                           dilation=1, dropout=0.2, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
                                           activation_fn=self.activation_fn)
            self.block_2_0 = Jasper_repeat(in_channels=512, out_channels=512, kernel_size=17, repeat_5=self.repeat_5, stride=1,
                                           dilation=1, dropout=0.2, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
                                           activation_fn=self.activation_fn)
            self.block_3_0 = Jasper_repeat(in_channels=640, out_channels=640, kernel_size=21, repeat_5=self.repeat_5, stride=1,
                                           dilation=1, dropout=0.3, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
                                           activation_fn=self.activation_fn)
            self.block_4_0 = Jasper_repeat(in_channels=768, out_channels=768, kernel_size=25, repeat_5=self.repeat_5, stride=1, dilation=1,
                                           dropout=0.3, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)

            self.block_0 = nn.Sequential(self.block_0, self.block_0_0)
            self.block_1 = nn.Sequential(self.block_1, self.block_1_0)
            self.block_2 = nn.Sequential(self.block_2, self.block_2_0)
            self.block_3 = nn.Sequential(self.block_3, self.block_3_0)
            self.block_4 = nn.Sequential(self.block_4, self.block_4_0)

        self.out_0 = Jasper_non_repeat(768, out_channels=896, kernel_size=29, stride=1, dilation=2,
                                       dropout=0.4, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)
        self.out_1 = Jasper_non_repeat(896, out_channels=1024, kernel_size=1, stride=1, dilation=1,
                                       dropout=0.4, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps, activation_fn=self.activation_fn)

    def forward(self, x, return_skips=False):
        if return_skips:
            skips = []
        x = self.init(x)
        x, skip1 = self.block_0(x, return_skips)
        assert skip1 == None
        x, skip2 = self.block_1(x, return_skips)
        x, skip3 = self.block_2(x, return_skips)
        x, skip4 = self.block_3(x, return_skips)
        x, skip5 = self.block_4(x, return_skips)
        x = self.out_0(x)
        x = self.out_1(x)
        return x


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
        if DEBUG: print('Padding {}'.format(pad_h, pad_w))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# a hack to use LayerNorm in sequential
# move the normalized dimension to the last dimension
class SeqLayerNormView(nn.Module):
	def __init__(self):
		super(SeqLayerNormView, self).__init__()
	def forward(self, x):
		return x.permute(0, 2, 3, 1) 


# restore the original order
class SeqLayerNormRestore(nn.Module):
	def __init__(self):
		super(SeqLayerNormRestore, self).__init__()
	def forward(self, x):
		return x.permute(0, 3, 1, 2).contiguous()
       

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
