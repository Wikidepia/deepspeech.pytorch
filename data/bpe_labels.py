import re
import json
import string
import pickle
import pandas as pd
from loguru import logger
import sentencepiece as sp
from functools import reduce
from string import punctuation, printable

russian_alphabet = 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'
punctuation = punctuation.replace('-', '')
printable = printable.replace('\n', '') + russian_alphabet

with open('phonemes_ru.json') as json_file:
    ru_phonemes = json.load(json_file)

# a hack to use BPE with phonemes
# you cannot just apply sp because phonemes have 2-3 letter codes
fake_alphabet = string.ascii_letters + russian_alphabet
phoneme_2_fake = {phoneme: fake_alphabet[i]
                  for i, phoneme in enumerate(ru_phonemes)}
fake_2_phoneme = {v: k for k, v in phoneme_2_fake.items()}

logger.add("bpe_labels.log", enqueue=True)


letter_type_dict = {
    # some random shit
    'ъ': 0, 'ь': 0,'-': 0,
    # vowels
    'а': 2, 'е': 2, 'ё': 2, 'и': 2, 'о': 2, 'у': 2, 'э': 2, 'ю': 2, 'я': 2, 'ы': 2,
    # consonants
    'м': 1, 'н': 1, 'л': 1, 'р': 1,
    'б': 1, 'в': 1, 'г': 1, 'д': 1, 'ж': 1, 'з': 1, 'й': 1, 'к': 1, 'п': 1,
    'с': 1, 'т': 1, 'ф': 1, 'х': 1, 'ц': 1, 'ч': 1, 'ш': 1, 'щ': 1,
}


def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)


class Labels:
    def __init__(self,
                 use_phonemes=False,
                 sp_model='data/spm_train_v05_cleaned_asr_10s_phoneme.model',
                 sp_model_phoneme='data/phoneme_cleaned_v5_spm_1000.model',
                 naive_split_list='data/naive_syllables.pickle',
                 naive_split=False,
                 sp_space_token='▁',
                 s2s_decoder=False,
                 double_supervision=False,
                 omit_spaces=False):
        kwargs = {
            'use_phonemes': use_phonemes,
            'sp_model': sp_model,
            'sp_model_phoneme': sp_model_phoneme,
            'sp_space_token': sp_space_token,
            's2s_decoder': s2s_decoder,
            'naive_split_list': naive_split_list,
            'naive_split': naive_split
        }
        print(kwargs)
        if use_phonemes:
            only_phonemes_kwargs = {**kwargs,
                                    'use_phonemes': True,
                                    'use_phonemes_wo_spaces': True}
            self.labels = _Labels(**only_phonemes_kwargs)
            self.label_list = self.labels.label_list
        elif (s2s_decoder and not double_supervision) or (not s2s_decoder and not double_supervision):
            # just an ordinary decoder
            kwargs = {**kwargs, 'omit_spaces': omit_spaces}
            self.labels = _Labels(**kwargs)
            self.label_list = self.labels.label_list
        elif not s2s_decoder and double_supervision:
            ctc_kwargs = {**kwargs, 's2s_decoder': False}
            s2s_kwargs = {**kwargs, 's2s_decoder': True}
            self.labels = [_Labels(**ctc_kwargs), _Labels(**s2s_kwargs)]
            self.label_list = self.labels[1].label_list
        elif s2s_decoder and double_supervision:
            raise NotImplementedError('This case should be impossible')

    def parse(self, text):
        if type(self.labels) != list:
            # ordinary case pass through
            return self.labels.parse(text)
        else:
            out = []
            for _labels_ in self.labels:
                out.append(_labels_.parse(text))
            return out


class _Labels:
    def __init__(self,
                 use_phonemes=False,
                 naive_split=False,
                 sp_model='data/spm_train_v05_cleaned_asr_10s_phoneme.model',
                 sp_model_phoneme='data/phoneme_cleaned_v5_spm_1000.model',
                 naive_split_list='data/naive_syllables.pickle',
                 sp_space_token='▁',
                 s2s_decoder=False,
                 use_phonemes_wo_spaces=False,
                 omit_spaces=False):
        self.omit_spaces = omit_spaces
        self.use_phonemes = use_phonemes
        self.use_phonemes_wo_spaces = use_phonemes_wo_spaces        
        if use_phonemes_wo_spaces:
            assert self.use_phonemes_wo_spaces == self.use_phonemes
        # will not be used
        # if sp is trained with coverage of 1.0
        # and default params
        self.remove_sp_tokens = ['<unk>', '<s>', '</s>', '2']
        # also reserve sp space token
        # and replace it with ordinary space later
        self.sp_space_token = sp_space_token
        self.s2s_decoder = s2s_decoder
        self.naive_split = naive_split

        assert self.naive_split + self.s2s_decoder < 2

        if self.naive_split:
            pieces = upkl(naive_split_list)

            sp_transcript = [naive_syllable_split(word) for word in 'пушистый рыжий котик'.split(' ')]
            sp_transcript = reduce(lambda a, b: a+[' ']+b,
                                   sp_transcript)
            print('Naive syllables loaded, {} tokens'.format(len(pieces)))
            print('Test naive syllable encoding {}'.format(sp_transcript))
        else:
            self.spm = sp.SentencePieceProcessor()
            if self.use_phonemes or self.use_phonemes_wo_spaces:
                self.spm.Load(sp_model_phoneme)
            else:
                self.spm.Load(sp_model)
            sp_tokens = self.spm.get_piece_size()

            if not self.use_phonemes:
                print('Sentencepiece model loaded, {} tokens'.format(sp_tokens))
                print('Test encoding of the sp model {}'.format(self.spm.encode_as_pieces('пушистый рыжий котик')))

            pieces = pd.DataFrame([{'piece_id': i,
                                    'piece_str': self.spm.IdToPiece(id=i),
                                    'piece_score': self.spm.GetScore(id=i)}
                                for i in range(0, sp_tokens)])
            pieces = pieces[~pieces.piece_str.isin(self.remove_sp_tokens+[self.sp_space_token])]
            pieces = list(pieces.piece_str.values)

        self.label_list = []

        # reserve 0 for CTC blank
        self.labels_map = {"_": 0}
        self.label_list.append("_")

        assert type(pieces) == list
        for key in list(pieces):
            self.labels_map[key] = len(self.labels_map)
            self.label_list.append(key)

        # only for ctc loss
        if not self.s2s_decoder:
            self.labels_map["2"] = len(self.labels_map)
            self.label_list.append("2")

        # both for ctc and attention
        # predict " " as a separate token

        # if not self.use_phonemes_wo_spaces:  - leave space dangling
        # but not
        self.labels_map[" "] = len(self.labels_map)
        self.label_list.append(" ")

        if self.s2s_decoder:
            self.labels_map["["] = len(self.labels_map)  # sos token
            self.label_list.append("[")
            self.labels_map["]"] = len(self.labels_map)  # eos token
            self.label_list.append("]")

        # print(self.labels_map)
        # print(self.label_list)
        assert len(self.labels_map) == len(self.label_list)

        self.labels_map_reverse = {v: k for k, v in self.labels_map.items()}
        if not self.use_phonemes:
            print('Test whole bpe class {}'.format(self.parse('пушистый рыжий котик')))

    def encode_phonemes(self, text):
        text = text.replace('\n', '')
        out = []
        words = text.split(' ')
        for i, word in enumerate(words):
            phonemes = word.split('-')
            for phoneme in phonemes:
                if phoneme in phoneme_2_fake:
                    out.append(phoneme_2_fake[phoneme])
                else:
                    print(phoneme, text)
                    raise ValueError('Phoneme not in dict')
            if i < len(words)-1:
                out.append(' ')
        return ''.join(out)

    def parse(self, text):
        if self.use_phonemes:
            text = ''.join([_ for _ in list(text)
                            if _ not in punctuation and _ in printable])
        else:
            text = ''.join([_ for _ in list(text)
                            if _ in russian_alphabet + '- '])
        text = remove_extra_spaces(text).strip()

        if not self.use_phonemes:
            text = text.lower()

        if self.naive_split:
            # a bit more cleaning
            # do not forget ё this time
            text = text.replace('\n','').replace('*','').replace('ё', 'е')
        else:
            text = text.replace('2',' ').replace('*',' ').replace('ё', 'е').strip()

        transcript = []

        if self.use_phonemes:
            # to fake alphabet
            fake_encoded = self.encode_phonemes(text)
            sp_transcript = self.spm.encode_as_pieces(fake_encoded)

            (check,
             original_trimmed,
             back_decoded) = self.check_phoneme_bpe_encoding(text,
                                                             sp_transcript)
            # strict
            assert check
        elif self.naive_split:
            sp_transcript = [naive_syllable_split(word) for word in text.split(' ')]
            sp_transcript = reduce(lambda a, b: a+[' ']+b,
                                   sp_transcript)
        else:
            sp_transcript = self.spm.encode_as_pieces(text)

        # print(sp_transcript)
        if self.s2s_decoder:
            transcript.append(self.labels_map['['])

        for i, token in enumerate(sp_transcript):
            try:
                code = None
                if token in self.remove_sp_tokens:
                    pass
                elif token == self.sp_space_token:
                    # replace spm space token with our space
                    # or just omit the space
                    if not self.use_phonemes_wo_spaces and not self.omit_spaces:
                        code = self.labels_map[' ']
                else:
                    code = self.labels_map[token]
                    if not self.s2s_decoder:
                        if transcript and transcript[-1] == code:
                            code = self.labels_map['2']  # double char for ctc
                if code:
                    transcript.append(code)
            except Exception as e:
                msg = 'Error {} with text {}, transcript {}'.format(str(e), text, sp_transcript)
                logger.error(msg, enqueue=True)

        if self.s2s_decoder:
            transcript.append(self.labels_map[']'])

        # print(transcript, self.render_transcript(transcript))
        return transcript

    def check_phoneme_bpe_encoding(self,
                                   text,
                                   sp_transcript):
        out = []
        for word in sp_transcript:
            if word == self.sp_space_token:
                out.append(' ')
            else:
                for char in word:
                    out.append(fake_2_phoneme[char])

        original_trimmed = str(text.replace('-', '')).strip()
        back_decoded = str(''.join(out)).strip()
        try:
            # convert back and check
            assert original_trimmed == back_decoded
            return True, original_trimmed, back_decoded
        except Exception as e:
            print('Error {} with {}'.format(str(e),
                                            text))
            return False, original_trimmed, back_decoded

    def render_transcript(self, codes):
        if self.use_phonemes:
            raise NotImplementedError('This method is not applicable for phoneme BPE')
        else:
            return ''.join([self.labels_map_reverse[i] for i in codes])


def naive_syllable_split(text,
                         debug=False):
    text = text.lower()

    text_len = len(text)
    son_list = [letter_type_dict[char] for char in text]

    splits = []

    is_syl = False

    if debug: print(son_list)

    for i in range(text_len - 1):
        if son_list[i] == 0:
            splits.append(i)
            is_syl = False
        elif son_list[i] in [1,2] and son_list[i+1] == 0:
            splits.append(i)
            is_syl = False
        elif not is_syl and son_list[i] != son_list[i+1]:
            is_syl = True
        elif is_syl and son_list[i] != son_list[i-1]:
            splits.append(i)
            is_syl = False
        elif not is_syl and son_list[i] == son_list[i+1]:
            splits.append(i)
            is_syl = False


        if debug: print(i, is_syl, son_list[i] == son_list[i-1])

    splits = lindexsplit(text, *splits)
    return [''.join(_).replace('-','') for _ in splits]


def lindexsplit(some_list, *args):
    # Checks to see if any extra arguments were passed. If so,
    # prepend the 0th index and append the final index of the
    # passed list. This saves from having to check for the beginning
    # and end of args in the for-loop. Also, increment each value in
    # args to get the desired behavior.
    if args:
        args = (0,) + tuple(data+1 for data in args) + (len(some_list)+1,)

    # For a little more brevity, here is the list comprehension of the following
    # statements:
    #    return [some_list[start:end] for start, end in zip(args, args[1:])]
    my_list = []
    for start, end in zip(args, args[1:]):
        my_list.append(some_list[start:end])

    if len(my_list) == 0:
        return [some_list]
    return my_list


def pckl(obj,path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def upkl(path):
    with open(path, 'rb') as handle:
        _ = pickle.load(handle)
    return _