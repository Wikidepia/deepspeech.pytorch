# mostly a hack, depends on russian_g2p
# use in a separate container with python 3.6, install dawg with conda
# g2p contains TF, therefore unclear how to utilize all CPU cores ...
import os
import sys
import warnings
import pandas as pd 
from tqdm import tqdm
sys.path.insert(0, '../russian_g2p/')
from multiprocessing import Pool
from russian_g2p.Transcription import Transcription
from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme
warnings.filterwarnings('ignore')


def list_multiprocessing(param_lst,
                         func,
                         **kwargs):

    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i,params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result=sorted(result,key=lambda x:x[0])
    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params,**kwargs)


def process_text_file(tup):
    (path, target_path) = tup
    global your_transcriptor

    with open(path, 'r', encoding="utf-8") as file:
        text = file.read().replace('\n', '')
        text = replace_encoded(text)

    try:
        if os.path.isfile(target_path):
            os.remove(target_path)
            print('{} was present, so removed'.format(target_path))

        transcription = ' '.join(['-'.join(_[0]) for _
                                  in your_transcriptor.transcribe(text.split(' '))])

        with open(target_path, "w") as transcription_file:
            print(transcription, file=transcription_file)
        return text, transcription
    except:
        return ''


def read_manifest(manifest_path):
    return pd.read_csv(manifest_path,
                       names=['wav_path','text_path','duration'])


def replace_encoded(text):
    text = text.lower()
    # handle stupid edge case
    while text.startswith('2'):
        text = text[1:]
    if '2' in text:
        text = list(text)
        _text = []
        for i, char in enumerate(text):
            if char == '2':
                try:
                    _text.extend([_text[-1]])
                except:
                    print(''.join(text))
            else:
                _text.extend([char])
        text = ''.join(_text)
    return text

manifests = ['../data/manifests/train_v05_cleaned_phone_calls.csv']

df = pd.concat([read_manifest(_) for _ in manifests])

text_paths = list(df.text_path.values)
phoneme_paths = [_.replace('.txt','_phoneme.txt') for _ in text_paths]

data = zip(text_paths,
           phoneme_paths)

data = list(data)

your_transcriptor = Transcription()

text_tuples = [process_text_file(tup) for tup in tqdm(data)]

proc_df = pd.DataFrame(text_tuples, columns=['text', 'phoneme'])
proc_df.to_feather('text_to_phoneme.feather')