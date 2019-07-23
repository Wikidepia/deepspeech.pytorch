import os
import re
import sys
import math
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
from loguru import logger
from bs4 import BeautifulSoup


logger.add("duma_download.log")
logger.remove(0)
logger.add(sys.stderr, level="ERROR")

def download_progress(download_url,
                      target_path,
                      disable_tqdm=False):

    r = requests.get(download_url,
                     stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024
    wrote = 0

    if r.status_code == 200:
        with open(target_path, 'wb') as f:
            for data in tqdm(r.iter_content(block_size),
                             total=math.ceil(total_size//block_size),
                             unit='MB', unit_scale=True,
                             disable=disable_tqdm):
                wrote = wrote + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong")                          
    else:
        msg = "Url {} responded with non 200 status".format(url)
        logger.error(msg)
        raise Exception(msg)


def process_one_video(video_url,
                      temp_video_file,
                      wav_path):
    """Download video, extract wav, delete video
    """
    
    ffmpeg_params = "ffmpeg -i '{}' -ac {} -ar {} -vn '{}'".format(
        temp_video_file,
        AUDIO_PARAMS['channels'],
        AUDIO_PARAMS['sampling_rate'],
        wav_path)

    if DEBUG:
        print(ffmpeg_params)

    if True:
        download_progress(video_url,
                          temp_video_file)    

    stdout = subprocess.Popen(ffmpeg_params,
                              shell=True,
                              stdout=subprocess.PIPE).stdout.read()

    msg = 'Decoding url {} stdout \n {}'.format(video_url,
                                                stdout)    
    logger.info(msg)

    if os.path.isfile(temp_video_file):
        os.remove(temp_video_file)
    else:
        logger.warn('File {} not found for deletion'.format(temp_video_file))
    return wav_path        


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ0-9!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c]')
    return reg.sub('', string)


def clean_name(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ ]')
    return remove_extra_spaces(reg.sub('', string).lower()).strip()


def make_text_plain(text):
    return remove_html_tags(
        remove_extra_spaces(
            remove_non_printed_chars(
                text
            ).replace('\n',' ').replace('  ',' ').replace('\t',' ')
        ).strip()
    )


def check_child(text):
    if text == '': return False
    if text == '/n': return False
    if text == ' ': return False
    return True


def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)


def process_transcript(video_urls,
                       script_url,
                       separators):
    '''Download and process a transcript with the corresponding videos
    '''
    assert 'START' == separators[0]
    assert len(video_urls) == len(separators)

    r = requests.get(script_url)
    html_doc = r.text
    soup = BeautifulSoup(html_doc, 'html.parser')

    utterances = soup.body.find(id="wrap").find(id="selectable-content").contents
    plain_utterances = [make_text_plain(str(utt)) for utt in utterances]

    msg = 'Script {} downloaded and processed'.format(script_url)
    if DEBUG:
        print(msg)
    logger.info(msg)       

    splits = []

    # find who the chairman is
    chairman = 'CHAIRMAN'
    for utt in utterances[0:5]:
        if 'Председатель Государственной Думы' in make_text_plain(str(utt)):
            # just last part of the string
            chairman = clean_name(
                make_text_plain(str(utt)).split('Председатель Государственной Думы')[1]
            )
            break

    # check that merged text contains the separator
    for _, sep in enumerate(separators):       
        sep = make_text_plain(sep)
        if sep=='START':
            splits.append(0)
        else:
            sep_lst = [1 if sep in utt else 0 for utt in plain_utterances]
            # try merging nearby texts
            if sum(sep_lst)==0:
                sep_lst = [1 
                           if sep in make_text_plain(' '.join(plain_utterances[i-WINDOW_SEARCH_WIDTH:i]) + \
                                                     ' ' + utt + ' ' + \
                                                     ' '.join(plain_utterances[i:i+WINDOW_SEARCH_WIDTH]))
                           else 0 
                           for i, utt in enumerate(plain_utterances)]
            if sum(sep_lst)>1:
                logger.warning('More than one split possible for {}'.format(script_url))

            if sum(sep_lst)>0:
                if DEBUG:
                    print(_, sep_lst.index(1),
                          len(plain_utterances) // len(separators) * _,
                          'cat')
                else:
                    msg = 'Sep idx {}, first split {}, equal split {}'.format(
                        _,
                        sep_lst.index(1),
                        len(plain_utterances) // len(separators) * _
                    )
                    logger.info(msg)
                if abs((sep_lst.index(1) - len(plain_utterances) // len(separators) * _)) > len(plain_utterances) * NAIVE_SPLIT_TOLERANCE:
                    msg = 'Script split too assymetric for {}. Falling back to naive split'.format(script_url)
                    logger.error(msg)
                    splits.append(int(len(plain_utterances) / len(separators) * _ ))
                else:
                    splits.append(sep_lst.index(1))            
            else:
                logger.error('Split not properly found for {}. Falling back to naive split'.format(script_url))
                splits.append(int(len(plain_utterances) / len(separators) * _ ))

    splits.append(len(utterances))
    utt_blocks = []
    for i,split in enumerate(splits[:-1]):
        utt_blocks.append(utterances[split:splits[i+1]])

    assert len(utt_blocks) == len(separators)
    
    msg = 'Script {} separated per video'.format(script_url)
    if DEBUG:
        print(msg)
    logger.info(msg)     

    if DEBUG:
        print(splits)

    data_dict_list = []
    assert len(video_urls) == len(utt_blocks)
    
    for utt_block, video_url in zip(utt_blocks,video_urls):
        if DEBUG:
            print(len(utt_block))
        
        video_filename = video_url.split('/')[-1]
        temp_video_file = DATA_PATH.format(video_filename)
        wav_path = temp_video_file + '.wav'
        
        if not os.path.isfile(wav_path):
            msg = 'Dowloading and processing video {}'.format(video_url)
            if DEBUG:
                print(msg)
            logger.info(msg)            
            wav_path = process_one_video(video_url,
                                         temp_video_file,
                                         wav_path)
            msg = 'Video {} downloaded and processed'.format(video_url)
            if DEBUG:
                print(msg)
            logger.info(msg)             
        else:
            msg = 'Video {} already processed, file {} exists'.format(video_url, wav_path)
            if DEBUG:
                print(msg)
            logger.info(msg)  
        

        cur_speech_utt_len = 0
        is_speech_block = False
        current_speaker = ''
        current_speaker_role = ''

        for i, utt in enumerate(utt_block):
            try:
                utt.children
                has_children = True
            except:
                has_children = False

            is_last = ( i == len(utt_block) - 1 ) 

            if has_children:
                children = list(utt)

                # speech block starts with speaker's name and optionally his position 
                has_speaker_name = str(children[0]).startswith('<b>') and str(children[0]).endswith('</b>')
                if len(children)>1:
                    has_speaker_position = str(children[1]).startswith('<i>') and str(children[1]).endswith('</i>')
                else:
                    has_speaker_position = False

                # we found a new speaker block
                if not is_last:
                    if has_speaker_name and not str(utt_block[i+1]).startswith('<b>'):
                        current_speaker = make_text_plain(str(children[0]))
                        cur_speech_utt_len = 0
                        if has_speaker_position:
                            current_speaker_role = make_text_plain(str(children[1]))
                        else:
                            current_speaker_role = ''

                for child in children:
                    text_child = make_text_plain(str(child))
                    if check_child(text_child) and \
                       not (str(child).startswith('<b>') and str(child).endswith('</b>')) and \
                       not (str(child).startswith('<i>') and str(child).endswith('</i>')) and \
                       cur_speech_utt_len < MAX_ONE_SPEAKER_UTT_LEN:
                            cur_speech_utt_len += 1
                            data_dict_list.append({
                                'speaker':clean_name(current_speaker).replace('председательствующий',chairman),
                                'speaker_role':current_speaker_role,
                                'video_url':video_url,
                                'script_url':script_url,
                                'wav_file':wav_path,
                                'text':text_child,
                                'idx_within_block':cur_speech_utt_len,
                            }) 
    return data_dict_list


DATA_PATH = '../../../nvme2/stt/duma_nvme2/{}'
AUDIO_PARAMS = {
    'sampling_rate':16000,
    'channels':1
}
DEBUG = False
MAX_ONE_SPEAKER_UTT_LEN = 10
WINDOW_SEARCH_WIDTH = 5
NAIVE_SPLIT_TOLERANCE = 0.2

df = pd.read_excel('../../../nvme2/stt/duma_nvme2/duma_300.xlsx')
RESULT_DF_FORMAT = '../../../nvme2/stt/duma_nvme2/duma_df_{}.feather'

grby = df.groupby(by='script_url')
for i, (script_url, group) in tqdm(enumerate(grby),
                                  total = len(grby)):
    try:
        _group = group
        separators = list(_group.text_start.values)
        video_urls = list(_group.video_url)

        _ = process_transcript(video_urls,
                               script_url,
                               separators)

        df_data = pd.DataFrame(_)
        df_data.reset_index(drop=True).to_feather(
            RESULT_DF_FORMAT.format(str(script_url.split('/')[-2]))
        )
    except Exception as e:
        msg = 'script_url {} caused and error {}'.format(script_url,
                                                         str(e))
        logger.error(msg)