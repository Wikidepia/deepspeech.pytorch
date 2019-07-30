import json
import os
import re
import math
import pickle
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from loguru import logger

logger.add("mosduma.log")

with open('../../../scrapy/tutorial/test.json', 'rb') as f:
    dct = json.load(f)
    
AUDIO_PARAMS = {
    'sampling_rate':16000,
    'channels':2
}
DEBUG = False
MAX_ONE_SPEAKER_UTT_LEN = 10
WINDOW_SEARCH_WIDTH = 5
NAIVE_SPLIT_TOLERANCE = 0.2
HOST = 'https://duma.mos.ru/'

def process_one_video(video_url,
                      temp_video_file,
                      wav_path):
    """Download video, extract wav, delete video
    """
    
        
    ffmpeg_params = "ffmpeg -y -i '{}' -ac {} -ar {} -vn '{}'".format(
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
                                                str(stdout))    
    logger.info(msg)

    if os.path.isfile(temp_video_file):
        os.remove(temp_video_file)
    else:
        logger.warn('File {} not found for deletion'.format(temp_video_file))
    return wav_path


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
        
videos = []
for small_dict in dct:
    if 'video' in small_dict:
        videos.append(small_dict)
        
video_df = pd.DataFrame(videos)
video_df = video_df[video_df['video'].apply(lambda x: '.bin' in x)].reset_index(drop=True)
video_df['hash'] = video_df['video'].apply(lambda x: x.split('/')[-1].split('.')[0])

video_df['wav_path'] = 'mosduma_' + video_df['page'].astype(str) + '_' + \
                        video_df['block'].astype(str) + '_' + video_df['hash'] + '.wav'

video_df.to_feather('video_df_bin.feather')

video_urls = list(video_df['video'].values)
wav_file_paths = list(video_df['wav_path'].values)

for (video_url,
     wav_file_path) in tqdm(zip(video_urls,
                                wav_file_paths), total=len(video_urls)):
    try:
        process_one_video(HOST + video_url,
                          'temp_video/temp_video_file.bin',
                          'wav_folder/' + wav_file_path)
    except Exception as e:
        msg = 'Video {} caused error {}'.format(video_url,str(e))    
        logger.error(msg)
        
msg = 'Done'    
logger.info(msg)