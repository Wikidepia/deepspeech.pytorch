import os
import re
import math
import time
import pickle
import requests
import subprocess
import random
import pandas as pd
from tqdm import tqdm
from loguru import logger
from bs4 import BeautifulSoup

logger.add("president_log.log")

AUDIO_PARAMS = {
    'sampling_rate':16000,
    'channels':1
}

HOST = 'http://kremlin.ru'



def pckl(obj,path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def upkl(path):
    with open(path, 'rb') as handle:
        _ = pickle.load(handle)
    return _

def get_html(script_url):
    r = requests.get(script_url)
    html_doc = r.text
    return html_doc

def download_audio(download_url,
                   target_path):
    
    r = requests.get(download_url,
                     stream=True)
    if r.status_code == 200:
        with open(target_path,'wb') as f:
            f.write(r.content)
    else:
        msg = "Url {} responded with non 200 status".format(url)
        logger.error(msg)
        raise Exception(msg)
        
def process_one_audio(audio_url,
                      temp_audio_file,
                      wav_path):
       
    ffmpeg_params = "ffmpeg -loglevel error -i '{}' -ac {} -ar {} -vn '{}'".format(
        temp_audio_file,
        AUDIO_PARAMS['channels'],
        AUDIO_PARAMS['sampling_rate'],
        wav_path)
    
    download_audio(audio_url,temp_audio_file)    
        
    stdout = subprocess.Popen(ffmpeg_params,
                              shell=True,
                              stdout=subprocess.PIPE).stdout.read()
    
    msg = 'Decoding url {} stdout \n {}'.format(audio_url,
                                                str(stdout))    
    logger.info(msg)

    if os.path.isfile(temp_audio_file):
        os.remove(temp_audio_file)
    else:
        logger.warn('File {} not found for deletion'.format(temp_audio_file))
    return wav_path        

def get_transcript(transcript_url):
    html = get_html(transcript_url)
    soup = BeautifulSoup(html, 'html.parser')
    article = soup.find("div",itemprop = 'articleBody')
    blocks = article.find_all('p')
    
    article_text = '\n'.join([block.getText().replace(u'\xa0', u' ') for block in blocks])
    return article_text

def add_spaces(string):
    no_space_finds = re.findall(r'[а-яё][А-ЯЁ]', string)
    for no_space_find in no_space_finds:
        string = string.replace(no_space_find,' '.join(no_space_find))
    return string

def remove_meta_below(string):
    return string.split('Опубликован в раздел')[0]

def find_speakers(string):
    res = re.compile(r'[А-ЯЁ]\.[А-ЯЁ][\w\s()]+:|Вопрос[\w\s()]+:')
    speaker_names = [word.split()[0].strip(':') for word in res.findall(string)]
    speaker_texts = [text.replace('\n',' ') for text in res.split(test_trans)[1:]]
    return zip(speaker_names,speaker_texts)

if False:
    audio_pagination = 'http://kremlin.ru/multimedia/audio/page/{}'
    audio_post_urls = []
    for i in tqdm(range(1,353)):
        html = get_html(audio_pagination.format(i))
        soup = BeautifulSoup(html, 'html.parser')
        blocks = soup.find_all("div", class_ = "media__top")
        #print(len(blocks))
        for block in blocks:
            try:
                audio_post_urls.append(block.a['href'])
            except:
                print(i)
                print(block)
        time_wait = random.randint(3,30)
        time.sleep(time_wait)
    pckl(audio_post_urls,'../data/president/audio_post_urls')  
    
if False:
    mp3_urls = []
    for audio_post_url in tqdm(audio_post_urls):
        audio_full_url = 'http://kremlin.ru'+audio_post_url
        html = get_html(audio_full_url)
        soup = BeautifulSoup(html, 'html.parser')
        block = soup.find("ul", class_ = "read__taglist")
        try:
            mp3_urls.append(block.a['href'])
            print(len(mp3_urls))
        except:
            print(i)
            print(block)
            time.sleep(300)
        time_wait = random.randint(1,3)
        time.sleep(time_wait)
    pckl(mp3_urls,'../data/president/mp3_urls')
    
if False:
    audio_post_urls = upkl('../data/president/audio_post_urls')
    mp3_urls=upkl('../data/president/mp3_urls')

    msg = 'Getting all audio urls and transcripts'    
    logger.info(msg)

    texts = []
    counter = 0
    for audio_post_url in tqdm(audio_post_urls):
        texts.append(get_transcript(HOST+audio_post_url[:-7]))
        counter += 1
        if counter%100 == 0:
            pckl(texts,'../data/president/texts')
        time_wait = random.randint(1,3)
        time.sleep(time_wait)
    pckl(texts,'../data/president/texts')    

    df = pd.DataFrame({'post_url':audio_post_urls,'mp3_url':mp3_urls,'transcript':texts})
    df['wav_file'] = df['mp3_url'].apply(lambda x: x.split('/')[-1].replace('.mp3','.wav'))
    df.to_feather('../data/president/final.feather')    
    
df = pd.read_feather('../data/president/final.feather')
msg = 'Downloading all audios'    
logger.info(msg)

mp3_urls = sorted(list(set(df.mp3_url.values)))
wav_file_paths = ['../data/president/wavs/{}'.format(_.split('/')[-1].replace('.mp3','.wav')) for _ in mp3_urls]
temp_filenames = ['temp_file.mp3'] * len(mp3_urls)


for i, (mp3_url,
        wav_file_path,
        temp_filename) in enumerate(tqdm(zip(mp3_urls,
                                              wav_file_paths,
                                              temp_filenames), total=len(mp3_urls))):
        try:
            if os.path.isfile(temp_filename):
                os.remove(temp_filename)
            
            process_one_audio(mp3_url,
                                  temp_filename,
                                  wav_file_path)
            time_wait = random.randint(1,3)
            time.sleep(time_wait)
        except Exception as e:
                print(str(e))
                time.sleep(300)
        
msg = 'Done'    
logger.info(msg)


if False:
    speaker_data = pd.DataFrame()
    for iterrow in tqdm(df.iterrows(),total=len(trans)):
        #print(iterrow[0])
        row =iterrow[1]
        temp_data = pd.DataFrame(list(find_speakers(row['transcript'])),columns=['speaker','speech'])
        temp_data['post_url']=row['post_url']
        temp_data['mp3_url']=row['mp3_url']
        temp_data['wav_file']=row['wav_file']
        speaker_data = speaker_data.append(temp_data)
    speaker_data.reset_index(drop=True).to_feather('../data/president/speaker_df.feather')