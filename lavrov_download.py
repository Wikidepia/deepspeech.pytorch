import os
import re
import math
import pickle
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
from loguru import logger
from bs4 import BeautifulSoup


logger.add("lavrov_log.log")


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
                             total=math.ceil(total_size//block_size) , unit='MB', unit_scale=True,
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
                                                str(stdout))    
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


def download_lavrov_post(lavrov_post_url):
    try:
        html = get_html(lavrov_post_url)
        soup = BeautifulSoup(html, 'html.parser')
        download_url = HOST+soup.find("a", class_ = "btn btn-single")['href']
        transcript = make_text_plain(soup.find("div", class_ = "text article-content").text)    
        data= {
            'video_url': download_url,
            'transcript': transcript,
        }
        return data
    except Exception as e:
        print(e)
        data= {
            'video_url': '',
            'transcript': '',
        }        
        return data


# 1 - 84
lavrov_pagination = 'http://www.mid.ru/vistupleniya_ministra?p_p_id=101_INSTANCE_MCZ7HQuMdqBY&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-1&p_p_col_pos=1&p_p_col_count=2&_101_INSTANCE_MCZ7HQuMdqBY_delta=30&_101_INSTANCE_MCZ7HQuMdqBY_keywords=&_101_INSTANCE_MCZ7HQuMdqBY_advancedSearch=false&_101_INSTANCE_MCZ7HQuMdqBY_andOperator=true&p_r_p_564233524_resetCur=false&_101_INSTANCE_MCZ7HQuMdqBY_cur={}'
lavrov_post_urls = []

# get all lavrov article urls
msg = 'Getting all urls'    
logger.info(msg)

for i in tqdm(range(0,84)):
    html = get_html(lavrov_pagination.format(i))
    soup = BeautifulSoup(html, 'html.parser')
    block = soup.find("ul", class_ = "anons-list news-anons-list")

    posts = list(block.children)
    for post in posts:
        try:
            lavrov_post_urls.append(post.a['href'])
        except:
            pass
        
pckl(lavrov_post_urls,'lavrov_speech_urls.pickle')
lavrov_post_urls = upkl('lavrov_speech_urls.pickle')

HOST = 'http://www.mid.ru/'
    
data = []

# get video urls and text urls
msg = 'Getting all video urls and transcripts'    
logger.info(msg)

с = 0
for lavrov_post_url in tqdm(lavrov_post_urls[0:5]):
    data.append(download_lavrov_post(lavrov_post_url))
    с += 1
    if c%100 == 0:
        pd.DataFrame(data).to_feather('lavrov_save.feather')
        
df = pd.DataFrame(data)
df['wav_file'] = df['video_url'].apply(lambda x: x.split('/')[-1]+'.wav')
df.to_feather('lavrov_final.feather')    

msg = 'Downloading all videos'    
logger.info(msg)

video_urls = list(df.video_url.values)
wav_file_paths = [_.split('/')[-1]+'.wav' for _ in video_urls]
temp_filenames = ['temp_video_file.mp4'] * len(video_urls)

for (video_url,
     wav_file_path,
     temp_filename) in tqdm(zip(video_urls,
                                wav_file_paths,
                                temp_filenames), total=len(video_urls)):
    try:
        process_one_video(video_url,
                          temp_filename,
                          wav_file_path)
    except Exception as e:
        msg = 'Video {} caused error {}'.format(video_url,str(e))    
        logger.error(msg)
        
msg = 'Done'    
logger.info(msg)        