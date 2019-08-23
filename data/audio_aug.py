import random
import librosa
import torchaudio  # used just as a proper wrapper around sox
import numpy as np
import subprocess
import pyrubberband as pyrb
from tempfile import NamedTemporaryFile
from data.audio_loader import load_audio_norm
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
from data.audio_loader import (float2int,
                               int2float)

use_shm = True
if use_shm:
    tempfile_dir = '/dev/shm'
else:
    tempfile_dir = None

"""
Librosa pitch and speed augs are low quality
TorchAudio has a proper SoxEffect wrapper, but it will not be supported
Also it only reads files on disk
Initially using sox for all files via subprocess caused problems
For the time being the most optimal strategy is just to try using pyrb
"""

class ChangeAudioSpeed:
    def __init__(self, limit=0.3, prob=0.5,
                 max_duration=10, sr=16000,
                 use_pyrb=False):
        self.limit = limit
        self.prob = prob
        self.max_duration = max_duration * sr
        self.use_pyrb = use_pyrb

    def __call__(self, wav=None,
                 sr=None):
        assert len(wav.shape)==1
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)
            if self.use_pyrb:
                _wav = pyrb.time_stretch(wav, sr, alpha)
            else:
                _wav = librosa.effects.time_stretch(wav, alpha)
            if _wav.shape[0] < self.max_duration:
                wav = _wav
        return {'wav':wav,'sr':sr}


class Shift:
    def __init__(self, limit=512, prob=0.5,
                 max_duration=10, sr=16000):
        self.limit = int(limit)
        self.prob = prob
        self.max_duration = max_duration * sr

    def __call__(self, wav=None,
                 sr=None):
        assert len(wav.shape)==1
        if random.random() < self.prob:
            limit = self.limit
            shift = round(random.uniform(0, limit))
            length = wav.shape[0]
            _wav = np.zeros(length+limit)
            _wav[shift:length+shift] = wav
            if _wav.shape[0]<self.max_duration:
                wav = _wav
        return {'wav':wav,'sr':sr}


class AudioDistort:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, wav=None,
                 sr=None):
        # simulate phone call clipping effect
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)
            maxval = np.max(wav)
            dtype = wav.dtype
            wav = clip(alpha * wav, dtype, maxval)
        return {'wav':wav,'sr':sr}


class PitchShift:
    def __init__(self, limit=5, prob=0.5,
                 use_pyrb=False):
        self.limit = abs(limit)
        self.prob = prob
        self.use_pyrb = use_pyrb


    def __call__(self, wav=None,
                 sr=22050):
        assert len(wav.shape)==1
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(-1, 1)
            if self.use_pyrb:
                wav = pyrb.pitch_shift(wav, sr, alpha)
            else:
                wav = librosa.effects.pitch_shift(wav, sr, n_steps=alpha)
        return {'wav':wav,'sr':sr}


class TorchAudioSoxChain:
    """Using a torchaudio proper C++ wrapper around soxi
    Also requires a file, but looks like it does not spawn processes
    """
    def __init__(self, speed_limit=0.3, prob=0.5,
                 pitch_limit=4, max_duration=10, sr=16000):
        self.speed_limit = speed_limit
        self.pitch_limit = pitch_limit
        self.prob = prob
        self.max_duration = max_duration * sr

    def __call__(self, wav=None, sr=None):
        assert len(wav.shape)==1
        _wav = None
        input_dtype = wav.dtype
        if random.random() < self.prob:
            speed_alpha = 1.0 + self.speed_limit * random.uniform(-1, 1)
            pitch_alpha = self.pitch_limit * random.uniform(-1, 1) * 100 # in cents
            #  https://github.com/carlthome/python-audio-effects/blob/master/pysndfx/dsp.py#L531
            with NamedTemporaryFile(suffix=".wav",
                                    dir=tempfile_dir) as temp_file:
                temp_filename = temp_file.name
                # always feed int16 to sox
                if wav.dtype == np.float32():
                    wav_int = float2int(wav)
                    wav_write(temp_filename,
                              sr,
                              wav_int)
                else:
                    wav_write(temp_filename,
                              sr,
                              wav)

                torchaudio.initialize_sox()
                effects = torchaudio.sox_effects.SoxEffectsChain()
                effects.append_effect_to_chain('pitch', pitch_alpha)
                effects.append_effect_to_chain('tempo', [speed_alpha])
                effects.append_effect_to_chain('rate', sr)
                effects.set_input_file(temp_filename)
                _wav, _sr = effects.sox_build_flow_effects()
                torchaudio.shutdown_sox()
                _wav = _wav.numpy().squeeze()
                assert sr == _sr
                # always float output
                if _wav.dtype == np.int16():
                    _wav = int2float(_wav)

        if _wav is not None:
            return {'wav': _wav,'sr': sr}
        else:
            return {'wav': wav,'sr': sr}


class SoxPhoneCodec:
    """Using a torchaudio proper C++ wrapper around soxi
    Also requires a file, but looks like it does not spawn processes
    """
    def __init__(self, prob=0.5,
                 sox_codec_list=['gsm', 'amr-nb', 'ogg'],
                 sox_sr_list=[8, 16],
                 quality_presets=list(range(1,8))):
        self.prob = prob
        self.sox_codec_list = sox_codec_list
        self.sox_sr_list = sox_sr_list
        self.quality_presets = quality_presets

    def __call__(self, wav=None, sr=None):

        assert len(wav.shape) == 1
        _wav = None

        if random.random() < self.prob:
            codec = random.choice(self.sox_codec_list)
            quality = random.choice(self.quality_presets)
            sox_sr = random.choice(self.sox_sr_list)
            # supress warnings
            if codec in ['amr-nb', 'gsm']:
                sox_sr = 8

            with NamedTemporaryFile(suffix="."+codec,
                                    dir=tempfile_dir) as codec_temp_file:
                with NamedTemporaryFile(suffix=".wav",
                                        dir=tempfile_dir) as wav_temp_file:

                    # save wav to disk
                    # transform using a codec
                    # convert back to wav and read

                    codec_temp_filename = codec_temp_file.name
                    wav_temp_filename = wav_temp_file.name

                    if wav.dtype == np.float32():
                        wav_int = float2int(wav)
                        wav_write(wav_temp_filename,
                                sr,
                                wav_int)
                    else:
                        wav_write(wav_temp_filename,
                                  sr,
                                  wav)                        

                    sox_params = 'sox {} -r {}k -c 1 -C {} -t {} {}'.format(
                        wav_temp_filename,
                        sox_sr,
                        quality,
                        codec,
                        codec_temp_filename
                    )
                    sox_reverse_params = 'sox {} -r {}k -c 1 -e signed-integer -t {} {}'.format(
                        codec_temp_filename,
                        str(int(sr//1000)),
                        'wav',
                        wav_temp_filename
                    )

                    # print(sox_params)
                    # print(sox_reverse_params)

                    _ = subprocess.Popen(sox_params,
                                         shell=True,
                                         stdout=subprocess.PIPE).stdout.read()

                    _ = subprocess.Popen(sox_reverse_params,
                                         shell=True,
                                         stdout=subprocess.PIPE).stdout.read()

                    _sr, _wav = wav_read(wav_temp_file)
                    assert _sr == sr
                    if _wav.dtype == np.int16():
                        _wav = int2float(_wav)

        if _wav is not None:
            return {'wav': _wav, 'sr': sr}
        else:
            return {'wav': wav, 'sr': sr}


class AddNoise:
    def __init__(self, limit=0.2, prob=0.5,
                 noise_samples=[]):
        self.limit = abs(limit)
        self.prob = prob
        self.noise_samples = noise_samples


    def __call__(self, wav=None,
                 sr=None):
        assert len(wav.shape) == 1
        # apply noise 2 times with some probability
        # audio and noise are both normalized
        for i in range(0, 2):
            if random.random() < self.prob:
                if i == 0:
                    _noise = get_stacked_noise(self.noise_samples,
                                               wav=wav,
                                               sr=sr)
                    # noise still should be longer than audio
                    if _noise.shape[0] < wav.shape[0]:
                        return {'wav': wav, 'sr':sr}
                else:
                    gaussian_noise = np.random.normal(0, 1, wav.shape[0]*2)
                    _noise = gaussian_noise
                alpha = self.limit * random.uniform(0, 1)
                pos = random.randint(0, _noise.shape[0]-wav.shape[0])
                wav = (wav + alpha * _noise[pos:pos+wav.shape[0]])/(1+alpha)

        return {'wav':wav, 'sr':sr}


class AddEcho:
    def __init__(self,
                 max_echos=100,
                 sound_speed=0.33,
                 echo_arrivals_ms=list(range(0, 400, 10)),
                 prob=0.5):
        self.prob = prob
        self.max_echos = max_echos
        self.sound_speed = sound_speed
        self.echo_arrivals_ms = echo_arrivals_ms

    @staticmethod
    def dampen(ms):
        if ms < 50:
            return 0.8
        if ms < 100:
            return 0.5
        return 0.3

    def __call__(self, wav=None,
                 sr=None):
        assert len(wav.shape) == 1
        # apply noise 2 times with some probability
        # audio and noise are both normalized
        _wav = None
        if random.random() < self.prob:
            for i in range(0, self.max_echos):
                if random.random() < self.prob:
                    # noise is audio itself shifted
                    echo_arrival = random.choice(self.echo_arrivals_ms)
                    shift_frames = int(sr * echo_arrival * self.sound_speed / 1000) + 1
                    if shift_frames < (wav.shape[0] - 1):
                        noise = np.zeros_like(wav)
                        noise[shift_frames:] = wav[:-shift_frames]
                        # vary dampening a bit
                        alpha = self.dampen(echo_arrival) * random.uniform(0.5, 1)
                        _wav = (wav + alpha * noise) / (1+alpha)
        if _wav is not None:
            return {'wav': _wav, 'sr': sr}
        else:
            return {'wav': wav, 'sr': sr}


def get_stacked_noise(noise_paths=None,
                      wav=None,
                      sr=16000):
    # randomly read noises to stack them
    # into one noise file longer than our audio
    # 10 files max
    for _ in range(0,10):
        noise_path = random.sample(noise_paths,
                                   k=1)[0]
        _noise, _sample_rate = load_audio_norm(noise_path)
        assert len(_noise.shape)==1
        if _sample_rate!=sr:
            _noise = librosa.resample(_noise, _sample_rate, sr)

        if _>0:
            noise = np.concatenate((noise, _noise),
                                   axis=0)
        else:
            noise = _noise
        assert len(noise.shape)==1
        if noise.shape[0]>wav.shape[0]:
            # we have enough noise already!
            break
        if _==10:
            print('Used 10 noise samples to construct noise')
    return noise


class Compose(object):
    def __init__(self, transforms, p=1.):
        self.transforms = [t for t in transforms if t is not None]
        self.p = p

    def __call__(self, **data):
        if np.random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
        return data


class OneOf(object):
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.p = prob
        transforms_ps = [t.prob for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, **data):
        if np.random.random() < self.p:
            t = np.random.choice(self.transforms, p=self.transforms_ps)
            t.prob = 1.
            data = t(**data)
        return data


class OneOrOther(object):
    def __init__(self, first, second, prob=.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.pprob = 1.
        self.p = prob

    def __call__(self, **data):
        return self.first(**data) if np.random.random() < self.p else self.second(**data)


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)