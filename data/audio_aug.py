import random
import librosa
import numpy as np
from data.audio_loader import load_audio_norm


class ChangeAudioSpeed:
    def __init__(self, limit=0.15, prob=0.5,
                 max_duration=10, sr=16000):
        self.limit = limit
        self.prob = prob
        self.max_duration = max_duration * sr

        
    def __call__(self, wav=None,
                 sr=None):
        assert len(wav.shape)==1
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)
            _wav = librosa.effects.time_stretch(wav, alpha)
            if _wav.shape[0]<self.max_duration:
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
    def __init__(self, limit=5, prob=0.5):
        self.limit = abs(limit)
        self.prob = prob

        
    def __call__(self, wav=None,
                 sr=22050):
        assert len(wav.shape)==1
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(-1, 1)
            wav = librosa.effects.pitch_shift(wav, sr, n_steps=alpha)
        return {'wav':wav,'sr':sr}   

    
class AddNoise:
    def __init__(self, limit=0.2, prob=0.5,
                 noise_samples=[]):
        self.limit = abs(limit)
        self.prob = prob
        self.noise_samples = noise_samples

        
    def __call__(self, wav=None,
                 sr=None):
        assert len(wav.shape)==1
        # apply noise 2 times with some probability
        # audio and noise are both normalized
        for i in range(0,2):
            if random.random() < self.prob:
                if i==0:
                    _noise = get_stacked_noise(noise_path=random.sample(self.noise_samples,k=1)[0],
                                               wav=wav,
                                               sr=sr)
                    # noise still should be longer than audio
                    if _noise.shape[0]<wav.shape[0]:
                        return {'wav':wav,'sr':sr}
                else:
                    gaussian_noise = np.random.normal(0, 1, wav.shape[0]*2)
                    _noise = gaussian_noise
                alpha = self.limit * random.uniform(0, 1)
                pos = random.randint(0,_noise.shape[0]-wav.shape[0])
                wav = (wav + alpha * _noise[pos:pos+wav.shape[0]])/(1+alpha)
            
        return {'wav':wav,'sr':sr}    


def get_stacked_noise(noise_path=None,
                      wav=None,
                      sr=16000):
    # randomly read noises to stack them
    # into one noise file longer than our audio
    # 10 files max
    for _ in range(0,10):
        _noise, _sample_rate = load_audio_norm(noise_path)
        assert len(_noise.shape)==1
        if _sample_rate!=sr:
            y = librosa.resample(y, _sample_rate, sample_rate)

        if _>0:
            noise = np.stack((noise, _noise),
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