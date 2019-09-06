import random
import cv2
import numpy as np
import math
cv2.setNumThreads(0)


class SCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class SOneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x = t(x)
        return x


class SComposePipelines:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        
        pipeline = random.choice(self.transforms)
        
        for t in pipeline:
            x = t(x)
        return x


class SOneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob:
            x = self.first(x)
        else:
            x = self.second(x)
        return x


class FrequencyMask:
    def __init__(self,
                 bands=2,
                 prob=.25,
                 dropout_width=10):
        assert dropout_width>0
        self.bands = bands
        self.prob = prob
        self.dropout_width = dropout_width

    def __call__(self, spect):
        assert len(spect.shape) == 2
        freqs, frames = spect.shape
        assert self.dropout_width < freqs
        for i in range(0,self.bands):
            # adding several dropot bands
            # becomes progressively harder
            if random.random() < self.prob:
                band_width = random.randint(0,
                                            int(self.dropout_width))
                band_center = random.randint(0, freqs)
                lower_band = max(0, int(band_center - band_width//2))
                higher_band = min(int(band_center + band_width//2),freqs)
                spect[lower_band:higher_band, :] = 0
        return spect
  

class TimeMask:
    def __init__(self,
                 bands=2,
                 prob=.25,
                 dropout_length=50,
                 max_dropout_ratio=.15):
        assert dropout_length > 0
        self.bands = bands
        self.prob = prob
        self.dropout_length = dropout_length
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, spect):
        assert len(spect.shape) == 2        
        freqs, frames = spect.shape
        for i in range(0,self.bands):
            # adding several dropot bands
            # becomes progressively harder
            if random.random() < self.prob:
                band_width = random.randint(0,
                                            int(self.dropout_length))
                # fix for very short files
                # dropout should not be more than some % of audio
                band_width = min(band_width,
                                 int(self.max_dropout_ratio*frames))
                band_center = random.randint(0, frames)
                lower_band = max(0, int(band_center - band_width//2))
                higher_band = min(int(band_center + band_width//2),frames)
                spect[:, lower_band:higher_band] = 0
        return spect

# TODO rewrite to be compatible with spectrograms
class ShiftScale:
    def __init__(self,
                 limit=4,
                 prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        limit = self.limit
        if random.random() < self.prob:
            height, width, channel = img.shape
            assert (width == height)
            size0 = width
            size1 = width + 2 * limit
            size = round(random.uniform(size0, size1))

            dx = round(random.uniform(0, size1 - size))
            dy = round(random.uniform(0, size1 - size))

            y1 = dy
            y2 = y1 + size
            x1 = dx
            x2 = x1 + size

            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
            img = (img1[y1:y2, x1:x2, :] if size == size0
            else cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))
        return img
    
# TODO also adapt this
# https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py#L807