import numpy as np
from scipy.io import wavfile

def load_audio_norm(path, channel=-1):
    # sound, sample_rate = torchaudio.load(path, normalization=lambda x: torch.abs(x).max())
    # sound = sound.numpy().T

    # use scipy, as all files are wavs for now
    # Fix https://github.com/pytorch/audio/issues/14 later
    sample_rate, sound = wavfile.read(path)

    try:
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/abs_max
    except Exception as e:
        print(path)
        raise ValueError('Mow')

    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        elif channel == -1:
            sound = sound.mean(axis=1)  # multiple channels, average
        else:
            sound = sound[:, channel]  # multiple channels, average
    return sound, sample_rate


def int2float(sound):
    _sound = np.copy(sound)
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1/abs_max
    _sound = _sound.squeeze()
    return _sound


def float2int(sound):
    _sound = np.copy(sound)
    _sound *= 16384
    _sound = _sound.astype('int16')
    _sound = _sound.squeeze()
    return _sound
