'''
class : preprocessor.py
description : all methods should return a torch vector
'''
from sklearn.preprocessing import maxabs_scale
import torch
import numpy as np
import librosa
from sklearn  import preprocessing

def transformAudioVecByTruncate(audio_vec, sr):
    MAX_SAMPLES = 50000
    signal_len = len(audio_vec)
    audio_arr  = np.zeros(MAX_SAMPLES)

    if signal_len > MAX_SAMPLES: # truncate
        audio_arr = audio_vec[:MAX_SAMPLES]
    else:                        # pad
        audio_arr[:signal_len] = audio_vec[:signal_len]

    # audio_arr = torch.asarray()      
    

def transformAudioVecByInterpolate(audio_vec, sr):
    pass

def transformAudioVecByEmbeddings(audio_vec, sr):
    pass

def transformMelSpecByTruncate(audio_vec, sr):
    FREQ_BAND = 13
    MAX_COMP_ALL_BANDS  = 630 * FREQ_BAND

    scaled_log_mel_features = extract_melspectrogram(audio_vec, sr, FREQ_BAND)




def extract_melspectrogram(signal, sr, num_mels):

    mel_features = librosa.feature.melspectrogram(y=signal,
        sr=sr,
        n_fft=200, # with sampling rate = 8000, this corresponds to 25 ms
        hop_length=80, # with sampling rate = 8000, this corresponds to 10 ms
        n_mels=num_mels, # number of frequency bins, use either 13 or 39
        fmin=50, # min frequency threshold
        fmax=4000 # max frequency threshold, set to SAMPLING_RATE/2
    )
    
    # for numerical stability added this line
    mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

    # 20 * log10 to convert to log scale
    log_mel_features = 20*np.log10(mel_features)

    # feature scaling
    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)
    
    return scaled_log_mel_features