'''
class : preprocessor.py
description : all methods should return a torch vector
'''
from sklearn.preprocessing import maxabs_scale
import torch
import numpy as np
import librosa
from sklearn  import preprocessing
import librosa
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def transformAudioVecByTruncate1D(audio_vec, sr):
    MAX_SAMPLES = 50000
    signal_len = len(audio_vec)
    audio_arr  = np.zeros(MAX_SAMPLES)

    if signal_len > MAX_SAMPLES: 
        audio_arr = audio_vec[:MAX_SAMPLES]
    else:                        
        audio_arr[:signal_len] = audio_vec[:signal_len]

    audio_torch = torch.reshape(torch.from_numpy(audio_arr), (-1, 1))

    return audio_torch     
    

def transformAudioVecByInterpolate(audio_vec, sr):
    pass

   

def transformMelSpecByTruncate2D(audio_vec, sr):
    '''
    data is transformed to 1 x K x T where K are the frequency components,
    T are the total components sampled. They are then padded or 
    truncated. 2D means the data is then processed in 2-dimensions.
    '''

    FREQ_BANDS = 13
    MAX_COMP_ALL_BANDS  = 630
    
    audio_arr           = np.zeros((FREQ_BANDS, MAX_COMP_ALL_BANDS))

    scaled_log_mel_features = extract_melspectrogram(audio_vec, sr, FREQ_BANDS)
    
    comp_len = scaled_log_mel_features.shape[1]

    if comp_len > MAX_COMP_ALL_BANDS:
        audio_arr = scaled_log_mel_features[:, :MAX_COMP_ALL_BANDS]
    else:
        audio_arr[ :, :comp_len] = scaled_log_mel_features[:, :comp_len]

    audio_torch = torch.reshape(torch.from_numpy(audio_arr), (1, FREQ_BANDS, MAX_COMP_ALL_BANDS))

    return audio_torch

def transformMelSpecByMeanPooling1D(audio_vec, N):

    FREQ_BANDS = 13
    SR         = 8000
    pooled     = np.zeros((FREQ_BANDS, N))
    stride     = 1

    scaled_log_mel_features = extract_melspectrogram(audio_vec, SR, FREQ_BANDS) # K x T
    T = scaled_log_mel_features.shape[1]

    # f = t - n + 1
    kernel_size = T - N + 1

    i = 0

    while i + kernel_size < T : 

        kernel_window = scaled_log_mel_features[ : , i : i + kernel_size ]
        mean_pool     = np.mean(kernel_window, axis= 1)
        pooled[:, i]  = mean_pool
        i += 1

    return pooled




def transformMelSpecByTruncate1D(audio_vec, sr):
    '''
    data is transformed to K x T where K are the frequency components,
    T are the total components sampled. They are then padded or 
    truncated. 1D means the data is then processed in 1-dimension 
    only i.e forward.
    '''

    FREQ_BANDS = 13
    MAX_COMP_ALL_BANDS  = 630

    audio_arr           = np.zeros((FREQ_BANDS, MAX_COMP_ALL_BANDS))

    scaled_log_mel_features = extract_melspectrogram(audio_vec, sr, FREQ_BANDS)
    
    comp_len = scaled_log_mel_features.shape[1]

    if comp_len > MAX_COMP_ALL_BANDS:
        audio_arr = scaled_log_mel_features[:, :MAX_COMP_ALL_BANDS]
    else:
        audio_arr[ :, :comp_len] = scaled_log_mel_features[:, :comp_len]

    audio_torch = torch.from_numpy(audio_arr)

    return audio_torch


def transformMFCCByTruncate1D(audio_vec, sr):
    pass

def extract_MFCC(signal, sr, num_mels):
    pass

def extract_melspectrogram(signal, sr, num_mels):

    mel_features = librosa.feature.melspectrogram(y=signal,
        sr=sr,
        n_fft=200, 
        hop_length=80, 
        n_mels=num_mels, 
        fmin=50, 
        fmax=4000 
    )
    
    mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

    log_mel_features = 20 * np.log10(mel_features)

    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)
    
    return scaled_log_mel_features



PTH = '/home/cepheus/My GIT/NNTI 22-23/speech_data'
for aud in os.listdir(PTH):
    x, sr  = librosa.load( os.path.join(PTH, aud), sr = 8000) 
    t = transformMelSpecByMeanPooling1D(x, 23)
    print(aud + " done ! with shape" + str(t.shape))
    # break