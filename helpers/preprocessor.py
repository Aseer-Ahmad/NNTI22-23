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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def transformAudioVecByTruncate(audio_vec, sr):
    MAX_SAMPLES = 50000
    signal_len = len(audio_vec)
    audio_arr  = np.zeros(MAX_SAMPLES)

    if signal_len > MAX_SAMPLES: 
        audio_arr = audio_vec[:MAX_SAMPLES]
    else:                        
        audio_arr[:signal_len] = audio_vec[:signal_len]

    audio_torch = torch.squeeze(torch.from_numpy(audio_arr))

    return audio_torch     
    

def transformAudioVecByInterpolate(audio_vec, sr):
    pass

def transformAudioVecByEmbeddings(audio_vec, sr):
    pass

def transformMelSpecByTruncate(audio_vec, sr):
    FREQ_BANDS = 13
    MAX_COMP_ALL_BANDS  = 630 * FREQ_BANDS
    audio_arr           = np.zeros(MAX_COMP_ALL_BANDS)

    scaled_log_mel_features = extract_melspectrogram(audio_vec, sr, FREQ_BANDS)
    
    scaled_log_mel_features_flat = scaled_log_mel_features.flatten()

    scaled_log_mel_features_len = scaled_log_mel_features_flat.shape[0]

    if scaled_log_mel_features_len > MAX_COMP_ALL_BANDS:
        audio_arr = scaled_log_mel_features_flat[:MAX_COMP_ALL_BANDS]
    else:
        audio_arr[:scaled_log_mel_features_len] = scaled_log_mel_features_flat[:scaled_log_mel_features_len]

    audio_torch = torch.squeeze(torch.from_numpy(audio_arr))

    return audio_torch

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

    log_mel_features = 20*np.log10(mel_features)

    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)
    
    return scaled_log_mel_features


x, sr  = librosa.load('/home/cepheus/My GIT/NNTI 22-23/speech_data/0_george_2.wav', sr = 8000) 
transformMelSpecByTruncate(x, sr)