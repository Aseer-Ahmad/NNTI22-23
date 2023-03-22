import numpy as np
import librosa
from sklearn  import preprocessing
import torch


class Compose(object):
    def __init__(self, transforms):
         self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

'''
References:
    NoiseTransform, TimeShiftTransform, TimeStretchingTransform, PitchShiftTransform
    -   https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47
    
    InvertPolarityTransform, RandomGainTransform
    -   https://www.youtube.com/watch?v=umAXGVzVvwQ
'''

class IdentityTransform:
    ''' Does nothing. Just a placeholder transform '''
    def __call__(self, wav: any):
        return wav

class NoiseTransform:
    ''' Adds noise to a raw audio file '''
    def __call__(self, wav: any, factor: float = 0.008):
        wav_n = wav + factor * np.random.normal(0,1,len(wav))
        return wav_n

class TimeShiftTransform:
    ''' Shifts the wave by sample_rate/10 factor. This will move the wave to the right by given factor along time axis. '''
    def __call__(self, wav: any, sr: int = 8000):
        wav_roll = np.roll(wav,int(sr/20))
        return wav_roll
    
class TimeStretchingTransform:
    ''' Takes wave samples and a factor by which to stretch the inputs. '''
    def __call__(self, wav:any, factor:float = 0.2):
        wav_time_stch = librosa.effects.time_stretch(wav,factor)
        return wav_time_stch
    
class PitchShiftTransform:
    '''
    Changes the pitch of sound without affecting its speed.
    Permissible n_steps values = -5 <= x <= 5
    '''
    def __call__(self, wav:any, sr:int = 8000, factor:int = -5):
        wav_pitch_sf = librosa.effects.pitch_shift(wav,sr,n_steps=-5)
        return wav_pitch_sf
    
class InvertPolarityTransform:
    ''' Inverts the polarity of the signal '''
    def __call__(self, wav:any):
        return wav * -1

class RandomGainTransform:
    ''' Adds a gain to the audio signal between a min and max range '''
    def __call__(self, wav:any, min_gain_factor:int=2, max_gain_factor:int=4):
        gain_factor = np.random.uniform(min_gain_factor,max_gain_factor)
        return wav * gain_factor
    

# class TrimMFCCs: 
# 	def __call__(self, batch): 
# 		return batch[1:, :]

# class Standardize:
# 	def __call__(self, batch): 
# 		for sequence in batch: 
# 			sequence -= sequence.mean(axis=0)
# 			sequence /= sequence.std(axis=0)
# 		return batch 

# class MFCC:
#     def __call__(self,signal, sr=8000, num_mels=39):
#         """
#         Given a time series speech signal (.wav), sampling rate (sr), 
#         and the number of mel coefficients, return a mel-scaled 
#         representation of the signal as numpy array.
#         """

#         mel_features = librosa.feature.melspectrogram(y=signal,
#             sr=sr,
#             n_fft=200, # with sampling rate = 8000, this corresponds to 25 ms
#             hop_length=80, # with sampling rate = 8000, this corresponds to 10 ms
#             n_mels=num_mels, # number of frequency bins, use either 13 or 39
#             fmin=50, # min frequency threshold
#             fmax=4000 # max frequency threshold, set to SAMPLING_RATE/2
#         )

#         # for numerical stability added this line
#         mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

#         # 20 * log10 to convert to log scale
#         log_mel_features = 20*np.log10(mel_features)

#         # feature scaling
#         scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)

#         return scaled_log_mel_features

class transformMelSpecByTruncate1D:
    def __call__(self, audio_vec, sr=8000):
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



class SpecAugment:
    '''
    Reference: https://www.kaggle.com/code/davids1992/specaugment-quick-implementation
    '''
    def __call__(self, spec: np.ndarray, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.clone()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape

            freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        
        return spec