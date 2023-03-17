from __future__ import print_function, division
from email.mime import audio
import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CustomAudioDataset(Dataset):

    def __init__(self, audio_dir, sr, transform):
        self.audio_dir   = audio_dir
        self.audio_files = os.listdir(audio_dir)
        self.transform   = transform
        self.sampling_rate = sr

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label      = int(audio_file.split('_')[0])
        AUDIO_PATH = os.path.join(self.audio_dir, audio_file)
        x, sr  = librosa.load(AUDIO_PATH, sr = self.sampling_rate) 
        out = self.transform(x)
        return out, label

        
        



