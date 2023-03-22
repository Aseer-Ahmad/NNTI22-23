from __future__ import print_function, division
from email.mime import audio
import os
import torch
from torch.utils.data import Dataset
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
        x, sr      = librosa.load(AUDIO_PATH, sr = self.sampling_rate) 
        out        = self.transform(x, sr)
        return out, label 


class CustomAudioDatasetAug(Dataset):
    def __init__(self, audio_dir, sr, raw_augment, transform, spec_augment):
        self.audio_dir   = audio_dir
        self.audio_files = os.listdir(audio_dir)
        self.raw_augment = raw_augment
        self.transform   = transform
        self.spec_augment = spec_augment
        self.sampling_rate = sr

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label      = int(audio_file.split('_')[0])
        AUDIO_PATH = os.path.join(self.audio_dir, audio_file)
        x, sr      = librosa.load(AUDIO_PATH, sr = self.sampling_rate) 
        x          = self.raw_augment(x)
        x          = self.transform(x)
        out        = self.spec_augment(x)
        return out, label 


class CustomAudioDatasetCLR(Dataset):
    def __init__(self, audio_dir, sr, raw_augment, transform, spec_augment):
        self.audio_dir   = audio_dir
        self.audio_files = os.listdir(audio_dir)
        self.raw_augment = raw_augment
        self.transform   = transform
        self.spec_augment = spec_augment
        self.sampling_rate = sr

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label      = int(audio_file.split('_')[0])
        AUDIO_PATH = os.path.join(self.audio_dir, audio_file)
        x, sr      = librosa.load(AUDIO_PATH, sr = self.sampling_rate) 
        x1, x2     = self.raw_augment(x), self.raw_augment(x)
        x1,x2          = self.transform(x1), self.transform(x2)
        out1,out2        = self.spec_augment(x1), self.spec_augment(x2)
        return out1, out2 