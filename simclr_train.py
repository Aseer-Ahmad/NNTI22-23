import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
# from torch.utils.data.dataset import Dataset

from helpers.CustomAudioDataset import CustomAudioDatasetCLR
from helpers.augmentations import NoiseTransform,TimeShiftTransform,TimeStretchingTransform
from helpers.augmentations import SpecAugment, transformMelSpecByTruncate1D
from models.simclr import AudioEncoder
from simclr import SimCLR

from pathlib import Path
# from pytorch_metric_learning import losses



# Declare all the PATHs
BASEDIR = Path(__file__).resolve().parent
METADATAFILE = BASEDIR / 'SDR_metadata.tsv'     # metadata file (.csv)
DATADIR = BASEDIR / 'speech_data'   # original data directory
TRAINDATA = BASEDIR / 'data' / 'test'       # training split data directory
DEVDATA = BASEDIR / 'data' / 'dev'       # training split data directory
TESTDATA = BASEDIR / 'data' / 'test'       # training split data directory
speaker = 'george'
SPEAKER_TRAINDATA = BASEDIR / 'data' / speaker / 'train'


# Define SimCLR hyperparameters
batch_size = 32
embedding_dim = 32
projection_size = 128
learning_rate = 0.001
num_epochs = 10
sr = 8000
n_mfcc = 39


# Define audio data path and augmentations
audio_data_path = SPEAKER_TRAINDATA

'''
Augmentation using SpecAugment:
This involves Frequency and Time Masking that randomly masks out either vertical (ie. Time Mask) or horizontal (ie. Frequency Mask) bands of information from the Spectrogram.
-   https://pytorch.org/audio/master/tutorials/audio_feature_augmentation_tutorial.html
-   https://towardsdatascience.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706
'''
spec_audio_transforms = transforms.Compose([ SpecAugment(),])

conv_to_spec_transforms = transforms.Compose([
                transformMelSpecByTruncate1D(),
			])

'''
Raw data augmentation
-   https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47
'''
raw_audio_transforms = transforms.Compose([NoiseTransform(),
                                           TimeShiftTransform(),
                                           TimeStretchingTransform()])

# Create audio dataset and data loader
# audio_files = [os.path.join(audio_data_path, f) for f in os.listdir(audio_data_path)]
train_dataset = CustomAudioDatasetCLR(audio_dir=audio_data_path, raw_augment=raw_audio_transforms,transform=conv_to_spec_transforms, spec_augment=spec_audio_transforms,sr=sr)
# train_dataset = CustomAudioDataset(audio_dir=audio_data_path, transform_raw=raw_audio_transforms, transform_conv=conv_to_spec_transforms, transform_spec=spec_audio_transforms, sr=sr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create encoder and SimCLR model
encoder = AudioEncoder(embedding_dim)
# model = SimCLR(encoder, projection_size)
model = SimCLR(encoder=encoder, projection_dim=64, n_features=10)

# Define loss function and optimizer
# criterion = nn.CosineSimilarity(dim=1)
''' 
Reference (Contrastive Learning Loss: NTXentLoss):
    - https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss 
NTXentLoss Implementation Explaination:
    - https://www.youtube.com/watch?v=_1eKr4rbgRI
'''
# criterion = losses.NTXentLoss(temperature=0.5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the training loop
num_epochs = 100
writer = SummaryWriter('logs')
for epoch in range(num_epochs):
    running_loss = 0.0
    for x1, x2 in train_loader:
        optimizer.zero_grad()
        h1,h2,z1,z2 = model(x1,x2)
        loss = criterion(z1, z2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    epoch_loss = running_loss / len(train_dataset.__len__())
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
