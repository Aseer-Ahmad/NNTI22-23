import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
# from torch.utils.data.dataset import Dataset

from helpers.CustomAudioDataset import CustomAudioDatasetCLR, CustomAudioDatasetAug
from helpers.augmentations import NoiseTransform,TimeShiftTransform,TimeStretchingTransform
from helpers.augmentations import SpecAugment, transformMelSpecByTruncate1D, IdentityTransform
from models.simclr import AudioEncoder,SimCLR
from helpers.metrics import audMetrics
# from simclr import SimCLR

from pathlib import Path
from pytorch_metric_learning import losses

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Declare all the PATHs
BASEDIR = Path(__file__).resolve().parent
METADATAFILE = BASEDIR / 'SDR_metadata.tsv'     # metadata file (.csv)
DATADIR = BASEDIR / 'speech_data'   # original data directory
TRAINDATA = BASEDIR / 'data' / 'test'       # training split data directory
DEVDATA = BASEDIR / 'data' / 'dev'       # training split data directory
TESTDATA = BASEDIR / 'data' / 'test'       # training split data directory
speaker = 'george'
SPEAKER_TRAINDATA = BASEDIR / 'data' / speaker / 'train'
SPEAKER_DEVDATA = BASEDIR / 'data' / speaker / 'DEV'


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
'''
Raw data augmentation
-   https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47
'''

raw_augment_train = transforms.RandomChoice([
                    IdentityTransform(),
                    NoiseTransform(),
                     TimeShiftTransform(),
                     TimeStretchingTransform(),
                    ])

raw_augment_test = transforms.Compose([
                    # NoiseTransform(),
                    #  TimeShiftTransform(),
                    #  TimeStretchingTransform(),
                    ])

transform = transforms.Compose([transformMelSpecByTruncate1D(),])

spec_augment_train = transforms.Compose([
                   SpecAugment()
                    ])

spec_augment_test = transforms.Compose([
                    # NoiseTransform(),
                    #  TimeShiftTransform(),
                    #  TimeStretchingTransform(),
                    ])



# Create audio dataset and data loader
train_dataset = CustomAudioDatasetCLR(audio_dir=audio_data_path, raw_augment=raw_augment_train,transform=transform, spec_augment=spec_augment_train,sr=sr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create encoder and SimCLR model
encoder = AudioEncoder(embedding_dim)
model = SimCLR(encoder=encoder)
model = model.float()
# model = SimCLR(encoder=encoder, projection_dim=64, n_features=10)

# Define loss function and optimizer
# criterion = nn.CosineSimilarity(dim=1)
''' 
Reference (Contrastive Learning Loss: NTXentLoss):
    - https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss 
NTXentLoss Implementation Explaination:
    - https://www.youtube.com/watch?v=_1eKr4rbgRI
NTXentLoss (Example usage)
    - https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
'''
criterion = losses.NTXentLoss(temperature=0.5)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the training loop
num_epochs = 10
# writer = SummaryWriter('logs')
for epoch in range(num_epochs):
    running_loss = 0.0
    for x1, x2 in train_loader: # custom dataloader gives 2 views of the audio data with random (and possibly different) transforms applied
        optimizer.zero_grad()
        # get the embeddings
        h1,z1 = model(x1.float())
        h2,z2 = model(x2.float())

        # create features and labels for the NTXentLoss
        z = torch.cat([z1,z2],dim=0)
        labels = torch.arange(z.shape[0])
        labels[z1.shape[0]:] = labels[:z1.shape[0]]
        loss = criterion(z, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / train_dataset.__len__()
    # writer.add_scalar('Loss/train', epoch_loss, epoch)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
    


# TODO:  write training loop


####################### APPROACH-2 #######################

# # Define the contrastive loss function
# def contrastive_loss(z_i, z_j, temperature=0.5):
#     '''
#     Reference:
#     https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
#     '''
#     # Compute the cosine similarity between the representations of the augmented views
#     sim_ij = F.cosine_similarity(z_i, z_j,dim=-1) / temperature
    
#     # Compute the similarity between the representations of all augmented views
#     # sim_all = torch.cat([sim_ij, sim_ij], dim=0)
    
#     # Create a mask to separate the positive pairs (i.e., two augmented views of the same image) from the negative pairs
#     mask = torch.eye(z_i.shape[0],dtype=torch.bool)
#     sim_ij.masked_fill_(mask, -9e15)
    
#     # Find positive example -> batch_size//2 away from the original example
#     pos_mask = mask.roll(shifts=sim_ij.shape[0]//2, dims=0)
#     # InfoNCE losse
#     nll = -sim_ij[pos_mask] + torch.logsumexp(sim_ij, dim=-1)
#     nll = nll.mean()

#     # # Compute the contrastive loss
#     # numerator = torch.exp(sim_ij)
#     # denominator = torch.exp(sim_all.masked_select(mask)).sum()
#     # loss = -torch.log(numerator / denominator)
    
#     return loss.mean()


# # Train the model using contrastive learning
# for epoch in range(10):
#     running_loss = 0.0
#     for batch in train_loader:
#         # Split the batch into two views of each image
#         x_i, x_j = batch
        
#         # Zero out the gradients
#         optimizer.zero_grad()
        
#         # Forward pass through the model to get the representations of the augmented views
#         h_i,z_i = model(x_i.float())
#         h_j,z_j = model(x_j.float())
        
#         # Compute the contrastive loss
#         loss = contrastive_loss(z_i, z_j)
        
#         # Backward pass and optimization step
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item() * x_i.size(0)
#     epoch_loss = running_loss / train_dataset.__len__()
# #     # writer.add_scalar('Loss/train', epoch_loss, epoch)
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))


###############################################################################
