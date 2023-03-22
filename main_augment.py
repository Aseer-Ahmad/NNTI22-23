from helpers.split_train_test import split_train_test_by_mtdata, split_train_test_by_user, split_train_test_speaker_by_mtdata
from helpers.train_augment import train, test
from models.cnn import CNN1D, CNN2D
# from models.rnn import RNN1D

import torch.nn as nn
from torch.optim import SGD, Adam
import torch
import torchvision.transforms as transforms
import torchaudio.transforms as T

# from helpers.preprocessor import transformMelSpecByTruncate1D, transformMelSpecByTruncate2D
from helpers.augmentations import transformMelSpecByTruncate1D, IdentityTransform, \
                                NoiseTransform,TimeShiftTransform, TimeStretchingTransform,\
                                SpecAugment
import os

# speaker to train on
speaker = 'theo'

# define paths
DATA_PATH = os.path.join(os.getcwd(), 'speech_data')
MDT_PATH  = os.path.join(os.getcwd(), 'SDR_metadata.tsv')

# split_train_test_by_mtdata(DATA_PATH, MDT_PATH)
split_train_test_speaker_by_mtdata(DATA_PATH, MDT_PATH)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
SR     = 8000
batch_size = 32


raw_augment_train = transforms.Compose([
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



model = CNN1D()
optimizer = SGD(model.parameters(), lr = 0.01)
loss  = nn.CrossEntropyLoss()

model = train(model, loss, optimizer, None, device, epochs, raw_augment_train, transform, spec_augment_train, SR, batch_size, speaker=speaker,val=False)


for speaker in os.listdir(os.path.join(os.getcwd(), 'data')):
    TEST_PTH  = os.path.join(os.getcwd(), 'data', speaker,'test')
    test(model, TEST_PTH, loss, raw_augment_test,transform, spec_augment_test, device, SR,speaker)

