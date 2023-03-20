from helpers.split_train_test import split_train_test_by_mtdata
from helpers.train import train
from models.cnn import CNN1D, CNN2D

import torch.nn as nn
from torch.optim import SGD, Adam
import torch

# DATA_PATH = os.path.join(os.getcwd(), 'speech_data')
# MDT_PATH  = os.path.join(os.getcwd(), 'SDR_metadata.tsv')

# split_train_test_by_mtdata(DATA_PATH, MDT_PATH)

model = CNN1D()
loss  = nn.CrossEntropyLoss()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = SGD(model.parameters(), lr = 0.01)
epochs = 10

model = train(model, loss, optimizer, None, device, epochs, True)