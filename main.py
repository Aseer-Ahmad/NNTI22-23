from helpers.split_train_test import split_train_test_by_mtdata, split_train_test_by_user
from helpers.train import train, test
from models.cnn import CNN1D, CNN2D
from models.rnn import RNN1D

import torch.nn as nn
from torch.optim import SGD, Adam
import torch

from helpers.preprocessor import transformMelSpecByTruncate1D, transformMelSpecByTruncate2D
import os

TEST_PTH  = os.path.join(os.getcwd(), 'data', 'test')
DATA_PATH = os.path.join(os.getcwd(), 'speech_data')
MDT_PATH  = os.path.join(os.getcwd(), 'SDR_metadata.tsv')

split_train_test_by_mtdata(DATA_PATH, MDT_PATH)
# split_train_test_by_user(DATA_PATH)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 4
SR     = 8000
batch_size = 32

model = CNN1D()
optimizer = SGD(model.parameters(), lr = 0.01)
loss  = nn.CrossEntropyLoss()

model = train(model, loss, optimizer, None, device, epochs, transformMelSpecByTruncate1D, SR, batch_size, False)
test(model, TEST_PTH, loss, transformMelSpecByTruncate1D, device, SR)

# model = CNN2D()
# optimizer = SGD(model.parameters(), lr = 0.01)
# loss  = nn.CrossEntropyLoss()

# model = train(model, loss, optimizer, None, device, epochs, transformMelSpecByTruncate2D, SR, batch_size, False)
# test(model, TEST_PTH, loss, transformMelSpecByTruncate2D, device, SR)

# model = RNN1D(input_size=630, hidden_size=32 , num_layers=2 , device=device)
# optimizer = Adam(model.parameters(), lr = 0.01)
# loss  = nn.CrossEntropyLoss()

# model = train(model, loss, optimizer, None, device, epochs, transformMelSpecByTruncate1D, SR, batch_size, False)
# test(model, TEST_PTH, loss, transformMelSpecByTruncate1D, device, SR)
