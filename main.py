from helpers.split_train_test import split_train_test_by_mtdata, split_train_test_by_user
from helpers.train import train, test
from helpers.tsne import  plot2D_tsne, plot3D_tsne
from models.cnn import CNN1D, CNN2D
from models.rnn import RNN1D
from models.baseline import BaseLine_Linear1D

import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
import torch
import numpy as np

from helpers.preprocessor import transformMelSpecByTruncate1D, transformMelSpecByTruncate2D, transformMelSpecByMeanPooling1D
import os


# Task I

# speakers = ['nicolas', 'theo' , 'jackson', 'george']

# TEST_PTH  = os.path.join(os.getcwd(), 'data', 'test')
# DATA_PATH = os.path.join(os.getcwd(), 'speech_data')
# MDT_PATH  = os.path.join(os.getcwd(), 'SDR_metadata.tsv')

# split_train_test_by_mtdata(DATA_PATH, MDT_PATH, speakers)

# device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# epochs = 2
# SR     = 8000
# batch_size = 32
# FREQ_BANDS  = 81
# DOWNSAMPLE_N = 15

# model     = BaseLine_Linear1D(FREQ_BANDS * DOWNSAMPLE_N)
# optimizer = SGD(model.parameters(), lr = 0.0009)
# loss      = nn.CrossEntropyLoss()

# model = train(model, loss, optimizer, None, device, epochs, transformMelSpecByMeanPooling1D, SR, batch_size, False)
# metrics_dict, test_x, test_y, preds = test(model, TEST_PTH, loss, transformMelSpecByMeanPooling1D, device, SR)



# TASK II
TEST_PTH  = os.path.join(os.getcwd(), 'data', 'test')
DATA_PATH = os.path.join(os.getcwd(), 'speech_data')
MDT_PATH  = os.path.join(os.getcwd(), 'SDR_metadata.tsv')

# print("creating new data splits")
# split_train_test_by_mtdata(DATA_PATH, MDT_PATH)
# print("splits created")

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
SR     = 8000
batch_size = 32

model = CNN1D()
optimizer = Adam(model.parameters(), lr = 0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
loss  = nn.CrossEntropyLoss()

model = train(model, loss, optimizer, None, device, epochs, transformMelSpecByTruncate1D, SR, batch_size, True)
metrics_dict, test_x, test_y, preds = test(model, TEST_PTH, loss, transformMelSpecByTruncate1D, device, SR)
# plot3D_tsne(test_x, test_y, preds, "CNN1D_epoch_2_3D.jpg")




# # model = CNN2D()
# # optimizer = SGD(model.parameters(), lr = 0.01)
# # loss  = nn.CrossEntropyLoss()

# # model = train(model, loss, optimizer, None, device, epochs, transformMelSpecByTruncate2D, SR, batch_size, False)
# # test(model, TEST_PTH, loss, transformMelSpecByTruncate2D, device, SR)



# # model = RNN1D(input_size=630, hidden_size=32 , num_layers=2 , device=device)
# # optimizer = Adam(model.parameters(), lr = 0.01)
# # loss  = nn.CrossEntropyLoss()

# # model = train(model, loss, optimizer, None, device, epochs, transformMelSpecByTruncate1D, SR, batch_size, False)
# # test(model, TEST_PTH, loss, transformMelSpecByTruncate1D, device, SR)
