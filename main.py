from helpers.split_train_test import split_train_test_by_mtdata
from helpers.train import train
from helpers.pValAnalysis import evaluate_Pvalue
from helpers.preprocessor import transformMelSpecByTruncate1D, transformMelSpecByTruncate2D
from models.cnn import CNN1D, CNN2D

import torch.nn as nn
from torch.optim import SGD, Adam
import torch

import os


# DATA_PATH = os.path.join(os.getcwd(), 'speech_data')
# MDT_PATH  = os.path.join(os.getcwd(), 'SDR_metadata.tsv')

# split_train_test_by_mtdata(DATA_PATH, MDT_PATH)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform1 = transformMelSpecByTruncate1D
transform2 = transformMelSpecByTruncate2D

model1 = CNN1D()
model2 = CNN2D()

opt1 = Adam(model1.parameters(), lr = 0.001) 
opt2 = Adam(model2.parameters(), lr = 0.001) 

loss  = nn.CrossEntropyLoss()
epochs = 10
batch_size = 32
num_runs = 20
SR = 8000

pvalue = evaluate_Pvalue(model1, model2, transform1, transform2, opt1, opt2, 
    loss, SR, batch_size, epochs, num_runs, device)

print(pvalue)