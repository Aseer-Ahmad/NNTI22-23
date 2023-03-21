from numpy import double
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, in_C = 1):
        super(CNN2D, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_channels=in_C, out_channels=32, kernel_size=3))
        layers.append(nn.BatchNorm2d(32))        
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
        layers.append(nn.BatchNorm2d(64))        
        layers.append(nn.Dropout(p = 0.2))
        layers.append(nn.ReLU())
        
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
        layers.append(nn.BatchNorm2d(128))        
        layers.append(nn.ReLU())
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear( 119808, 64, dtype=float))
        
        layers.append(nn.ReLU())
        layers.append(nn.Linear( 64, 10))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.layers(x)
        return x


class CNN1D(nn.Module):
    def __init__(self, in_C = 13):
        super(CNN1D, self).__init__()

        layers = []
        
        layers.append(nn.Conv1d(in_channels=in_C, out_channels=32, kernel_size=3))
        layers.append(nn.BatchNorm1d(32))        
        layers.append(nn.ReLU())

        layers.append(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3))
        layers.append(nn.BatchNorm1d(64))        
        layers.append(nn.Dropout(p = 0.2))
        layers.append(nn.ReLU())
        
        layers.append(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3))
        layers.append(nn.BatchNorm1d(128))        
        layers.append(nn.ReLU())
        
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear( 39936, 64, dtype=float))
        
        layers.append(nn.ReLU())
        layers.append(nn.Linear( 64, 10))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        x = self.layers(x)
        
        return x


class CNNEnsemble(nn.Module):
    def __init__(self, in_C = 13):
        super(CNNEnsemble, self).__init__()
        
    def forward(self, x):
        pass

