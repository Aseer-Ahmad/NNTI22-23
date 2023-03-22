from numpy import double
import torch.nn as nn
import torch.nn.functional as F

class BaseLine_Linear1D(nn.Module):
    def __init__(self, input_size):
        super(BaseLine_Linear1D, self).__init__()

        layers = []

        layers.append(nn.Linear( input_size, 128))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(128))

        layers.append(nn.Linear( 128, 64))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(64))


        layers.append(nn.Linear( 64, 32))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(32))

        layers.append(nn.Linear( 32, 10))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        x = self.layers(x)
        
        return x
