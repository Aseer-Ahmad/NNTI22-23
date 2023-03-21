from numpy import double
import torch.nn as nn
import torch.nn.functional as F

class BaseLine_Linear1D(nn.Module):
    def __init__(self, in_C = 1):
        super(BaseLine_Linear1D, self).__init__()

        layers = []
        
        layers.append(nn.Linear( 39936, 64, dtype=float))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear( 64, 10))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        x = self.layers(x)
        
        return x
