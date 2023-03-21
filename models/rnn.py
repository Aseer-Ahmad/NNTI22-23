from numpy import float64
import torch.nn as nn
import torch

class RNN1D(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(RNN1D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)
        self.device = device

    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.fc(out[:, -1, :])

        return out