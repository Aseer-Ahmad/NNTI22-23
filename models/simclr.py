import torch.nn as nn

'''
Reference (Theory):
    https://amitness.com/2020/03/illustrated-simclr/#:~:text=The%20idea%20of%20SimCLR%20framework,applied%20to%20get%20representations%20z.
    
'''


class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super(AudioEncoder,self).__init__()
        self.embedding_dim = embedding_dim
        self.conv_layers = nn.Sequential(nn.Conv1d(13, 64, 3, padding=1),
                         nn.BatchNorm1d(64),
                         nn.ReLU(),
                         nn.Conv1d(64, 128, 3, padding=1),
                         nn.BatchNorm1d(128),
                         nn.ReLU(),
                         nn.Conv1d(128, 256, 3, padding=1),
                         nn.BatchNorm1d(256),
                         nn.ReLU(),
                         nn.Conv1d(256, 512, 3, padding=1),
                         nn.BatchNorm1d(512),
                         nn.ReLU(),
                         nn.AdaptiveAvgPool1d(1),
                         nn.Flatten(),
                         nn.Linear(512, self.embedding_dim))
    
    def forward(self,x):
        return self.conv_layers(x)
        

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_size=128):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(nn.Linear(encoder.embedding_dim, encoder.embedding_dim),
                                              nn.ReLU(),
                                              nn.Linear(encoder.embedding_dim, projection_size))

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

