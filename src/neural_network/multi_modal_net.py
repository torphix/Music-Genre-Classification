import torch
import torch.nn as nn
from .resnet import ResNet1d, ResNet2d

class MultiModalNet(nn.Module):
    def __init__(self, n_layers, out_d=10):
        super().__init__()
        '''
        Uses multiple input datatypes for prediction
        '''
        self.img_emb = ResNet2d(4, n_layers, 512) 
        self.mel_emb = ResNet1d(128, n_layers, 512) 
        self.out_layer = nn.Sequential(
            nn.Linear(512* 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_d)
        )

    def forward(self, data):
        img, mel = data['images'], data['mels']
        img_x = self.img_emb(img)
        mel_x = self.mel_emb(mel)
        x = torch.cat([img_x, mel_x], dim=1)
        return self.out_layer(x)