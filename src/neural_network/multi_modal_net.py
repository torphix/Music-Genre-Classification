import torch
import torch.nn as nn
from .resnet import ResNet1d, ResNet2d

class MultiModalNet(nn.Module):
    def __init__(self, out_d=10):
        super().__init__()
        '''
        Uses multiple input datatypes for prediction
        '''
        self.img_emb = ResNet2d(4, 512, 6)
        self.audio_emb = ResNet1d(1, 512, 6)
        self.mel_emb = ResNet1d(128, 512, 6)
        self.out_layer = nn.Sequential(
            nn.Linear(512* 3, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, out_d)
        )

    def forward(self, data):
        img, audio, mel = data['images'], data['audio'], data['mels']
        img_x = self.img_emb(img)
        audio_x = self.audio_emb(audio)
        mel_x = self.mel_emb(mel)
        x = torch.cat([img_x, audio_x, mel_x], dim=1)
        return self.out_layer(x)