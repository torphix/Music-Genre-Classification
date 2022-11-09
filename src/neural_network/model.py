import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_d, k_size=3, stride=1, padding=1):
        super().__init__()
        
        # Layer 1 may be used to downsample
        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_d, in_d, k_size, stride, padding),
            nn.BatchNorm1d(in_d),
            nn.ReLU(),
        )
        # No downsampling in layer 2
        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_d, in_d, k_size, 1, 1),
            nn.BatchNorm1d(in_d),
            nn.ReLU(),
        )
      
        # Downsample residual
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_d, in_d, k_size, stride, padding),
                nn.BatchNorm1d(in_d),
                nn.ReLU(),
            )
        else:
            self.downsample = False
    
    def forward(self, x):
        # Create a new tensor in memory
        residual = x.clone()
        x = self.layer_1(x)
        x = self.layer_2(x)
        if self.downsample:
            residual = self.downsample(residual)
        return x + residual


class ResNet(nn.Module):
    def __init__(self, in_d, out_d, n_blocks):
        super().__init__()

        self.blocks = nn.ModuleList()

        self.in_layer = nn.Sequential(
            nn.Conv1d(in_d, 64, 7, 2, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        for i in range(n_blocks-1):
            in_channels = int(64*(2**i))
            out_channels = int(64*(2**(i+1)))
            self.blocks.append(
                nn.Sequential(
                    ResidualBlock(in_channels, k_size=3, stride=2, padding=0),
                    ResidualBlock(in_channels, k_size=3, stride=1, padding=1),
                    nn.Conv1d(in_channels, out_channels, 1, 1, 0) # Cast to next in_d size
                ))

        self.avg_pool = nn.AvgPool1d(kernel_size=39)
        self.out_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.out_layer(x)
        return x
