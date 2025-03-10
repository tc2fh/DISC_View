"""
Sources: 
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
https://www.geeksforgeeks.org/u-net-architecture-explained/
https://www.kaggle.com/code/arnavjain1/unet-segmentation-of-oc-od 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # add batch normalization?
        self.block_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.block_down(x)

class Contracting(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.contract_layer = nn.Sequential(
            ConvBlockDown(in_channels, out_channels), # (1, 112)
            ConvBlockDown(out_channels, out_channels*2), # (112, 224)
            ConvBlockDown(out_channels*2, out_channels*4), # (224, 448)
        )

        self.connecting_layer = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.contract_layer(x)
        return self.connecting_layer(x)
    
class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block_up = nn.Sequential(
            # Upsampling layer
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=1),
            # TO DO: Add concatenation layer
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block_up(x)

class Expanding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.expanding_layer = nn.Sequential(
            ConvBlockUp(in_channels, out_channels), # (448, 224)
            ConvBlockUp(out_channels, out_channels // 2), # (224, 112)
            ConvBlockUp(out_channels // 2, out_channels // 2) # (112, 112)
        )

        self.sigmoid_layer = nn.Sequential(
            nn.Conv2d(out_channels // 2, 1, kernel_size=1), # need padding?
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.expanding_layer(x)
        return self.sigmoid_layer(x)
