"""
Sources: 
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
https://www.geeksforgeeks.org/u-net-architecture-explained/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # add batch normalization?
        self.block_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # bias true or false?
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.block_down(x)

class Contracting(nn.Module):
    pass 
