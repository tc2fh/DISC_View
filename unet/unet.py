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
    """ A single contracting block with two convolutions + max pooling. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # add batch normalization?
        self.block_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Keep output before pooling for future skip connections
        skip = self.block_down(x)
        pooled = self.pool(skip)
        return skip, pooled

class Contracting(nn.Module):
    """ Encoder that stores feature maps for later concatenation. """
    def __init__(self, in_channels):
        super().__init__()

        self.block1 = ConvBlockDown(in_channels, 112) 
        self.block2 = ConvBlockDown(112, 224) 
        self.block3 = ConvBlockDown(224, 448)

        self.connecting_layer = nn.Sequential(
            nn.Conv2d(448, 448, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(448, 448, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Save feature maps for skip connections
        skip1, x = self.block1(x)
        skip2, x = self.block2(x)
        skip3, x = self.block3(x)

        # Bottleneck layer
        x = self.connecting_layer(x)
        return x, [skip3, skip2, skip1]
    
class ConvBlockUp(nn.Module):
    """ A single expanding block that upsamples and concatenates with corresponding contracting block. """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        self.block_up = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1), # in_channels*2 to account for concatenation
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        # Upsample to match skip connection size
        x = self.upsample(x)

        # Concatenate along channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        x = self.dropout(x)
        x = self.block_up(x)
        return x

class Expanding(nn.Module):
    """ Decoder that gradually reconstructs the segmented mask. """
    def __init__(self):
        super().__init__()

        self.block1 = ConvBlockUp(448, 224)
        self.block2 = ConvBlockUp(224, 112)
        self.block3 = ConvBlockUp(112, 112) 

        self.final_conv = nn.Conv2d(112, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, skip_connections):
        x = self.block1(x, skip_connections[0])
        x = self.block2(x, skip_connections[1])
        x = self.block3(x, skip_connections[2])
        x = self.sigmoid(self.final_conv(x))
        return x
    
class UNet(nn.Module):
    """ Full UNet combining encoder and decoder. """
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = Contracting(in_channels)
        self.decoder = Expanding()

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        return self.decoder(x, skip_connections)
    