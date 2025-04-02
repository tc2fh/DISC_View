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

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        self.block_up = nn.Sequential(
            # in_channels*2 to account for concatenation
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1),
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

        # final layer should have 2 channels (od and oc masks)
        self.final_conv = nn.Conv2d(112, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip_connections):
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


class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block for Channel Attention """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Pool across spatial dims
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)  # Global pooling
        y = self.fc(y).view(b, c, 1, 1)  # Fully connected layers
        return x * y.expand_as(x)  # Re-weight feature maps


class AttentionGate(nn.Module):
    """ Attention Gate for Spatial Feature Selection in Skip Connections """

    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # **Fix: Ensure W_g expects the correct number of input channels**
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        """
        x: Skip connection feature map (from encoder)
        g: Gating feature map (from decoder)
        """

        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)  # Upsample gating signal
        print(f'x: {x.shape}. g: {g.shape}')
        x1 = self.W_x(x)  # Transform skip connection
        g1 = self.W_g(g)  # **Ensure W_g matches g's channels**
        psi = self.relu(x1 + g1)  # Element-wise addition
        psi = self.sigmoid(self.psi(psi))  # Generate attention map
        
        return x * psi  # Weight input feature map by attention


class UNetWithAttention(nn.Module):
    """ UNet with Attention Gates in Skip Connections and SEBlock in Bottleneck """
    
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        self.encoder = Contracting(in_channels)
        self.decoder = Expanding()

        # **Attention Gates for Skip Connections**
        self.attn_gate1 = AttentionGate(448, 448, 224)  # Largest skip connection
        self.attn_gate2 = AttentionGate(224, 224, 112)
        self.attn_gate3 = AttentionGate(112, 112, 56)   # Smallest skip connection

        # **Squeeze-and-Excitation Block in Bottleneck**
        self.se_block = SEBlock(448)  

    def forward(self, x):
        # **Encoder Forward Pass**
        x, skip_connections = self.encoder(x)

        # **Apply SEBlock in the Bottleneck**
        x = self.se_block(x)

        # **Apply Attention Gates to Skip Connections**
        print(f"skip_connections[0]: {skip_connections[0].shape}. x: {x.shape}")
        skip_connections[0] = self.attn_gate1(skip_connections[0], x)
        print(f"skip_connections[1]: {skip_connections[1].shape}. x: {x.shape}")
        skip_connections[1] = self.attn_gate2(skip_connections[1], x)
        print(f"skip_connections[2]: {skip_connections[2].shape}. x: {x.shape}")
        skip_connections[2] = self.attn_gate3(skip_connections[2], x)

        # **Decoder Forward Pass**
        x = self.decoder.block1(x, skip_connections[0])
        x = self.decoder.block2(x, skip_connections[1])
        x = self.decoder.block3(x, skip_connections[2])

        return self.decoder.final_conv(x)  # Output segmentation mask
