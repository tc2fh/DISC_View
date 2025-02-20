"""
Sources: 
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
https://www.geeksforgeeks.org/u-net-architecture-explained/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def contracting(self):
        pass

    def connecting(self):
        pass

    def expansive(self):
        pass

    def forward(self):
        pass
