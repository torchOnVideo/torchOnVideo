import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from .Efficient_ResBlocks import CasResB

class SRnet(nn.Module):
    def __init__(self, scale, channels, n_frames):
        super(SRnet, self).__init__()
        body = []
        body.append(nn.Conv2d(1 * scale ** 2 * (n_frames-1) + 1, channels, 3, 1, 1, bias=False))
        body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(CasResB(8, channels))
        if scale == 4:
            body.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            body.append(nn.PixelShuffle(2))
            body.append(nn.LeakyReLU(0.1, inplace=True))
            body.append(nn.Conv2d(64, 64 * 4, 1, 1, 0, bias=False))
            body.append(nn.PixelShuffle(2))
            body.append(nn.LeakyReLU(0.1, inplace=True))
        elif scale == 3:
            body.append(nn.Conv2d(channels, 64 * 9, 1, 1, 0, bias=False))
            body.append(nn.PixelShuffle(3))
            body.append(nn.LeakyReLU(0.1, inplace=True))
        elif scale == 2:
            body.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            body.append(nn.PixelShuffle(2))
            body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(nn.Conv2d(64, 1, 3, 1, 1, bias=True))

        self.body = nn.Sequential(*body)

    def __call__(self, x):
        out = self.body(x)
        return out
