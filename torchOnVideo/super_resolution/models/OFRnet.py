import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from .Efficient_ResBlocks import CasResB
from ..utils import optical_flow_warp


class OFRnet(nn.Module):
    def __init__(self, scale, channels):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.scale = scale

        ## RNN part
        self.RNN1 = nn.Sequential(
            nn.Conv2d(4, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            CasResB(3, channels)
        )
        self.RNN2 = nn.Sequential(
            nn.Conv2d(channels, 2, 3, 1, 1, bias=False),
        )

        # SR part
        SR = []
        SR.append(CasResB(3, channels))
        if self.scale == 4:
            SR.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(2))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
            SR.append(nn.Conv2d(64, 64 * 4, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(2))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
        elif self.scale == 3:
            SR.append(nn.Conv2d(channels, 64 * 9, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(3))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
        elif self.scale == 2:
            SR.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(2))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
        SR.append(nn.Conv2d(64, 2, 3, 1, 1, bias=False))

        self.SR = nn.Sequential(*SR)

    def __call__(self, x):  # x: b*2*h*w
        # Part 1
        x_L1 = self.pool(x)
        b, c, h, w = x_L1.size()

        # Shardul Change
        # input_L1 = torch.cat((x_L1, torch.zeros(b, 2, h, w).cuda()), 1)
        input_L1 = torch.cat((x_L1, torch.zeros(b, 2, h, w)), 1)
        optical_flow_L1 = self.RNN2(self.RNN1(input_L1))
        optical_flow_L1_upscaled = F.interpolate(optical_flow_L1, scale_factor=2, mode='bilinear',
                                                 align_corners=False) * 2

        # Part 2
        x_L2 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L1_upscaled)
        input_L2 = torch.cat((x_L2, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L1_upscaled), 1)
        optical_flow_L2 = self.RNN2(self.RNN1(input_L2)) + optical_flow_L1_upscaled

        # Part 3
        x_L3 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L2)
        input_L3 = torch.cat((x_L3, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L2), 1)
        optical_flow_L3 = self.SR(self.RNN1(input_L3)) + \
                          F.interpolate(optical_flow_L2, scale_factor=self.scale, mode='bilinear',
                                        align_corners=False) * self.scale
        return optical_flow_L1, optical_flow_L2, optical_flow_L3
