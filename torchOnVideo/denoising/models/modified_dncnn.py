import torch
import torch.nn as nn


class ModifiedDnCNN(nn.Module):
    def __init__(self, input_channels, output_channels, nlconv_features, nlconv_layers, dnnconv_features, dnnconv_layers):
        super(ModifiedDnCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nlconv_features = nlconv_features
        self.nlconv_layers = nlconv_layers
        self.dnnconv_features = dnnconv_features
        self.dnnconv_layers = dnnconv_layers

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_channels,\
                                out_channels=self.nlconv_features,\
                                kernel_size=1,\
                                padding=0,\
                                bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.nlconv_layers-1):
            layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                    out_channels=self.nlconv_features,\
                                    kernel_size=1,\
                                    padding=0,\
                                    bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Shorter DnCNN
        layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                out_channels=self.dnnconv_features,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.dnnconv_layers-2):
            layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                    out_channels=self.dnnconv_features,\
                                    kernel_size=3,\
                                    padding=1,\
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.dnnconv_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                out_channels=self.output_channels,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        return out
