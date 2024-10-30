import torch
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn

# Reference : https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
class ResidualConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, residual=False) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.residual = residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU()
        )

    def forward(self, x):
        if self.residual:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if x.shape[1] == x2.shape[1]:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x


class Unet_encoder(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__()
        self.net = nn.Sequential(   ResidualConvBlock(input_channels, output_channels),
                                    nn.MaxPool2d(2))

    def forward(self, x):
        return self.net(x)


class Unet_decoder(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__()
        self.net = nn.Sequential(   nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2),
                                    ResidualConvBlock(output_channels, output_channels),
                                    ResidualConvBlock(output_channels, output_channels))

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.net(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dimension, output_dimension) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.net = nn.Sequential(   nn.Linear(input_dimension, output_dimension),
                                    nn.GELU(),
                                    nn.Linear(output_dimension, output_dimension))

    def forward(self, x):
        x = x.reshape(-1, self.input_dimension)
        return self.net(x)


class Unet(nn.Module):
    def __init__(self, input_channels, num_features, num_classes) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_features = num_features
        self.num_classes = num_classes

        self.init_conv = ResidualConvBlock(input_channels, num_features, residual=True)

        self.encode1 = Unet_encoder(num_features, num_features)
        self.encode2 = Unet_encoder(num_features, 2 * num_features)

        self.to_vec = nn.Sequential( nn.AvgPool2d(7), nn.GELU())

        self.time_embed1 = EmbedFC(1, 2 * num_features)
        self.time_embed2 = EmbedFC(1, num_features)

        self.contextembed1 = EmbedFC(num_classes, 2 * num_features)
        self.contextembed2 = EmbedFC(num_classes, num_features)

        self.decode0 = nn.Sequential(
            nn.ConvTranspose2d(2 * num_features, 2 * num_features, kernel_size=7, stride=7),
            nn.GroupNorm(8, 2 * num_features),
            nn.ReLU(True),
        )
        self.decode1 = Unet_decoder(4 * num_features, num_features)
        self.decode2 = Unet_decoder(2 * num_features, num_features)
        self.out = nn.Sequential(
            nn.Conv2d(2 * num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, num_features),
            nn.ReLU(True),
            nn.Conv2d(num_features, self.input_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        enc1 = self.encode1(x)
        enc2 = self.encode2(enc1)
        hidden_vec = self.to_vec(enc2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.num_classes).type(torch.float)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.num_classes)
        context_mask = -(1 - context_mask)  # flip 01
        # context_mask is for context dropout
        c *= context_mask

        # embed context and time step
        c_emb1  = self.contextembed1(c).reshape(-1, self.num_features * 2, 1, 1)
        t_emb1  = self.time_embed1(t).reshape(-1, self.num_features * 2, 1, 1)
        c_emb2  = self.contextembed2(c).reshape(-1, self.num_features, 1, 1)
        t_emb2  = self.time_embed2(t).reshape(-1, self.num_features, 1, 1)

        dec1    = self.decode0(hidden_vec)
        dec2    = self.decode1(x=c_emb1 * dec1 + t_emb1, skip=enc2)
        dec3    = self.decode2(x=c_emb2 * dec2 + t_emb2, skip=enc1)
        out     = self.out(torch.cat((dec3, x), 1))

        return out