import torch
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, residual=False) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
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


class U_encoder(nn.Module):
    def __init__(self, in_chans, out_chans) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_chans, out_chans),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


class U_decoder(nn.Module):
    def __init__(self, in_chans, out_chans) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2),
            ConvBlock(out_chans, out_chans),
            ConvBlock(out_chans, out_chans)
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.net(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        return self.net(x)


class Unet(nn.Module):
    def __init__(self, in_channels, n_features, n_classes) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_features = n_features
        self.n_classes = n_classes

        self.init_conv = ConvBlock(in_channels, n_features, residual=True)

        self.encode1 = U_encoder(n_features, n_features)
        self.encode2 = U_encoder(n_features, 2 * n_features)

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(7),
            nn.GELU()
        )

        self.time_embed1 = EmbedFC(1, 2 * n_features)
        self.time_embed2 = EmbedFC(1, n_features)

        self.contextembed1 = EmbedFC(n_classes, 2 * n_features)
        self.contextembed2 = EmbedFC(n_classes, n_features)

        self.decode0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_features, 2 * n_features,
                               kernel_size=7, stride=7),
            nn.GroupNorm(8, 2 * n_features),
            nn.ReLU(True),
        )
        self.decode1 = U_decoder(4 * n_features, n_features)
        self.decode2 = U_decoder(2 * n_features, n_features)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_features, n_features,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, n_features),
            nn.ReLU(True),
            nn.Conv2d(n_features, self.in_channels,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        enc1 = self.encode1(x)
        enc2 = self.encode2(enc1)
        hidden_vec = self.to_vec(enc2)

        # convert context
        c = nn.functional.one_hot(
            c, num_classes=self.n_classes).type(torch.float)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = -(1 - context_mask)  # flip 01
        # context_mask is for context dropout
        c *= context_mask

        # embed context and time step
        c_emb1 = self.contextembed1(c).reshape(-1, self.n_features * 2, 1, 1)
        t_emb1 = self.time_embed1(t).reshape(-1, self.n_features * 2, 1, 1)
        c_emb2 = self.contextembed2(c).reshape(-1, self.n_features, 1, 1)
        t_emb2 = self.time_embed2(t).reshape(-1, self.n_features, 1, 1)

        dec1 = self.decode0(hidden_vec)
        dec2 = self.decode1(x=c_emb1 * dec1 + t_emb1, skip=enc2)
        dec3 = self.decode2(x=c_emb2 * dec2 + t_emb2, skip=enc1)
        out = self.out(torch.cat((dec3, x), 1))

        return out