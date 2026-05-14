"""ShallowConvNet: Shallow Convolutional Network for EEG decoding.

Reference: Schirrmeister et al., "Deep learning with convolutional neural
networks for EEG decoding and visualization", Human Brain Mapping, 2017.

Architecture mimics FBCSP:
  1. Temporal conv (broad kernel, 25 samples)
  2. Spatial conv (per-channel)
  3. Square activation → avg pooling → log activation
  4. Dense classifier
"""

import torch
import torch.nn as nn


def _glorot_weight_zero_bias(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ShallowConvNet(nn.Module):
    """Shallow FBCSP-inspired ConvNet for motor imagery.

    Args:
        chans: number of EEG channels
        num_classes: number of classes
        time_point: number of time samples
        n_filters_time: number of temporal filters (default 40)
        filter_time_length: temporal kernel size (default 25)
        pool_time_length: pooling kernel size (default 75)
        pool_time_stride: pooling stride (default 15)
        drop_prob: dropout probability (default 0.5)
    """

    def __init__(self, chans=22, num_classes=4, time_point=800,
                 n_filters_time=40, filter_time_length=25,
                 pool_time_length=75, pool_time_stride=15,
                 drop_prob=0.5):
        super().__init__()

        # Input: (B, C, T) → (B, 1, T, C) — temporal dim on height axis
        self.conv_time = nn.Conv2d(1, n_filters_time,
                                   (filter_time_length, 1), bias=True)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_time,
                                   (1, chans), bias=False)
        self.bnorm = nn.BatchNorm2d(n_filters_time)

        self.pool = nn.AvgPool2d((pool_time_length, 1),
                                 stride=(pool_time_stride, 1))
        self.dropout = nn.Dropout(drop_prob)

        # Compute output size after conv + pool for final dense layer
        out_time = time_point - filter_time_length + 1
        out_time = int((out_time - pool_time_length) / pool_time_stride + 1)

        # Dense classifier via Conv2d: maps (B, F, out_time, 1) → (B, num_classes, 1, 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(n_filters_time, num_classes, (out_time, 1)),
            nn.Flatten(start_dim=1),
        )

        _glorot_weight_zero_bias(self)

    def forward(self, x):
        # x: (B, C, T) → (B, 1, T, C)
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, T, C)

        x = self.conv_time(x)   # (B, F, T', C)
        x = self.conv_spat(x)   # (B, F, T', 1)
        x = self.bnorm(x)
        x = torch.square(x)
        x = self.pool(x)        # (B, F, T'', 1)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return self.classifier(x)
