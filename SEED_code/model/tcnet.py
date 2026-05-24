"""EEG-TCNet: EEGNet + Temporal Convolutional Network.

Reference: Ingolfsson et al., "EEG-TCNet: An Accurate Temporal Convolutional
Network for Embedded Motor-Imagery Brain-Machine Interfaces", 2020.

Architecture:
  1. EEGNet frontend (temporal + spatial conv)
  2. TCN blocks (causal dilated convolutions with residual connections)
  3. Global average pooling + classifier
"""

import torch
import torch.nn as nn


class _Chomp1d(nn.Module):
    """Chomp padding for causal convolutions."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class _TCNBlock(nn.Module):
    """Temporal Convolutional Network block with dilated causal conv + residual."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 dropout=0.3):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.relu(out + residual)


class EEGTCNet(nn.Module):
    """EEG-TCNet: EEGNet frontend + TCN backend.

    Args:
        chans: number of EEG channels
        num_classes: number of classes
        time_point: number of time samples
        f1: EEGNet F1 filter count
        d: depth multiplier
        tcn_channels: TCN hidden channels
        tcn_layers: number of TCN blocks
        tcn_kernel: TCN kernel size
        dropout: dropout rate
    """

    def __init__(self, chans=22, num_classes=4, time_point=800,
                 f1=8, d=2, tcn_channels=32, tcn_layers=2,
                 tcn_kernel=3, dropout=0.3):
        super().__init__()

        f2 = f1 * d

        # --- EEGNet frontend ---
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(dropout),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding=(0, 8)),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(dropout),
        )

        # --- TCN backend --- processed temporal length after EEGNet pooling
        # time_point → pool4 → time_point//4 → pool8 → time_point//32
        tcn_input_len = time_point // 32
        tcn_input_dim = f2 * tcn_input_len

        # Project to TCN channel dimension
        self.tcn_proj = nn.Conv1d(f2, tcn_channels, kernel_size=1)

        # TCN blocks with exponentially increasing dilation
        tcn_blocks = []
        for i in range(tcn_layers):
            tcn_blocks.append(
                _TCNBlock(tcn_channels, tcn_channels, tcn_kernel,
                          dilation=2**i, dropout=dropout)
            )
        self.tcn = nn.Sequential(*tcn_blocks)

        # Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(tcn_channels, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(dim=1)  # (B, 1, C, T)

        # EEGNet frontend
        x = self.temporal_conv(x)   # (B, f1, 1, T)
        x = self.spatial_conv(x)    # (B, d*f1, 1, T//4)
        x = self.separable_conv(x)  # (B, f2, 1, T//32)

        # Remove spatial dim and permute for TCN: (B, f2, T//32)
        x = x.squeeze(dim=2)

        # Project to TCN channels
        x = self.tcn_proj(x)  # (B, tcn_channels, T//32)

        # TCN blocks
        x = self.tcn(x)  # (B, tcn_channels, T//32)

        # Global average pooling + classifier
        x = self.gap(x).squeeze(dim=-1)  # (B, tcn_channels)
        return self.classifier(x)
