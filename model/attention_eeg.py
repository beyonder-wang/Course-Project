"""Attention-enhanced EEG classification models.

Building blocks:
  - SEBlock: Squeeze-and-Excitation channel attention (learnable)
  - SimAM: parameter-free 3D attention based on neuroscience theory
  - SpatialAttention: spatial attention via conv

Models:
  - EEGNet_SE:  EEGNet + SE blocks after each conv block
  - EEGNet_SimAM: EEGNet + SimAM attention (no extra params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Args:
        channels: number of input channels
        reduction: reduction ratio for bottleneck (default 4)
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class SimAM(nn.Module):
    """Parameter-free attention module based on SimAM (Yang et al., 2021).

    Computes an energy function per neuron: low-energy neurons are more
    important. Zero additional parameters.

    Works on (B, C, H, W) or (B, C, T) after reshape.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Apply SimAM attention.

        Args:
            x: (B, C, H, W) — for EEG, use (B, 1, C, T) or adapt

        Returns:
            attention-weighted tensor, same shape
        """
        n = x.shape[2] * x.shape[3] - 1
        if n <= 0:
            return x

        # Per-channel mean
        mu = x.mean(dim=[2, 3], keepdim=True)
        # (x - mu)^2
        square = (x - mu) ** 2
        # Per-channel variance
        var = square.sum(dim=[2, 3], keepdim=True) / n + 1e-4

        # Energy: (x - mu)^2 / (4 * (var + λ)) + 0.5 * log(2π*var)
        # Simplified: attention = sigmoid(-energy)
        energy_inv = 4.0 * (var + 1e-4) / (square + 1e-4) + 2.0 * torch.log(var + 1e-4)
        energy_inv = energy_inv.clamp(min=-1e4, max=1e4)

        # Sigmoid gating
        attn = torch.sigmoid(energy_inv)
        return x * attn


class SpatialAttention1D(nn.Module):
    """1D spatial attention for EEG: learn which channels to focus on.

    Args:
        kernel_size: conv kernel for generating attention map
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        avg_out = x.mean(dim=1, keepdim=True)  # (B, 1, T)
        max_out, _ = x.max(dim=1, keepdim=True)  # (B, 1, T)
        y = torch.cat([avg_out, max_out], dim=1)  # (B, 2, T)
        y = self.conv(y)  # (B, 1, T)
        return x * self.sigmoid(y)


# ---------------------------------------------------------------------------
# Attention-enhanced EEG models
# ---------------------------------------------------------------------------

class EEGNet_SE(nn.Module):
    """EEGNet with Squeeze-and-Excitation attention after each conv block.

    Args: same as EEGNet
    """

    def __init__(self, chans, num_classes, time_point=200, f1=8, d=2,
                 pk1=4, pk2=8, dp=0.5, max_norm1=1, norm=None):
        super().__init__()
        f2 = f1 * d
        if norm is None:
            norm = nn.Identity()
        self.norm = norm

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.se1 = SEBlock(f1)

        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1), stride=pk1),
            nn.Dropout(dp),
        )
        self.se2 = SEBlock(d * f1)

        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding=(0, 8)),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2), stride=pk2),
            nn.Dropout(dp),
        )
        self.se3 = SEBlock(f2)

        self._apply_max_norm(self.block2[0], max_norm1)
        self.embed_dim = f2 * ((time_point // pk1) // pk2)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.se1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.block3(x)
        x = self.se3(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class EEGNet_SimAM(nn.Module):
    """EEGNet with SimAM (parameter-free) attention after each conv block.

    Args: same as EEGNet
    """

    def __init__(self, chans, num_classes, time_point=200, f1=8, d=2,
                 pk1=4, pk2=8, dp=0.5, max_norm1=1, norm=None):
        super().__init__()
        f2 = f1 * d
        if norm is None:
            norm = nn.Identity()
        self.norm = norm

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.simam1 = SimAM()

        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1), stride=pk1),
            nn.Dropout(dp),
        )
        self.simam2 = SimAM()

        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding=(0, 8)),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2), stride=pk2),
            nn.Dropout(dp),
        )
        self.simam3 = SimAM()

        self._apply_max_norm(self.block2[0], max_norm1)
        self.embed_dim = f2 * ((time_point // pk1) // pk2)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.simam1(x)
        x = self.block2(x)
        x = self.simam2(x)
        x = self.block3(x)
        x = self.simam3(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class EEGNet_SimAM_SE(nn.Module):
    """EEGNet with both SimAM + SE attention (strongest attention variant)."""

    def __init__(self, chans, num_classes, time_point=200, f1=8, d=2,
                 pk1=4, pk2=8, dp=0.5, max_norm1=1, norm=None):
        super().__init__()
        f2 = f1 * d
        if norm is None:
            norm = nn.Identity()
        self.norm = norm

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.se1 = SEBlock(f1)
        self.simam1 = SimAM()

        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1), stride=pk1),
            nn.Dropout(dp),
        )
        self.se2 = SEBlock(d * f1)
        self.simam2 = SimAM()

        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding=(0, 8)),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2), stride=pk2),
            nn.Dropout(dp),
        )
        self.se3 = SEBlock(f2)
        self.simam3 = SimAM()

        self._apply_max_norm(self.block2[0], max_norm1)
        self.embed_dim = f2 * ((time_point // pk1) // pk2)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.se1(x)
        x = self.simam1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.simam2(x)
        x = self.block3(x)
        x = self.se3(x)
        x = self.simam3(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
