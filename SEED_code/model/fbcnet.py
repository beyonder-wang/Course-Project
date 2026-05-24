"""FBCNet: Filter Bank Convolutional Network for Motor Imagery.

Reference: Ravi et al., "FBCNet: A Multi-view Convolutional Neural Network
for Brain-Computer Interface", 2021.

Architecture:
  1. Multi-band FFT filtering → N frequency bands
  2. Per-band depthwise spatial convolution
  3. Variance pooling over time
  4. FC classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .band_decomposition import BandDecomposition


class FBCNet(nn.Module):
    """Filter Bank Convolutional Network for BCIC2A motor imagery."""

    def __init__(self, chans=22, num_classes=4, fs=200, time_point=800,
                 num_filters=32, band_width=4, dropout=0.3):
        super().__init__()

        # Multi-band decomposition (delta, theta, alpha, beta, gamma)
        self.band_decomp = BandDecomposition(fs=fs)
        num_bands = len(self.band_decomp.band_names)  # 5

        # Per-band spatial convolution: depthwise (separate filters per band)
        # Input: (B, num_bands * C, 1, T) after stacking
        # Conv: groups=num_bands, so each band gets its own filters
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_bands * chans,
                out_channels=num_bands * num_filters,
                kernel_size=(1, 1),
                groups=num_bands,
                bias=False,
            ),
            nn.BatchNorm2d(num_bands * num_filters),
            nn.ELU(),
        )

        # Variance pooling: compute log-variance over time
        # No learnable params — just computes var along time axis

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_bands * num_filters, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape

        # 1. Band decomposition → list of (B, C, T)
        bands = self.band_decomp.forward(x)
        band_list = [bands[name] for name in self.band_decomp.band_names]

        # 2. Stack bands: (B, num_bands * C, T)
        x = torch.cat(band_list, dim=1)  # (B, num_bands*C, T)

        # 3. Add spatial dimension for Conv2d: (B, num_bands*C, 1, T)
        x = x.unsqueeze(dim=2)

        # 4. Spatial conv: (B, num_bands*num_filters, 1, T)
        x = self.spatial_conv(x)

        # 5. Variance pooling: var over time dim
        # x shape: (B, num_bands*num_filters, 1, T)
        x = x.squeeze(dim=2)  # (B, num_bands*num_filters, T)
        # Log variance
        x = torch.log(x.var(dim=-1) + 1e-8)  # (B, num_bands*num_filters)

        # 6. Classifier
        return self.classifier(x)
