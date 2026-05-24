"""Custom Motor Imagery CNN optimized for BCIC2A.

Design:
  - Three parallel temporal conv branches (different kernel sizes for mu/beta rhythms)
  - Depthwise spatial convolution (per-channel filtering)
  - Global average pooling
  - Minimal parameter count to avoid overfitting
"""

import torch
import torch.nn as nn


class MICNN(nn.Module):
    """Motor Imagery CNN — designed for BCIC2A (22ch, 800pts, 4-class)."""

    def __init__(self, chans=22, num_classes=4, dropout=0.3):
        super().__init__()

        # Multi-scale temporal convolutions
        self.temp1 = nn.Sequential(
            nn.Conv1d(chans, 8, kernel_size=16, padding=8),
            nn.BatchNorm1d(8),
            nn.ELU(),
        )
        self.temp2 = nn.Sequential(
            nn.Conv1d(chans, 8, kernel_size=32, padding=16),
            nn.BatchNorm1d(8),
            nn.ELU(),
        )
        self.temp3 = nn.Sequential(
            nn.Conv1d(chans, 8, kernel_size=64, padding=32),
            nn.BatchNorm1d(8),
            nn.ELU(),
        )

        # Fusion conv: combines multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(24, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Depthwise spatial: per-channel filtering
        self.spatial = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout),
        )

        # Classifier
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: (B, C, T) e.g. (32, 22, 800)
        # Multi-scale temporal
        t1 = self.temp1(x)   # (B, 8, 800)
        t2 = self.temp2(x)   # (B, 8, 800)
        t3 = self.temp3(x)   # (B, 8, 800)

        # Concatenate along channel dim
        f = torch.cat([t1, t2, t3], dim=1)  # (B, 24, 800)

        # Fusion + spatial + pooling
        f = self.fusion(f)     # (B, 16, 800)
        f = self.spatial(f)    # (B, 16, 1)

        # Classify
        f = f.flatten(start_dim=1)
        return self.classifier(f)
