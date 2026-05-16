import torch
import torch.nn as nn


class EmotionDLHead(nn.Module):
    """Auxiliary head for label-distribution learning."""

    def __init__(self, feature_dim, num_classes, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features)
