"""KAN-classifier variants of existing EEG backbone models.

Each class inherits from its parent and only replaces the classifier MLP
with a KANMLP, keeping the feature extractor unchanged. This allows
isolated measurement of KAN's contribution.

Adds to MODEL_DICT:
  EEGLSTM_KAN, EEGGRU_KAN, EEGNet_KAN, EEGMamba_KAN, SimpleMLP_KAN
"""

import torch.nn as nn

from .simple import SimpleMLP
from .eegnet import EEGNet
from .rnn import EEGGRU, EEGLSTM
from .mamba_model import EEGMamba
from .kan import KANMLP


class _KANClassifierMixin:
    """Mixin: replace self.classifier with KANMLP after __init__."""

    def _replace_classifier_with_kan(self, in_dim, num_classes, hidden_dims=(64,),
                                     dropout=0.3):
        layers = [in_dim] + list(hidden_dims) + [num_classes]
        self.classifier = KANMLP(layers, dropout=dropout)


# ---------------------------------------------------------------------------
# SimpleMLP_KAN — KAN replaces all MLP layers
# ---------------------------------------------------------------------------

class SimpleMLP_KAN(nn.Module):
    """Flattened input → KANMLP classifier (no CNN/RNN feature extractor)."""

    def __init__(self, input_channels=20, time_points=200, num_classes=3,
                 hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        self.flatten_dim = input_channels * time_points
        layers = [self.flatten_dim] + list(hidden_dims) + [num_classes]
        self.kan = KANMLP(layers, dropout=dropout)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.kan(x)


# ---------------------------------------------------------------------------
# LSTM / GRU KAN variants
# ---------------------------------------------------------------------------

class EEGLSTM_KAN(EEGLSTM):
    """EEGLSTM with KAN classifier head."""

    def __init__(self, chans=20, hidden_dim=64, num_layers=2, num_classes=3,
                 dropout=0.3, bidirectional=True, grad_clip=1.0):
        super().__init__(chans, hidden_dim, num_layers, num_classes,
                         dropout, bidirectional, grad_clip)
        feat_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = KANMLP([feat_dim, 64, num_classes], dropout=dropout)


class EEGGRU_KAN(EEGGRU):
    """EEGGRU with KAN classifier head."""

    def __init__(self, chans=20, hidden_dim=64, num_layers=2, num_classes=3,
                 dropout=0.3, bidirectional=True, grad_clip=1.0):
        super().__init__(chans, hidden_dim, num_layers, num_classes,
                         dropout, bidirectional, grad_clip)
        feat_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = KANMLP([feat_dim, 64, num_classes], dropout=dropout)


# ---------------------------------------------------------------------------
# EEGNet KAN variant
# ---------------------------------------------------------------------------

class EEGNet_KAN(EEGNet):
    """EEGNet with KAN classifier head."""

    def __init__(self, chans, num_classes, time_point=200, f1=8, d=2,
                 pk1=4, pk2=8, dp=0.5, max_norm1=1, norm=None):
        if norm is None:
            norm = nn.Identity()
        super().__init__(chans, num_classes, time_point, f1, d, pk1, pk2,
                         dp, max_norm1, norm)
        self.classifier = KANMLP([self.embed_dim, num_classes], dropout=dp)


# ---------------------------------------------------------------------------
# Mamba KAN variant
# ---------------------------------------------------------------------------

class EEGMamba_KAN(EEGMamba):
    """EEGMamba with KAN classifier head."""

    def __init__(self, chans=20, d_model=64, num_layers=2, d_state=16,
                 num_classes=3, dropout=0.3):
        super().__init__(chans, d_model, num_layers, d_state, num_classes, dropout)
        self.classifier = KANMLP([d_model, 64, num_classes], dropout=dropout)
