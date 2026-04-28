"""Multi-band SimCLR encoder with per-band projection heads.

Architecture:
  1. FFT band decomposition → {delta, theta, alpha, beta, gamma}
  2. Stack bands → shared LSTM → per-band features
  3. Per-band projection heads → contrastive loss per band

The LSTM backbone mirrors EEGLSTM.lstm for direct weight transfer
during fine-tuning. Projection heads are discarded after pre-training.

Supports optional MoE layer between LSTM and projection heads.
"""

import torch
import torch.nn as nn

from .band_decomposition import BandDecomposition


class MultiBandSimCLREncoder(nn.Module):
    """Multi-band SimCLR encoder with shared LSTM and per-band projections.

    Args:
        chans: number of input EEG channels
        hidden_dim: LSTM hidden dimension (default 64)
        num_layers: LSTM layers (default 2)
        proj_dim: projection output dimension per band (default 64)
        dropout: dropout rate (default 0.3)
        bidirectional: use bidirectional LSTM (default True)
        bands: dict {name: (low_hz, high_hz)}. Default: 5 standard bands.
    """

    DEFAULT_BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 75.0),
    }

    def __init__(
        self,
        chans,
        hidden_dim=64,
        num_layers=2,
        proj_dim=64,
        dropout=0.3,
        bidirectional=True,
        bands=None,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Frequency band config
        self.bands = bands if bands is not None else self.DEFAULT_BANDS
        self.band_names = list(self.bands.keys())
        self.filter = BandDecomposition(self.bands)

        # Shared LSTM encoder (same structure as EEGLSTM.lstm)
        self.lstm = nn.LSTM(
            input_size=chans,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        feat_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Per-band projection heads
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, proj_dim),
            )
            for name in self.band_names
        })

    def _lstm_encode(self, x):
        """Run LSTM and return final hidden state features.

        Args:
            x: (N, T, C) — stacked bands from potentially multiple samples

        Returns:
            (N, feat_dim)
        """
        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            feat = h_n[-1]
        return feat

    def forward(self, x):
        """Forward pass returning per-band projected features.

        Args:
            x: (B, C, T) single augmented view

        Returns:
            dict {band_name: (B, proj_dim)} projected features
        """
        B = x.size(0)

        # 1. Decompose into bands
        bands = self.filter(x)  # {name: (B, C, T)}

        # 2. Stack all bands: (B * num_bands, C, T)
        stacked = torch.cat([bands[name] for name in self.band_names], dim=0)

        # 3. Shared LSTM: (B * num_bands, T, C)
        stacked = stacked.transpose(1, 2)
        feats = self._lstm_encode(stacked)  # (B * num_bands, feat_dim)

        # 4. Per-band projection
        result = {}
        for i, name in enumerate(self.band_names):
            feat = feats[i * B : (i + 1) * B]  # (B, feat_dim)
            result[name] = self.projections[name](feat)  # (B, proj_dim)

        return result

    def get_encoder_state_dict(self):
        """Return LSTM-only state dict for weight transfer to EEGLSTM."""
        return {
            k.removeprefix("lstm."): v
            for k, v in self.state_dict().items()
            if k.startswith("lstm.")
        }


class MultiBandMoESimCLREncoder(MultiBandSimCLREncoder):
    """Multi-band SimCLR encoder with MoE between LSTM and projections.

    Additional args:
        moe_num_experts: number of MoE experts (default 4)
        moe_top_k: top-k experts activated (default 2)
        moe_expert_mult: hidden multiplier per expert (default 4)
    """

    def __init__(
        self,
        chans,
        hidden_dim=64,
        num_layers=2,
        proj_dim=64,
        dropout=0.3,
        bidirectional=True,
        bands=None,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_mult=4,
    ):
        super().__init__(
            chans=chans,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            dropout=dropout,
            bidirectional=bidirectional,
            bands=bands,
        )

        feat_dim = hidden_dim * 2 if bidirectional else hidden_dim
        from .moe import MoELayer
        self.moe = MoELayer(
            dim=feat_dim,
            num_experts=moe_num_experts,
            expert_mult=moe_expert_mult,
            top_k=moe_top_k,
            dropout=dropout,
        )

    def forward(self, x):
        """Returns (band_features, balance_loss)."""
        B = x.size(0)
        bands = self.filter(x)

        stacked = torch.cat([bands[name] for name in self.band_names], dim=0)
        stacked = stacked.transpose(1, 2)
        feats = self._lstm_encode(stacked)  # (B * num_bands, feat_dim)

        # MoE
        moe_out, balance_loss = self.moe(feats)

        # Per-band projections
        result = {}
        for i, name in enumerate(self.band_names):
            feat = moe_out[i * B : (i + 1) * B]
            result[name] = self.projections[name](feat)

        return result, balance_loss

    def get_encoder_state_dict(self):
        """Return LSTM + MoE state dict (exclude projection heads)."""
        return {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("projections.")
        }
