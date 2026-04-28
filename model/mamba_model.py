"""Mamba-based EEG classifier — bidirectional selective state-space model.

Architecture: input embedding → BiMamba blocks → pooling → classifier.
Follows the same interface as EEGLSTM for drop-in compatibility with MODEL_DICT.

NOTE: The default implementation uses a pure-PyTorch sequential scan for
correctness on CPU. For training on sequences longer than ~200 steps or with
large batch sizes, install mamba-ssm for the parallel (associative) scan:
    pip install mamba-ssm

Without mamba-ssm, use --device cuda for reasonable training speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Minimal selective-scan SSM (pure PyTorch fallback when mamba-ssm unavailable)
# ---------------------------------------------------------------------------

class SelectiveScan(nn.Module):
    """Selective state-space scan (S6 block from Mamba).

    Implements: h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t
                y_t = C_t h_t + D x_t

    The selective scan uses sequential recurrence. For long sequences
    (>1000 steps), install mamba-ssm for the parallel (associative) scan.
    """

    def __init__(self, d_model, d_state=16, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        dt_rank = dt_rank or max(d_model // 16, 1)

        # A: diagonal state matrix (decays)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # Δ projection: dt_rank → d_model (with softplus for positivity)
        self.dt_proj = nn.Sequential(
            nn.Linear(dt_rank, d_model),
            nn.Softplus(),
        )

        # B, C projections
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2)

        self.dt_rank = dt_rank

    def forward(self, x):
        """Selective scan forward.

        Args:
            x: (B, L, D)

        Returns:
            (B, L, D)
        """
        B, L, D = x.shape

        # Project input
        proj = self.x_proj(x)  # (B, L, dt_rank + d_state * 2)
        delta_before = proj[:, :, :self.dt_rank]
        B_tilde = proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_tilde = proj[:, :, self.dt_rank + self.d_state:]

        # Δ (discretization step)
        delta = self.dt_proj(delta_before)  # (B, L, d_model)

        # A discretization: A_bar = exp(Δ * A)
        A = -torch.exp(self.A_log.float())  # (D, N)
        A_bar = torch.exp(torch.einsum("bld,dn->bldn", delta, A))

        # B discretization: B_bar = Δ * B
        B_bar = torch.einsum("bld,bln->bldn", delta, B_tilde)

        # Selective scan (sequential)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y = torch.einsum("bdn,bn->bd", h, C_tilde[:, t]) + self.D * x[:, t]
            outputs.append(y)

        out = torch.stack(outputs, dim=1)  # (B, L, D)
        return out


# ---------------------------------------------------------------------------
# BiMamba block
# ---------------------------------------------------------------------------

class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block with forward + backward SSM passes.

    Uses element-wise addition of forward/backward features to maintain
    dimension consistency for stacking multiple blocks.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        inner_dim = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)

        self.ssm_fwd = SelectiveScan(inner_dim, d_state)
        self.ssm_bwd = SelectiveScan(inner_dim, d_state)

        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, L, D)

        Returns:
            (B, L, D)
        """
        residual = x
        x_norm = self.norm(x)

        proj = self.in_proj(x_norm)
        x_fwd, x_bwd = proj.chunk(2, dim=-1)

        # Forward SSM
        fwd_out = self.ssm_fwd(x_fwd)

        # Backward SSM (reverse sequence)
        bwd_out = self.ssm_bwd(torch.flip(x_bwd, [1]))
        bwd_out = torch.flip(bwd_out, [1])

        # Combine: element-wise addition
        merged = F.silu(fwd_out + bwd_out)
        out = self.out_proj(merged)
        out = self.dropout(out)

        return out + residual


# ---------------------------------------------------------------------------
# EEGMamba classifier
# ---------------------------------------------------------------------------

class EEGMamba(nn.Module):
    """Bidirectional Mamba for EEG classification.

    Compatible with MODEL_DICT: same interface as EEGLSTM.

    Args:
        chans: number of input EEG channels
        d_model: model dimension (default 64)
        num_layers: number of BiMamba blocks (default 2)
        d_state: SSM state dimension (default 16)
        num_classes: number of output classes
        dropout: dropout rate (default 0.3)
    """

    def __init__(
        self,
        chans=20,
        d_model=64,
        num_layers=2,
        d_state=16,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.chans = chans
        self.d_model = d_model

        # Input projection: channels → d_model
        self.input_proj = nn.Linear(chans, d_model)

        # Stacked BiMamba blocks
        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.grad_clip = 1.0

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T) EEG input

        Returns:
            (B, num_classes) logits
        """
        # (B, C, T) → (B, T, C) → (B, T, D)
        x = x.transpose(1, 2)
        x = self.input_proj(x)

        # BiMamba blocks
        for block in self.blocks:
            x = block(x)

        # Pool: mean over time dimension
        feat = x.mean(dim=1)  # (B, D)

        # LayerNorm before classifier
        feat = self.norm_out(feat)

        return self.classifier(feat)

    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
