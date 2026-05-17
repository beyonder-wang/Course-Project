"""ATCNet: Attention Temporal Convolutional Network for EEG Motor Imagery.

Reference: Altaheri et al., "Physics-informed attention temporal convolutional
network for EEG-based motor imagery classification", IEEE TNSRE, 2023.

Architecture:
  1. ConvBlock: EEGNet-style frontend (temporal + spatial + separable conv)
  2. Sliding window over time dimension
  3. Per-window AttentionBlock (MHA + residual) → TCN (causal dilated conv)
  4. Average window predictions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constrained layers (max-norm weight regularization)
# ---------------------------------------------------------------------------

class _Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class _LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class _CausalConv1d(nn.Conv1d):
    """Causal 1D convolution (pads only left side)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        self._padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return super().forward(F.pad(x, (self._padding, 0)))


def _glorot_init(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# ATCNet sub-modules
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """EEGNet-style convolutional frontend for ATCNet."""

    def __init__(self, F1=16, kernel_length=64, pool_length=8, D=2,
                 in_channels=22, dropout=0.3):
        super().__init__()

        self.temporal_conv = nn.Conv2d(
            1, F1, (1, kernel_length),
            padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.spatial_conv = _Conv2dWithConstraint(
            F1, F1 * D, (in_channels, 1),
            groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_length))
        self.drop1 = nn.Dropout(dropout)

        self.conv = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                              padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 7))
        self.drop2 = nn.Dropout(dropout)

        _glorot_init(self)

    def forward(self, x):
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)

        x = self.temporal_conv(x)   # (B, F1, C, T)
        x = self.bn1(x)

        x = self.spatial_conv(x)    # (B, F1*D, 1, T)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)           # (B, F1*D, 1, T//pool)
        x = self.drop1(x)

        x = self.conv(x)            # (B, F1*D, 1, T//pool)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)           # (B, F1*D, 1, T//pool//7)
        x = self.drop2(x)

        return x


class _AttentionBlock(nn.Module):
    """Multi-head self-attention with residual connection + LayerNorm.

    Optional attention weight dropout and extra FC-output dropout
    (from the TF reference, disabled by default — enable via
    attn_drop and residual_drop when complementary regularization is used).
    """

    def __init__(self, d_model=32, key_dim=8, n_head=2, dropout=0.5,
                 attn_drop=0.0, residual_drop=0.0):
        super().__init__()
        self.n_head = n_head
        self.key_dim = key_dim
        self.inner_dim = n_head * key_dim
        self.attn_drop = attn_drop

        self.w_q = nn.Linear(d_model, self.inner_dim)
        self.w_k = nn.Linear(d_model, self.inner_dim)
        self.w_v = nn.Linear(d_model, self.inner_dim)
        self.fc = nn.Linear(self.inner_dim, d_model)
        self.dropout_fc = nn.Dropout(dropout)
        self.dropout_residual = nn.Dropout(residual_drop)
        self.norm = nn.LayerNorm(d_model)

        _glorot_init(self)

    def forward(self, x):
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        q = self.w_q(x).view(B, L, self.n_head, self.key_dim).permute(2, 0, 1, 3)
        k = self.w_k(x).view(B, L, self.n_head, self.key_dim).permute(2, 0, 1, 3)
        v = self.w_v(x).view(B, L, self.n_head, self.key_dim).permute(2, 0, 1, 3)

        scale = math.sqrt(self.key_dim)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / scale
        attn = torch.softmax(attn, dim=-1)
        # Attention weight dropout (optional, default off)
        attn = F.dropout(attn, p=self.attn_drop, training=self.training)

        out = torch.einsum('hblt,hbtv->hblv', attn, v)
        out = out.permute(1, 2, 0, 3).reshape(B, L, self.inner_dim)
        out = self.dropout_fc(self.fc(out))
        out = self.dropout_residual(out)  # extra dropout before residual (optional, default off)
        return out + residual


def _drop_path(x, drop_prob=0.0, training=False):
    """Stochastic Depth / DropPath per sample (standard from Deep Residual Learning)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class _TCNBlock(nn.Module):
    """Single TCN block: 2 causal dilated convs + residual."""

    def __init__(self, n_filters=32, kernel_size=4, dilation=1,
                 dropout=0.3, drop_path_prob=0.0):
        super().__init__()
        self.drop_path_prob = drop_path_prob

        self.conv1 = nn.Sequential(
            _CausalConv1d(n_filters, n_filters, kernel_size, dilation=dilation),
            nn.BatchNorm1d(n_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            _CausalConv1d(n_filters, n_filters, kernel_size, dilation=dilation),
            nn.BatchNorm1d(n_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.act = nn.ELU()

        nn.init.constant_(self.conv1[0].bias, 0)
        nn.init.constant_(self.conv2[0].bias, 0)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x + residual)
        return _drop_path(x, self.drop_path_prob, self.training)


class _TCN(nn.Module):
    """Stack of TCN blocks with exponentially increasing dilation."""

    def __init__(self, depth=2, kernel_size=4, n_filters=32,
                 dropout=0.3, drop_path_prob=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TCNBlock(n_filters, kernel_size, dilation=2 ** i,
                      dropout=dropout, drop_path_prob=drop_path_prob)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Main ATCNet model
# ---------------------------------------------------------------------------

class ATCNet(nn.Module):
    """ATCNet for EEG motor imagery classification.

    Args:
        chans: number of EEG channels
        num_classes: number of classes
        time_point: number of time samples
        F1: conv block F1 filter count (default 16)
        D: depth multiplier (default 2)
        kernel_length: temporal conv kernel (default 64)
        pool_length: first pooling size (default 8)
        dropout_conv: conv block dropout (default 0.3)
        d_model: attention/TCN hidden dim (default 32)
        key_dim: attention key dim (default 8)
        n_head: number of attention heads (default 2)
        dropout_attn: attention dropout (default 0.5)
        tcn_depth: number of TCN blocks (default 2)
        kernel_tcn: TCN kernel size (default 4)
        dropout_tcn: TCN dropout (default 0.3)
        n_windows: number of sliding windows (default 5)
    """

    def __init__(self, chans=22, num_classes=4, time_point=800,
                 F1=16, D=2, kernel_length=64, pool_length=8,
                 dropout_conv=0.3, d_model=32, key_dim=8, n_head=2,
                 dropout_attn=0.5, attn_drop=0.0, residual_drop=0.0,
                 tcn_depth=2, kernel_tcn=4, dropout_tcn=0.3,
                 drop_path_prob=0.0, n_windows=5):
        super().__init__()

        self.n_windows = n_windows
        self.num_classes = num_classes

        # --- Conv frontend ---
        self.conv_block = _ConvBlock(
            F1=F1, kernel_length=kernel_length, pool_length=pool_length,
            D=D, in_channels=chans, dropout=dropout_conv)

        # --- ATC blocks (one per sliding window) ---
        # Each ATC block: Attention → TCN → Linear classifier
        for w in range(n_windows):
            attn = _AttentionBlock(d_model, key_dim, n_head, dropout_attn,
                                   attn_drop=attn_drop, residual_drop=residual_drop)
            tcn = _TCN(tcn_depth, kernel_tcn, d_model, dropout_tcn,
                       drop_path_prob=drop_path_prob)
            linear = _LinearWithConstraint(d_model, num_classes, max_norm=0.25)
            self.add_module(f'attn_{w}', attn)
            self.add_module(f'tcn_{w}', tcn)
            self.add_module(f'linear_{w}', linear)

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv_block(x)           # (B, d_model, 1, L)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, L, d_model)

        B, L, _ = x.shape
        window_len = L - self.n_windows + 1

        outputs = torch.zeros(B, self.num_classes,
                              dtype=x.dtype, device=x.device)
        for w in range(self.n_windows):
            # Extract window: (B, window_len, d_model)
            win = x[:, w:w + window_len, :]

            # Attention
            attn = getattr(self, f'attn_{w}')(win)  # (B, window_len, d_model)

            # TCN: (B, d_model, window_len)
            tcn_out = getattr(self, f'tcn_{w}')(attn.permute(0, 2, 1))
            # Classifier on last timestep
            linear = getattr(self, f'linear_{w}')
            outputs = outputs + linear(tcn_out[:, :, -1])

        return outputs / self.n_windows


# ---------------------------------------------------------------------------
# Capacity presets
# ---------------------------------------------------------------------------

ATCNET_PRESETS = {
    "base": dict(
        F1=16, D=2, d_model=32, n_head=2, n_windows=5, tcn_depth=2,
        dropout_conv=0.3, dropout_attn=0.5, dropout_tcn=0.3,
    ),
    "large": dict(
        F1=24, D=2, d_model=48, n_head=4, n_windows=7, tcn_depth=3,
        dropout_conv=0.3, dropout_attn=0.5, dropout_tcn=0.3,
    ),
    "xl": dict(
        F1=32, D=2, d_model=64, n_head=4, n_windows=9, tcn_depth=4,
        dropout_conv=0.3, dropout_attn=0.5, dropout_tcn=0.3,
    ),
}
