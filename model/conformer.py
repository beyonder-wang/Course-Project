"""EEG-Conformer: Convolution-augmented Transformer for EEG classification.

Architecture based on the Conformer paper (Gulati et al. 2020, INTERSPEECH),
adapted for EEG motor imagery classification:

  PatchEmbedding → Dropout → N×ConformerBlock → LayerNorm → GAP → Classifier

Each ConformerBlock:
    FFN(half) → MHSA(relative pos) → ConvModule → FFN(half) → LayerNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _FeedForwardModule(nn.Module):
    """Half-step feed-forward: Linear(factor) → Swish → Dropout → Linear → Dropout."""

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        hidden = dim * expansion_factor
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        # x: (B, L, dim)
        x = self.linear1(x)
        x = F.silu(x)  # Swish / SiLU
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class _RelativePositionBias(nn.Module):
    """Learned relative position bias for multi-head attention (T5-style)."""

    def __init__(self, n_head, max_len=200):
        super().__init__()
        # 2*max_len - 1 possible relative offsets
        self.bias = nn.Parameter(torch.randn(2 * max_len - 1, n_head))
        nn.init.xavier_uniform_(self.bias)

    def forward(self, length, device=None):
        """Returns bias of shape (1, n_head, length, length)."""
        dtype = self.bias.dtype
        bias_dev = self.bias.device
        pos = torch.arange(length, device=bias_dev)
        rel = pos[:, None] - pos[None, :]  # (L, L)
        rel = rel + (length - 1)            # shift to [0, 2*L-2]
        bias_table = self.bias[rel]          # (L, L, n_head)
        return bias_table.permute(2, 0, 1).unsqueeze(0)  # (1, n_head, L, L)


class _MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with pre-LayerNorm, residual, and relative pos bias."""

    def __init__(self, dim, n_head=4, dropout=0.1, max_len=200):
        super().__init__()
        assert dim % n_head == 0, f"dim {dim} not divisible by n_head {n_head}"
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.rel_bias = _RelativePositionBias(n_head, max_len)

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x: (B, L, dim)
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, L, L)
        attn = attn + self.rel_bias(L, x.device)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout.p, training=self.training)

        out = attn @ v                                    # (B, n_head, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, -1)       # (B, L, dim)
        out = self.dropout(self.fc(out))
        return out + residual


class _ConformerConvModule(nn.Module):
    """Convolution module: PointwiseConv1d → GLU → DepthwiseConv1d → BN → Swish → PointwiseConv1d."""

    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        inner = dim * expansion_factor
        self.pointwise_conv1 = nn.Conv1d(dim, inner, 1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.bn = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        nn.init.xavier_uniform_(self.pointwise_conv1.weight)
        nn.init.xavier_uniform_(self.depthwise_conv.weight)
        nn.init.xavier_uniform_(self.pointwise_conv2.weight)
        nn.init.constant_(self.pointwise_conv1.bias, 0)
        nn.init.constant_(self.depthwise_conv.bias, 0)
        nn.init.constant_(self.pointwise_conv2.bias, 0)

    def forward(self, x):
        # x: (B, L, dim)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, dim, L)

        x = self.pointwise_conv1(x)   # (B, inner, L)
        x = x[:, :x.shape[1] // 2, :] * F.sigmoid(x[:, x.shape[1] // 2:, :])  # GLU
        # After GLU: (B, dim, L)

        x = self.depthwise_conv(x)    # (B, dim, L)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)   # (B, dim, L)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # (B, L, dim)
        return x + residual


class _ConformerBlock(nn.Module):
    """Full Conformer block: half-FFN → MHSA → ConvMod → half-FFN → post-LayerNorm."""

    def __init__(self, dim, n_head=4, kernel_size=31, ff_expansion=4,
                 conv_expansion=2, dropout=0.1, max_len=200):
        super().__init__()
        self.ffn1 = _FeedForwardModule(dim, ff_expansion, dropout)
        self.mhsa = _MultiHeadSelfAttention(dim, n_head, dropout, max_len)
        self.conv = _ConformerConvModule(dim, conv_expansion, kernel_size, dropout)
        self.ffn2 = _FeedForwardModule(dim, ff_expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Main EEG-Conformer model
# ---------------------------------------------------------------------------

class EEGConformer(nn.Module):
    """EEG-Conformer: patch embedding + Conformer blocks + global pooling.

    Args:
        chans: number of EEG channels
        num_classes: number of classes
        time_point: number of time samples
        dim: model dimension (default 64)
        n_blocks: number of Conformer blocks (default 4)
        n_head: attention heads per block (default 4)
        kernel_size: depthwise conv kernel size (default 31, ~155ms at 200 Hz)
        ff_expansion: feed-forward expansion factor (default 4)
        conv_expansion: conv-module expansion factor (default 2)
        patch_kernel: patch embedding conv kernel size (default 25)
        patch_stride: patch embedding stride (default 10)
        dropout: global dropout rate (default 0.1)
    """

    def __init__(self, chans=22, num_classes=4, time_point=800,
                 dim=64, n_blocks=4, n_head=4, kernel_size=31,
                 ff_expansion=4, conv_expansion=2,
                 patch_kernel=25, patch_stride=10, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        # --- Patch embedding ---
        self.patch_conv = nn.Conv1d(
            chans, dim, kernel_size=patch_kernel, stride=patch_stride)
        self.patch_norm = nn.LayerNorm(dim)
        # Compute sequence length after patching
        with torch.no_grad():
            dummy = torch.zeros(1, chans, time_point)
            patch_out = self.patch_conv(dummy)
            seq_len = patch_out.shape[-1]

        self.dropout = nn.Dropout(dropout)

        # --- Conformer blocks ---
        self.blocks = nn.ModuleList([
            _ConformerBlock(dim, n_head, kernel_size, ff_expansion,
                            conv_expansion, dropout, max_len=seq_len)
            for _ in range(n_blocks)
        ])

        # --- Head ---
        self.head_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.classifier:
                # already handled in sub-modules
                pass
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # x: (B, C, T)
        x = self.patch_conv(x)        # (B, dim, L)
        x = x.transpose(1, 2)         # (B, L, dim)
        x = self.patch_norm(x)
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.head_norm(x)
        x = x.mean(dim=1)             # global average pooling: (B, dim)
        x = self.classifier(x)        # (B, num_classes)
        return x
