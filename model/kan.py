"""KAN (Kolmogorov-Arnold Network) layer — learnable B-spline activations.

Replaces standard MLP layers with learnable spline functions.
Pure PyTorch implementation, no extra dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """Linear layer with learnable B-spline activation functions.

    Each weight is replaced by a learnable B-spline curve phi(x) = base_weight * x
    + sum_k spline_weight[k] * B_k(x), where B_k are B-spline basis functions.

    Args:
        in_features: input dimension
        out_features: output dimension
        grid_size: number of grid intervals (default 5)
        spline_order: B-spline order, 3 = cubic (default 3)
        scale_base: initial scale for base activation (default 1.0)
        scale_spline: initial scale for spline coefficients (default 1.0)
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_base=1.0,
        scale_spline=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Base linear weight (standard linear contribution)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Spline coefficients: (out_features, in_features, grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # Learnable grid: uniformly initialized, then adapted during training
        h = (torch.arange(grid_size + 1) / grid_size * 2 - 1).repeat(in_features, 1)
        self.register_buffer("grid", h)  # (in_features, grid_size + 1)

        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1)

    def _b_splines(self, x: torch.Tensor):
        """Evaluate B-spline basis functions at x.

        Args:
            x: (B, in_features), values in [-1, 1]

        Returns:
            (B, in_features, grid_size + spline_order) basis matrix
        """
        # Ensure x is in [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)

        grid = self.grid  # (in_features, grid_size + 1)
        # Extend grid with uniform spacing for higher-order splines
        h = grid[:, 1] - grid[:, 0]
        left_ext = grid[:, :1] - h.unsqueeze(1) * torch.arange(
            self.spline_order, 0, -1, device=x.device
        ).unsqueeze(0)
        right_ext = grid[:, -1:] + h.unsqueeze(1) * torch.arange(
            1, self.spline_order + 1, device=x.device
        ).unsqueeze(0)
        extended_grid = torch.cat([left_ext, grid, right_ext], dim=1)

        # Zeroth-order basis (step functions)
        bases = ((x.unsqueeze(-1) >= extended_grid[:, :-1]) &
                 (x.unsqueeze(-1) < extended_grid[:, 1:])).float()

        # Recursively build higher-order splines
        for k in range(1, self.spline_order + 1):
            left = extended_grid[:, : -(k + 1)]
            right = extended_grid[:, k + 1 :]
            term0 = (x.unsqueeze(-1) - left) / (extended_grid[:, k:-1] - left + 1e-8)
            term1 = (right - x.unsqueeze(-1)) / (right - extended_grid[:, k:-1] + 1e-8)

            bases = term0 * bases[:, :, :-1] + term1 * bases[:, :, 1:]

        return bases

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (..., in_features)

        Returns:
            (..., out_features)
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Base linear contribution
        base_out = F.linear(x_flat, self.base_weight)

        # B-spline contribution
        spline_basis = self._b_splines(x_flat)  # (N, in, grid+order)
        spline_out = torch.einsum("nik,oik->no", spline_basis, self.spline_weight)

        out = self.scale_base * base_out + self.scale_spline * spline_out
        return out.reshape(*shape[:-1], self.out_features)

    def update_grid(self, x: torch.Tensor, margin=0.01):
        """Update spline grid based on input activations (for grid extension).

        Call periodically during training to adapt grid to input distribution.
        """
        with torch.no_grad():
            x_sorted = x.sort(dim=0)[0]
            n = x_sorted.shape[0]
            indices = torch.linspace(0, n - 1, self.grid_size + 1, device=x.device).long()
            new_grid = x_sorted[indices].t()
            # Smooth update
            eps = margin * (new_grid[:, -1] - new_grid[:, 0]).unsqueeze(1)
            new_grid[:, 0] -= eps[:, 0]
            new_grid[:, -1] += eps[:, 0]
            self.grid.copy_(new_grid)


class KANMLP(nn.Sequential):
    """MLP composed entirely of KAN layers.

    Args:
        layers: list of integers [in_dim, hidden_1, ..., out_dim]
        grid_size: B-spline grid size
        spline_order: B-spline order (3 = cubic)
        dropout: dropout rate between layers (default 0.0)
    """

    def __init__(self, layers, grid_size=5, spline_order=3, dropout=0.0):
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(KANLinear(layers[i], layers[i + 1],
                                    grid_size=grid_size, spline_order=spline_order))
            if dropout > 0 and i < len(layers) - 2:
                blocks.append(nn.Dropout(dropout))
        super().__init__(*blocks)
