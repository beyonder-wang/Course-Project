"""Multi-head contrastive loss for multi-band SimCLR.

Computes NT-Xent loss separately for each frequency band,
then averages across bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBandNTXentLoss(nn.Module):
    """Multi-head NT-Xent loss for multi-band contrastive learning.

    Computes the normalized temperature-scaled cross-entropy loss per
    frequency band, then returns the mean across all bands.

    Args:
        temperature: softmax temperature (default 0.1)
        band_weights: optional dict of per-band weights for weighted averaging.
            If None, all bands weighted equally.
    """

    def __init__(self, temperature=0.1, band_weights=None):
        super().__init__()
        self.temperature = temperature
        self.band_weights = band_weights  # dict or None

    def _ntxent(self, z_i, z_j):
        """Standard NT-Xent loss for one band.

        Args:
            z_i, z_j: (B, proj_dim) projected features from two augmented views

        Returns:
            scalar loss
        """
        n = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        z = F.normalize(z, dim=-1)

        # Cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Positive pairs: (i, i+n) and (i+n, i)
        pos_mask = torch.zeros(2 * n, 2 * n, device=z.device, dtype=torch.bool)
        pos_mask[:n, n:] = torch.eye(n, device=z.device, dtype=torch.bool)
        pos_mask[n:, :n] = torch.eye(n, device=z.device, dtype=torch.bool)

        # Negatives: exclude self
        neg_mask = ~torch.eye(2 * n, device=z.device, dtype=torch.bool)

        # For numerical stability, subtract max per row
        sim_max = sim.max(dim=1, keepdim=True)[0].detach()
        sim = sim - sim_max

        # Log-sum-exp over negatives
        sim_exp = sim.exp() * neg_mask.float()
        log_sum_exp = sim_exp.sum(dim=1, keepdim=True).log()

        # Log-probability of positives
        log_prob = sim - log_sum_exp
        pos_log_prob = log_prob[pos_mask]

        # Mean over positive pairs
        loss = -pos_log_prob.mean()
        return loss

    def forward(self, z_i_dict, z_j_dict):
        """Compute multi-band NT-Xent loss.

        Args:
            z_i_dict: dict {band_name: (B, proj_dim)} view 1
            z_j_dict: dict {band_name: (B, proj_dim)} view 2

        Returns:
            scalar: mean NT-Xent loss across all bands
        """
        band_names = list(z_i_dict.keys())
        losses = []
        weights = []

        for name in band_names:
            loss_val = self._ntxent(z_i_dict[name], z_j_dict[name])
            losses.append(loss_val)
            w = 1.0 if self.band_weights is None else self.band_weights.get(name, 1.0)
            weights.append(w)

        weights = torch.tensor(weights, device=losses[0].device)
        weights = weights / weights.sum()

        total = sum(w * l for w, l in zip(weights, losses))
        return total


class MultiBandInfoNCELoss(nn.Module):
    """Lighter alternative: InfoNCE computed jointly across all bands.

    Concatenates all band features and computes a single NT-Xent loss.
    This encourages the encoder to produce complementary representations
    across bands rather than redundant ones.

    Args:
        temperature: softmax temperature (default 0.1)
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i_dict, z_j_dict):
        """Joint InfoNCE across all bands.

        Args:
            z_i_dict, z_j_dict: dict {band_name: (B, proj_dim)}

        Returns:
            scalar loss
        """
        band_names = sorted(z_i_dict.keys())
        all_z_i = []
        all_z_j = []
        for name in band_names:
            all_z_i.append(F.normalize(z_i_dict[name], dim=-1))
            all_z_j.append(F.normalize(z_j_dict[name], dim=-1))

        z_i = torch.cat(all_z_i, dim=-1)  # (B, num_bands * proj_dim)
        z_j = torch.cat(all_z_j, dim=-1)

        n = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D_total)
        sim = torch.mm(z, z.t()) / self.temperature

        pos_mask = torch.zeros(2 * n, 2 * n, device=z.device, dtype=torch.bool)
        pos_mask[:n, n:] = torch.eye(n, device=z.device, dtype=torch.bool)
        pos_mask[n:, :n] = torch.eye(n, device=z.device, dtype=torch.bool)
        neg_mask = ~torch.eye(2 * n, device=z.device, dtype=torch.bool)

        sim_max = sim.max(dim=1, keepdim=True)[0].detach()
        sim = sim - sim_max

        sim_exp = sim.exp() * neg_mask.float()
        log_sum_exp = sim_exp.sum(dim=1, keepdim=True).log()
        log_prob = sim - log_sum_exp
        pos_log_prob = log_prob[pos_mask]

        return -pos_log_prob.mean()
