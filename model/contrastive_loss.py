import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss from SimCLR.

    Args:
        temperature: softmax temperature (default 0.1)
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute contrastive loss between two views.

        Args:
            z_i: projection from view 1, shape (B, proj_dim)
            z_j: projection from view 2, shape (B, proj_dim)

        Returns:
            scalar loss
        """
        B = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        z = torch.cat([z_i, z_j], dim=0)  # (2B, proj_dim)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Positive pairs: (i, i+B) and (i+B, i) for i in [0, B)
        pos_mask = torch.zeros(2 * B, 2 * B, device=z.device)
        pos_mask[torch.arange(B), torch.arange(B, 2 * B)] = 1
        pos_mask[torch.arange(B, 2 * B), torch.arange(B)] = 1

        # Mask out self-contrast
        self_mask = torch.eye(2 * B, device=z.device)
        sim = sim - self_mask * 1e9

        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
        loss = F.cross_entropy(sim, labels)
        return loss
