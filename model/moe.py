import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """Mixture of Experts with top-k gating and load balancing.

    Each expert is a small MLP: Linear → GELU → Dropout → Linear.
    The router selects the top-k experts and computes a weighted sum.

    Args:
        dim: input/output feature dimension
        num_experts: number of expert networks (default 4)
        expert_mult: hidden dim multiplier for experts (default 4)
        top_k: number of experts to activate per token (default 2)
        dropout: dropout rate inside experts (default 0.1)
    """

    def __init__(self, dim, num_experts=4, expert_mult=4, top_k=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts, bias=False)

        hidden_dim = dim * expert_mult
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """Forward pass returning MoE output and load balancing loss.

        Args:
            x: (B, dim) input features

        Returns:
            out: (B, dim) MoE output
            balance_loss: scalar load balancing loss
        """
        B, D = x.shape

        gate_logits = self.router(x)  # (B, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k selection
        topk_gates, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Normalize selected gates to sum to 1
        topk_gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)

        # Compute expert outputs and weighted sum
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # (B,)
            gate = topk_gates[:, k]  # (B,)

            # Group tokens by expert assignment
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    out[mask] += gate[mask].unsqueeze(-1) * expert_out

        balance_loss = self._load_balance_loss(gate_probs, topk_indices)
        return out, balance_loss

    def _load_balance_loss(self, gate_probs, topk_indices):
        """Compute load balancing loss to encourage uniform expert usage.

        loss = num_experts * sum(f_i * g_i)
        where f_i is the fraction of tokens routed to expert i,
              g_i is the mean router probability for expert i.
        """
        B = gate_probs.size(0)
        num_experts = self.num_experts

        # Fraction of tokens dispatched to each expert
        f = torch.zeros(num_experts, device=gate_probs.device)
        for e in range(num_experts):
            f[e] = (topk_indices == e).sum().float() / (B * self.top_k)

        # Mean gate probability per expert
        g = gate_probs.mean(dim=0)  # (num_experts,)

        return num_experts * (f * g).sum()
