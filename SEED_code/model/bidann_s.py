"""BiDANN-S style hemisphere-aware encoder for SEED DE features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_decomposition import BandDecomposition
from .dgcnn import GraphConv
from .rgnn import SEED_CHANNELS
from .seed_asymnet import HEMISPHERE_PAIRS


MIDLINE_CHANNELS = {"FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ"}


class _HemisphereGraphEncoder(nn.Module):
    def __init__(self, num_nodes, input_features, hidden_dim, graph_layers, dropout):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_proj = nn.Linear(input_features, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(graph_layers)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(graph_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        adj_init = torch.eye(num_nodes) + 0.01 * torch.randn(num_nodes, num_nodes)
        self.adj = nn.Parameter(adj_init)

    def _normalized_adj(self):
        adj = 0.5 * (self.adj + self.adj.transpose(0, 1))
        adj = F.relu(adj)
        adj = adj + torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)

    def forward(self, x):
        x = self.input_proj(x)
        adj = self._normalized_adj()
        for layer, norm in zip(self.graph_layers, self.norm_layers):
            x = layer(x, adj)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
        return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)


class BiDANN_S(nn.Module):
    """Simplified hemisphere-aware backbone for subject-adversarial DE training."""

    def __init__(
        self,
        chans,
        num_classes,
        hidden_dim=32,
        graph_layers=2,
        classifier_hidden=128,
        dropout=0.3,
        fs=200,
        return_features=False,
    ):
        super().__init__()
        self.decomp = BandDecomposition(fs=fs)
        self.input_features = len(self.decomp.band_names)
        self.return_features = return_features

        channel_to_idx = {name: i for i, name in enumerate(SEED_CHANNELS[:chans])}
        pair_indices = [
            (channel_to_idx[left], channel_to_idx[right])
            for left, right in HEMISPHERE_PAIRS
            if left in channel_to_idx and right in channel_to_idx
        ]
        left_indices = sorted({left for left, _ in pair_indices})
        right_indices = sorted({right for _, right in pair_indices})
        mid_indices = [
            idx for name, idx in channel_to_idx.items()
            if name in MIDLINE_CHANNELS and idx not in left_indices and idx not in right_indices
        ]

        self.register_buffer("pair_index", torch.tensor(pair_indices, dtype=torch.long))
        self.register_buffer("left_index", torch.tensor(left_indices, dtype=torch.long))
        self.register_buffer("right_index", torch.tensor(right_indices, dtype=torch.long))
        self.register_buffer("mid_index", torch.tensor(mid_indices, dtype=torch.long))

        self.left_encoder = _HemisphereGraphEncoder(
            num_nodes=len(left_indices),
            input_features=self.input_features,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
        )
        self.right_encoder = _HemisphereGraphEncoder(
            num_nodes=len(right_indices),
            input_features=self.input_features,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
        )
        self.midline_encoder = nn.Sequential(
            nn.Linear(self.input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.asym_mlp = nn.Sequential(
            nn.Linear(len(pair_indices) * self.input_features, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, hidden_dim * 2),
            nn.ReLU(),
        )

        self.feature_dim = hidden_dim * 7
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def _extract_de_features(self, x):
        if x.dim() == 3 and x.size(-1) == self.input_features:
            return x
        band_signals = self.decomp(x)
        de_features = []
        for name in self.decomp.band_names:
            band_x = band_signals[name]
            band_var = band_x.var(dim=-1, unbiased=False)
            de_features.append(0.5 * torch.log(band_var + 1e-6))
        return torch.stack(de_features, dim=-1)

    def forward(self, x):
        x = self._extract_de_features(x)

        left_feat = self.left_encoder(x[:, self.left_index])
        right_feat = self.right_encoder(x[:, self.right_index])

        mid_x = x[:, self.mid_index]
        mid_feat = self.midline_encoder(mid_x).mean(dim=1)

        left_pairs = x[:, self.pair_index[:, 0]]
        right_pairs = x[:, self.pair_index[:, 1]]
        asym_feat = self.asym_mlp((left_pairs - right_pairs).reshape(x.size(0), -1))

        features = torch.cat([left_feat, right_feat, mid_feat, asym_feat], dim=1)
        logits = self.classifier(features)
        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
