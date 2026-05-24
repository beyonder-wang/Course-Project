"""Lightweight SOGNN-style graph classifier for DE-based SEED features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_decomposition import BandDecomposition
from .dgcnn import GraphConv


class SOGNN(nn.Module):
    """Sample-organized sparse graph network on DE features."""

    def __init__(
        self,
        chans,
        num_classes,
        hidden_dim=48,
        graph_layers=3,
        classifier_hidden=128,
        dropout=0.3,
        fs=200,
        top_k=8,
        return_features=False,
    ):
        super().__init__()
        self.decomp = BandDecomposition(fs=fs)
        self.num_nodes = chans
        self.input_features = len(self.decomp.band_names)
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.return_features = return_features

        self.input_proj = nn.Linear(self.input_features, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(graph_layers)]
        )
        self.score_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(graph_layers)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(graph_layers)]
        )
        self.gate_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(graph_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        self.feature_dim = hidden_dim * 2
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

    def _sparsify(self, adj):
        if self.top_k is None or self.top_k <= 0 or self.top_k >= adj.size(-1):
            return adj
        values, indices = torch.topk(adj, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(adj)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return 0.5 * (sparse + sparse.transpose(-1, -2))

    def _build_adj(self, x, scorer):
        proj = scorer(x)
        proj = F.normalize(proj, dim=-1)
        adj = torch.bmm(proj, proj.transpose(1, 2))
        adj = F.relu(adj)
        adj = self._sparsify(adj)
        eye = torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye
        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)

    def forward(self, x):
        x = self._extract_de_features(x)
        x = self.input_proj(x)

        for graph_layer, score_layer, norm_layer, gate_layer in zip(
            self.graph_layers,
            self.score_layers,
            self.norm_layers,
            self.gate_layers,
        ):
            adj = self._build_adj(x, score_layer)
            residual = x
            x = graph_layer(x, adj)
            x = norm_layer(x)
            x = F.elu(x + residual)
            x = x * torch.sigmoid(gate_layer(x))
            x = self.dropout(x)

        features = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        logits = self.classifier(features)
        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
