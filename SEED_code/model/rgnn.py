"""RGNN-style SEED model with internal DE features and sparse graph priors."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_decomposition import BandDecomposition
from .dgcnn import GraphConv


SEED_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2",
    "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5",
    "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1",
    "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
    "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]


def _add_named_edges(edge_set, channel_to_idx, names):
    for a, b in zip(names[:-1], names[1:]):
        if a in channel_to_idx and b in channel_to_idx:
            ia, ib = channel_to_idx[a], channel_to_idx[b]
            edge_set.add((ia, ib))
            edge_set.add((ib, ia))


def _build_seed_prior_mask(chans):
    mask = torch.eye(chans, dtype=torch.float32)
    if chans != len(SEED_CHANNELS):
        return mask

    idx = {name: i for i, name in enumerate(SEED_CHANNELS)}
    edges = set()

    rows = [
        ["FP1", "FPZ", "FP2"],
        ["AF3", "AF4"],
        ["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"],
        ["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"],
        ["T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"],
        ["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"],
        ["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"],
        ["PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"],
        ["CB1", "O1", "OZ", "O2", "CB2"],
    ]
    for row in rows:
        _add_named_edges(edges, idx, row)

    chains = [
        ["FP1", "AF3", "F3", "FC3", "C3", "CP3", "P3", "PO3", "O1"],
        ["FP2", "AF4", "F4", "FC4", "C4", "CP4", "P4", "PO4", "O2"],
        ["F7", "FT7", "T7", "TP7", "P7", "PO7"],
        ["F8", "FT8", "T8", "TP8", "P8", "PO8"],
        ["F5", "FC5", "C5", "CP5", "P5", "PO5"],
        ["F6", "FC6", "C6", "CP6", "P6", "PO6"],
        ["F1", "FC1", "C1", "CP1", "P1"],
        ["F2", "FC2", "C2", "CP2", "P2"],
        ["FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ"],
    ]
    for chain in chains:
        _add_named_edges(edges, idx, chain)

    hemisphere_pairs = [
        ("FP1", "FP2"), ("AF3", "AF4"), ("F7", "F8"), ("F5", "F6"), ("F3", "F4"),
        ("F1", "F2"), ("FT7", "FT8"), ("FC5", "FC6"), ("FC3", "FC4"), ("FC1", "FC2"),
        ("T7", "T8"), ("C5", "C6"), ("C3", "C4"), ("C1", "C2"), ("TP7", "TP8"),
        ("CP5", "CP6"), ("CP3", "CP4"), ("CP1", "CP2"), ("P7", "P8"), ("P5", "P6"),
        ("P3", "P4"), ("P1", "P2"), ("PO7", "PO8"), ("PO5", "PO6"), ("PO3", "PO4"),
        ("O1", "O2"), ("CB1", "CB2"),
    ]
    for a, b in hemisphere_pairs:
        ia, ib = idx[a], idx[b]
        edges.add((ia, ib))
        edges.add((ib, ia))

    for ia, ib in edges:
        mask[ia, ib] = 1.0
    return mask


class RGNN(nn.Module):
    """SEED-oriented RGNN with DE extraction and biologically inspired sparse priors."""

    def __init__(
        self,
        chans,
        num_classes,
        hidden_dim=32,
        graph_layers=2,
        classifier_hidden=128,
        dropout=0.3,
        fs=200,
        top_k=8,
        dyn_alpha=0.15,
        return_features=False,
    ):
        super().__init__()
        self.decomp = BandDecomposition(fs=fs)
        self.num_nodes = chans
        self.input_features = len(self.decomp.band_names)
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k
        self.dyn_alpha = dyn_alpha
        self.return_features = return_features

        self.register_buffer("prior_mask", _build_seed_prior_mask(chans))
        self.edge_logits = nn.Parameter(0.01 * torch.randn(chans, chans))
        self.de_proj = nn.Linear(self.input_features, self.input_features, bias=False)

        self.graph_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        in_dim = self.input_features
        for _ in range(graph_layers):
            self.graph_layers.append(GraphConv(in_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.feature_dim = chans * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, classifier_hidden),
            nn.ELU(),
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

    def _build_adj(self, de_feats):
        static_adj = F.softplus(self.edge_logits) * self.prior_mask
        static_adj = static_adj.unsqueeze(0).expand(de_feats.size(0), -1, -1)

        proj = self.de_proj(de_feats)
        proj = F.normalize(proj, dim=-1)
        dynamic_adj = torch.bmm(proj, proj.transpose(1, 2))
        dynamic_adj = F.relu(dynamic_adj) * self.prior_mask.unsqueeze(0)

        adj = static_adj + self.dyn_alpha * dynamic_adj
        adj = self._sparsify(adj)
        eye = torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye
        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)

    def forward(self, x):
        x = self._extract_de_features(x)
        adj = self._build_adj(x)

        for layer, norm in zip(self.graph_layers, self.norm_layers):
            x = layer(x, adj)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)

        features = x.reshape(x.size(0), -1)
        logits = self.classifier(features)
        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
