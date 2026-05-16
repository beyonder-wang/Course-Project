"""SEEDAsymNet: DE + asymmetry + graph fusion model for SEED."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_decomposition import BandDecomposition
from .dgcnn import GraphConv
from .rgnn import SEED_CHANNELS, _build_seed_prior_mask


HEMISPHERE_PAIRS = [
    ("FP1", "FP2"), ("AF3", "AF4"), ("F7", "F8"), ("F5", "F6"), ("F3", "F4"),
    ("F1", "F2"), ("FT7", "FT8"), ("FC5", "FC6"), ("FC3", "FC4"), ("FC1", "FC2"),
    ("T7", "T8"), ("C5", "C6"), ("C3", "C4"), ("C1", "C2"), ("TP7", "TP8"),
    ("CP5", "CP6"), ("CP3", "CP4"), ("CP1", "CP2"), ("P7", "P8"), ("P5", "P6"),
    ("P3", "P4"), ("P1", "P2"), ("PO7", "PO8"), ("PO5", "PO6"), ("PO3", "PO4"),
    ("O1", "O2"), ("CB1", "CB2"),
]


class SEEDAsymNet(nn.Module):
    """SEED model using official-style DE and asymmetry cues from raw windows."""

    def __init__(
        self,
        chans,
        num_classes,
        hidden_dim=64,
        graph_layers=2,
        asym_hidden=256,
        fusion_hidden=256,
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
        self.top_k = top_k
        self.dyn_alpha = dyn_alpha
        self.return_features = return_features
        self.dropout = nn.Dropout(dropout)

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

        channel_to_idx = {name: i for i, name in enumerate(SEED_CHANNELS[:chans])}
        pair_indices = [
            (channel_to_idx[left], channel_to_idx[right])
            for left, right in HEMISPHERE_PAIRS
            if left in channel_to_idx and right in channel_to_idx
        ]
        self.register_buffer("pair_index", torch.tensor(pair_indices, dtype=torch.long))

        asym_input_dim = len(pair_indices) * self.input_features * 2
        self.asym_mlp = nn.Sequential(
            nn.Linear(asym_input_dim, asym_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(asym_hidden, hidden_dim),
            nn.ReLU(),
        )

        self.feature_dim = chans * hidden_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def _extract_de_and_var(self, x):
        band_signals = self.decomp(x)
        de_features = []
        var_features = []
        for name in self.decomp.band_names:
            band_x = band_signals[name]
            band_var = band_x.var(dim=-1, unbiased=False) + 1e-6
            var_features.append(band_var)
            de_features.append(0.5 * torch.log(band_var))
        de = torch.stack(de_features, dim=-1)
        var = torch.stack(var_features, dim=-1)
        return de, var

    def _sparsify(self, adj):
        if self.top_k is None or self.top_k <= 0 or self.top_k >= adj.size(-1):
            return adj
        values, indices = torch.topk(adj, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(adj)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return 0.5 * (sparse + sparse.transpose(0, 1))

    def _build_adj(self, de_feats):
        static_adj = F.softplus(self.edge_logits) * self.prior_mask
        proj = self.de_proj(de_feats)
        proj = F.normalize(proj, dim=-1)
        dynamic_adj = torch.bmm(proj, proj.transpose(1, 2)).mean(dim=0)
        dynamic_adj = F.relu(dynamic_adj) * self.prior_mask
        adj = static_adj + self.dyn_alpha * dynamic_adj
        adj = self._sparsify(adj)
        adj = adj + torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)

    def _extract_asymmetry(self, de_feats, var_feats):
        left = self.pair_index[:, 0]
        right = self.pair_index[:, 1]
        de_left = de_feats[:, left]
        de_right = de_feats[:, right]
        var_left = var_feats[:, left]
        var_right = var_feats[:, right]

        dasm = de_left - de_right
        rasm = torch.log(var_left / var_right)
        asym = torch.cat([dasm, rasm], dim=-1)
        return asym.reshape(asym.size(0), -1)

    def forward(self, x):
        de_feats, var_feats = self._extract_de_and_var(x)
        adj = self._build_adj(de_feats)

        graph_x = de_feats
        for layer, norm in zip(self.graph_layers, self.norm_layers):
            graph_x = layer(graph_x, adj)
            graph_x = norm(graph_x)
            graph_x = F.elu(graph_x)
            graph_x = self.dropout(graph_x)
        graph_feat = graph_x.reshape(graph_x.size(0), -1)

        asym_feat = self.asym_mlp(self._extract_asymmetry(de_feats, var_feats))
        features = torch.cat([graph_feat, asym_feat], dim=1)
        logits = self.classifier(features)

        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
