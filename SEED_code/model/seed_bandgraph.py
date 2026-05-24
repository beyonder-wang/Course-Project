"""SEEDBandGraphNet: multi-band graph fusion model for SEED."""

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


class SEEDBandGraphNet(nn.Module):
    """Multi-band SEED model with per-band graph encoders and band attention."""

    def __init__(
        self,
        chans,
        num_classes,
        hidden_dim=48,
        graph_layers=2,
        band_hidden=96,
        asym_hidden=96,
        fusion_hidden=192,
        dropout=0.3,
        fs=200,
        top_k=10,
        dyn_alpha=0.2,
        return_features=False,
    ):
        super().__init__()
        self.decomp = BandDecomposition(fs=fs)
        self.num_nodes = chans
        self.num_bands = len(self.decomp.band_names)
        self.top_k = top_k
        self.dyn_alpha = dyn_alpha
        self.return_features = return_features
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("prior_mask", _build_seed_prior_mask(chans))
        self.edge_logits = nn.Parameter(0.01 * torch.randn(chans, chans))
        self.de_proj = nn.Linear(self.num_bands, self.num_bands, bias=False)

        self.band_input_proj = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(self.num_bands)]
        )
        self.band_graph_layers = nn.ModuleList()
        self.band_norm_layers = nn.ModuleList()
        for _ in range(self.num_bands):
            band_layers = nn.ModuleList()
            band_norms = nn.ModuleList()
            for _ in range(graph_layers):
                band_layers.append(GraphConv(hidden_dim, hidden_dim))
                band_norms.append(nn.LayerNorm(hidden_dim))
            self.band_graph_layers.append(band_layers)
            self.band_norm_layers.append(band_norms)

        channel_to_idx = {name: i for i, name in enumerate(SEED_CHANNELS[:chans])}
        pair_indices = [
            (channel_to_idx[left], channel_to_idx[right])
            for left, right in HEMISPHERE_PAIRS
            if left in channel_to_idx and right in channel_to_idx
        ]
        self.register_buffer("pair_index", torch.tensor(pair_indices, dtype=torch.long))
        asym_input_dim = len(pair_indices) * 2

        self.band_asym_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(asym_input_dim, asym_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(asym_hidden, band_hidden),
                nn.ReLU(),
            )
            for _ in range(self.num_bands)
        ])

        self.band_fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + band_hidden, band_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_bands)
        ])

        attn_hidden = max(16, band_hidden // 2)
        self.band_attention = nn.Sequential(
            nn.Linear(band_hidden, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        self.feature_dim = band_hidden * 2
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

    def _band_asymmetry(self, de_feats, var_feats, band_idx):
        left = self.pair_index[:, 0]
        right = self.pair_index[:, 1]
        de_left = de_feats[:, left, band_idx]
        de_right = de_feats[:, right, band_idx]
        var_left = var_feats[:, left, band_idx]
        var_right = var_feats[:, right, band_idx]
        dasm = de_left - de_right
        rasm = torch.log(var_left / var_right)
        return torch.cat([dasm, rasm], dim=1)

    def forward(self, x):
        de_feats, var_feats = self._extract_de_and_var(x)
        adj = self._build_adj(de_feats)

        band_features = []
        for band_idx in range(self.num_bands):
            band_x = de_feats[:, :, band_idx].unsqueeze(-1)
            band_x = self.band_input_proj[band_idx](band_x)
            for layer, norm in zip(
                self.band_graph_layers[band_idx],
                self.band_norm_layers[band_idx],
            ):
                band_x = layer(band_x, adj)
                band_x = norm(band_x)
                band_x = F.elu(band_x)
                band_x = self.dropout(band_x)

            graph_pool = torch.cat(
                [band_x.mean(dim=1), band_x.max(dim=1).values],
                dim=1,
            )
            asym_feat = self.band_asym_mlps[band_idx](
                self._band_asymmetry(de_feats, var_feats, band_idx)
            )
            band_features.append(
                self.band_fusion_mlps[band_idx](torch.cat([graph_pool, asym_feat], dim=1))
            )

        band_features = torch.stack(band_features, dim=1)
        band_scores = self.band_attention(band_features)
        band_weights = torch.softmax(band_scores, dim=1)
        weighted_band = (band_weights * band_features).sum(dim=1)
        max_band = band_features.max(dim=1).values
        features = torch.cat([weighted_band, max_band], dim=1)
        logits = self.classifier(features)

        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
