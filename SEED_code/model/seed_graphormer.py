"""SEEDGraphormer: multiband graph-transformer for emotion EEG."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_decomposition import BandDecomposition
from .dgcnn import GraphConv
from .rgnn import _build_seed_prior_mask


class SEEDGraphormer(nn.Module):
    """Heavy SEED-oriented model combining DE features, graph priors and Transformer."""

    def __init__(
        self,
        chans,
        num_classes,
        d_model=128,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.3,
        attn_dropout=0.1,
        fs=200,
        top_k=12,
        dyn_alpha=0.2,
        return_features=False,
    ):
        super().__init__()
        self.decomp = BandDecomposition(fs=fs)
        self.num_nodes = chans
        self.input_features = len(self.decomp.band_names)
        self.top_k = top_k
        self.dyn_alpha = dyn_alpha
        self.return_features = return_features

        self.register_buffer("prior_mask", _build_seed_prior_mask(chans))
        self.edge_logits = nn.Parameter(0.01 * torch.randn(chans, chans))
        self.dynamic_proj = nn.Linear(self.input_features, self.input_features, bias=False)

        self.token_proj = nn.Linear(self.input_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, chans + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.pre_graph = GraphConv(d_model, d_model)
        self.post_graph = GraphConv(d_model, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)

        self.feature_dim = d_model * 3
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _extract_de_features(self, x):
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

        proj = self.dynamic_proj(de_feats)
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
        de_feats = self._extract_de_features(x)
        adj = self._build_adj(de_feats)

        x = self.token_proj(de_feats)
        x = self.pre_graph(x, adj)
        x = self.pre_norm(F.gelu(x))

        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.attn_dropout(tokens)
        tokens = self.transformer(tokens)

        cls_token = tokens[:, 0]
        node_tokens = tokens[:, 1:]
        node_tokens = self.post_graph(node_tokens, adj)
        node_tokens = self.post_norm(F.gelu(node_tokens))

        mean_pool = node_tokens.mean(dim=1)
        max_pool = node_tokens.max(dim=1).values
        features = torch.cat([cls_token, mean_pool, max_pool], dim=1)
        logits = self.classifier(features)

        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
