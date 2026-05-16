"""DGCNN-style graph classifier for SEED-like EEG inputs.

This version keeps the current raw waveform pipeline intact by extracting
band-DE features internally, then applying a lightweight learnable graph over
electrodes before classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_decomposition import BandDecomposition


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        out = torch.matmul(x, self.weight)
        out = torch.matmul(adj, out)
        if self.bias is not None:
            out = out + self.bias
        return out


class DGCNN(nn.Module):
    """Learnable-graph EEG classifier with internal DE feature extraction."""

    def __init__(
        self,
        chans,
        num_classes,
        hidden_dim=32,
        graph_layers=2,
        classifier_hidden=64,
        dropout=0.3,
        fs=200,
        return_features=False,
    ):
        super().__init__()
        if graph_layers < 1:
            raise ValueError("graph_layers must be >= 1")

        self.decomp = BandDecomposition(fs=fs)
        self.num_nodes = chans
        self.input_features = len(self.decomp.band_names)
        self.return_features = return_features
        self.dropout = nn.Dropout(dropout)

        adj_init = torch.eye(chans) + 0.01 * torch.randn(chans, chans)
        self.adj = nn.Parameter(adj_init)

        layers = [GraphConv(self.input_features, hidden_dim)]
        for _ in range(graph_layers - 1):
            layers.append(GraphConv(hidden_dim, hidden_dim))
        self.graph_layers = nn.ModuleList(layers)

        self.feature_dim = chans * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(chans * hidden_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def _extract_de_features(self, x):
        band_signals = self.decomp(x)
        de_features = []
        for name in self.decomp.band_names:
            band_x = band_signals[name]
            band_var = band_x.var(dim=-1, unbiased=False)
            de_features.append(0.5 * torch.log(band_var + 1e-6))
        return torch.stack(de_features, dim=-1)

    def _normalized_adj(self):
        adj = 0.5 * (self.adj + self.adj.transpose(0, 1))
        adj = F.relu(adj)
        adj = adj + torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)

    def forward(self, x):
        x = self._extract_de_features(x)
        adj = self._normalized_adj()

        for layer in self.graph_layers:
            x = layer(x, adj)
            x = F.elu(x)
            x = self.dropout(x)

        features = x.reshape(x.size(0), -1)
        logits = self.classifier(features)
        if self.return_features:
            return {"logits": logits, "features": features}
        return logits


class DGCNN_RG(DGCNN):
    """RGNN-inspired DGCNN with dynamic residual adjacency and sparsification."""

    def __init__(
        self,
        chans,
        num_classes,
        hidden_dim=32,
        graph_layers=2,
        classifier_hidden=64,
        dropout=0.3,
        fs=200,
        top_k=12,
        dropedge=0.1,
        dyn_alpha=0.2,
        return_features=False,
    ):
        super().__init__(
            chans=chans,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            classifier_hidden=classifier_hidden,
            dropout=dropout,
            fs=fs,
            return_features=return_features,
        )
        self.top_k = top_k
        self.dropedge = dropedge
        self.dyn_alpha = dyn_alpha
        self.de_proj = nn.Linear(self.input_features, self.input_features, bias=False)

    def _sparsify(self, adj):
        if self.top_k is None or self.top_k <= 0 or self.top_k >= adj.size(-1):
            return adj
        k = min(self.top_k, adj.size(-1))
        values, indices = torch.topk(adj, k=k, dim=-1)
        sparse = torch.zeros_like(adj)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return 0.5 * (sparse + sparse.transpose(-1, -2))

    def _dropedge(self, adj):
        if not self.training or self.dropedge <= 0:
            return adj
        keep = torch.rand_like(adj).ge(self.dropedge).to(adj.dtype)
        keep = torch.triu(keep, diagonal=1)
        keep = keep + keep.transpose(-1, -2)
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        if adj.dim() == 3:
            eye = eye.unsqueeze(0)
        return adj * keep + eye * adj

    def _build_adj(self, de_feats):
        static_adj = 0.5 * (self.adj + self.adj.transpose(0, 1))
        static_adj = F.relu(static_adj)
        static_adj = static_adj.unsqueeze(0).expand(de_feats.size(0), -1, -1)

        proj = self.de_proj(de_feats)
        proj = F.normalize(proj, dim=-1)
        dynamic_adj = torch.bmm(proj, proj.transpose(1, 2))
        dynamic_adj = F.relu(dynamic_adj)

        adj = static_adj + self.dyn_alpha * dynamic_adj
        adj = self._sparsify(adj)
        adj = self._dropedge(adj)
        eye = torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye

        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)

    def forward(self, x):
        x = self._extract_de_features(x)
        adj = self._build_adj(x)

        for layer in self.graph_layers:
            x = layer(x, adj)
            x = F.elu(x)
            x = self.dropout(x)

        features = x.reshape(x.size(0), -1)
        logits = self.classifier(features)
        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
