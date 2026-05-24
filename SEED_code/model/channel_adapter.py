import torch
import torch.nn as nn


class ChannelAdapter(nn.Module):
    """Per-dataset 1x1 Conv1d to map different channel counts to a unified dimension.

    Args:
        dataset_channels: dict mapping dataset name → channel count,
            e.g. {"MDD": 20, "SEED": 62}
        unified_dim: target channel dimension (default 64)
    """

    def __init__(self, dataset_channels, unified_dim=64):
        super().__init__()
        self.unified_dim = unified_dim
        self.dataset_names = list(dataset_channels.keys())

        adapters = {}
        for name in self.dataset_names:
            in_ch = dataset_channels[name]
            adapters[name] = nn.Conv1d(in_ch, unified_dim, kernel_size=1)
        self.adapters = nn.ModuleDict(adapters)

    def forward(self, x, source_idx):
        """Apply per-sample channel adapter based on source dataset index.

        Args:
            x: (B, C_i, T) — all samples in the batch must be from the same dataset
            source_idx: (B,) — source dataset indices

        Returns:
            (B, unified_dim, T)
        """
        # All samples in a batch should be from the same dataset
        # (DataLoader shuffles individual samples, so mixed-dataset batches can occur)
        unique_src = source_idx.unique()
        if len(unique_src) == 1:
            name = self.dataset_names[unique_src.item()]
            return self.adapters[name](x)
        else:
            # Mixed-dataset batch: process each subset separately
            out = torch.zeros(x.size(0), self.unified_dim, x.size(2),
                              device=x.device, dtype=x.dtype)
            for src in unique_src:
                mask = source_idx == src
                name = self.dataset_names[src.item()]
                out[mask] = self.adapters[name](x[mask])
            return out


class Phase2SimCLR(nn.Module):
    """ChannelAdapter + SimCLREncoder wrapper for Phase 2 pre-training.

    Args:
        dataset_channels: dict mapping dataset name → channel count
        unified_dim: target channel dimension for adapter output
        encoder_kwargs: kwargs passed to SimCLREncoder (chans=unified_dim set automatically)
    """

    def __init__(self, dataset_channels, unified_dim=64, **encoder_kwargs):
        super().__init__()
        self.adapter = ChannelAdapter(dataset_channels, unified_dim=unified_dim)

        from .simclr_model import SimCLREncoder
        self.encoder = SimCLREncoder(chans=unified_dim, **encoder_kwargs)

    def forward(self, x, source_idx):
        x = self.adapter(x, source_idx)
        return self.encoder(x)

    def get_encoder_state_dict(self):
        return self.encoder.get_encoder_state_dict()

    def get_adapter_state_dict(self):
        return self.adapter.state_dict()


class Phase2MoESimCLR(nn.Module):
    """ChannelAdapter + MoESimCLREncoder wrapper for Phase 2 + MoE pre-training.

    Args:
        dataset_channels: dict mapping dataset name → channel count
        unified_dim: target channel dimension for adapter output
        encoder_kwargs: kwargs passed to MoESimCLREncoder
    """

    def __init__(self, dataset_channels, unified_dim=64, **encoder_kwargs):
        super().__init__()
        self.adapter = ChannelAdapter(dataset_channels, unified_dim=unified_dim)

        from .simclr_model import MoESimCLREncoder
        self.encoder = MoESimCLREncoder(chans=unified_dim, **encoder_kwargs)

    def forward(self, x, source_idx):
        x = self.adapter(x, source_idx)
        return self.encoder(x)

    def get_encoder_state_dict(self):
        return self.encoder.get_encoder_state_dict()

    def get_adapter_state_dict(self):
        return self.adapter.state_dict()
