import torch
import torch.nn as nn


class SimCLREncoder(nn.Module):
    """LSTM encoder + projection head for SimCLR contrastive pre-training.

    The LSTM backbone mirrors EEGLSTM.lstm exactly for direct weight transfer
    after pre-training. The projection head is discarded during fine-tuning.

    Args:
        chans: number of input channels
        hidden_dim: LSTM hidden dimension (default 64)
        num_layers: LSTM layers (default 2)
        proj_dim: projection output dimension (default 128)
        dropout: dropout rate (default 0.3)
        bidirectional: use bidirectional LSTM (default True)
    """

    def __init__(
        self,
        chans,
        hidden_dim=64,
        num_layers=2,
        proj_dim=128,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=chans,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        feat_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def _lstm_forward(self, x):
        """Extract LSTM features. Shared by SimCLREncoder and MoESimCLREncoder."""
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            feat = h_n[-1]
        return feat

    def forward(self, x):
        """Forward pass returning projected features for contrastive loss.

        Args:
            x: (B, C, T) input

        Returns:
            (B, proj_dim) projected features
        """
        feat = self._lstm_forward(x)
        return self.projection(feat)

    def get_encoder_state_dict(self):
        """Return LSTM-only state dict for loading into downstream EEGLSTM."""
        prefix = "lstm."
        return {k[len(prefix):]: v for k, v in self.state_dict().items()
            if k.startswith(prefix)}


class MoESimCLREncoder(SimCLREncoder):
    """SimCLR encoder with MoE layer between LSTM and projection head.

    During pre-training: LSTM → MoE → Projection → NT-Xent loss
    During fine-tuning:  LSTM → MoE → Classifier (projection discarded)

    Additional args:
        moe_num_experts: number of MoE experts (default 4)
        moe_top_k: top-k experts to activate (default 2)
        moe_expert_mult: hidden dim multiplier per expert (default 4)
    """

    def __init__(
        self,
        chans,
        hidden_dim=64,
        num_layers=2,
        proj_dim=128,
        dropout=0.3,
        bidirectional=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_mult=4,
    ):
        super().__init__(
            chans=chans,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        from .moe import MoELayer

        feat_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.moe = MoELayer(
            dim=feat_dim,
            num_experts=moe_num_experts,
            expert_mult=moe_expert_mult,
            top_k=moe_top_k,
            dropout=dropout,
        )

    def forward(self, x):
        """Returns (projection, balance_loss) for contrastive pre-training."""
        feat = self._lstm_forward(x)
        moe_out, balance_loss = self.moe(feat)
        proj = self.projection(moe_out)
        return proj, balance_loss

    def get_encoder_state_dict(self):
        """Return LSTM + MoE state dict for downstream MoE-equipped model."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("projection.")}
