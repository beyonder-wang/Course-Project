from .simple import SimpleLinear, SimpleMLP, DENet
from .eegnet import EEGNet
from .rnn import EEGGRU, EEGLSTM
from .mamba_model import EEGMamba
from .kan_backbones import (
    SimpleMLP_KAN, EEGLSTM_KAN, EEGGRU_KAN, EEGNet_KAN, EEGMamba_KAN,
)
from .attention_eeg import (
    EEGNet_SE, EEGNet_SimAM, EEGNet_SimAM_SE,
    SEBlock, SimAM, SpatialAttention1D,
)
from .simclr_model import SimCLREncoder, MoESimCLREncoder
from .contrastive_loss import NTXentLoss
from .augmentations import (
    GaussianNoise, ChannelDropout, TimeShift, Compose, SimCLRTransform,
)
from .channel_adapter import ChannelAdapter, Phase2SimCLR, Phase2MoESimCLR
from .moe import MoELayer
from .kan import KANLinear, KANMLP
from .band_decomposition import BandDecomposition
from .multiband_loss import MultiBandNTXentLoss
from .multiband_simclr import MultiBandSimCLREncoder, MultiBandMoESimCLREncoder
from .mi_cnn import MICNN
from .fbcnet import FBCNet
from .tcnet import EEGTCNet
from .shallownet import ShallowConvNet
from .atcnet import ATCNet
from .conformer import EEGConformer
from .dgcnn import DGCNN, DGCNN_RG
from .rgnn import RGNN
from .emotion_dl import EmotionDLHead
from .domain_adversarial import DomainAdversarialHead
from .seed_graphormer import SEEDGraphormer
from .seed_asymnet import SEEDAsymNet
from .seed_bandgraph import SEEDBandGraphNet

MODEL_DICT = {
    # Original baselines
    "SimpleLinear": SimpleLinear,
    "SimpleMLP": SimpleMLP,
    "DENet": DENet,
    "EEGNet": EEGNet,
    "EEGGRU": EEGGRU,
    "EEGLSTM": EEGLSTM,
    "EEGMamba": EEGMamba,
    # KAN classifier variants
    "SimpleMLP_KAN": SimpleMLP_KAN,
    "EEGLSTM_KAN": EEGLSTM_KAN,
    "EEGGRU_KAN": EEGGRU_KAN,
    "EEGNet_KAN": EEGNet_KAN,
    "EEGMamba_KAN": EEGMamba_KAN,
    # Attention-enhanced variants
    "EEGNet_SE": EEGNet_SE,
    "EEGNet_SimAM": EEGNet_SimAM,
    "EEGNet_SimAM_SE": EEGNet_SimAM_SE,
    "DGCNN": DGCNN,
    "DGCNN_RG": DGCNN_RG,
    "RGNN": RGNN,
    "SEEDGraphormer": SEEDGraphormer,
    "SEEDAsymNet": SEEDAsymNet,
    "SEEDBandGraphNet": SEEDBandGraphNet,
    # Custom motor imagery CNN
    "MICNN": MICNN,
    # FBCNet: multi-band + spatial conv + variance pooling
    "FBCNet": FBCNet,
    # EEG-TCNet: EEGNet + TCN
    "EEGTCNet": EEGTCNet,
    # ShallowConvNet: FBCSP-inspired shallow CNN
    "ShallowConvNet": ShallowConvNet,
    # ATCNet: Attention TCN for motor imagery
    "ATCNet": ATCNet,
    # EEGConformer: CNN + Transformer hybrid
    "EEGConformer": EEGConformer,
}

__all__ = [
    # Supervised baselines
    "SimpleLinear", "SimpleMLP", "DENet", "EEGNet", "EEGGRU", "EEGLSTM", "EEGMamba",
    # KAN classifier variants
    "SimpleMLP_KAN", "EEGLSTM_KAN", "EEGGRU_KAN", "EEGNet_KAN", "EEGMamba_KAN",
    # Attention variants
    "EEGNet_SE", "EEGNet_SimAM", "EEGNet_SimAM_SE",
    "DGCNN", "DGCNN_RG", "RGNN", "SEEDGraphormer", "SEEDAsymNet", "SEEDBandGraphNet",
    "EmotionDLHead", "DomainAdversarialHead",
    "SEBlock", "SimAM", "SpatialAttention1D",
    # SimCLR encoders
    "SimCLREncoder", "MoESimCLREncoder",
    "MultiBandSimCLREncoder", "MultiBandMoESimCLREncoder",
    # Loss functions
    "NTXentLoss", "MultiBandNTXentLoss",
    # Augmentations
    "GaussianNoise", "ChannelDropout", "TimeShift", "Compose", "SimCLRTransform",
    # Channel adaptation
    "ChannelAdapter", "Phase2SimCLR", "Phase2MoESimCLR",
    # MoE
    "MoELayer",
    # KAN
    "KANLinear", "KANMLP",
    # Band decomposition
    "BandDecomposition",
    # ShallowConvNet & ATCNet
    "ShallowConvNet", "ATCNet",
    # Registry
    "MODEL_DICT",
]
