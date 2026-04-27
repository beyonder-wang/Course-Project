from .simple import SimpleLinear, SimpleMLP
from .eegnet import EEGNet
from .rnn import EEGGRU, EEGLSTM
from .simclr_model import SimCLREncoder, MoESimCLREncoder
from .contrastive_loss import NTXentLoss
from .augmentations import (
    GaussianNoise, ChannelDropout, TimeShift, Compose, SimCLRTransform,
)
from .channel_adapter import ChannelAdapter, Phase2SimCLR, Phase2MoESimCLR
from .moe import MoELayer

MODEL_DICT = {
    "SimpleLinear": SimpleLinear,
    "SimpleMLP": SimpleMLP,
    "EEGNet": EEGNet,
    "EEGGRU": EEGGRU,
    "EEGLSTM": EEGLSTM,
}

__all__ = [
    "SimpleLinear", "SimpleMLP", "EEGNet", "EEGGRU", "EEGLSTM",
    "SimCLREncoder", "MoESimCLREncoder",
    "NTXentLoss",
    "GaussianNoise", "ChannelDropout", "TimeShift", "Compose", "SimCLRTransform",
    "ChannelAdapter", "Phase2SimCLR", "Phase2MoESimCLR",
    "MoELayer", "MODEL_DICT",
]
