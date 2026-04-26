from .simple import SimpleLinear, SimpleMLP
from .eegnet import EEGNet
from .rnn import EEGGRU, EEGLSTM

MODEL_DICT = {
    "SimpleLinear": SimpleLinear,
    "SimpleMLP": SimpleMLP,
    "EEGNet": EEGNet,
    "EEGGRU": EEGGRU,
    "EEGLSTM": EEGLSTM,
}

__all__ = ["SimpleLinear", "SimpleMLP", "EEGNet", "EEGGRU", "EEGLSTM", "MODEL_DICT"]
