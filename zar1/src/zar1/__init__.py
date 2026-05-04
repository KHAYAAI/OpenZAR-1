"""ZAR-1: Sovereign verifiable Recurrent-Depth MoE Transformer."""

from zar1.act import ACTHalting
from zar1.model import ZAR1Config, ZAR1Model
from zar1.moe import MoEFeedForward
from zar1.recurrent_block import RecurrentTransformerBlock

__version__ = "0.1.0"

__all__ = [
    "ACTHalting",
    "MoEFeedForward",
    "RecurrentTransformerBlock",
    "ZAR1Config",
    "ZAR1Model",
]
