"""ZAR-1: Sovereign verifiable Recurrent-Depth MoE Transformer."""

from zar1.act import ACTHalting
from zar1.model import ZAR1Config, ZAR1Model
from zar1.moe import MoEFeedForward, MLPRouter, PIDBalancer
from zar1.recurrent_block import LoRAAdapter, RecurrentTransformerBlock

__version__ = "0.3.0"

__all__ = [
    "ACTHalting",
    "LoRAAdapter",
    "MLPRouter",
    "MoEFeedForward",
    "PIDBalancer",
    "RecurrentTransformerBlock",
    "ZAR1Config",
    "ZAR1Model",
]
