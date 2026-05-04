"""Export a (small) ZAR-1 model to ONNX for downstream Zenith canister inference.

Note: full 7B export to ONNX is impractical for on-chain inference; this script
is intended for distilled student variants (e.g. 1.3B) targeting WASM/candle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.model import ZAR1Config, ZAR1Model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--output", default="zar1.onnx")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-loops", type=int, default=4)
    args = p.parse_args()

    config = ZAR1Config.from_yaml(args.config)
    model = ZAR1Model(config).eval()
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)

    # Wrap to fix inference_n_loops and return only logits.
    class Exportable(torch.nn.Module):
        def __init__(self, m, loops):
            super().__init__()
            self.m = m
            self.loops = loops

        def forward(self, input_ids):
            return self.m(input_ids, inference_n_loops=self.loops).logits

    wrap = Exportable(model, args.n_loops)
    dummy = torch.randint(0, config.vocab_size, (1, args.seq_len), dtype=torch.long)
    torch.onnx.export(
        wrap,
        (dummy,),
        args.output,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
    )
    print(f"Exported ONNX -> {args.output}")


if __name__ == "__main__":
    main()
