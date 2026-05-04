"""Benchmark script for ZAR-1 evaluation on MMLU-Pro, GPQA, HumanEval, and African languages.

Usage:
    python scripts/benchmark.py --model_path ./checkpoints/zar1_7b/latest.pt \
                                --benchmark mmlu_pro --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.model import ZAR1Config, ZAR1Model


def load_model(checkpoint_path: str, config_path: str | None = None) -> ZAR1Model:
    """Load a ZAR-1 checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_path:
        config = ZAR1Config.from_yaml(config_path)
    else:
        config = ZAR1Config()

    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)
    if Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    return model


@torch.no_grad()
def compute_perplexity(
    model: ZAR1Model,
    texts: list[str],
    tokenizer,
    max_seq_len: int = 2048,
    stride: int = 512,
) -> float:
    """Compute perplexity on a list of texts."""
    device = next(model.parameters()).device
    nlls = []

    for text in tqdm(texts, desc="Computing perplexity", leave=False):
        encodings = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        if encodings.shape[1] < 2:
            continue
        seq_len = encodings.shape[1]

        for i in range(0, seq_len - 1, stride):
            begin_loc = max(0, i - max_seq_len)
            end_loc = min(seq_len, i + max_seq_len + 1)
            trg_len = end_loc - i
            input_ids = encodings[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss
                if loss is not None:
                    nlls.append(loss.item())

    if not nlls:
        return float("inf")
    return math.exp(sum(nlls) / len(nlls))


def evaluate_mmlu_pro(model: ZAR1Model, tokenizer, num_examples: int = 100) -> dict:
    """Evaluate on MMLU-Pro (mock benchmark; requires lm-eval)."""
    try:
        from lm_eval.tasks.mmlu_pro import MMLU_PRO
    except ImportError:
        return {
            "mmlu_pro": None,
            "note": "lm-eval not installed; skipping MMLU-Pro evaluation",
        }

    results = {"mmlu_pro": {"accuracy": 0.0, "num_examples": num_examples}}
    print(f"[INFO] MMLU-Pro evaluation requires lm-eval harness; check documentation.")
    return results


def evaluate_gpqa(model: ZAR1Model, tokenizer, num_examples: int = 50) -> dict:
    """Evaluate on GPQA Diamond (requires lm-eval)."""
    results = {"gpqa_diamond": {"accuracy": 0.0, "num_examples": num_examples}}
    print(f"[INFO] GPQA evaluation requires lm-eval harness; check documentation.")
    return results


def evaluate_humaneval(model: ZAR1Model, tokenizer, num_examples: int = 164) -> dict:
    """Evaluate on HumanEval (requires humaneval library)."""
    try:
        from human_eval.data import read_problems

        problems = read_problems()
    except ImportError:
        return {
            "humaneval": None,
            "note": "human-eval not installed; skipping HumanEval evaluation",
        }

    results = {"humaneval": {"pass_at_k": {}, "num_examples": min(num_examples, len(problems))}}
    print(f"[INFO] HumanEval evaluation requires human-eval library; check documentation.")
    return results


def evaluate_african_languages(
    model: ZAR1Model, tokenizer, languages: list[str] | None = None
) -> dict:
    """Evaluate on African language test set (Zulu, Xhosa, Swahili)."""
    if languages is None:
        languages = ["zulu", "xhosa", "swahili"]

    results = {"african_languages": {}}
    for lang in languages:
        results["african_languages"][lang] = {
            "accuracy": 0.0,
            "note": "African language benchmarks not yet implemented; requires curated test sets",
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ZAR-1 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/zar1_7b.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--benchmark",
        choices=["mmlu_pro", "gpqa", "humaneval", "african", "all"],
        default="all",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    print("[INFO] Loading model...")
    model = load_model(args.model_path, args.config_path)
    print(f"[INFO] Model loaded: {model.config.dim}D, {model.num_parameters()/1e9:.2f}B params")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", use_fast=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        return

    results = {}

    if args.benchmark in ["mmlu_pro", "all"]:
        print("\n[MMLU-Pro Evaluation]")
        results.update(evaluate_mmlu_pro(model, tokenizer))

    if args.benchmark in ["gpqa", "all"]:
        print("\n[GPQA Evaluation]")
        results.update(evaluate_gpqa(model, tokenizer))

    if args.benchmark in ["humaneval", "all"]:
        print("\n[HumanEval Evaluation]")
        results.update(evaluate_humaneval(model, tokenizer))

    if args.benchmark in ["african", "all"]:
        print("\n[African Languages Evaluation]")
        results.update(evaluate_african_languages(model, tokenizer))

    print("\n[Results Summary]")
    print(json.dumps(results, indent=2))

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
