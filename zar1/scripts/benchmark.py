"""Evaluation script for ZAR-1.

Runs MMLU-Pro, GPQA Diamond, HumanEval (via lm-evaluation-harness when
available), and a custom Zulu/Xhosa MMLU translation set. Logs to WandB and
writes results to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.model import ZAR1Config, ZAR1Model  # noqa: E402


def load_model(config_path: str, ckpt_path: str | None) -> ZAR1Model:
    config = ZAR1Config.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
    model.eval()
    return model


def _logprob_of_choice(model: ZAR1Model, tokenizer, prompt: str, choice: str) -> float:
    """Score ``prompt + choice`` with summed log-probabilities of ``choice`` tokens."""
    full = tokenizer(prompt + choice, return_tensors="pt").input_ids
    plen = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    full = full.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(full).logits
    log_probs = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
    target = full[0, 1:]
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(token_lp[plen - 1 :].sum().item())


def run_multichoice(model, tokenizer, examples: list[dict]) -> float:
    """Compute accuracy on a list of {question, choices, answer_idx} dicts."""
    correct = 0
    for ex in examples:
        scores = [
            _logprob_of_choice(model, tokenizer, ex["question"] + "\nAnswer: ", c)
            for c in ex["choices"]
        ]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == ex["answer_idx"]:
            correct += 1
    return correct / max(1, len(examples))


def try_lm_eval(model: ZAR1Model, tokenizer, tasks: list[str]) -> dict:
    """Best-effort handoff to lm-evaluation-harness; falls back to a stub."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print(f"lm-eval not installed; skipping tasks: {tasks}")
        return {t: None for t in tasks}

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks)
    return {t: res["results"].get(t) for t in tasks}


def load_african_mmlu(path: str) -> list[dict]:
    """Load Zulu/Xhosa-translated MMLU JSONL (one example per line)."""
    examples = []
    if not os.path.exists(path):
        return examples
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if "question" in ex and "choices" in ex and "answer_idx" in ex:
                examples.append(ex)
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default="benchmark_results.json")
    ap.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--zulu-set", default="data/african/mmlu_zu.jsonl")
    ap.add_argument("--xhosa-set", default="data/african/mmlu_xh.jsonl")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model = load_model(args.config, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    results: dict = {}

    print("Running standard benchmarks via lm-eval...")
    results["lm_eval"] = try_lm_eval(
        model, tokenizer, ["mmlu_pro", "gpqa_diamond", "humaneval"]
    )

    print("Running African-language evaluations...")
    zu = load_african_mmlu(args.zulu_set)
    xh = load_african_mmlu(args.xhosa_set)
    results["zulu_mmlu_acc"] = run_multichoice(model, tokenizer, zu) if zu else None
    results["xhosa_mmlu_acc"] = run_multichoice(model, tokenizer, xh) if xh else None

    print(json.dumps(results, indent=2))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    if args.wandb:
        import wandb

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run = wandb.init(
            project=cfg.get("logging", {}).get("wandb_project", "zar1"),
            name="benchmark",
            config=cfg,
        )
        run.log(_flatten(results))
        run.finish()


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)) or v is None:
            out[key] = v
    return out


if __name__ == "__main__":
    main()
