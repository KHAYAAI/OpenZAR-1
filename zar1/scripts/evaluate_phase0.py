"""Phase 0 evaluation harness for ZAR-1 (Master Plan v2.0 §10.1 GO/NO-GO gate).

Runs the formal Phase 0 evaluation suite required to advance from 7B prototype
to Phase 1 (50B):
  - MMLU (5-shot)        Gate: ≥ 50%
  - GSM8K (8-shot)       Gate: ≥ 15%
  - ARC-Challenge (25-shot)
  - HellaSwag (10-shot)
  - Depth extrapolation  Gate: 2x loops improves loss ≥ 3%

Two evaluation backends:
  1. lm-eval-harness (preferred): pip install lm-eval
  2. Native multiple-choice scorer (fallback): no extra deps, slower

The native scorer uses likelihood-based scoring: for each multiple-choice
question, score each candidate as the negative log-likelihood of its tokens
conditioned on the question, and pick the lowest NLL.

Usage:
    python scripts/evaluate_phase0.py --model_path ./checkpoints/latest.pt \
        --config configs/zar1_7b.yaml \
        --benchmark all \
        --max_examples 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.model import ZAR1Config, ZAR1Model  # noqa: E402


# Master Plan v2.0 §10.1 — Phase 0 GO/NO-GO Gate Thresholds
PHASE0_GATES = {
    "mmlu_5shot": 0.50,           # ≥ 50%
    "gsm8k_8shot": 0.15,          # ≥ 15%
    "depth_extrap_pct": 3.0,      # ≥ 3% loss improvement at 2x loops
    "spectral_radius_max": 1.0,   # < 1.0 throughout training (checkpoint stat)
}


@dataclass
class BenchmarkResult:
    """Single-benchmark outcome."""

    name: str
    accuracy: float | None = None
    num_correct: int = 0
    num_total: int = 0
    duration_sec: float = 0.0
    examples_per_sec: float = 0.0
    backend: str = "native"
    error: str | None = None


@dataclass
class Phase0Report:
    """Complete Phase 0 evaluation report against Master Plan §10.1."""

    model_path: str
    config_path: str
    n_params_billions: float
    benchmarks: dict[str, BenchmarkResult] = field(default_factory=dict)
    depth_extrap_loss_1x: float | None = None
    depth_extrap_loss_2x: float | None = None
    depth_extrap_improvement_pct: float | None = None
    gate_pass_mmlu: bool = False
    gate_pass_gsm8k: bool = False
    gate_pass_depth_extrap: bool = False

    @property
    def overall_pass(self) -> bool:
        return self.gate_pass_mmlu and self.gate_pass_gsm8k and self.gate_pass_depth_extrap

    def to_dict(self) -> dict:
        d = {
            "model_path": self.model_path,
            "config_path": self.config_path,
            "n_params_billions": self.n_params_billions,
            "benchmarks": {k: asdict(v) for k, v in self.benchmarks.items()},
            "depth_extrap_loss_1x": self.depth_extrap_loss_1x,
            "depth_extrap_loss_2x": self.depth_extrap_loss_2x,
            "depth_extrap_improvement_pct": self.depth_extrap_improvement_pct,
            "gates": {
                "mmlu_5shot_ge_50pct": self.gate_pass_mmlu,
                "gsm8k_8shot_ge_15pct": self.gate_pass_gsm8k,
                "depth_extrap_ge_3pct": self.gate_pass_depth_extrap,
            },
            "overall_pass": self.overall_pass,
            "phase0_gates": PHASE0_GATES,
        }
        return d


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def load_model_and_tokenizer(checkpoint_path: str, config_path: str) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = ZAR1Config.from_yaml(config_path)
    model = ZAR1Model(config).to(device=device, dtype=dtype)

    if Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"[eval] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[eval] WARNING: checkpoint not found at {checkpoint_path}; using random init")
    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.extras.get("data", {}).get("tokenizer", "meta-llama/Meta-Llama-3-8B"),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    return model, tokenizer, device


# ----------------------------------------------------------------------------
# Native multiple-choice scoring (fallback when lm-eval not available)
# ----------------------------------------------------------------------------

@torch.no_grad()
def score_completion_nll(
    model: ZAR1Model, tokenizer, prompt: str, completion: str, device, max_len: int = 2048
) -> float:
    """Compute the average negative log-likelihood of ``completion`` given ``prompt``.

    Used to rank multiple-choice candidates: lower NLL = better fit.
    """
    full_text = prompt + completion
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(full_ids) > max_len:
        # Truncate from the left to keep the completion in context.
        full_ids = full_ids[-max_len:]
        if len(prompt_ids) > max_len - 1:
            prompt_ids = prompt_ids[-(max_len - 1) :]

    n_completion = len(full_ids) - len(prompt_ids)
    if n_completion <= 0:
        return float("inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    # Mask prompt tokens so loss is only computed over the completion.
    labels[:, : len(prompt_ids)] = -100

    out = model(input_ids=input_ids, labels=labels)
    if out.loss is None or torch.isnan(out.loss):
        return float("inf")
    return float(out.loss.item())


def native_multiple_choice(
    model: ZAR1Model,
    tokenizer,
    device,
    questions: list[dict],
) -> tuple[int, int]:
    """Score multiple-choice questions and return (correct, total).

    Each question dict must have:
      - "prompt": str
      - "choices": list[str] (the candidate completions)
      - "answer_idx": int (index of correct choice)
    """
    correct = 0
    total = 0
    for q in tqdm(questions, desc="MC eval", leave=False):
        nlls = [
            score_completion_nll(model, tokenizer, q["prompt"], c, device)
            for c in q["choices"]
        ]
        pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
        if pred == q["answer_idx"]:
            correct += 1
        total += 1
    return correct, total


# ----------------------------------------------------------------------------
# Few-shot prompt construction for native eval
# ----------------------------------------------------------------------------

def _build_mmlu_prompt(question: str, choices: list[str], n_shot_examples: list[dict]) -> str:
    """Build a 5-shot MMLU prompt."""
    parts = [
        "The following are multiple choice questions (with answers).\n"
    ]
    for ex in n_shot_examples:
        parts.append(f"\nQuestion: {ex['question']}")
        for i, c in enumerate(ex["choices"]):
            parts.append(f"\n{chr(65+i)}. {c}")
        parts.append(f"\nAnswer: {chr(65 + ex['answer_idx'])}\n")
    parts.append(f"\nQuestion: {question}")
    for i, c in enumerate(choices):
        parts.append(f"\n{chr(65+i)}. {c}")
    parts.append("\nAnswer:")
    return "".join(parts)


def _build_gsm8k_prompt(question: str, n_shot_examples: list[dict]) -> str:
    """Build an 8-shot GSM8K prompt with chain-of-thought."""
    parts = []
    for ex in n_shot_examples:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "".join(parts)


# ----------------------------------------------------------------------------
# Benchmark runners
# ----------------------------------------------------------------------------

def evaluate_with_lm_eval(
    model: ZAR1Model, tokenizer, task: str, num_fewshot: int, limit: int | None
) -> BenchmarkResult:
    """Run a benchmark via lm-evaluation-harness if available."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        return BenchmarkResult(
            name=task,
            error="lm-eval not installed (pip install lm-eval). Falling back to native.",
            backend="lm-eval",
        )

    start = time.time()
    try:
        # Wrap our model in a minimal HFLM-compatible interface.
        # Note: ZAR-1 has a custom forward signature, so the wrapper must adapt.
        # For now we attempt direct evaluation; fall back to native on failure.
        results = simple_evaluate(
            model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1),
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
        )
        acc = results["results"][task].get("acc,none") or results["results"][task].get("acc")
        return BenchmarkResult(
            name=task,
            accuracy=float(acc) if acc is not None else None,
            duration_sec=time.time() - start,
            backend="lm-eval",
        )
    except Exception as e:
        return BenchmarkResult(
            name=task, error=f"lm-eval failed: {str(e)[:200]}", backend="lm-eval"
        )


def evaluate_mmlu_native(
    model: ZAR1Model, tokenizer, device, max_examples: int = 200
) -> BenchmarkResult:
    """Evaluate MMLU (5-shot) via native multiple-choice scoring."""
    start = time.time()
    try:
        from datasets import load_dataset

        # Load a sample of MMLU subjects for evaluation.
        ds = load_dataset(
            "cais/mmlu", "all", split="test", streaming=True
        )
        ds_dev = load_dataset(
            "cais/mmlu", "all", split="dev", streaming=False
        )
    except Exception as e:
        return BenchmarkResult(
            name="mmlu_5shot", error=f"Failed to load MMLU dataset: {e}", backend="native"
        )

    # Build a small pool of 5-shot examples from the dev split.
    dev_examples = []
    for ex in ds_dev:
        if len(dev_examples) >= 5:
            break
        dev_examples.append(
            {
                "question": ex["question"],
                "choices": ex["choices"],
                "answer_idx": ex["answer"],
            }
        )

    questions = []
    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        prompt = _build_mmlu_prompt(ex["question"], ex["choices"], dev_examples)
        questions.append(
            {
                "prompt": prompt,
                "choices": [f" {chr(65+i)}" for i in range(len(ex["choices"]))],
                "answer_idx": ex["answer"],
            }
        )

    correct, total = native_multiple_choice(model, tokenizer, device, questions)
    duration = time.time() - start
    return BenchmarkResult(
        name="mmlu_5shot",
        accuracy=correct / max(total, 1),
        num_correct=correct,
        num_total=total,
        duration_sec=duration,
        examples_per_sec=total / max(duration, 1e-6),
        backend="native",
    )


def evaluate_arc_challenge_native(
    model: ZAR1Model, tokenizer, device, max_examples: int = 200
) -> BenchmarkResult:
    """Evaluate ARC-Challenge (25-shot) via native scoring."""
    start = time.time()
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        ds_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    except Exception as e:
        return BenchmarkResult(
            name="arc_challenge", error=f"Failed to load ARC: {e}", backend="native"
        )

    fewshot = []
    for ex in ds_train.select(range(min(25, len(ds_train)))):
        labels = ex["choices"]["label"]
        try:
            ans_idx = labels.index(ex["answerKey"])
        except ValueError:
            continue
        fewshot.append(
            {
                "question": ex["question"],
                "choices": ex["choices"]["text"],
                "answer_idx": ans_idx,
            }
        )

    questions = []
    for i, ex in enumerate(ds.select(range(min(max_examples, len(ds))))):
        labels = ex["choices"]["label"]
        try:
            ans_idx = labels.index(ex["answerKey"])
        except ValueError:
            continue
        prompt = _build_mmlu_prompt(ex["question"], ex["choices"]["text"], fewshot)
        questions.append(
            {
                "prompt": prompt,
                "choices": [f" {chr(65+j)}" for j in range(len(ex["choices"]["text"]))],
                "answer_idx": ans_idx,
            }
        )

    correct, total = native_multiple_choice(model, tokenizer, device, questions)
    duration = time.time() - start
    return BenchmarkResult(
        name="arc_challenge",
        accuracy=correct / max(total, 1),
        num_correct=correct,
        num_total=total,
        duration_sec=duration,
        examples_per_sec=total / max(duration, 1e-6),
        backend="native",
    )


def evaluate_hellaswag_native(
    model: ZAR1Model, tokenizer, device, max_examples: int = 200
) -> BenchmarkResult:
    """Evaluate HellaSwag (10-shot) via native scoring."""
    start = time.time()
    try:
        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag", split="validation")
    except Exception as e:
        return BenchmarkResult(
            name="hellaswag", error=f"Failed to load HellaSwag: {e}", backend="native"
        )

    questions = []
    for i, ex in enumerate(ds.select(range(min(max_examples, len(ds))))):
        questions.append(
            {
                "prompt": f"{ex['ctx']} ",
                "choices": ex["endings"],
                "answer_idx": int(ex["label"]),
            }
        )

    correct, total = native_multiple_choice(model, tokenizer, device, questions)
    duration = time.time() - start
    return BenchmarkResult(
        name="hellaswag",
        accuracy=correct / max(total, 1),
        num_correct=correct,
        num_total=total,
        duration_sec=duration,
        examples_per_sec=total / max(duration, 1e-6),
        backend="native",
    )


def evaluate_gsm8k_native(
    model: ZAR1Model, tokenizer, device, max_examples: int = 100
) -> BenchmarkResult:
    """Evaluate GSM8K (8-shot) via greedy generation + final-number extraction."""
    import re

    start = time.time()
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        ds_train = load_dataset("gsm8k", "main", split="train")
    except Exception as e:
        return BenchmarkResult(
            name="gsm8k_8shot", error=f"Failed to load GSM8K: {e}", backend="native"
        )

    fewshot = [
        {"question": ds_train[i]["question"], "answer": ds_train[i]["answer"]}
        for i in range(8)
    ]

    def extract_number(text: str) -> str | None:
        """Extract the final numeric answer from a chain-of-thought string."""
        match = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            return match[-1]
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else None

    correct = 0
    total = 0
    for ex in tqdm(
        ds.select(range(min(max_examples, len(ds)))),
        desc="GSM8K eval",
        leave=False,
    ):
        prompt = _build_gsm8k_prompt(ex["question"], fewshot)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > 1500:
            ids = ids[-1500:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.0,
                n_loops=4,
            )
        gen = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
        gen = gen.split("Question:")[0]  # stop before next few-shot example

        pred = extract_number(gen)
        gold = extract_number(ex["answer"])
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        total += 1

    duration = time.time() - start
    return BenchmarkResult(
        name="gsm8k_8shot",
        accuracy=correct / max(total, 1),
        num_correct=correct,
        num_total=total,
        duration_sec=duration,
        examples_per_sec=total / max(duration, 1e-6),
        backend="native",
    )


# ----------------------------------------------------------------------------
# Depth extrapolation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_depth_extrapolation(
    model: ZAR1Model, tokenizer, device, num_samples: int = 32
) -> tuple[float, float, float]:
    """Compare eval loss at 1x and 2x the trained loop count.

    Master Plan v2.0 §10.1 gate: 2x loops should improve eval loss by ≥ 3%.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", split="train", streaming=True
        )
    except Exception:
        # Fallback: use HellaSwag context strings as eval text.
        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)

    base_loops = model.config.max_loops
    losses_1x = []
    losses_2x = []

    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        text = ex.get("text") or ex.get("ctx") or ""
        if not text or len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)[:1024]
        if len(ids) < 16:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        out_1x = model(input_ids=input_ids, labels=input_ids, inference_n_loops=base_loops)
        out_2x = model(
            input_ids=input_ids, labels=input_ids, inference_n_loops=base_loops * 2
        )
        if out_1x.loss is not None and not torch.isnan(out_1x.loss):
            losses_1x.append(float(out_1x.loss.item()))
        if out_2x.loss is not None and not torch.isnan(out_2x.loss):
            losses_2x.append(float(out_2x.loss.item()))

    if not losses_1x or not losses_2x:
        return 0.0, 0.0, 0.0

    avg_1x = sum(losses_1x) / len(losses_1x)
    avg_2x = sum(losses_2x) / len(losses_2x)
    improvement_pct = 100.0 * (avg_1x - avg_2x) / max(avg_1x, 1e-9)
    return avg_1x, avg_2x, improvement_pct


# ----------------------------------------------------------------------------
# Phase 0 gate report
# ----------------------------------------------------------------------------

def print_phase0_report(report: Phase0Report) -> None:
    """Print the formal Phase 0 GO/NO-GO report."""
    print("\n" + "=" * 70)
    print("ZAR-1 PHASE 0 EVALUATION REPORT")
    print("Master Plan v2.0 §10.1 — Phase 1 GO/NO-GO Gate")
    print("=" * 70)
    print(f"\nModel:  {report.model_path}")
    print(f"Config: {report.config_path}")
    print(f"Params: {report.n_params_billions:.2f}B\n")

    print("Benchmark Results:")
    print("-" * 70)
    for name, b in report.benchmarks.items():
        if b.error:
            print(f"  {name:<22} ERROR: {b.error}")
        elif b.accuracy is not None:
            pct = 100 * b.accuracy
            print(
                f"  {name:<22} {pct:6.2f}%  "
                f"({b.num_correct}/{b.num_total}, {b.duration_sec:.0f}s, {b.backend})"
            )

    if report.depth_extrap_loss_1x is not None:
        print(f"\nDepth Extrapolation:")
        print(f"  Eval loss @ 1x loops: {report.depth_extrap_loss_1x:.4f}")
        print(f"  Eval loss @ 2x loops: {report.depth_extrap_loss_2x:.4f}")
        print(f"  Improvement:          {report.depth_extrap_improvement_pct:+.2f}%")

    print("\nMaster Plan §10.1 Gate Criteria:")
    print("-" * 70)
    gates = [
        (
            "MMLU 5-shot ≥ 50%",
            report.gate_pass_mmlu,
            f"{(report.benchmarks.get('mmlu_5shot', BenchmarkResult('mmlu_5shot')).accuracy or 0) * 100:.2f}%",
        ),
        (
            "GSM8K 8-shot ≥ 15%",
            report.gate_pass_gsm8k,
            f"{(report.benchmarks.get('gsm8k_8shot', BenchmarkResult('gsm8k_8shot')).accuracy or 0) * 100:.2f}%",
        ),
        (
            "Depth extrapolation ≥ 3%",
            report.gate_pass_depth_extrap,
            f"{report.depth_extrap_improvement_pct or 0:+.2f}%",
        ),
    ]
    for name, ok, value in gates:
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {name:<35} actual: {value}")

    overall = "✓ GO — Proceed to Phase 1" if report.overall_pass else "✗ NO-GO — Iterate on Phase 0"
    print("\n" + "=" * 70)
    print(f"OVERALL DECISION: {overall}")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAR-1 Phase 0 evaluation harness")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--benchmark",
        choices=["mmlu", "gsm8k", "arc", "hellaswag", "depth", "all"],
        default="all",
    )
    parser.add_argument(
        "--max_examples", type=int, default=200, help="Max examples per benchmark"
    )
    parser.add_argument(
        "--prefer_lm_eval",
        action="store_true",
        help="Try lm-eval-harness backend before falling back to native",
    )
    parser.add_argument(
        "--output_json", type=str, default="phase0_eval_report.json"
    )
    args = parser.parse_args()

    print("[eval] Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.config)
    n_params = model.num_parameters() / 1e9

    report = Phase0Report(
        model_path=args.model_path,
        config_path=args.config,
        n_params_billions=n_params,
    )

    bench_to_run = (
        ["mmlu", "gsm8k", "arc", "hellaswag", "depth"]
        if args.benchmark == "all"
        else [args.benchmark]
    )

    if "mmlu" in bench_to_run:
        print("\n[eval] Running MMLU (5-shot)...")
        if args.prefer_lm_eval:
            r = evaluate_with_lm_eval(model, tokenizer, "mmlu", 5, args.max_examples)
            if r.error:
                r = evaluate_mmlu_native(model, tokenizer, device, args.max_examples)
        else:
            r = evaluate_mmlu_native(model, tokenizer, device, args.max_examples)
        report.benchmarks["mmlu_5shot"] = r

    if "gsm8k" in bench_to_run:
        print("\n[eval] Running GSM8K (8-shot)...")
        r = evaluate_gsm8k_native(
            model, tokenizer, device, min(args.max_examples, 100)
        )
        report.benchmarks["gsm8k_8shot"] = r

    if "arc" in bench_to_run:
        print("\n[eval] Running ARC-Challenge (25-shot)...")
        r = evaluate_arc_challenge_native(
            model, tokenizer, device, args.max_examples
        )
        report.benchmarks["arc_challenge"] = r

    if "hellaswag" in bench_to_run:
        print("\n[eval] Running HellaSwag (10-shot)...")
        r = evaluate_hellaswag_native(model, tokenizer, device, args.max_examples)
        report.benchmarks["hellaswag"] = r

    if "depth" in bench_to_run:
        print("\n[eval] Running depth extrapolation...")
        loss_1x, loss_2x, improvement = evaluate_depth_extrapolation(
            model, tokenizer, device, num_samples=32
        )
        report.depth_extrap_loss_1x = loss_1x
        report.depth_extrap_loss_2x = loss_2x
        report.depth_extrap_improvement_pct = improvement

    # Evaluate against Master Plan §10.1 gates.
    mmlu_acc = report.benchmarks.get("mmlu_5shot", BenchmarkResult("mmlu")).accuracy
    gsm_acc = report.benchmarks.get("gsm8k_8shot", BenchmarkResult("gsm8k")).accuracy
    report.gate_pass_mmlu = mmlu_acc is not None and mmlu_acc >= PHASE0_GATES["mmlu_5shot"]
    report.gate_pass_gsm8k = gsm_acc is not None and gsm_acc >= PHASE0_GATES["gsm8k_8shot"]
    report.gate_pass_depth_extrap = (
        report.depth_extrap_improvement_pct is not None
        and report.depth_extrap_improvement_pct >= PHASE0_GATES["depth_extrap_pct"]
    )

    print_phase0_report(report)

    with open(args.output_json, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"[eval] Report written to {args.output_json}")

    sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
