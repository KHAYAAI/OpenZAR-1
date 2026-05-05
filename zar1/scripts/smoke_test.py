"""1B smoke test for ZAR-1 (Master Plan v2.0 Month 1 gate).

Validates that the architecture trains stably before committing to a full 7B run:
  - No NaN/Inf in loss across 100 steps
  - Loss decreases from initialization
  - Spectral radius of recurrent_proj stays < 1.0 throughout
  - Expert load distribution within 3x of uniform
  - Memory usage fits on a single 8-GPU node

Usage:
    python scripts/smoke_test.py --config configs/zar1_1b_smoke_test.yaml
    python scripts/smoke_test.py --config configs/zar1_1b_smoke_test.yaml --steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.data import build_attention_mask_from_doc_ids, build_train_val_loaders  # noqa: E402
from zar1.model import ZAR1Config, ZAR1Model  # noqa: E402


class SmokeTestResult:
    """Container for smoke test outcomes against Master Plan v2.0 §10.1 gate."""

    def __init__(self) -> None:
        self.no_nan: bool = True
        self.loss_decreasing: bool = True
        self.spectral_radius_ok: bool = True
        self.expert_load_ok: bool = True
        self.memory_ok: bool = True

        self.initial_loss: float | None = None
        self.final_loss: float | None = None
        self.max_spectral_radius: float = 0.0
        self.expert_load_ratio: float = 1.0  # max/min ratio
        self.peak_memory_gb: float = 0.0
        self.tokens_per_sec: float = 0.0
        self.errors: list[str] = []

    @property
    def passed(self) -> bool:
        return all(
            [
                self.no_nan,
                self.loss_decreasing,
                self.spectral_radius_ok,
                self.expert_load_ok,
                self.memory_ok,
            ]
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": {
                "no_nan": self.no_nan,
                "loss_decreasing": self.loss_decreasing,
                "spectral_radius_lt_1": self.spectral_radius_ok,
                "expert_load_within_3x": self.expert_load_ok,
                "memory_under_80gb": self.memory_ok,
            },
            "metrics": {
                "initial_loss": self.initial_loss,
                "final_loss": self.final_loss,
                "max_spectral_radius": self.max_spectral_radius,
                "expert_load_max_min_ratio": self.expert_load_ratio,
                "peak_memory_gb": self.peak_memory_gb,
                "tokens_per_sec": self.tokens_per_sec,
            },
            "errors": self.errors,
        }


def _check_for_nans(t: torch.Tensor) -> bool:
    """Return True if tensor contains any NaN or Inf values."""
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def _track_expert_load(routing_probs: torch.Tensor, num_experts: int) -> tuple[float, float]:
    """Compute (max/min) and entropy of top-1 expert assignments.

    Master Plan v2.0 §10.1 gate: All experts within 3x of uniform.
    """
    top1 = routing_probs.argmax(dim=-1).flatten()
    counts = torch.bincount(top1, minlength=num_experts).float()
    counts = counts.clamp(min=1.0)  # avoid div by zero in ratio
    ratio = (counts.max() / counts.min()).item()
    probs = counts / counts.sum()
    entropy = -(probs * probs.log()).sum().item()
    return ratio, entropy


def run_smoke_test(config_path: str, num_steps: int | None = None) -> SmokeTestResult:
    """Run the 1B smoke test and return results."""
    result = SmokeTestResult()

    print(f"[smoke] Loading config: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if num_steps is not None:
        cfg["training"]["steps"] = num_steps

    # Build model on GPU (or CPU if no CUDA).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[smoke] Device: {device}, dtype: {dtype}")

    config = ZAR1Config(
        vocab_size=cfg["model"]["vocab_size"],
        dim=cfg["model"]["dim"],
        num_heads=cfg["model"]["n_heads"],
        num_kv_heads=cfg["model"]["n_kv_heads"],
        prelude_layers=cfg["model"]["prelude_layers"],
        coda_layers=cfg["model"]["coda_layers"],
        max_loops=cfg["model"]["max_loops"],
        num_experts=cfg["model"]["num_experts"],
        top_k=cfg["model"]["top_k"],
        max_seq_len=cfg["model"]["max_seq_len"],
        act_epsilon=cfg["model"]["act_epsilon"],
        act_ponder_tau=cfg["model"]["act_ponder_tau"],
        spectral_max=cfg["model"]["spectral_max"],
        spectral_iters=cfg["model"]["spectral_iters"],
        lora_rank=cfg["model"].get("lora_rank", 16),
        use_lora=cfg["model"].get("use_lora", True),
    )

    model = ZAR1Model(config).to(device=device, dtype=dtype)
    n_params = model.num_parameters()
    print(f"[smoke] Model params: {n_params / 1e9:.3f}B")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Build data loaders.
    train_loader, _val_loader = build_train_val_loaders(
        cfg=cfg["data"],
        seq_len=cfg["training"]["max_seq_len"],
        micro_batch_size=cfg["training"]["micro_batch_size"],
    )

    # Run smoke training loop.
    total_steps = cfg["training"]["steps"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    grad_clip = cfg["training"]["gradient_clip"]
    num_experts = cfg["model"]["num_experts"]

    losses: list[float] = []
    spectral_radii: list[float] = []
    expert_load_ratios: list[float] = []

    model.train()
    optim.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)

    start_time = time.time()
    total_tokens = 0

    for step in range(total_steps):
        accum_loss = 0.0
        for _micro in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            out = model(input_ids=input_ids, labels=labels)

            if _check_for_nans(out.loss):
                result.no_nan = False
                result.errors.append(f"NaN/Inf in loss at step {step}")
                print(f"[smoke] FAIL: NaN/Inf at step {step}")
                return result

            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += float(out.loss.detach().item()) / grad_accum
            total_tokens += input_ids.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # Apply spectral constraint and track radius.
        sigma = float(model.apply_spectral_constraints().item())
        spectral_radii.append(sigma)

        losses.append(accum_loss)

        # Probe expert load distribution every 10 steps.
        if step % 10 == 0:
            with torch.no_grad():
                # Sample a small input to measure router behavior.
                probe_input = input_ids[:1, :128]
                probe_h = model.embed(probe_input)
                for blk in model.prelude:
                    probe_h = blk(probe_h, None)
                _, routing_probs, _ = model.recurrent_block.moe(
                    model.recurrent_block.norm2(probe_h)
                )
                ratio, entropy = _track_expert_load(routing_probs, num_experts)
                expert_load_ratios.append(ratio)

        if step % cfg["training"]["log_every"] == 0:
            print(
                f"[smoke] step={step:3d} loss={accum_loss:.4f} "
                f"sigma={sigma:.4f} "
                f"mem={(torch.cuda.max_memory_allocated() / 1024**3) if device.type == 'cuda' else 0:.2f}GB"
            )

    elapsed = time.time() - start_time
    result.tokens_per_sec = total_tokens / max(elapsed, 1e-6)

    # Validate against Master Plan v2.0 §10.1 gate criteria.
    result.initial_loss = losses[0] if losses else None
    result.final_loss = losses[-1] if losses else None

    # Check 1: Loss decreases (compare last 10% to first 10%).
    if len(losses) >= 10:
        early = sum(losses[: max(1, len(losses) // 10)]) / max(1, len(losses) // 10)
        late = sum(losses[-max(1, len(losses) // 10) :]) / max(1, len(losses) // 10)
        if late >= early:
            result.loss_decreasing = False
            result.errors.append(
                f"Loss did not decrease: early={early:.4f}, late={late:.4f}"
            )

    # Check 2: Spectral radius stays < 1.0.
    result.max_spectral_radius = max(spectral_radii) if spectral_radii else 0.0
    if result.max_spectral_radius >= 1.0:
        result.spectral_radius_ok = False
        result.errors.append(
            f"Spectral radius exceeded 1.0: max={result.max_spectral_radius:.4f}"
        )

    # Check 3: Expert load within 3x of uniform.
    if expert_load_ratios:
        result.expert_load_ratio = max(expert_load_ratios)
        if result.expert_load_ratio > 3.0:
            result.expert_load_ok = False
            result.errors.append(
                f"Expert load imbalance > 3x: max_ratio={result.expert_load_ratio:.2f}"
            )

    # Check 4: Memory under 80GB (per-GPU H100/A100 limit).
    if device.type == "cuda":
        result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        if result.peak_memory_gb > 80.0:
            result.memory_ok = False
            result.errors.append(
                f"Peak memory exceeded 80GB: {result.peak_memory_gb:.2f}GB"
            )

    return result


def print_report(result: SmokeTestResult) -> None:
    """Print a formatted gate report to stdout."""
    print("\n" + "=" * 60)
    print("ZAR-1 1B SMOKE TEST REPORT (Master Plan v2.0 Month 1 Gate)")
    print("=" * 60)

    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"\nOverall Status: {status}\n")

    print("Gate Criteria:")
    checks = [
        ("No NaN/Inf in loss", result.no_nan),
        ("Loss decreases over training", result.loss_decreasing),
        ("Spectral radius < 1.0", result.spectral_radius_ok),
        ("Expert load within 3x of uniform", result.expert_load_ok),
        ("Memory < 80GB per GPU", result.memory_ok),
    ]
    for name, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {name}")

    print("\nMetrics:")
    if result.initial_loss is not None:
        print(f"  Initial loss:           {result.initial_loss:.4f}")
        print(f"  Final loss:             {result.final_loss:.4f}")
    print(f"  Max spectral radius:    {result.max_spectral_radius:.4f}")
    print(f"  Expert load max/min:    {result.expert_load_ratio:.2f}x")
    print(f"  Peak memory:            {result.peak_memory_gb:.2f} GB")
    print(f"  Throughput:             {result.tokens_per_sec:.0f} tokens/sec")

    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAR-1 1B smoke test")
    parser.add_argument(
        "--config", type=str, default="configs/zar1_1b_smoke_test.yaml"
    )
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    parser.add_argument(
        "--output_json",
        type=str,
        default="smoke_test_results.json",
        help="Where to write JSON results",
    )
    args = parser.parse_args()

    result = run_smoke_test(args.config, num_steps=args.steps)
    print_report(result)

    with open(args.output_json, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[smoke] Results written to {args.output_json}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
