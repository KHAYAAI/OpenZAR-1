"""Hyperparameter ablation sweep harness for ZAR-1 (Master Plan v2.0 Month 2).

Sweeps over:
  - Learning rate ∈ {1e-4, 3e-4, 5e-4}
  - MoE TopK ∈ {4, 8}
  - ACT threshold (epsilon) ∈ {0.95, 0.99}  (i.e., halt_at = 1 - epsilon)
  - LoRA rank ∈ {8, 16, 32}

For each combination, runs a short training (default 200 steps) at 1B scale and
records final loss, depth-extrapolation eval, and stability metrics. Results are
written to a JSON ledger and printed as a comparison table.

Usage:
    # Quick sweep (fewer combinations, fewer steps)
    python scripts/ablation_sweep.py --base_config configs/zar1_1b_smoke_test.yaml \
        --steps 100 --quick

    # Full sweep (all combinations)
    python scripts/ablation_sweep.py --base_config configs/zar1_1b_smoke_test.yaml \
        --steps 500
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.data import build_train_val_loaders  # noqa: E402
from zar1.model import ZAR1Config, ZAR1Model  # noqa: E402


# Sweep grids (Master Plan v2.0 §6.1 Month 2)
DEFAULT_GRID = {
    "lr": [1e-4, 3e-4, 5e-4],
    "top_k": [4, 8],
    "act_epsilon": [0.05, 0.01],  # halt at >= 1-eps, so 0.05 = halt at 0.95
    "lora_rank": [8, 16, 32],
}

QUICK_GRID = {
    "lr": [3e-4],
    "top_k": [4, 8],
    "act_epsilon": [0.01],
    "lora_rank": [8, 16],
}


@dataclass
class AblationResult:
    """Outcome of a single ablation run."""

    config_id: str
    lr: float
    top_k: int
    act_epsilon: float
    lora_rank: int

    final_loss: float | None = None
    initial_loss: float | None = None
    val_loss: float | None = None
    depth_extrap_loss_1x: float | None = None
    depth_extrap_loss_2x: float | None = None
    depth_extrap_improvement_pct: float | None = None

    max_spectral_radius: float = 0.0
    expert_load_ratio: float = 1.0
    avg_act_steps: float | None = None
    tokens_per_sec: float = 0.0

    diverged: bool = False
    error: str | None = None


def make_config_id(combo: dict) -> str:
    """Generate a short descriptive ID for a combo."""
    return (
        f"lr{combo['lr']:.0e}_k{combo['top_k']}_"
        f"e{combo['act_epsilon']:.2f}_r{combo['lora_rank']}"
    )


def build_combo_config(base_cfg: dict, combo: dict) -> dict:
    """Apply ablation combo overrides to a base config (deep-copy)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["lr"] = combo["lr"]
    cfg["model"]["top_k"] = combo["top_k"]
    cfg["model"]["act_epsilon"] = combo["act_epsilon"]
    cfg["model"]["lora_rank"] = combo["lora_rank"]
    return cfg


def run_single_ablation(cfg: dict, combo: dict, num_steps: int) -> AblationResult:
    """Run one ablation training and return its result."""
    config_id = make_config_id(combo)
    result = AblationResult(
        config_id=config_id,
        lr=combo["lr"],
        top_k=combo["top_k"],
        act_epsilon=combo["act_epsilon"],
        lora_rank=combo["lora_rank"],
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

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
            lora_rank=cfg["model"]["lora_rank"],
            use_lora=cfg["model"].get("use_lora", True),
        )

        model = ZAR1Model(config).to(device=device, dtype=dtype)
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
            weight_decay=cfg["training"]["weight_decay"],
        )

        train_loader, val_loader = build_train_val_loaders(
            cfg=cfg["data"],
            seq_len=cfg["training"]["max_seq_len"],
            micro_batch_size=cfg["training"]["micro_batch_size"],
        )

        losses = []
        spectral_radii = []
        act_steps_log = []
        total_tokens = 0

        model.train()
        train_iter = iter(train_loader)
        start_time = time.time()

        for step in range(num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, labels=labels)

            if torch.isnan(out.loss) or torch.isinf(out.loss):
                result.diverged = True
                result.error = f"NaN/Inf at step {step}"
                return result

            optim.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            optim.step()

            sigma = float(model.apply_spectral_constraints().item())
            spectral_radii.append(sigma)
            losses.append(float(out.loss.item()))
            act_steps_log.append(out.n_loops)
            total_tokens += input_ids.numel()

        elapsed = time.time() - start_time
        result.tokens_per_sec = total_tokens / max(elapsed, 1e-6)
        result.initial_loss = losses[0]
        result.final_loss = losses[-1]
        result.max_spectral_radius = max(spectral_radii) if spectral_radii else 0.0
        result.avg_act_steps = sum(act_steps_log) / max(len(act_steps_log), 1)

        # Depth extrapolation: evaluate val loss at 1x and 2x trained loop count.
        model.eval()
        with torch.no_grad():
            val_iter = iter(val_loader)
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_batch = batch  # fallback

            v_ids = val_batch["input_ids"].to(device)
            v_lbl = val_batch["labels"].to(device)

            base_loops = config.max_loops
            out_1x = model(input_ids=v_ids, labels=v_lbl, inference_n_loops=base_loops)
            out_2x = model(input_ids=v_ids, labels=v_lbl, inference_n_loops=base_loops * 2)

            result.val_loss = float(out_1x.loss.item())
            result.depth_extrap_loss_1x = float(out_1x.loss.item())
            result.depth_extrap_loss_2x = float(out_2x.loss.item())
            if result.depth_extrap_loss_1x and result.depth_extrap_loss_1x > 0:
                result.depth_extrap_improvement_pct = (
                    100.0
                    * (result.depth_extrap_loss_1x - result.depth_extrap_loss_2x)
                    / result.depth_extrap_loss_1x
                )

    except Exception as e:
        result.diverged = True
        result.error = f"{type(e).__name__}: {str(e)[:200]}"

    return result


def print_results_table(results: list[AblationResult]) -> None:
    """Pretty-print results as a comparison table."""
    print("\n" + "=" * 110)
    print("ABLATION SWEEP RESULTS (Master Plan v2.0 §6.1 Month 2)")
    print("=" * 110)

    # Header
    cols = [
        ("config_id", 28),
        ("final_loss", 10),
        ("val_loss", 10),
        ("d.extrap%", 10),
        ("σ_max", 8),
        ("act_n", 6),
        ("status", 12),
    ]
    print(" | ".join(f"{name:<{w}}" for name, w in cols))
    print("-" * 110)

    # Sort by final loss (best first)
    successful = [r for r in results if not r.diverged and r.final_loss is not None]
    failed = [r for r in results if r.diverged or r.final_loss is None]
    successful.sort(key=lambda r: r.final_loss if r.final_loss is not None else math.inf)

    for r in successful + failed:
        status = "diverged" if r.diverged else "ok"
        row = [
            (r.config_id, 28),
            (f"{r.final_loss:.4f}" if r.final_loss is not None else "N/A", 10),
            (f"{r.val_loss:.4f}" if r.val_loss is not None else "N/A", 10),
            (
                f"{r.depth_extrap_improvement_pct:.2f}%"
                if r.depth_extrap_improvement_pct is not None
                else "N/A",
                10,
            ),
            (f"{r.max_spectral_radius:.3f}", 8),
            (f"{r.avg_act_steps:.1f}" if r.avg_act_steps is not None else "N/A", 6),
            (status, 12),
        ]
        print(" | ".join(f"{val:<{w}}" for val, w in row))
    print("=" * 110)

    if successful:
        best = successful[0]
        print(f"\n[BEST] {best.config_id} -- final_loss={best.final_loss:.4f}")
        print(
            f"  Depth extrapolation: {best.depth_extrap_improvement_pct:.2f}%"
            if best.depth_extrap_improvement_pct is not None
            else ""
        )
        # Master Plan v2.0 §10.1 Gate Criterion: depth extrapolation ≥ 3%
        if best.depth_extrap_improvement_pct is not None:
            gate_pass = best.depth_extrap_improvement_pct >= 3.0
            print(
                f"  Master Plan §10.1 depth extrapolation gate (≥3%): "
                f"{'✓ PASS' if gate_pass else '✗ FAIL'}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAR-1 hyperparameter ablation sweep")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/zar1_1b_smoke_test.yaml",
        help="Base config to derive ablation variants from",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Steps per ablation run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small grid (4 combinations) instead of full grid",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="ablation_results.json",
        help="Where to write JSON results",
    )
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    grid = QUICK_GRID if args.quick else DEFAULT_GRID
    combinations = [
        dict(zip(grid.keys(), values))
        for values in itertools.product(*grid.values())
    ]

    print(f"[sweep] {len(combinations)} combinations × {args.steps} steps each")
    print(f"[sweep] Grid: {grid}\n")

    results: list[AblationResult] = []
    for i, combo in enumerate(combinations):
        print(f"\n[sweep] Run {i+1}/{len(combinations)}: {combo}")
        cfg = build_combo_config(base_cfg, combo)
        result = run_single_ablation(cfg, combo, args.steps)
        results.append(result)
        if result.diverged:
            print(f"[sweep]   DIVERGED: {result.error}")
        else:
            print(
                f"[sweep]   final_loss={result.final_loss:.4f}, "
                f"σ_max={result.max_spectral_radius:.3f}, "
                f"d.extrap={result.depth_extrap_improvement_pct:.2f}%"
                if result.depth_extrap_improvement_pct is not None
                else f"[sweep]   final_loss={result.final_loss:.4f}"
            )

        # Free GPU memory between runs.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_results_table(results)

    # Persist results.
    serialized = [asdict(r) for r in results]
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "grid": grid,
                "steps": args.steps,
                "base_config": args.base_config,
                "results": serialized,
            },
            f,
            indent=2,
        )
    print(f"\n[sweep] Results written to {args.output_json}")


if __name__ == "__main__":
    main()
