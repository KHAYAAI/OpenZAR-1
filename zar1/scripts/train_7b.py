"""Distributed FSDP training script for ZAR-1 7B.

Launch with:
    torchrun --nproc_per_node=8 scripts/train_7b.py --config configs/zar1_7b.yaml
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Allow running as a script: ``python scripts/train_7b.py``
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.data import build_attention_mask_from_doc_ids, build_train_val_loaders  # noqa: E402
from zar1.model import ZAR1Config, ZAR1Model  # noqa: E402
from zar1.recurrent_block import (  # noqa: E402
    RecurrentTransformerBlock,
    StandardTransformerBlock,
)


_SHOULD_STOP = False


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        global _SHOULD_STOP
        _SHOULD_STOP = True
        print(f"[rank {os.environ.get('RANK', '0')}] received signal {signum}; stopping.")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _setup_distributed() -> tuple[int, int, int]:
    """Initialise process group from torchrun env vars."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _is_main(rank: int) -> bool:
    return rank == 0


def cosine_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _wandb_init(cfg: dict, rank: int):
    if not _is_main(rank):
        return None
    try:
        import wandb
    except ImportError:
        return None
    log_cfg = cfg.get("logging", {})
    return wandb.init(
        project=log_cfg.get("wandb_project", "zar1"),
        entity=log_cfg.get("wandb_entity"),
        name=log_cfg.get("run_name"),
        config=cfg,
        resume="allow",
    )


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(min=1e-9)
    return -(p * p.log()).sum(dim=-1).mean()


def build_model(cfg: dict, device: torch.device) -> ZAR1Model:
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
        spectral_max=cfg["model"].get("spectral_max", 0.99),
        spectral_iters=cfg["model"].get("spectral_iters", 5),
        pad_token_id=cfg["model"].get("pad_token_id", 0),
        tie_word_embeddings=cfg["model"].get("tie_word_embeddings", True),
    )
    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)
    return model


def wrap_fsdp(model: ZAR1Model, cfg: dict) -> FSDP:
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={StandardTransformerBlock, RecurrentTransformerBlock},
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    sharding = (
        ShardingStrategy.HYBRID_SHARD
        if cfg["distributed"].get("hybrid_shard", False)
        else ShardingStrategy.FULL_SHARD
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
    )


def save_checkpoint(model: FSDP, optim, step: int, output_dir: Path, rank: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()
    if _is_main(rank):
        ckpt_path = output_dir / f"step_{step:07d}.pt"
        torch.save({"step": step, "model": state}, ckpt_path)
        latest = output_dir / "latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(ckpt_path.name)


def load_checkpoint(model: FSDP, path: str, rank: int) -> int:
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    if not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu")
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model.load_state_dict(state["model"], strict=False)
    if _is_main(rank):
        print(f"Resumed from {path} at step {state.get('step', 0)}")
    return int(state.get("step", 0))


def evaluate(model: FSDP, val_loader, device, num_heads: int, max_batches: int = 20) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            doc_ids = batch["doc_ids"].to(device, non_blocking=True)
            attn_mask = build_attention_mask_from_doc_ids(doc_ids, num_heads=1)
            attn_mask = attn_mask.to(dtype=torch.bfloat16)
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            total_loss += float(out.loss.item())
            n += 1
    model.train()
    return total_loss / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _install_signal_handlers()
    rank, world_size, local_rank = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if _is_main(rank):
        print(f"World size: {world_size}, device: {device}")

    model = build_model(cfg, device)
    if _is_main(rank):
        print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")

    if cfg["distributed"].get("fsdp", True) and world_size > 1:
        model = wrap_fsdp(model, cfg)

    if cfg["distributed"].get("activation_checkpointing", False):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        non_reentrant = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant,
            check_fn=lambda m: isinstance(
                m, (StandardTransformerBlock, RecurrentTransformerBlock)
            ),
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        eps=cfg["training"]["eps"],
        weight_decay=cfg["training"]["weight_decay"],
        fused=torch.cuda.is_available(),
    )

    train_loader, val_loader = build_train_val_loaders(
        cfg=cfg["data"],
        seq_len=cfg["training"]["max_seq_len"],
        micro_batch_size=cfg["training"]["micro_batch_size"],
    )

    start_step = 0
    resume = cfg["training"].get("resume_from")
    if resume:
        start_step = load_checkpoint(model, resume, rank)

    run = _wandb_init(cfg, rank)

    grad_accum = int(cfg["training"]["grad_accum_steps"])
    total_steps = int(cfg["training"]["steps"])
    warmup = int(cfg["training"]["warmup_steps"])
    base_lr = float(cfg["training"]["lr"])
    min_lr = float(cfg["training"]["min_lr"])
    grad_clip = float(cfg["training"]["gradient_clip"])
    log_every = int(cfg["training"]["log_every"])
    eval_every = int(cfg["training"]["eval_every"])
    save_every = int(cfg["training"]["save_every"])
    output_dir = Path(cfg["training"]["output_dir"])
    patience = int(cfg["training"].get("early_stop_patience", 10))
    best_val = float("inf")
    plateau = 0

    model.train()
    optim.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    step = start_step
    t0 = time.time()

    while step < total_steps:
        if _SHOULD_STOP:
            if _is_main(rank):
                print("Stop signal received; saving checkpoint before exit.")
            save_checkpoint(model, optim, step, output_dir, rank)
            break

        # Set LR for this optimizer step.
        lr = cosine_lr(step, warmup, total_steps, base_lr, min_lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        accum_aux = 0.0
        accum_ponder = 0.0
        accum_steps_act = 0
        accum_router_entropy = 0.0

        for micro in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            doc_ids = batch["doc_ids"].to(device, non_blocking=True)

            attn_mask = build_attention_mask_from_doc_ids(doc_ids).to(dtype=torch.bfloat16)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()

            accum_loss += float(out.loss.detach().item()) / grad_accum
            accum_aux += float(out.aux_loss.detach().item()) / grad_accum
            accum_ponder += float(out.ponder_cost.detach().item()) / grad_accum
            accum_steps_act += out.n_loops

        if isinstance(model, FSDP):
            model.clip_grad_norm_(grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optim.step()
        optim.zero_grad(set_to_none=True)

        # Spectral-radius constraint after each optimizer step.
        if isinstance(model, FSDP):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType  # noqa: F401
            with FSDP.summon_full_params(model, writeback=True):
                model.module.apply_spectral_constraints()
        else:
            model.apply_spectral_constraints()

        step += 1

        if _is_main(rank) and step % log_every == 0:
            elapsed = time.time() - t0
            tps = (
                log_every
                * grad_accum
                * cfg["training"]["micro_batch_size"]
                * cfg["training"]["max_seq_len"]
                * world_size
                / max(1e-9, elapsed)
            )
            mem = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            log_data = {
                "loss": accum_loss,
                "perplexity": math.exp(min(20.0, accum_loss)),
                "lr": lr,
                "aux_loss": accum_aux,
                "ponder_cost": accum_ponder,
                "act_steps": accum_steps_act / grad_accum,
                "tokens_per_sec": tps,
                "mem_gb": mem,
                "step": step,
            }
            print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in log_data.items()))
            if run is not None:
                run.log(log_data, step=step)
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        if step % eval_every == 0:
            val_loss = evaluate(
                model, val_loader, device, num_heads=cfg["model"]["n_heads"]
            )
            if _is_main(rank):
                print(f"[step {step}] val_loss={val_loss:.4f}")
                if run is not None:
                    run.log({"val_loss": val_loss}, step=step)
            if val_loss < best_val - 1e-3:
                best_val = val_loss
                plateau = 0
            else:
                plateau += 1
                if plateau >= patience:
                    if _is_main(rank):
                        print(f"Early stop: no val improvement for {patience} evals.")
                    save_checkpoint(model, optim, step, output_dir, rank)
                    break

        if step % save_every == 0:
            save_checkpoint(model, optim, step, output_dir, rank)

    save_checkpoint(model, optim, step, output_dir, rank)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
