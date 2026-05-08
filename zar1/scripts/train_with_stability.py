"""Training script with stability features: spectral constraint, ACT curriculum, expert monitoring.

Usage:
    # Dry run (100 steps)
    python scripts/train_with_stability.py --config configs/zar1_7b.yaml --max_steps 100
    
    # Full training
    python scripts/train_with_stability.py --config configs/zar1_7b.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.model import ZAR1Config, ZAR1Model
from zar1.data import build_train_val_loaders
from zar1.training_stability import (
    SpectralConstraintController,
    ACTCurriculum,
    ExpertLoadMonitor,
)
from zar1.validation import (
    compute_validation_loss,
    compute_perplexity,
    EarlyStoppingCallback,
    should_validate,
    should_save_checkpoint,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ZAR-1 training with stability")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="100-step validation run")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override from command line
    if args.max_steps is not None:
        cfg["training"]["steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["training"]["micro_batch_size"] = args.batch_size
    if args.dry_run:
        cfg["training"]["steps"] = 100
        logger.info("[DRY RUN] Limited to 100 steps")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("Loading model...")
    config = ZAR1Config.from_yaml(args.config)
    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)
    n_params = model.num_parameters()
    logger.info(f"Model: {n_params/1e9:.2f}B parameters")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Setup scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["steps"],
        eta_min=cfg["training"]["min_lr"],
    )

    # Setup data loaders
    logger.info("Loading data...")
    train_loader, val_loader = build_train_val_loaders(
        cfg=cfg["data"],
        seq_len=cfg["training"]["max_seq_len"],
        micro_batch_size=cfg["training"]["micro_batch_size"],
    )

    # Setup stability features
    logger.info("Setting up stability features...")
    spectral_controller = SpectralConstraintController(
        weight=model.recurrent_block.recurrent_proj.weight,
        max_sigma=config.spectral_max,
        n_iters=5,
        device=device,
    )

    act_curriculum = ACTCurriculum(curriculum_steps=5000)
    expert_monitor = ExpertLoadMonitor(n_experts=config.num_experts)

    # Setup early stopping
    early_stopping = EarlyStoppingCallback(patience=3, min_delta=0.001)

    # Training loop
    logger.info("Starting training...")
    model.train()
    start_time = time.time()
    total_tokens = 0

    for step in range(cfg["training"]["steps"]):
        try:
            batch = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        labels = batch["labels"].to(device, dtype=torch.long)

        # Forward pass with ACT curriculum
        logits = act_curriculum.forward(model, input_ids)

        # Compute loss
        loss = logits.view(-1, config.vocab_size).loss
        if loss is None:
            # Manual loss computation
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
                reduction="mean",
            )

        # Backward pass
        loss.backward()

        # Apply spectral constraint BEFORE optimizer step
        sigma = spectral_controller.apply_constraint()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate decay
        scheduler.step()

        total_tokens += input_ids.numel()

        # Logging
        if step % cfg["training"]["log_every"] == 0:
            elapsed = time.time() - start_time
            throughput = total_tokens / elapsed / 1e3  # k tokens/sec
            logger.info(
                f"Step {step:6d} | Loss: {loss:.4f} | "
                f"σ={sigma:.4f} | Throughput: {throughput:.1f}k tokens/sec | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Validation
        if should_validate(step, cfg["training"]["eval_every"]):
            val_loss = compute_validation_loss(
                model, val_loader, device, max_batches=50, vocab_size=config.vocab_size
            )
            val_ppl = compute_perplexity(val_loss)
            logger.info(f"  Val loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

            # Early stopping check
            if early_stopping.on_validation(val_loss, step, model):
                logger.warning(f"Early stopping triggered at step {step}")
                break

        # Checkpoint save
        if should_save_checkpoint(step, cfg["training"]["save_every"]):
            checkpoint_dir = Path(cfg["training"]["output_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "config": config,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Training complete
    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete!")
    logger.info(f"  Total time: {elapsed/3600:.1f} hours")
    logger.info(f"  Total tokens: {total_tokens/1e9:.2f}B")
    logger.info(f"  Throughput: {total_tokens/elapsed/1e3:.1f}k tokens/sec")

    # Early stopping summary
    if early_stopping.best_epoch > 0:
        summary = early_stopping.get_summary()
        logger.info(f"  Best validation loss: {summary['best_loss']:.4f} at step {summary['best_epoch']}")


if __name__ == "__main__":
    main()
