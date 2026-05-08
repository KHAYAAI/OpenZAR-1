"""Validation utilities for training: perplexity computation, early stopping."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def compute_validation_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    vocab_size: int = 128_256,
) -> float:
    """Compute validation perplexity on held-out set.
    
    Args:
        model: Language model
        val_loader: Validation data loader
        device: Device to compute on
        max_batches: Max batches to evaluate (None = all)
        vocab_size: Vocabulary size for loss computation
    
    Returns:
        Validation loss (cross-entropy)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(tqdm(
        val_loader,
        desc="Validation",
        total=max_batches,
        leave=False,
        disable=max_batches is None,
    )):
        if max_batches is not None and i >= max_batches:
            break

        # Handle both (input, target) and dict-style batches
        if isinstance(batch, (tuple, list)):
            input_ids, labels = batch
        else:
            input_ids = batch.get("input_ids", batch.get("x"))
            labels = batch.get("labels", batch.get("y"))

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        if hasattr(model, "forward"):
            output = model(input_ids)
        else:
            output = model(input_ids)

        # Extract logits
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += labels.numel()

    model.train()
    return total_loss / max(total_tokens, 1)


def compute_perplexity(val_loss: float) -> float:
    """Convert validation loss to perplexity.
    
    Perplexity = exp(loss)
    """
    return math.exp(min(val_loss, 100.0))  # Cap to avoid overflow


class EarlyStoppingCallback:
    """Early stopping based on validation loss plateau.
    
    Stops training if validation loss doesn't improve for patience steps.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        restore_best: bool = True,
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of validation checks without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_validation(
        self,
        val_loss: float,
        epoch: int,
        model: Optional[torch.nn.Module] = None,
    ) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Validation loss
            epoch: Current epoch/step
            model: Model for weight restoration
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait_count = 0

            if self.restore_best and model is not None:
                self.best_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }

            return False
        else:
            # No improvement
            self.wait_count += 1

            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                return True

            return False

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        """Restore model to best epoch weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

    def get_summary(self) -> dict:
        """Return early stopping summary."""
        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
            "total_patience": self.patience,
        }


def should_validate(step: int, val_every: int) -> bool:
    """Check if validation should run at this step."""
    return step > 0 and step % val_every == 0


def should_save_checkpoint(step: int, save_every: int) -> bool:
    """Check if checkpoint should be saved at this step."""
    return step > 0 and step % save_every == 0
