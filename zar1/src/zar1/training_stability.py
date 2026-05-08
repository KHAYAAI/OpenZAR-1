"""Training stability utilities: spectral constraint enforcement, ACT curriculum, expert monitoring."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _spectral_norm_power_iter(
    weight: torch.Tensor,
    u: torch.Tensor,
    n_iters: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate largest singular value via power iteration.
    
    Returns (sigma, new_u). Updates u estimate in-place.
    """
    w = weight
    if w.dim() != 2:
        w = w.reshape(w.shape[0], -1)
    
    for _ in range(n_iters):
        v = F.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
    
    sigma = torch.dot(u, torch.mv(w, v))
    return sigma, u


class SpectralConstraintController:
    """Applies spectral norm constraint to recurrent weight matrix.
    
    Ensures ρ(W_recurrent) ≤ max_sigma for training stability.
    Call before optimizer.step() to include constraint in gradient flow.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        max_sigma: float = 0.99,
        n_iters: int = 5,
        device: str = "cuda",
    ):
        self.weight = weight
        self.max_sigma = max_sigma
        self.n_iters = n_iters
        self.device = device

        # Cache for left singular vector (not a parameter)
        u_init = torch.randn(weight.shape[0], device=device)
        self._u = F.normalize(u_init, dim=0)

    @torch.no_grad()
    def apply_constraint(self) -> torch.Tensor:
        """Rescale weight if σ > max_sigma. Returns current sigma value."""
        w = self.weight.data
        u = self._u.to(dtype=w.dtype, device=w.device)

        sigma, new_u = _spectral_norm_power_iter(w, u, self.n_iters)
        self._u.copy_(new_u.to(self._u.dtype))

        sigma_val = sigma.detach().abs()
        if sigma_val > self.max_sigma:
            w.mul_(self.max_sigma / (sigma_val + 1e-12))

        return sigma_val


class ACTCurriculum:
    """Curriculum learning for ACT (Adaptive Computation Time).
    
    Disables halting in early training to allow model to explore all loops.
    Gradually enables halting after curriculum_steps.
    """

    def __init__(self, curriculum_steps: int = 5000):
        self.curriculum_steps = curriculum_steps
        self.current_step = 0

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        original_halt_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Forward pass with curriculum-gated halting.
        
        Args:
            model: ZAR1Model with ACT module
            x: Input tensor
            original_halt_threshold: Original ACT threshold (restored after forward)
        
        Returns:
            logits: Model output
        """
        if self.current_step < self.curriculum_steps:
            # Disable ACT by setting impossibly high threshold
            # (ACT only halts if halt_prob > threshold)
            if hasattr(model, "act"):
                old_epsilon = model.act.epsilon
                model.act.epsilon = 999.0  # Effectively never halts
                logits = model(x)
                model.act.epsilon = old_epsilon
            else:
                logits = model(x)
        else:
            # ACT enabled, uses original threshold
            logits = model(x)

        self.current_step += 1
        return logits

    def is_curriculum_active(self) -> bool:
        """Returns True if still in curriculum phase."""
        return self.current_step < self.curriculum_steps

    def curriculum_progress(self) -> float:
        """Returns curriculum progress as fraction [0, 1]."""
        return min(1.0, self.current_step / self.curriculum_steps)


class ExpertLoadMonitor:
    """Tracks expert utilization and flags dead experts.
    
    Requires integration with MoEFFN routing decisions.
    """

    def __init__(self, n_experts: int, death_threshold: float = 0.05):
        self.n_experts = n_experts
        self.death_threshold = death_threshold

        # Running counters
        self.expert_counts = torch.zeros(n_experts)
        self.total_tokens = 0

    def update(self, routing_indices: torch.Tensor) -> None:
        """Update expert counts from routing decisions.
        
        Args:
            routing_indices: Flat tensor of expert indices assigned this batch
        """
        self.total_tokens += routing_indices.numel()
        self.expert_counts += torch.bincount(
            routing_indices.flatten(),
            minlength=self.n_experts,
        ).float()

    def get_stats(self) -> dict:
        """Return expert utilization statistics."""
        if self.total_tokens == 0:
            return {
                "active_experts": 0,
                "dead_experts": self.n_experts,
                "dead_ratio": 1.0,
                "entropy": 0.0,
            }

        # Compute fraction of tokens sent to each expert
        expert_fractions = self.expert_counts / self.total_tokens

        # Count "dead" experts (below threshold)
        dead_mask = expert_fractions < (1.0 / self.n_experts) * (1 - self.death_threshold)
        dead_experts = dead_mask.sum().item()
        dead_ratio = dead_experts / self.n_experts

        # Compute entropy of expert load distribution
        # (higher = more balanced, lower = more concentrated)
        probs = expert_fractions / (expert_fractions.sum() + 1e-10)
        entropy = -(probs * (probs.log() + 1e-10)).sum().item()

        return {
            "active_experts": self.n_experts - dead_experts,
            "dead_experts": dead_experts,
            "dead_ratio": dead_ratio,
            "entropy": entropy,
            "expert_fractions": expert_fractions,
        }

    def reset(self) -> None:
        """Reset counters for next monitoring window."""
        self.expert_counts.zero_()
        self.total_tokens = 0


def setup_training_stability(
    model: nn.Module,
    device: str = "cuda",
    max_sigma: float = 0.99,
    curriculum_steps: int = 5000,
    n_experts: int = 256,
) -> tuple[SpectralConstraintController, ACTCurriculum, ExpertLoadMonitor]:
    """Initialize all stability components.
    
    Returns:
        (spectral_controller, act_curriculum, expert_monitor)
    """
    # Get recurrent weight from model
    recurrent_weight = model.recurrent_block.recurrent_proj.weight

    spectral = SpectralConstraintController(
        recurrent_weight,
        max_sigma=max_sigma,
        n_iters=5,
        device=device,
    )

    act_curriculum = ACTCurriculum(curriculum_steps=curriculum_steps)
    expert_monitor = ExpertLoadMonitor(n_experts=n_experts, death_threshold=0.05)

    return spectral, act_curriculum, expert_monitor
