"""Adaptive Computation Time (ACT / PonderNet-style) halting.

Given a stream of intermediate hidden states produced by a recurrent block,
ACT learns to halt early. A small MLP predicts a per-step halting probability
``p_t`` from the current hidden state. We accumulate ``Σ p_t`` and stop when
it exceeds ``1 - epsilon``. The final output is the weighted average of all
intermediate outputs (weights = ``p_t`` normalized).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting controller.

    Args:
        dim: Hidden dimension of the model.
        epsilon: Halting threshold; halt when cumulative probability >= 1 - epsilon.
        ponder_tau: Coefficient on the residual ponder cost.
        max_steps: Hard cap on the number of recurrent iterations.
    """

    def __init__(
        self,
        dim: int,
        epsilon: float = 0.01,
        ponder_tau: float = 0.01,
        max_steps: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.ponder_tau = ponder_tau
        self.max_steps = max_steps

        self.halt_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        step_fn: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
        max_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Run the recurrent loop with early halting.

        Args:
            hidden: Initial hidden states ``(B, T, dim)``.
            step_fn: Callable ``(hidden, loop_index) -> (new_hidden, aux_loss)``.
                Typically a closure around a :class:`RecurrentTransformerBlock`.
            max_steps: Optional override for ``self.max_steps``.

        Returns:
            output: Weighted average of all intermediate outputs.
            ponder_cost: Scalar; ``ponder_tau * mean(remaining_probability)``.
            n_steps: Number of recurrence steps actually executed (Python int).
            aux_loss_sum: Sum of MoE auxiliary losses across executed steps.
        """
        n = max_steps if max_steps is not None else self.max_steps

        bsz, seq, dim = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        # Per-(B, T) cumulative halting probability and running weighted sum.
        cum_p = torch.zeros(bsz, seq, device=device, dtype=torch.float32)
        running = torch.zeros_like(hidden, dtype=torch.float32)
        # ``still_running`` tracks positions that have not yet halted.
        still_running = torch.ones(bsz, seq, device=device, dtype=torch.bool)

        aux_loss_sum = hidden.new_zeros(())
        steps_taken = 0

        for t in range(n):
            new_hidden, aux_loss = step_fn(hidden, t)
            aux_loss_sum = aux_loss_sum + aux_loss
            steps_taken = t + 1

            # Predict per-token halting probability from updated state.
            p_t = torch.sigmoid(self.halt_mlp(new_hidden).squeeze(-1))  # (B, T)
            p_t = p_t.float()

            is_last_step = t == n - 1
            # If we're at the last step or this push would cross 1 - eps, halt now.
            would_finish = (cum_p + p_t) >= (1.0 - self.epsilon)
            halt_now = (would_finish | is_last_step) & still_running

            # Tokens that continue: use predicted p_t. Tokens that halt now:
            # absorb the entire remaining probability mass (1 - cum_p).
            remainder = (1.0 - cum_p).clamp(min=0.0)
            weight = torch.where(halt_now, remainder, p_t)
            # Mask out tokens that already halted previously.
            weight = weight * still_running.float()

            running = running + weight.unsqueeze(-1) * new_hidden.float()
            cum_p = cum_p + weight

            still_running = still_running & (~halt_now)
            hidden = new_hidden

            if not still_running.any():
                break

        # Residual ponder cost: probability mass left unused averaged across tokens.
        residual = (1.0 - cum_p).clamp(min=0.0)
        ponder_cost = self.ponder_tau * residual.mean()

        output = running.to(dtype)
        return output, ponder_cost, steps_taken, aux_loss_sum
