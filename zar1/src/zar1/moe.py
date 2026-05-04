"""Mixture-of-Experts feed-forward layer with top-k gating and load balancing.

Implements a Switch/Shazeer-style sparse MoE FFN where each expert is a SwiGLU
block. Routing uses top-k softmax gating with a capacity factor and a
load-balancing auxiliary loss to prevent expert collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """A single SwiGLU FFN expert: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFeedForward(nn.Module):
    """Sparse Mixture-of-Experts feed-forward with top-k gating.

    Args:
        dim: Model hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts to route each token to.
        capacity_factor: Multiplier on tokens-per-expert capacity. Tokens beyond
            capacity for an expert are dropped (zeros are returned for that slot).
        hidden_dim: Per-expert hidden dimension. Defaults to ``dim * 4 // num_experts``.
        aux_loss_weight: Coefficient applied to the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 256,
        top_k: int = 8,
        capacity_factor: float = 1.25,
        hidden_dim: int | None = None,
        aux_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

        if hidden_dim is None:
            hidden_dim = max(1, (dim * 4) // num_experts)
        self.hidden_dim = hidden_dim

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(dim, hidden_dim) for _ in range(num_experts)]
        )

    def _compute_aux_loss(
        self, router_probs: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """Shazeer load-balancing loss: N * sum_i (f_i * P_i).

        ``f_i`` is the fraction of tokens routed (top-1 hard) to expert i and
        ``P_i`` is the mean router probability for expert i.
        """
        # router_probs: (T, E), expert_mask: (T, E) one-hot of selected experts
        f_i = expert_mask.float().mean(dim=0)  # (E,)
        p_i = router_probs.mean(dim=0)  # (E,)
        return self.num_experts * torch.sum(f_i * p_i)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens through top-k experts.

        Args:
            x: Input tensor of shape ``(batch, seq, dim)``.

        Returns:
            output: Tensor of shape ``(batch, seq, dim)``.
            routing_probs: Tensor of shape ``(batch, seq, num_experts)``.
            aux_loss: Scalar auxiliary load-balancing loss (already weighted).
        """
        bsz, seq, dim = x.shape
        num_tokens = bsz * seq
        x_flat = x.reshape(num_tokens, dim)

        router_logits = self.router(x_flat)  # (T, E)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k experts per token; renormalize the chosen probabilities.
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Hard top-1 mask (used for load-balancing statistics).
        top1_mask = F.one_hot(topk_idx[:, 0], num_classes=self.num_experts)

        # Capacity per expert (rounded up).
        capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        capacity = max(capacity, 1)

        output = torch.zeros_like(x_flat)

        # Dispatch to each expert in turn. For bf16/large models this is
        # acceptable; high-throughput kernels (e.g. fused_moe) can replace this.
        for e_idx in range(self.num_experts):
            # Find (token, slot) positions selecting this expert.
            slot_mask = topk_idx == e_idx  # (T, K) bool
            if not slot_mask.any():
                continue
            token_pos, slot_pos = torch.nonzero(slot_mask, as_tuple=True)

            if token_pos.numel() > capacity:
                # Drop tokens beyond capacity (deterministic order).
                token_pos = token_pos[:capacity]
                slot_pos = slot_pos[:capacity]

            expert_in = x_flat[token_pos]
            expert_out = self.experts[e_idx](expert_in)
            weights = topk_probs[token_pos, slot_pos].unsqueeze(-1)
            output.index_add_(0, token_pos, expert_out * weights)

        aux_loss = self.aux_loss_weight * self._compute_aux_loss(router_probs, top1_mask)

        output = output.reshape(bsz, seq, dim)
        routing_probs = router_probs.reshape(bsz, seq, self.num_experts)
        return output, routing_probs, aux_loss
