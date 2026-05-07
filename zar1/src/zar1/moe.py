"""Mixture-of-Experts feed-forward layer with top-k gating and load balancing.

Implements a Switch/Shazeer-style sparse MoE FFN where each expert is a SwiGLU
block. Routing uses a 2-layer MLP router with PID-controlled bias to prevent
expert collapse. The PID controller dynamically adjusts router logits based on
expert load deviation from uniform distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class MLPRouter(nn.Module):
    """2-layer MLP router for expert gating.

    Replaces simple linear routing with learned nonlinear projection to improve
    discrimination between experts and reduce collapse risk.

    Args:
        dim: Model hidden dimension.
        num_experts: Total number of experts.
        hidden_dim: Hidden dimension of MLP. Defaults to dim.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.dim = dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        self.layer1 = nn.Linear(dim, hidden_dim, bias=True)
        self.layer2 = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens through MLP.

        Args:
            x: Input tensor of shape ``(num_tokens, dim)``.

        Returns:
            logits: Tensor of shape ``(num_tokens, num_experts)``.
        """
        h = F.relu(self.layer1(x))
        logits = self.layer2(h)
        return logits


class PIDBalancer(nn.Module):
    """PID controller for expert load balancing.

    Maintains exponential moving average of expert load and applies proportional-
    integral-derivative control to compute a bias term that encourages uniform
    expert utilization without requiring an auxiliary loss.

    Args:
        num_experts: Total number of experts.
        kp: Proportional gain (default 0.5).
        ki: Integral gain (default 0.1).
        kd: Derivative gain (default 0.1).
        ema_decay: Exponential moving average decay for load tracking (default 0.99).
    """

    def __init__(
        self,
        num_experts: int,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.1,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ema_decay = ema_decay

        # Exponential moving average of expert load (fraction of tokens routed).
        self.register_buffer(
            "ema_load", torch.ones(num_experts) / num_experts, persistent=False
        )
        # Integral error accumulation.
        self.register_buffer("integral_error", torch.zeros(num_experts), persistent=False)
        # Previous error for derivative computation.
        self.register_buffer(
            "prev_error", torch.zeros(num_experts), persistent=False
        )

    def forward(
        self, routing_probs: torch.Tensor, topk_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute PID-controlled bias for router logits.

        Args:
            routing_probs: Softmax routing probabilities, shape ``(num_tokens, num_experts)``.
            topk_idx: Hard expert assignments (top-1), shape ``(num_tokens,)``.

        Returns:
            bias: Bias term to add to router logits, shape ``(num_experts,)``.
        """
        # Compute current expert load (fraction of tokens assigned to each expert).
        num_tokens = routing_probs.shape[0]
        current_load = torch.zeros(
            self.num_experts, device=routing_probs.device, dtype=routing_probs.dtype
        )
        current_load.scatter_add_(0, topk_idx, torch.ones(num_tokens, device=routing_probs.device, dtype=routing_probs.dtype))
        current_load = current_load / max(num_tokens, 1)

        # Update EMA.
        self.ema_load = (
            self.ema_decay * self.ema_load + (1 - self.ema_decay) * current_load
        )

        # Target load: uniform distribution.
        target_load = torch.ones_like(self.ema_load) / self.num_experts

        # Compute error (positive = over-utilized, negative = under-utilized).
        error = self.ema_load - target_load

        # Update integral and derivative terms.
        self.integral_error += error
        derivative_error = error - self.prev_error
        self.prev_error = error

        # PID formula: bias = Kp*error + Ki*integral + Kd*derivative.
        bias = (
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * derivative_error
        )

        return bias


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
    """Sparse Mixture-of-Experts feed-forward with MLP routing and PID balancing.

    Args:
        dim: Model hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts to route each token to.
        capacity_factor: Multiplier on tokens-per-expert capacity. Tokens beyond
            capacity for an expert are dropped (zeros are returned for that slot).
        hidden_dim: Per-expert hidden dimension. Defaults to ``dim * 4 // num_experts``.
        aux_loss_weight: Coefficient applied to the load-balancing auxiliary loss.
        router_hidden_dim: MLP router hidden dimension. Defaults to dim.
        use_pid: Enable PID-based load balancing. If False, reverts to aux_loss only.
        pid_kp: PID proportional gain (default 0.5).
        pid_ki: PID integral gain (default 0.1).
        pid_kd: PID derivative gain (default 0.1).
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 256,
        top_k: int = 8,
        capacity_factor: float = 1.25,
        hidden_dim: int | None = None,
        aux_loss_weight: float = 0.01,
        router_hidden_dim: int | None = None,
        use_pid: bool = True,
        pid_kp: float = 0.5,
        pid_ki: float = 0.1,
        pid_kd: float = 0.1,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.use_pid = use_pid

        if hidden_dim is None:
            hidden_dim = max(1, (dim * 4) // num_experts)
        self.hidden_dim = hidden_dim

        if router_hidden_dim is None:
            router_hidden_dim = dim

        self.router = MLPRouter(dim, num_experts, hidden_dim=router_hidden_dim)
        if use_pid:
            self.pid = PIDBalancer(
                num_experts, kp=pid_kp, ki=pid_ki, kd=pid_kd
            )
        else:
            self.pid = None

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
        """Route tokens through top-k experts with MLP router and PID balancing.

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

        # MLP router logits.
        router_logits = self.router(x_flat)  # (T, E)

        # Apply PID bias if enabled (encourages uniform expert utilization).
        if self.use_pid and self.training:
            # Compute bias based on expert load deviation.
            router_probs_temp = F.softmax(router_logits, dim=-1)
            top1_hard = torch.argmax(router_probs_temp, dim=-1)
            pid_bias = self.pid(router_probs_temp, top1_hard)
            router_logits = router_logits + pid_bias.unsqueeze(0)

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
