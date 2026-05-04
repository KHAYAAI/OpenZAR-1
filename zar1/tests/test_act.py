"""Tests for Adaptive Computation Time halting."""

import torch

from zar1.act import ACTHalting
from zar1.recurrent_block import RecurrentTransformerBlock


def test_act_basic_forward():
    """Test ACT forward pass returns properly-shaped outputs."""
    act = ACTHalting(dim=64, epsilon=0.01, max_steps=4)
    hidden = torch.randn(2, 8, 64)  # (B, T, dim)

    def dummy_step(h: torch.Tensor, t: int):
        return h + 0.1 * torch.randn_like(h), torch.tensor(0.0)

    output, ponder_cost, n_steps, aux_loss = act(hidden, dummy_step, max_steps=4)

    assert output.shape == hidden.shape
    assert ponder_cost.dim() == 0  # scalar
    assert 1 <= n_steps <= 4


def test_act_early_exit():
    """Test that ACT halts early when cumulative probability exceeds threshold."""
    act = ACTHalting(dim=32, epsilon=0.05, max_steps=8)  # high epsilon -> easier to halt
    hidden = torch.randn(1, 4, 32)

    steps_taken = []

    def recording_step(h: torch.Tensor, t: int):
        steps_taken.append(t)
        return h + 0.01 * torch.randn_like(h), torch.tensor(0.0)

    output, ponder_cost, n_steps, _ = act(hidden, recording_step, max_steps=8)
    assert n_steps == len(steps_taken)
    # With high epsilon, we should exit early (not all 8 steps)
    assert n_steps < 8


def test_act_ponder_cost_positive():
    """Ponder cost should be non-negative."""
    act = ACTHalting(dim=16, epsilon=0.01, ponder_tau=0.1, max_steps=4)
    hidden = torch.randn(1, 4, 16)

    def step(h: torch.Tensor, t: int):
        return h, torch.tensor(0.0)

    _, ponder_cost, _, _ = act(hidden, step, max_steps=4)
    assert ponder_cost.item() >= 0


def test_act_with_recurrent_block():
    """Integration test: ACT + RecurrentTransformerBlock."""
    block = RecurrentTransformerBlock(
        dim=64,
        num_heads=4,
        num_kv_heads=2,
        max_loops=4,
        num_experts=8,
        top_k=2,
    )
    act = ACTHalting(dim=64, epsilon=0.01, max_steps=4)
    hidden = torch.randn(1, 8, 64)

    def step(h: torch.Tensor, t: int):
        return block(h, t)

    output, ponder_cost, n_steps, aux_loss = act(hidden, step, max_steps=4)
    assert output.shape == hidden.shape
    assert n_steps >= 1
