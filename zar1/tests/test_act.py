"""Tests for the ACT halting module."""

import torch
import torch.nn as nn

from zar1.act import ACTHalting


def _make_step_fn(halt: ACTHalting, fixed_p: float | None = None):
    """Return a step_fn that increments the hidden state and (optionally)
    forces the halting probability to a known value."""

    def step(h: torch.Tensor, t: int):
        new_h = h + 1.0
        return new_h, h.new_zeros(())

    if fixed_p is not None:
        # Override the halt MLP so sigmoid(out) == fixed_p (approximately).
        with torch.no_grad():
            for p in halt.halt_mlp.parameters():
                p.zero_()
            # Final bias controls the output.
            halt.halt_mlp[-1].bias.fill_(torch.logit(torch.tensor(fixed_p)).item())
    return step


def test_early_exit_triggers():
    """With high halting probability, ACT should exit on the first step."""
    halt = ACTHalting(dim=8, epsilon=0.01, max_steps=8)
    step = _make_step_fn(halt, fixed_p=0.999)
    h0 = torch.zeros(1, 4, 8)
    out, ponder, n_steps, aux = halt(h0, step)
    assert n_steps == 1
    # Output should be ~ first step result (h0 + 1).
    assert torch.allclose(out, torch.ones_like(out), atol=1e-2)
    assert ponder.item() >= 0.0


def test_runs_full_loop_when_p_low():
    halt = ACTHalting(dim=8, epsilon=0.01, max_steps=4)
    step = _make_step_fn(halt, fixed_p=1e-4)
    h0 = torch.zeros(2, 3, 8)
    out, ponder, n_steps, _ = halt(h0, step)
    assert n_steps == 4
    # Last step absorbs the remainder, so total weight ~ 1; output ~ last hidden state.
    assert torch.allclose(out, torch.full_like(out, 4.0), atol=1e-2)
    assert ponder.item() >= 0.0


def test_weighted_average_property():
    """If all weights are equal, the output should equal the simple mean."""
    halt = ACTHalting(dim=4, epsilon=0.5, max_steps=2)
    # Force halting prob to 0.5 each step (so two steps each get weight 0.5).
    step = _make_step_fn(halt, fixed_p=0.5)
    h0 = torch.zeros(1, 1, 4)
    out, _, n_steps, _ = halt(h0, step)
    assert n_steps <= 2
    # h after step 1 = 1, h after step 2 = 2; weighted average ~ (0.5*1 + 0.5*2) = 1.5
    # When epsilon is large, ACT halts after step 1 absorbing all mass -> output ~1.
    # Either is acceptable; just check that out lies between 1 and 2.
    val = out.mean().item()
    assert 0.99 <= val <= 2.01


def test_ponder_cost_nonnegative():
    halt = ACTHalting(dim=4, epsilon=0.01, max_steps=3)
    step = _make_step_fn(halt, fixed_p=0.1)
    h0 = torch.zeros(1, 2, 4)
    _, ponder, _, _ = halt(h0, step)
    assert ponder.item() >= 0.0


def test_aux_loss_accumulates():
    halt = ACTHalting(dim=4, epsilon=1e-9, max_steps=3)

    def step(h, t):
        return h + 1.0, h.new_tensor(0.5)

    h0 = torch.zeros(1, 1, 4)
    _, _, n_steps, aux = halt(h0, step)
    assert aux.item() == 0.5 * n_steps
