"""Tests for the Mixture-of-Experts FFN."""

import torch

from zar1.moe import MoEFeedForward


def test_moe_output_shape():
    moe = MoEFeedForward(dim=64, num_experts=8, top_k=2, capacity_factor=2.0)
    x = torch.randn(2, 16, 64)
    out, probs, aux = moe(x)
    assert out.shape == (2, 16, 64)
    assert probs.shape == (2, 16, 8)
    assert aux.dim() == 0  # scalar
    # Routing probs sum to 1 along expert dim
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-4)


def test_moe_aux_loss_positive():
    moe = MoEFeedForward(dim=32, num_experts=4, top_k=1, capacity_factor=4.0)
    x = torch.randn(1, 8, 32)
    _, _, aux = moe(x)
    assert aux.item() > 0


def test_moe_capacity_drop():
    """With a tiny capacity factor, some tokens should be dropped (output zeros)."""
    torch.manual_seed(0)
    moe = MoEFeedForward(
        dim=16, num_experts=2, top_k=1, capacity_factor=0.1, hidden_dim=8
    )
    # Force a many-to-one routing scenario by zeroing one router weight column.
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, :] = 10.0  # all tokens prefer expert 0
    x = torch.randn(1, 32, 16)
    out, _, _ = moe(x)
    # Most tokens dropped -> many output rows are exactly zero.
    zeros = (out.abs().sum(-1) == 0).float().sum().item()
    assert zeros > 0


def test_moe_gradients_flow():
    moe = MoEFeedForward(dim=16, num_experts=4, top_k=2, capacity_factor=2.0)
    x = torch.randn(1, 8, 16, requires_grad=True)
    out, _, aux = moe(x)
    (out.sum() + aux).backward()
    assert x.grad is not None
    assert moe.router.weight.grad is not None
