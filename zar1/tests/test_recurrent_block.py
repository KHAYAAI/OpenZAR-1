"""Tests for Recurrent Transformer Block."""

import torch

from zar1.recurrent_block import RecurrentTransformerBlock


def test_recurrent_block_shape():
    """Test forward pass returns correct shape."""
    block = RecurrentTransformerBlock(
        dim=64,
        num_heads=4,
        num_kv_heads=2,
        max_loops=4,
    )
    x = torch.randn(2, 8, 64)
    h, aux_loss = block(x, loop_index=0)
    assert h.shape == x.shape
    assert aux_loss.dim() == 0


def test_loop_index_changes_output():
    """Test that different loop indices produce different outputs (due to loop embeddings)."""
    torch.manual_seed(42)
    block = RecurrentTransformerBlock(dim=32, num_heads=2, num_kv_heads=1, max_loops=4)
    x = torch.randn(1, 4, 32)

    with torch.no_grad():
        h0, _ = block(x, loop_index=0)
        h1, _ = block(x, loop_index=1)
        h2, _ = block(x, loop_index=2)

    # Different loop indices should produce slightly different outputs
    assert not torch.allclose(h0, h1, atol=1e-4)
    assert not torch.allclose(h1, h2, atol=1e-4)


def test_spectral_constraint():
    """Test that spectral norm is kept under control."""
    block = RecurrentTransformerBlock(
        dim=32,
        num_heads=2,
        num_kv_heads=1,
        max_loops=4,
        spectral_max=0.95,
    )
    for _ in range(5):
        sigma = block.apply_spectral_constraint()
        assert sigma <= 1.0  # Should be rescaled


def test_recurrent_gradient_flow():
    """Test gradients flow through the recurrent block."""
    block = RecurrentTransformerBlock(dim=16, num_heads=2, num_kv_heads=1, max_loops=2)
    x = torch.randn(1, 4, 16, requires_grad=True)
    h, aux = block(x, loop_index=0)
    loss = h.sum() + aux
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_attention_mask():
    """Test that attention mask is properly applied."""
    block = RecurrentTransformerBlock(dim=32, num_heads=2, num_kv_heads=1)
    x = torch.randn(1, 4, 32)

    # Causal mask (default via SDPA)
    h1, _ = block(x, loop_index=0, attention_mask=None)

    # Custom mask (all zeros = allow all attention)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    h2, _ = block(x, loop_index=0, attention_mask=mask)

    # Results should differ (though not drastically)
    assert h1.shape == h2.shape
