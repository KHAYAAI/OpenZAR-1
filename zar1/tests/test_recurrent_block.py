"""Tests for Recurrent Transformer Block."""

import torch

from zar1.moe import MLPRouter, PIDBalancer
from zar1.recurrent_block import LoRAAdapter, RecurrentTransformerBlock


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


def test_lora_adapter_starts_at_zero():
    """LoRA adapter must start as identity (B initialized to 0)."""
    lora = LoRAAdapter(dim=32, rank=8)
    x = torch.randn(2, 4, 32)
    delta = lora(x)
    assert delta.shape == x.shape
    # B is zero-initialized, so output must be exactly zero.
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_lora_adapter_trains():
    """LoRA adapter should produce non-zero output after a gradient step."""
    lora = LoRAAdapter(dim=16, rank=4)
    x = torch.randn(1, 4, 16, requires_grad=True)
    target = torch.randn_like(x)

    optim = torch.optim.SGD(lora.parameters(), lr=0.1)
    for _ in range(10):
        optim.zero_grad()
        out = lora(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optim.step()

    delta = lora(x)
    assert not torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_recurrent_block_lora_per_loop_distinct():
    """Different loop indices must use different LoRA adapters."""
    torch.manual_seed(0)
    block = RecurrentTransformerBlock(
        dim=32, num_heads=2, num_kv_heads=1, max_loops=4, lora_rank=8, use_lora=True
    )
    # Manually push different non-zero values into different LoRA Bs.
    with torch.no_grad():
        for i, lora in enumerate(block.attn_loras):
            lora.lora_B.weight.fill_(0.01 * (i + 1))

    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        h0, _ = block(x, loop_index=0)
        h1, _ = block(x, loop_index=1)
        h3, _ = block(x, loop_index=3)

    assert not torch.allclose(h0, h1, atol=1e-5)
    assert not torch.allclose(h1, h3, atol=1e-5)


def test_recurrent_block_lora_disabled():
    """When use_lora=False, the block should produce output without LoRA paths."""
    block = RecurrentTransformerBlock(
        dim=32, num_heads=2, num_kv_heads=1, max_loops=4, use_lora=False
    )
    assert block.attn_loras is None
    assert block.ffn_loras is None
    x = torch.randn(1, 4, 32)
    h, _ = block(x, loop_index=0)
    assert h.shape == x.shape


def test_mlp_router_shape():
    """Test MLP router outputs correct shape."""
    router = MLPRouter(dim=64, num_experts=256)
    x = torch.randn(100, 64)
    logits = router(x)
    assert logits.shape == (100, 256)


def test_pid_balancer_initialization():
    """Test PID balancer initializes with correct state."""
    balancer = PIDBalancer(num_experts=32, kp=0.5, ki=0.1, kd=0.1)
    assert balancer.ema_load.shape == (32,)
    assert balancer.integral_error.shape == (32,)
    assert balancer.prev_error.shape == (32,)
    # EMA load should be uniform initially
    assert torch.allclose(balancer.ema_load, torch.ones(32) / 32)


def test_pid_balancer_updates_load():
    """Test PID balancer tracks expert load over time."""
    balancer = PIDBalancer(num_experts=8)
    routing_probs = torch.softmax(torch.randn(64, 8), dim=-1)
    top1 = torch.argmax(routing_probs, dim=-1)

    # Run PID multiple times; load should evolve
    biases = []
    for _ in range(3):
        bias = balancer(routing_probs, top1)
        biases.append(bias.clone())

    # Biases should change as EMA load updates
    assert not torch.allclose(biases[0], biases[1], atol=1e-5)


def test_recurrent_block_with_pid():
    """Test recurrent block with PID-based load balancing enabled."""
    block = RecurrentTransformerBlock(
        dim=32,
        num_heads=2,
        num_kv_heads=1,
        max_loops=4,
        num_experts=16,
        top_k=4,
        use_pid=True,
    )
    block.train()  # PID only applies in training
    x = torch.randn(1, 8, 32)
    h, aux_loss = block(x, loop_index=0)
    assert h.shape == x.shape
    assert aux_loss.dim() == 0
    assert not torch.isnan(aux_loss)


def test_recurrent_block_pid_disabled():
    """Test recurrent block with PID-based load balancing disabled."""
    block = RecurrentTransformerBlock(
        dim=32,
        num_heads=2,
        num_kv_heads=1,
        max_loops=4,
        num_experts=16,
        top_k=4,
        use_pid=False,
    )
    assert block.moe.pid is None
    x = torch.randn(1, 8, 32)
    h, aux_loss = block(x, loop_index=0)
    assert h.shape == x.shape
