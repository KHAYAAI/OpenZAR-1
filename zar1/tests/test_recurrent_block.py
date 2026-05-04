"""Tests for the recurrent transformer block."""

import torch

from zar1.recurrent_block import RecurrentTransformerBlock, _spectral_norm_power_iter


def _make_block(max_loops: int = 4) -> RecurrentTransformerBlock:
    return RecurrentTransformerBlock(
        dim=64,
        num_heads=8,
        num_kv_heads=4,
        max_loops=max_loops,
        num_experts=4,
        top_k=2,
        max_seq_len=32,
    )


def test_weight_tying_across_loops():
    block = _make_block()
    x = torch.randn(2, 8, 64)
    # Snapshot parameters before/after multiple loops; they must not change.
    before = {k: v.detach().clone() for k, v in block.named_parameters()}
    h = x
    for t in range(4):
        h, _ = block(h, loop_index=t)
    for k, v in block.named_parameters():
        assert torch.equal(before[k], v.detach()), f"parameter {k} changed across loops"


def test_loop_embedding_changes_per_iteration():
    block = _make_block()
    embeds = block.loop_embed.weight  # (max_loops, dim)
    for i in range(embeds.shape[0]):
        for j in range(i + 1, embeds.shape[0]):
            # Random init means embeddings should be distinct.
            assert not torch.allclose(embeds[i], embeds[j])


def test_spectral_norm_constraint():
    block = _make_block()
    # Blow up the recurrent_proj to force a rescale.
    with torch.no_grad():
        block.recurrent_proj.weight.mul_(50.0)
    # Run several times so the cached power-iteration vector converges.
    for _ in range(4):
        block.apply_spectral_constraint()
    w = block.recurrent_proj.weight.data
    # Validate via SVD (ground truth) — small slack since power iter is approximate.
    sigma = torch.linalg.svdvals(w)[0].item()
    assert sigma <= block.spectral_max + 5e-2, f"sigma={sigma}"


def test_power_iteration_matches_svd():
    torch.manual_seed(0)
    w = torch.randn(32, 32)
    u = torch.randn(32)
    u = u / u.norm()
    sigma_pi, _ = _spectral_norm_power_iter(w, u, n_iters=50)
    sigma_svd = torch.linalg.svdvals(w)[0]
    assert abs(sigma_pi.item() - sigma_svd.item()) / sigma_svd.item() < 0.05


def test_forward_shapes_and_aux_loss():
    block = _make_block()
    x = torch.randn(2, 8, 64)
    out, aux = block(x, loop_index=0)
    assert out.shape == x.shape
    assert aux.dim() == 0
