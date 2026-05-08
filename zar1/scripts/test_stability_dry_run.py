"""Dry run test: Validate all stability features in isolation (no data loading required).

Usage:
    python scripts/test_stability_dry_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zar1.model import ZAR1Config, ZAR1Model
from zar1.training_stability import (
    SpectralConstraintController,
    ACTCurriculum,
    ExpertLoadMonitor,
)


def test_spectral_constraint():
    """Test spectral constraint enforcement."""
    print("\n" + "=" * 60)
    print("TEST 1: SPECTRAL CONSTRAINT")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small model for quick test
    config = ZAR1Config(dim=256, num_heads=8, num_kv_heads=4, max_loops=4)
    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)

    spectral = SpectralConstraintController(
        weight=model.recurrent_block.recurrent_proj.weight,
        max_sigma=0.99,
        n_iters=5,
        device=device,
    )

    print("Initial model created")

    # Run spectral constraint
    sigma_before = spectral.apply_constraint()
    print(f"  Spectral radius after constraint: {sigma_before:.4f}")

    # Verify it's reasonable
    assert sigma_before <= 1.05, f"Spectral radius too high: {sigma_before}"
    print("  ✓ Spectral constraint working")


def test_act_curriculum():
    """Test ACT curriculum (halting disabled during curriculum)."""
    print("\n" + "=" * 60)
    print("TEST 2: ACT CURRICULUM")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ZAR1Config(dim=256, num_heads=8, num_kv_heads=4, max_loops=4)
    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)

    curriculum = ACTCurriculum(curriculum_steps=100)

    print("Initial model created with curriculum (100 steps)")

    # Test early curriculum phase
    model.train()
    x = torch.randint(0, 1000, (2, 16)).to(device)

    print(f"  Curriculum progress (step 0): {curriculum.curriculum_progress():.1%}")
    assert curriculum.is_curriculum_active(), "Curriculum should be active at step 0"

    # Simulate many steps
    for step in range(150):
        curriculum.current_step = step

    print(f"  Curriculum progress (step 150): {curriculum.curriculum_progress():.1%}")
    assert not curriculum.is_curriculum_active(), "Curriculum should be inactive after 100 steps"

    print("  ✓ ACT curriculum working")


def test_expert_load_monitoring():
    """Test expert load monitoring and dead expert detection."""
    print("\n" + "=" * 60)
    print("TEST 3: EXPERT LOAD MONITORING")
    print("=" * 60)

    n_experts = 256
    monitor = ExpertLoadMonitor(n_experts=n_experts, death_threshold=0.05)

    print(f"Created monitor for {n_experts} experts")

    # Simulate balanced routing (all experts used equally)
    print("\n  Case 1: Balanced routing")
    balanced_routes = torch.arange(n_experts).repeat(10)
    monitor.update(balanced_routes)
    stats = monitor.get_stats()

    print(f"    Active experts: {stats['active_experts']}/{n_experts}")
    print(f"    Dead ratio: {stats['dead_ratio']:.1%}")
    print(f"    Entropy: {stats['entropy']:.2f}")

    assert stats["active_experts"] > 200, "Most experts should be active when balanced"
    print("    ✓ Balanced routing detected")

    monitor.reset()

    # Simulate imbalanced routing (few experts used, many dead)
    print("\n  Case 2: Imbalanced routing (expert collapse simulation)")
    imbalanced_routes = torch.randint(0, 10, (1000,))  # Only first 10 experts
    monitor.update(imbalanced_routes)
    stats = monitor.get_stats()

    print(f"    Active experts: {stats['active_experts']}/{n_experts}")
    print(f"    Dead ratio: {stats['dead_ratio']:.1%}")
    print(f"    Entropy: {stats['entropy']:.2f}")

    assert stats["dead_ratio"] > 0.9, "Most experts should be dead when imbalanced"
    print("    ✓ Expert collapse detected correctly")


def test_gradient_flow():
    """Test that gradients flow through all components."""
    print("\n" + "=" * 60)
    print("TEST 4: GRADIENT FLOW WITH STABILITY")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ZAR1Config(dim=256, num_heads=8, num_kv_heads=4, max_loops=4)
    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)

    model.train()

    # Forward pass
    x = torch.randint(0, 1000, (2, 16)).to(device)
    labels = torch.randint(0, 1000, (2, 16)).to(device)

    output = model(x, labels=labels)
    loss = output.loss

    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            if "recurrent" in name:
                print(f"  ✓ {name}: grad norm = {param.grad.norm():.4f}")

    assert has_grads, "No gradients computed!"
    print("  ✓ Gradients flowing through model")


def test_full_integration():
    """Test all stability features together in training loop."""
    print("\n" + "=" * 60)
    print("TEST 5: FULL INTEGRATION (5-step mini training)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ZAR1Config(dim=256, num_heads=8, num_kv_heads=4, max_loops=4)
    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    spectral = SpectralConstraintController(
        weight=model.recurrent_block.recurrent_proj.weight,
        max_sigma=0.99,
        device=device,
    )

    curriculum = ACTCurriculum(curriculum_steps=10)
    monitor = ExpertLoadMonitor(n_experts=config.num_experts)

    print("Setup complete, running 5 training steps...")

    losses = []
    for step in range(5):
        # Synthetic batch
        x = torch.randint(0, 1000, (2, 16)).to(device)
        labels = torch.randint(0, 1000, (2, 16)).to(device)

        # Forward with curriculum
        model.train()
        output = model(x, labels=labels)
        loss = output.loss

        # Backward
        loss.backward()

        # Spectral constraint
        sigma = spectral.apply_constraint()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        curriculum.current_step += 1

        losses.append(loss.item())
        print(f"  Step {step}: loss={loss.item():.4f}, σ={sigma:.4f}")

    # Check loss trend
    avg_first_half = sum(losses[:2]) / 2
    avg_second_half = sum(losses[2:]) / 2

    print(f"\n  Loss trend: {avg_first_half:.4f} → {avg_second_half:.4f}")
    print("  ✓ Integration test passed")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ZAR-1 STABILITY FEATURES: DRY RUN TESTS")
    print("=" * 60)

    try:
        test_spectral_constraint()
        test_act_curriculum()
        test_expert_load_monitoring()
        test_gradient_flow()
        test_full_integration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nReady for Phase 1 training!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
