# ZAR-1 Phase 1 Pretraining Launch Checklist

## Infrastructure & Environment

### ✅ Item 1: WandB Configuration
- [x] Project name: `zar1`
- [x] Config path: `configs/zar1_7b.yaml`
- [ ] **Action Required**: Authenticate WandB before launch
  ```bash
  wandb login
  # Paste your API key from https://wandb.ai/settings/api
  ```

### ❓ Item 2: Checkpoint Storage
- [x] Output directory: `./checkpoints/zar1_7b`
- [x] Estimated checkpoint size: ~8.9 GB per checkpoint (7B model in bfloat16)
- [x] Estimated total storage (100 checkpoints @ save_every=1000): ~890 GB
- [ ] **Action Required**: Verify available disk space
  ```bash
  df -h . | tail -1  # Current directory
  # Ensure at least 1TB available
  ```

### ⚠️ Item 3: GPU Cluster Reservation
- [x] Required GPUs: 8 × H100 (80GB HBM)
- [x] Recommended: NVLink/NVSwitch for 2.2TB/s inter-GPU bandwidth
- [x] Alternative: A100-80GB (slightly slower, ~95 tokens/sec vs 110 tokens/sec)
- [ ] **Action Required**: Reserve cluster
  - [ ] Verify 8×H100 availability
  - [ ] Check cluster interconnect (NVLink preferred)
  - [ ] Verify no maintenance windows during 7-10 day training window
  - [ ] Ensure CUDA 12.1+ and NCCL 2.18+ installed

### ✅ Item 4: Model Architecture Validation
- [x] Smoke test: All components validated
- [x] Gradient flow: Verified
- [x] Spectral constraint: Enforced
- [x] Features: LoRA ✓ | PID ✓ | LRS ✓ | ACT ✓

### ❓ Item 5: Dataset Access
- [ ] **Action Required**: Verify HuggingFace Hub access
  ```bash
  python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True); print(next(iter(ds)))" 
  # Should print a sample document in <1 second
  ```
  - If this fails: either offline mode or network restriction
  - Workaround: Pre-download shard files (~50GB per shard)

---

## Training Parameters

### Default Configuration (zar1_7b.yaml)
```yaml
Training Steps:     500,000
Batch Tokens:       2,097,152 (2M per step)
Gradient Accum:     16 steps
Learning Rate:      3e-4 (cosine decay to 3e-5)
Warmup Steps:       2,000

Model:
  Dimension:        4,096
  Heads:            32 (num_heads=32, num_kv_heads=8 → 4x GQA)
  Layers:           4 prelude + 8 recurrent + 4 coda = 16 total
  Experts:          256 (top-8, PID load balancing)
  Max Loops:        8
  
Data:
  Source:           FineWeb-Edu (FP32 → bfloat16)
  Tokenizer:        Llama-3 (128,256 tokens)
  Max Seq Length:   8,192
  Pack Sequences:   true (cross-document masking)
```

### Estimated Wall-Clock Time
- **Per 1000 steps**: ~6-8 hours on 8×H100
- **Total (500k steps)**: 7-10 days
- **Estimated cost**: $50-75 USD (at $2/H100-hour)

---

## Pre-Launch Validation Steps

### Step 1: Verify Python Environment
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```
Requirements:
- PyTorch ≥ 2.5
- CUDA 12.1+
- transformers ≥ 4.44
- datasets ≥ 2.20
- wandb ≥ 0.17

### Step 2: Test Data Loading
```bash
python -c "
from zar1.data import build_train_val_loaders
import yaml

with open('configs/zar1_7b.yaml') as f:
    cfg = yaml.safe_load(f)

train_loader, val_loader = build_train_val_loaders(
    cfg=cfg['data'],
    seq_len=cfg['training']['max_seq_len'],
    micro_batch_size=cfg['training']['micro_batch_size'],
)

# Get first batch
batch = next(iter(train_loader))
print(f'Batch shape: {batch[\"input_ids\"].shape}')
print(f'Sample tokens: {batch[\"input_ids\"][0, :10].tolist()}')
"
```

### Step 3: Test Single Training Step
```bash
# Single-GPU test (no distributed)
python scripts/train_7b.py --config configs/zar1_7b.yaml --steps 1

# Should complete in <2 minutes, print one loss value
```

### Step 4: Test Multi-GPU Setup (Optional, Recommended)
```bash
# Test with 2 GPUs (simulating 8-GPU setup)
torchrun --nproc_per_node=2 scripts/train_7b.py \
  --config configs/zar1_7b.yaml --steps 1

# Should distribute and print loss from all ranks
```

---

## Launch Commands

### Option A: Dry Run (Validation)
```bash
# Test 100 steps on 8 GPUs to validate stability
torchrun --nproc_per_node=8 scripts/train_7b.py \
  --config configs/zar1_7b.yaml \
  --override training.steps=100
```
**Expected output**: Loss decreases, no NaNs, runs for ~1 hour

### Option B: Full Training (Production)
```bash
# Full 500k step run
torchrun --nproc_per_node=8 scripts/train_7b.py \
  --config configs/zar1_7b.yaml

# Watch training:
# 1. WandB dashboard: https://wandb.ai/[username]/zar1
# 2. Local logs: tail checkpoints/zar1_7b/train.log
# 3. Checkpoints: ls -lh checkpoints/zar1_7b/
```

### Option C: Resume from Checkpoint
```bash
# If training is interrupted
torchrun --nproc_per_node=8 scripts/train_7b.py \
  --config configs/zar1_7b.yaml \
  --resume checkpoints/zar1_7b/step_100000.pt
```

---

## Monitoring During Training

### WandB Metrics to Watch
- `loss/train`: Should decrease smoothly (target: 2.5 → 1.8 over 500k steps)
- `aux_loss`: MoE load-balancing loss (target: < 0.05)
- `ponder_cost`: ACT halting penalty (target: 0.01-0.02)
- `n_loops`: Average recurrence depth (target: 3-5 loops)
- `throughput`: Tokens/sec (target: > 10k tokens/sec)
- `expert_entropy`: Router entropy (target: > 0.9 = well-balanced)

### Red Flags (Stop Training If You See)
- [ ] Loss becomes NaN or Inf → learning rate too high
- [ ] Loss plateaus without decreasing → might need LR adjustment
- [ ] Memory OOM after step N → reduce micro_batch_size
- [ ] expert_entropy < 0.5 → expert collapse (shouldn't happen with PID)
- [ ] Throughput drops below 5k tokens/sec → check GPU utilization

---

## Post-Training

After 500k steps (~7-10 days):

### 1. Select Checkpoint
```bash
# Usually pick latest or best val_loss
cp checkpoints/zar1_7b/step_500000.pt checkpoints/zar1_7b/latest.pt
```

### 2. Evaluate Phase 0 Gates
```bash
python scripts/evaluate_phase0.py \
  --model_path checkpoints/zar1_7b/latest.pt \
  --config configs/zar1_7b.yaml \
  --benchmark all

# Expected gates:
# - MMLU 5-shot: ≥ 50% ✓
# - GSM8K 8-shot: ≥ 15% ✓
# - Depth extrapolation: ≥ 3% ✓
```

### 3. Begin Phase 2 (SFT with Markovian RSA)
- While pretraining runs, design RSA synthetic examples (1-2 days)
- After Phase 1 completes, run SFT for 10k-20k steps
- Enable RSA token during generation

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA out of memory | Batch size too large | Reduce `micro_batch_size` in config |
| All GPUs show 0% usage | Data loading bottleneck | Increase `num_workers` in config |
| Loss doesn't decrease | Learning rate incorrect | Adjust `lr` (default 3e-4) |
| Training hangs after step N | NCCL timeout | Increase `NCCL_TIMEOUT=3600` |
| Checkpoint save is slow | Disk I/O bottleneck | Save to local NVMe if possible |
| Expert collapse (low entropy) | Shouldn't happen with PID | Check `use_pid=true` in config |

---

## Estimated Timeline

- **Days 1-7**: Phase 1 pretraining (500k steps)
- **Days 3-4 (parallel)**: Design Markovian RSA data
- **Days 8-9**: Phase 2 SFT (10k-20k steps with RSA)
- **Days 10-12**: Phase 3 RL fine-tuning
- **Total**: ~2 weeks to full training completion

---

## Sign-Off Before Launch

Before running the full training, confirm:

- [ ] Disk space verified (1TB+ available)
- [ ] GPU cluster reserved (8×H100, 7-10 days)
- [ ] WandB authenticated (`wandb login`)
- [ ] HuggingFace Hub access verified
- [ ] Single-GPU test passed (`python scripts/train_7b.py --steps 1`)
- [ ] Multi-GPU test passed (`torchrun --nproc_per_node=8 ... --steps 1`)
- [ ] All team members notified (cluster in use)
- [ ] Slack/email alerts configured (job completion)

**Ready to launch**: Run `torchrun --nproc_per_node=8 scripts/train_7b.py --config configs/zar1_7b.yaml`

