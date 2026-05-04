# ZAR-1

A sovereign, verifiable Recurrent-Depth Mixture-of-Experts Transformer from
South Africa. ZAR-1 combines:

- **Mixture of 256 Experts** with top-8 gating and load-balancing loss
- **Adaptive Computation Time (ACT)** halting for variable inference depth
- **Spectral-radius constraint** on the recurrent injection for training stability
- **Loop-index embeddings** that condition the shared-weight recurrent block
  on its current depth
- **Grouped Query Attention (GQA)** with RoPE in every block
- **FSDP** training pipeline for 8x H100

## Layout

```
zar1/
├── src/zar1/
│   ├── moe.py               # MoE FFN (top-k gating, capacity, aux loss)
│   ├── recurrent_block.py   # GQA + MoE + loop embeds + spectral constraint
│   ├── act.py               # PonderNet-style halting
│   ├── model.py             # Prelude -> Recurrent -> Coda assembly
│   ├── data.py              # FineWeb-Edu + African JSONL packed loader
│   └── server.py            # FastAPI inference server
├── configs/zar1_7b.yaml     # 7B training config
├── scripts/
│   ├── train_7b.py          # FSDP distributed training
│   ├── benchmark.py         # MMLU-Pro/GPQA/HumanEval + Zulu/Xhosa
│   └── export_onnx.py       # ONNX export for Zenith canister
├── tests/                   # pytest unit tests
├── zenith/                  # ZK-proof canister (Rust + WASM + Plonk)
├── Dockerfile
└── pyproject.toml
```

## Install

```bash
pip install -e .[dev]
```

## Train

```bash
torchrun --nproc_per_node=8 scripts/train_7b.py --config configs/zar1_7b.yaml
```

The script:
- wraps the model with FSDP (`HYBRID_SHARD` if `distributed.hybrid_shard: true`)
- uses bfloat16 mixed precision
- applies the spectral-radius constraint after each optimizer step
- handles `SIGTERM` / `SIGINT` (saves a checkpoint before exiting)
- supports resume via `training.resume_from`
- logs `loss`, `perplexity`, `lr`, `aux_loss`, `ponder_cost`,
  ACT step count, and GPU memory to WandB

## Evaluate

```bash
python scripts/benchmark.py \
    --config configs/zar1_7b.yaml \
    --checkpoint checkpoints/zar1_7b/latest.pt \
    --wandb
```

## Serve

```bash
ZAR1_CONFIG=configs/zar1_7b.yaml \
ZAR1_CHECKPOINT=checkpoints/zar1_7b/latest.pt \
python -m zar1.server
```

Endpoints: `POST /v1/infer`, `POST /v1/infer_batch`, `GET /v1/stream` (SSE).

## Docker

```bash
docker build -t zar1:latest .
docker run --gpus all -p 8000:8000 \
    -v $PWD/checkpoints:/app/checkpoints \
    -e ZAR1_CHECKPOINT=/app/checkpoints/zar1_7b/latest.pt \
    zar1:latest
```

## Zenith canister

```bash
bash zenith/deploy.sh
```

Exports the model to ONNX, builds the Rust canister to `wasm32-unknown-unknown`,
and deploys to the Zenith testnet. The canister batches up to 1000 inference
requests, runs them with `candle-onnx`, and posts a single aggregated Plonk
proof on-chain.

## Tests

```bash
pytest tests/
```

## Architecture details

### Recurrent core

A single `RecurrentTransformerBlock` is applied for up to `max_loops`
iterations with weight tying. Each loop:

1. Inject the loop-index embedding (`loop_embed[t]`).
2. Run the GQA attention with RoPE.
3. Run the MoE FFN.
4. After the optimizer step, project the recurrent injection matrix back into
   the spectral ball of radius `0.99` via 5 power-iteration steps.

ACT halts the loop early when the cumulative sigmoid halting probability
exceeds `1 - epsilon`. The ponder cost penalizes unused probability mass.

### MoE FFN

Each token is routed to its top-8 of 256 SwiGLU experts. The Shazeer
load-balancing auxiliary loss (`N * sum(f_i * P_i)`) prevents collapse, and a
capacity factor of 1.25 caps tokens per expert (overflow is dropped).
