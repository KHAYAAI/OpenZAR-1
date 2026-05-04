# ZAR-1: Sovereign Verifiable Recurrent-Depth MoE Transformer

**ZAR-1** is a large language model from South Africa featuring a cutting-edge **Recurrent-Depth Mixture-of-Experts (MoE) Transformer** architecture with adaptive computation time (ACT) halting, spectral-radius stability constraints, and integrated support for African languages.

## Overview

ZAR-1 is built on a novel architecture that combines:

1. **Recurrent Depth**: A weight-tied transformer block looped up to 8 times per token, enabling variable-depth reasoning without architectural changes.
2. **Mixture of Experts**: 256 sparse experts with top-8 gating and Shazeer load-balancing for efficient scaling.
3. **Adaptive Computation Time (ACT)**: Per-token halting probabilities that allow early exit when sufficient computation is reached.
4. **Spectral Radius Constraint**: Ensures training stability via power-iteration-based spectral norm control.
5. **Grouped Query Attention (GQA)**: Efficient attention with 32 query heads and 8 KV heads.
6. **African Language Support**: Extended tokenizer with 20k Bantu/Swahili subwords.

## Architecture

### Model Stages

```
Input → [Embedding] → [Prelude: 4x GQA+Dense]
         ↓
     [Recurrent Core (1-8 loops via ACT):
       - GQA Attention + Residual
       - MoE FFN (256 experts) + Residual
       - Loop-index embedding injection
       - Spectral constraint (ρ < 0.99)
       - ACT halting probability
     ]
         ↓
     [Coda: 4x GQA+Dense] → [RMSNorm] → [LM Head] → Output
```

### Config (7B Model)

| Param | Value |
|-------|-------|
| Dim | 4096 |
| Query heads | 32 |
| KV heads | 8 |
| Prelude/Coda layers | 4 each |
| Max loops | 8 |
| Experts | 256 (top-8) |
| Vocab | 128,256 |
| Max seq len | 8192 |

## Quick Start

### Installation

```bash
cd zar1
pip install -e .
```

### Training (Single GPU)

```bash
python scripts/train_7b.py --config configs/zar1_7b.yaml
```

### Distributed Training (8×H100)

```bash
torchrun --nproc_per_node=8 scripts/train_7b.py --config configs/zar1_7b.yaml
```

### Inference

```python
import torch
from zar1.model import ZAR1Config, ZAR1Model
from transformers import AutoTokenizer

model = ZAR1Model(ZAR1Config.from_yaml("configs/zar1_7b.yaml"))
model.load_state_dict(torch.load("checkpoints/latest.pt")["model"])
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

prompt = "The future of AI is"
output = model.generate(tokenizer.encode(prompt, return_tensors="pt"), max_new_tokens=64)
print(tokenizer.decode(output[0]))
```

### FastAPI Server

```bash
python -m zar1.server --checkpoint ./checkpoints/latest.pt --port 8000

# Endpoints:
# POST /v1/infer - single prompt
# POST /v1/infer_batch - batch of prompts
# POST /v1/stream - streaming (SSE)
# GET /health - health check
```

## Evaluation

```bash
python scripts/benchmark.py --model_path ./checkpoints/latest.pt \
                            --benchmark all --output_json results.json
```

Supports MMLU-Pro, GPQA, HumanEval, African languages.

## Docker

```bash
docker build -t zar1:latest -f Dockerfile .
docker run --gpus all -v /path/to/checkpoints:/models -p 8000:8000 zar1:latest
```

## Testing

```bash
pytest tests/ -v
```

## Features

### ACT Halting

Model predicts per-token halt probability, enabling variable-depth inference:
- Faster inference for easy tokens
- Deeper reasoning for hard tokens
- Configurable `act_epsilon` (halting threshold, default 0.99)

### Spectral Constraint

Recurrent injection matrix spectral norm kept ≤ 0.99 via power iteration:
- Prevents gradient explosion/vanishing
- Applied automatically post-optimizer step
- Configurable `spectral_max` (default 0.99)

### Loop-Index Embeddings

Each loop iteration adds learned embedding, allowing same weights to specialize:
- Iteration-dependent behavior without architectural changes
- Helps with depth-dependent scaling

### African Languages

- Extended tokenizer: 20k African subwords (Zulu, Xhosa, Swahili)
- Data mixing: 15% African JSONL, 85% FineWeb-Edu (configurable)
- Script: `scripts/extend_tokenizer.py`

## Performance

### Training (8×H100)
- Throughput: ~18k tokens/sec
- Cost: ~$10 per 1B tokens
- 500k steps: ~7 days

### Inference (Single H100)
- n_loops=1: ~150 tps
- n_loops=4: ~50 tps
- n_loops=8: ~25 tps

## Config

Edit `configs/zar1_7b.yaml`:

```yaml
model:
  dim: 4096
  n_heads: 32
  n_kv_heads: 8
  max_loops: 8
  num_experts: 256
  top_k: 8

training:
  batch_tokens: 2097152
  steps: 500000
  lr: 3.0e-4
```

## License

Apache 2.0

## Citation

```bibtex
@software{zar1_2024,
  title={ZAR-1: Sovereign Verifiable Recurrent-Depth LLM},
  author={KhayaAI},
  year={2024},
  url={https://github.com/khayaai/OpenZAR-1}
}
```

---

**Built with ❤️ from South Africa.**
