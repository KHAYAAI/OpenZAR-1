"""FastAPI inference server for ZAR-1.

Usage:
    python -m zar1.server --checkpoint ./checkpoints/latest.pt --port 8000

Or with uvicorn:
    uvicorn zar1.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zar1.model import ZAR1Config, ZAR1Model

app = FastAPI(title="ZAR-1 Inference", version="0.1.0")

_model: ZAR1Model | None = None
_tokenizer = None
_device = None


def _ensure_model_loaded():
    global _model, _tokenizer, _device
    if _model is None:
        raise RuntimeError("Model not loaded. Call /health or initialize via command-line.")
    return _model, _tokenizer, _device


class InferRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(64, ge=1, le=512, description="Max tokens to generate")
    temperature: float = Field(1.0, ge=0.01, le=2.0, description="Sampling temperature")
    top_k: int | None = Field(None, ge=1, le=100, description="Top-K sampling")
    n_loops: int | None = Field(None, ge=1, le=64, description="Override loop count")


class InferResponse(BaseModel):
    prompt: str
    response: str
    n_tokens: int
    n_loops: int


class BatchInferRequest(BaseModel):
    prompts: list[str] = Field(..., min_items=1, max_items=32)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(1.0, ge=0.01, le=2.0)
    top_k: int | None = None
    n_loops: int | None = None


class BatchInferResponse(BaseModel):
    results: list[InferResponse]


@app.on_event("startup")
async def startup():
    """Placeholder for startup events."""
    pass


@app.get("/health")
async def health():
    """Health check endpoint."""
    model, _, device = _ensure_model_loaded()
    return {
        "status": "ok",
        "model_dim": model.config.dim,
        "device": str(device),
        "params": model.num_parameters(),
    }


@app.post("/v1/infer", response_model=InferResponse)
async def infer(request: InferRequest) -> InferResponse:
    """Single prompt inference endpoint."""
    model, tokenizer, device = _ensure_model_loaded()

    try:
        input_ids = torch.tensor(
            tokenizer.encode(request.prompt, add_special_tokens=False), dtype=torch.long
        ).unsqueeze(0)
        if input_ids.shape[1] > model.config.max_seq_len:
            input_ids = input_ids[:, -model.config.max_seq_len :]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization failed: {str(e)}")

    try:
        output = model.generate(
            input_ids.to(device),
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            n_loops=request.n_loops,
        )
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return InferResponse(
        prompt=request.prompt,
        response=response_text,
        n_tokens=output.shape[1],
        n_loops=request.n_loops or model.config.max_loops,
    )


@app.post("/v1/infer_batch", response_model=BatchInferResponse)
async def infer_batch(request: BatchInferRequest) -> BatchInferResponse:
    """Batch inference endpoint."""
    results = []
    for prompt in request.prompts:
        single_req = InferRequest(
            prompt=prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            n_loops=request.n_loops,
        )
        result = await infer(single_req)
        results.append(result)
    return BatchInferResponse(results=results)


@app.post("/v1/stream")
async def stream_infer(request: InferRequest):
    """Streaming inference endpoint (Server-Sent Events)."""
    model, tokenizer, device = _ensure_model_loaded()

    async def generate_stream():
        try:
            input_ids = torch.tensor(
                tokenizer.encode(request.prompt, add_special_tokens=False),
                dtype=torch.long,
            ).unsqueeze(0)
            if input_ids.shape[1] > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len :]

            output = model.generate(
                input_ids.to(device),
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                n_loops=request.n_loops,
            )
            response_text = tokenizer.decode(output[0], skip_special_tokens=True)
            for token in response_text.split():
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def load_model_from_checkpoint(checkpoint_path: str, config_path: str | None = None):
    """Load model from checkpoint."""
    global _model, _tokenizer, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_path and Path(config_path).exists():
        _model = ZAR1Model(ZAR1Config.from_yaml(config_path)).to(
            device=_device, dtype=torch.bfloat16
        )
    else:
        _model = ZAR1Model(ZAR1Config()).to(device=_device, dtype=torch.bfloat16)

    if Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(state.get("model", state), strict=False)
    _model.eval()

    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", use_fast=True
        )
    except Exception as e:
        print(f"Warning: could not load tokenizer: {e}")
        _tokenizer = None


def main():
    parser = argparse.ArgumentParser(description="ZAR-1 Inference Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    load_model_from_checkpoint(args.checkpoint, args.config)
    print(f"[INFO] Model loaded from {args.checkpoint}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
