"""FastAPI inference server for ZAR-1.

Endpoints:
    POST /v1/infer        - single-prompt generation
    POST /v1/infer_batch  - batched generation
    GET  /v1/stream       - SSE streaming generation (?prompt=...&n_loops=...)
    GET  /healthz         - liveness probe

The server loads the model in bfloat16 on a single GPU. For multi-GPU serving,
wrap with vLLM/TGI or use tensor parallel.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator

import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zar1.model import ZAR1Config, ZAR1Model


class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    n_loops: int | None = Field(default=None, ge=1, le=64)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int | None = Field(default=50, ge=1)


class InferBatchRequest(BaseModel):
    prompts: list[str]
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    n_loops: int | None = Field(default=None, ge=1, le=64)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int | None = Field(default=50, ge=1)


class InferResponse(BaseModel):
    text: str
    n_loops: int


class InferBatchResponse(BaseModel):
    texts: list[str]


_app_state: dict = {"model": None, "tokenizer": None, "device": None}


def _load_model(config_path: str, ckpt_path: str | None) -> tuple[ZAR1Model, object, torch.device]:
    """Load tokenizer + ZAR-1 model in bf16 onto a single CUDA device (or CPU)."""
    from transformers import AutoTokenizer

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    config = ZAR1Config.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA unavailable; running on CPU (slow).")

    model = ZAR1Model(config).to(device=device, dtype=torch.bfloat16)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"Loaded checkpoint from {ckpt_path}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["data"].get("tokenizer", "meta-llama/Meta-Llama-3-8B"), use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    return model, tokenizer, device


app = FastAPI(title="ZAR-1 Inference Server", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    cfg_path = os.environ.get("ZAR1_CONFIG", "configs/zar1_7b.yaml")
    ckpt = os.environ.get("ZAR1_CHECKPOINT")
    try:
        model, tok, dev = _load_model(cfg_path, ckpt)
    except Exception as e:  # pragma: no cover
        print(f"[startup] Failed to load model: {e}")
        return
    _app_state["model"] = model
    _app_state["tokenizer"] = tok
    _app_state["device"] = dev


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": _app_state["model"] is not None,
        "device": str(_app_state.get("device")),
    }


def _ensure_loaded() -> None:
    if _app_state["model"] is None:
        raise HTTPException(503, "Model not loaded; check server logs.")


@torch.no_grad()
def _generate_sync(
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    n_loops: int | None,
) -> list[str]:
    model: ZAR1Model = _app_state["model"]
    tok = _app_state["tokenizer"]
    dev = _app_state["device"]

    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(dev)
    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        n_loops=n_loops,
        eos_token_id=tok.eos_token_id,
    )
    new_ids = out_ids[:, input_ids.shape[1] :]
    return tok.batch_decode(new_ids, skip_special_tokens=True)


@app.post("/v1/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    _ensure_loaded()
    try:
        text = _generate_sync(
            [req.prompt], req.max_new_tokens, req.temperature, req.top_k, req.n_loops
        )[0]
    except torch.cuda.OutOfMemoryError as e:  # pragma: no cover
        torch.cuda.empty_cache()
        raise HTTPException(503, f"GPU OOM: {e}") from e
    return InferResponse(text=text, n_loops=req.n_loops or 0)


@app.post("/v1/infer_batch", response_model=InferBatchResponse)
def infer_batch(req: InferBatchRequest) -> InferBatchResponse:
    _ensure_loaded()
    if not req.prompts:
        raise HTTPException(400, "prompts must be non-empty")
    texts = _generate_sync(
        req.prompts, req.max_new_tokens, req.temperature, req.top_k, req.n_loops
    )
    return InferBatchResponse(texts=texts)


async def _sse_stream(
    prompt: str, max_new_tokens: int, temperature: float, top_k: int | None, n_loops: int | None
) -> AsyncGenerator[bytes, None]:
    """Token-by-token Server-Sent Events stream."""
    model: ZAR1Model = _app_state["model"]
    tok = _app_state["tokenizer"]
    dev = _app_state["device"]

    input_ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    out = input_ids
    last_text_len = 0
    for _ in range(max_new_tokens):
        ctx = out[:, -model.config.max_seq_len :]
        with torch.no_grad():
            logits = model(ctx, inference_n_loops=n_loops).logits[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
        out = torch.cat([out, nxt], dim=-1)
        full = tok.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
        delta = full[last_text_len:]
        last_text_len = len(full)
        if delta:
            yield f"data: {delta}\n\n".encode()
        if tok.eos_token_id is not None and nxt.item() == tok.eos_token_id:
            break
        await asyncio.sleep(0)
    yield b"data: [DONE]\n\n"


@app.get("/v1/stream")
def stream(
    prompt: str = Query(...),
    max_new_tokens: int = Query(128, ge=1, le=4096),
    n_loops: int | None = Query(None, ge=1, le=64),
    temperature: float = Query(0.7, ge=0.0, le=2.0),
    top_k: int | None = Query(50, ge=1),
) -> StreamingResponse:
    _ensure_loaded()
    return StreamingResponse(
        _sse_stream(prompt, max_new_tokens, temperature, top_k, n_loops),
        media_type="text/event-stream",
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "zar1.server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
