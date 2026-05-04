"""Dataset and data loading for ZAR-1.

Combines streamed FineWeb-Edu (English) with locally-stored African-language
JSONL corpora. Supports on-the-fly sequence packing with cross-document
attention masking via document-id tracking.
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class PackedSample:
    """A single packed training sample."""

    input_ids: torch.Tensor   # (T,)
    labels: torch.Tensor      # (T,)
    doc_ids: torch.Tensor     # (T,) integer doc id, used for cross-doc masking


def _load_tokenizer(name_or_path: str):
    """Load a Hugging Face tokenizer; raises a helpful error if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError("transformers is required for the data loader") from e
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    return tok


def _iter_jsonl_files(pattern: str) -> Iterator[str]:
    """Yield text fields from JSONL files matching a glob pattern."""
    paths = sorted(glob.glob(pattern))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("content")
                if text:
                    yield text


def _iter_hf_stream(name: str, split: str) -> Iterator[str]:
    """Yield text fields from a streamed Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("datasets is required for the data loader") from e
    ds = load_dataset(name, split=split, streaming=True)
    for ex in ds:
        text = ex.get("text") or ex.get("content")
        if text:
            yield text


class PackedTextDataset(IterableDataset):
    """Iterable dataset that streams, tokenizes and packs text into fixed-length blocks.

    Args:
        tokenizer_name: HF tokenizer identifier (e.g. ``meta-llama/Meta-Llama-3-8B``).
        seq_len: Output sequence length per packed sample.
        hf_dataset: Optional HF dataset name to stream.
        hf_split: HF split to use.
        african_glob: Glob for local JSONL files of African-language data.
        mix_african_prob: Probability of pulling the next document from the
            African-language pool (rest from HF). Set to 0 to disable.
        validation_fraction: Reserve a deterministic fraction for validation.
        is_validation: Whether this instance is the validation split.
        seed: Shuffling/sampling seed.
    """

    def __init__(
        self,
        tokenizer_name: str,
        seq_len: int = 8192,
        hf_dataset: str | None = "HuggingFaceFW/fineweb-edu",
        hf_split: str = "train",
        african_glob: str | None = "data/african/*.jsonl",
        mix_african_prob: float = 0.15,
        validation_fraction: float = 0.01,
        is_validation: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self.african_glob = african_glob
        self.mix_african_prob = mix_african_prob
        self.validation_fraction = validation_fraction
        self.is_validation = is_validation
        self.seed = seed

        self._tokenizer = None
        self._eos_id = None

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.tokenizer_name)
            self._eos_id = self._tokenizer.eos_token_id or 0

    def _docs(self) -> Iterator[str]:
        """Yield raw document texts, mixing HF and local African JSONLs."""
        info = get_worker_info()
        worker_id = info.id if info else 0
        rng = random.Random(self.seed + worker_id + (1 if self.is_validation else 0))

        hf_iter = (
            _iter_hf_stream(self.hf_dataset, self.hf_split) if self.hf_dataset else iter(())
        )
        african_iter = (
            _iter_jsonl_files(self.african_glob) if self.african_glob else iter(())
        )

        while True:
            doc_idx = next(self._counter)
            # Deterministic train/val split based on doc index.
            is_val_doc = (doc_idx % max(1, int(1 / self.validation_fraction))) == 0
            if is_val_doc != self.is_validation:
                continue

            use_african = african_iter is not None and rng.random() < self.mix_african_prob
            try:
                if use_african:
                    yield next(african_iter)
                else:
                    yield next(hf_iter)
            except StopIteration:
                # Try the other source as a fallback before terminating.
                try:
                    yield next(african_iter if not use_african else hf_iter)
                except StopIteration:
                    return

    def __iter__(self) -> Iterator[PackedSample]:
        self._ensure_tokenizer()
        self._counter = iter(range(10**12))

        buf_ids: list[int] = []
        buf_docs: list[int] = []
        doc_id_counter = 0

        for text in self._docs():
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            ids.append(self._eos_id)
            doc_id_counter += 1

            buf_ids.extend(ids)
            buf_docs.extend([doc_id_counter] * len(ids))

            while len(buf_ids) >= self.seq_len + 1:
                chunk_ids = buf_ids[: self.seq_len + 1]
                chunk_docs = buf_docs[: self.seq_len + 1]
                buf_ids = buf_ids[self.seq_len :]
                buf_docs = buf_docs[self.seq_len :]

                input_ids = torch.tensor(chunk_ids[:-1], dtype=torch.long)
                labels = torch.tensor(chunk_ids[1:], dtype=torch.long)
                # Mask cross-document targets so loss respects boundaries.
                doc_t = torch.tensor(chunk_docs, dtype=torch.long)
                same_doc = doc_t[1:] == doc_t[:-1]
                labels[~same_doc] = -100
                yield PackedSample(
                    input_ids=input_ids,
                    labels=labels,
                    doc_ids=doc_t[:-1],
                )


def collate(batch: list[PackedSample]) -> dict[str, torch.Tensor]:
    """Stack PackedSamples into a batch dict."""
    return {
        "input_ids": torch.stack([b.input_ids for b in batch], dim=0),
        "labels": torch.stack([b.labels for b in batch], dim=0),
        "doc_ids": torch.stack([b.doc_ids for b in batch], dim=0),
    }


def build_attention_mask_from_doc_ids(
    doc_ids: torch.Tensor, num_heads: int = 1
) -> torch.Tensor:
    """Build an additive attention mask that prevents cross-document attention.

    Args:
        doc_ids: ``(B, T)`` integer document ids.
        num_heads: Number of attention heads (mask broadcast across heads).

    Returns:
        ``(B, num_heads, T, T)`` additive mask of 0s (allowed) and -inf (blocked),
        with a causal lower-triangular constraint also applied.
    """
    bsz, seq = doc_ids.shape
    same = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)  # (B, T, T)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=doc_ids.device))
    allowed = same & causal
    mask = torch.zeros(bsz, 1, seq, seq, dtype=torch.float32, device=doc_ids.device)
    mask = mask.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    if num_heads > 1:
        mask = mask.expand(bsz, num_heads, seq, seq)
    return mask


def build_train_val_loaders(
    cfg: dict, seq_len: int, micro_batch_size: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Helper to construct train/val DataLoaders from a config dict."""
    common = dict(
        tokenizer_name=cfg.get("tokenizer", "meta-llama/Meta-Llama-3-8B"),
        seq_len=seq_len,
        hf_dataset=cfg.get("hf_dataset"),
        hf_split=cfg.get("hf_split", "train"),
        african_glob=cfg.get("african_glob"),
        validation_fraction=cfg.get("validation_fraction", 0.01),
    )
    train_ds = PackedTextDataset(is_validation=False, **common)
    val_ds = PackedTextDataset(is_validation=True, **common)

    num_workers = int(os.environ.get("ZAR1_NUM_WORKERS", "2"))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=micro_batch_size,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate,
        pin_memory=True,
    )
    return train_loader, val_loader
