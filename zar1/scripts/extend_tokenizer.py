"""Extend Llama-3 tokenizer with African language subwords.

This script loads the Llama-3 tokenizer and extends it with additional tokens
for African languages (Zulu, Xhosa, Swahili, etc.).

Usage:
    python scripts/extend_tokenizer.py \
        --base_tokenizer meta-llama/Meta-Llama-3-8B \
        --african_vocab data/african_vocab.txt \
        --output_dir tokenizers/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_base_tokenizer(model_name: str):
    """Load a HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


def load_african_vocab(vocab_path: str) -> list[str]:
    """Load African language vocabulary from a file (one token per line)."""
    path = Path(vocab_path)
    if not path.exists():
        print(f"[WARNING] Vocabulary file not found: {vocab_path}")
        print("[INFO] Generating synthetic African language tokens for demo...")
        # Generate synthetic Bantu morphemes as placeholder
        african_tokens = [
            "uku",
            "aba",
            "umu",
            "ama",
            "ubu",
            "aka",
            "isi",
            "inxi",
            "impi",
            "indoda",
        ]
        return african_tokens

    with open(path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens


def extend_tokenizer(tokenizer, new_tokens: list[str], num_new: int = 20000) -> tuple:
    """Add new tokens to tokenizer.

    Args:
        tokenizer: HF tokenizer instance
        new_tokens: List of tokens to add
        num_new: Target number of new tokens (pad with <AFR_X> placeholders if needed)

    Returns:
        (updated_tokenizer, num_added)
    """
    # Trim or pad the token list
    if len(new_tokens) < num_new:
        remaining = num_new - len(new_tokens)
        new_tokens = new_tokens + [f"<AFR_{i}>" for i in range(remaining)]
    else:
        new_tokens = new_tokens[:num_new]

    num_added = tokenizer.add_tokens(new_tokens)
    print(f"[INFO] Added {num_added} new tokens")
    print(f"[INFO] New vocabulary size: {len(tokenizer)}")
    return tokenizer, num_added


def main():
    parser = argparse.ArgumentParser(
        description="Extend tokenizer with African language subwords"
    )
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Base tokenizer model name",
    )
    parser.add_argument(
        "--african_vocab",
        type=str,
        default="data/african_vocab.txt",
        help="Path to African vocabulary file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizers/",
        help="Output directory for extended tokenizer",
    )
    parser.add_argument(
        "--num_new",
        type=int,
        default=20000,
        help="Target number of new tokens to add",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading base tokenizer: {args.base_tokenizer}")
    tokenizer = load_base_tokenizer(args.base_tokenizer)
    if tokenizer is None:
        return

    print(f"[INFO] Base vocabulary size: {len(tokenizer)}")

    print(f"[INFO] Loading African vocabulary: {args.african_vocab}")
    african_vocab = load_african_vocab(args.african_vocab)
    print(f"[INFO] Loaded {len(african_vocab)} African tokens")

    print(f"[INFO] Extending tokenizer with {args.num_new} tokens...")
    tokenizer, num_added = extend_tokenizer(tokenizer, african_vocab, args.num_new)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "zar1_tokenizer"
    print(f"[INFO] Saving extended tokenizer to {output_path}")
    tokenizer.save_pretrained(str(output_path))

    print(f"\n[SUCCESS] Tokenizer extended and saved!")
    print(f"  Base vocab size: {len(tokenizer) - num_added}")
    print(f"  New tokens added: {num_added}")
    print(f"  Total vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    main()
