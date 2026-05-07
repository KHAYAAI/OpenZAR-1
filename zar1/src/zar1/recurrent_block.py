"""Recurrent transformer block with GQA, RoPE, MoE, loop-index embeddings,
LoRA depth adapters, and a power-iteration spectral-radius constraint on the
recurrent injection.

The block is intended to be applied repeatedly with weight tying. Each loop
gets a unique loop-index embedding plus its own LoRA adapter so the same weights
can specialize their behavior across recurrence depth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zar1.moe import MoEFeedForward


class LoRAAdapter(nn.Module):
    """Low-rank adapter (LoRA) for per-loop expressiveness.

    Implements y = x + (B @ A @ x) * (alpha / rank), where A and B are
    low-rank decomposition matrices. ``A`` is initialized to a small random
    value, ``B`` to zero, so the adapter starts as identity.

    Args:
        dim: Input/output dimension.
        rank: LoRA rank (typically 8-32). Master Plan v2.0 specifies rank 16.
        alpha: Scaling factor (typically equal to rank).
        dropout: Optional dropout on the LoRA path.
    """

    def __init__(self, dim: int, rank: int = 16, alpha: float | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / max(rank, 1)

        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Standard LoRA init: A ~ Kaiming, B = 0 so adapter starts at zero.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA delta: ``B(A(x)) * scaling``.

        The caller is responsible for adding this to the base output.
        """
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (LLaMA-style)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Pre-compute cos/sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(B, H, T, D)`` using ``cos/sin (T, D)``."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotated * sin)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with RoPE.

    ``num_kv_heads`` must divide ``num_heads``. Q heads in the same group share
    a single K/V head.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 8192,
        rope_base: float = 500_000.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.group_size = num_heads // num_kv_heads
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = _build_rope_cache(seq, self.head_dim, self.rope_base, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Repeat KV heads to match Q heads (GQA expansion).
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        # Use SDPA: enables FlashAttention/MemEfficient kernels on H100.
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.o_proj(attn_out)


class DenseFFN(nn.Module):
    """Standard SwiGLU dense FFN (for prelude/coda blocks)."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8 / 3)
            hidden_dim = (hidden_dim + 255) // 256 * 256
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class StandardTransformerBlock(nn.Module):
    """Pre-norm transformer block (GQA + dense FFN) for prelude/coda."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 8192,
        ffn_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = DenseFFN(dim, ffn_hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


@torch.no_grad()
def _spectral_norm_power_iter(
    weight: torch.Tensor, u: torch.Tensor, n_iters: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the largest singular value of a 2D matrix via power iteration.

    Returns ``(sigma, new_u)``. ``u`` is a left singular vector estimate that is
    updated in-place style (returned for caching across calls).
    """
    w = weight
    if w.dim() != 2:
        w = w.reshape(w.shape[0], -1)
    for _ in range(n_iters):
        v = F.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
    sigma = torch.dot(u, torch.mv(w, v))
    return sigma, u


class RecurrentTransformerBlock(nn.Module):
    """Weight-tied recurrent block with loop-index embeddings.

    The same parameters are applied across ``max_loops`` iterations. Each loop:
      1. Add the loop-index embedding to the residual stream.
      2. Pre-norm GQA attention + residual.
      3. Pre-norm MoE FFN + residual.

    A spectral-radius constraint is applied to the *recurrent injection matrix*
    (``recurrent_proj``) after each loop to keep the looped operator's largest
    singular value <= ``spectral_max`` (default 0.99) for training stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_loops: int = 8,
        num_experts: int = 256,
        top_k: int = 8,
        max_seq_len: int = 8192,
        spectral_max: float = 0.99,
        spectral_iters: int = 5,
        lora_rank: int = 16,
        use_lora: bool = True,
        use_pid: bool = True,
        pid_kp: float = 0.5,
        pid_ki: float = 0.1,
        pid_kd: float = 0.1,
        router_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_loops = max_loops
        self.spectral_max = spectral_max
        self.spectral_iters = spectral_iters
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.moe = MoEFeedForward(
            dim,
            num_experts=num_experts,
            top_k=top_k,
            use_pid=use_pid,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            router_hidden_dim=router_hidden_dim,
        )

        # Loop-index embedding table; one vector per recurrence step.
        self.loop_embed = nn.Embedding(max_loops, dim)
        nn.init.normal_(self.loop_embed.weight, mean=0.0, std=0.02)

        # Recurrent injection: blends previous-loop hidden with current input.
        self.recurrent_proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.recurrent_proj.weight, mean=0.0, std=1.0 / math.sqrt(dim))

        # Per-loop LoRA depth adapters (Master Plan v2.0 §2.1).
        # One LoRA pair per loop, applied to attention output and FFN output.
        # Adds (max_loops * 2 * 2 * rank * dim) params: ~2M for 7B at rank 16.
        if use_lora:
            self.attn_loras = nn.ModuleList(
                [LoRAAdapter(dim, rank=lora_rank) for _ in range(max_loops)]
            )
            self.ffn_loras = nn.ModuleList(
                [LoRAAdapter(dim, rank=lora_rank) for _ in range(max_loops)]
            )
        else:
            self.attn_loras = None
            self.ffn_loras = None

        # Cached left singular vector for power iteration (not a parameter).
        self.register_buffer(
            "_u", F.normalize(torch.randn(dim), dim=0), persistent=False
        )

    @torch.no_grad()
    def apply_spectral_constraint(self) -> torch.Tensor:
        """Rescale ``recurrent_proj`` so its spectral norm is <= ``spectral_max``."""
        w = self.recurrent_proj.weight.data
        u = self._u.to(dtype=w.dtype, device=w.device)
        sigma, new_u = _spectral_norm_power_iter(w, u, self.spectral_iters)
        self._u.copy_(new_u.to(self._u.dtype))
        sigma_val = sigma.detach().abs()
        if sigma_val > self.spectral_max:
            w.mul_(self.spectral_max / (sigma_val + 1e-12))
        return sigma_val

    def forward(
        self,
        x: torch.Tensor,
        loop_index: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single recurrent iteration.

        Args:
            x: Hidden states from the previous loop, shape ``(B, T, dim)``.
            loop_index: 0-based recurrence index in ``[0, max_loops)``.
            attention_mask: Optional attention mask for the GQA layer.

        Returns:
            output: Updated hidden states, shape ``(B, T, dim)``.
            aux_loss: MoE load-balancing loss for this loop.
        """
        idx = max(0, min(loop_index, self.max_loops - 1))
        loop_emb = self.loop_embed.weight[idx]  # (dim,)

        # Recurrent injection + loop-index conditioning.
        h = self.recurrent_proj(x) + loop_emb.view(1, 1, -1)

        # Attention sublayer with per-loop LoRA adapter.
        attn_in = self.norm1(h)
        attn_out = self.attn(attn_in, attention_mask)
        if self.use_lora:
            attn_out = attn_out + self.attn_loras[idx](attn_in)
        h = h + attn_out

        # MoE FFN sublayer with per-loop LoRA adapter.
        ffn_in = self.norm2(h)
        ffn_out, _routing, aux_loss = self.moe(ffn_in)
        if self.use_lora:
            ffn_out = ffn_out + self.ffn_loras[idx](ffn_in)
        h = h + ffn_out
        return h, aux_loss
