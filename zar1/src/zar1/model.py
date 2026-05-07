"""Full ZAR-1 Recurrent-Depth MoE Transformer model.

Architecture:
    embedding -> [prelude_layers x StandardTransformerBlock]
              -> recurrent core (RecurrentTransformerBlock looped via ACT)
              -> [coda_layers x StandardTransformerBlock]
              -> RMSNorm -> tied LM head -> logits

The LM head shares weights with the input embedding (Llama-style tying).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from zar1.act import ACTHalting
from zar1.recurrent_block import (
    RecurrentTransformerBlock,
    RMSNorm,
    StandardTransformerBlock,
)


@dataclass
class ZAR1Config:
    """Configuration for the ZAR-1 model."""

    vocab_size: int = 128_256
    dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    prelude_layers: int = 4
    coda_layers: int = 4
    max_loops: int = 8
    num_experts: int = 256
    top_k: int = 8
    max_seq_len: int = 8192
    act_epsilon: float = 0.01
    act_ponder_tau: float = 0.01
    spectral_max: float = 0.99
    spectral_iters: int = 5
    lora_rank: int = 16
    use_lora: bool = True
    use_pid: bool = True
    pid_kp: float = 0.5
    pid_ki: float = 0.1
    pid_kd: float = 0.1
    router_hidden_dim: int | None = None
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ZAR1Config":
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("model", {})
        return cls(
            vocab_size=m.get("vocab_size", 128_256),
            dim=m.get("dim", 4096),
            num_heads=m.get("n_heads", 32),
            num_kv_heads=m.get("n_kv_heads", 8),
            prelude_layers=m.get("prelude_layers", 4),
            coda_layers=m.get("coda_layers", 4),
            max_loops=m.get("max_loops", 8),
            num_experts=m.get("num_experts", 256),
            top_k=m.get("top_k", 8),
            max_seq_len=m.get("max_seq_len", 8192),
            act_epsilon=m.get("act_epsilon", 0.01),
            act_ponder_tau=m.get("act_ponder_tau", 0.01),
            lora_rank=m.get("lora_rank", 16),
            use_lora=m.get("use_lora", True),
            use_pid=m.get("use_pid", True),
            pid_kp=m.get("pid_kp", 0.5),
            pid_ki=m.get("pid_ki", 0.1),
            pid_kd=m.get("pid_kd", 0.1),
            router_hidden_dim=m.get("router_hidden_dim", None),
            spectral_max=m.get("spectral_max", 0.99),
            spectral_iters=m.get("spectral_iters", 5),
            tie_word_embeddings=m.get("tie_word_embeddings", True),
            pad_token_id=m.get("pad_token_id", 0),
            extras=cfg,
        )


@dataclass
class ZAR1Output:
    """Container for forward-pass outputs."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    ponder_cost: torch.Tensor | None = None
    n_loops: int = 0


class ZAR1Model(nn.Module):
    """Recurrent-Depth MoE Transformer."""

    def __init__(self, config: ZAR1Config) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)

        self.prelude = nn.ModuleList(
            [
                StandardTransformerBlock(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.prelude_layers)
            ]
        )

        self.recurrent_block = RecurrentTransformerBlock(
            dim=config.dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_loops=config.max_loops,
            num_experts=config.num_experts,
            top_k=config.top_k,
            max_seq_len=config.max_seq_len,
            spectral_max=config.spectral_max,
            spectral_iters=config.spectral_iters,
            lora_rank=config.lora_rank,
            use_lora=config.use_lora,
            use_pid=config.use_pid,
            pid_kp=config.pid_kp,
            pid_ki=config.pid_ki,
            pid_kd=config.pid_kd,
            router_hidden_dim=config.router_hidden_dim,
        )

        self.act = ACTHalting(
            dim=config.dim,
            epsilon=config.act_epsilon,
            ponder_tau=config.act_ponder_tau,
            max_steps=config.max_loops,
        )

        self.coda = nn.ModuleList(
            [
                StandardTransformerBlock(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.coda_layers)
            ]
        )

        self.final_norm = RMSNorm(config.dim)

        if config.tie_word_embeddings:
            self.lm_head = None  # Use self.embed.weight directly.
        else:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return F.linear(hidden, self.embed.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_n_loops: int | None = None,
    ) -> ZAR1Output:
        """Run a forward pass.

        Args:
            input_ids: ``(B, T)`` token ids.
            attention_mask: Optional additive attention mask broadcastable to
                ``(B, num_heads, T, T)``. ``None`` => causal mask via SDPA.
            labels: Optional ``(B, T)`` labels for cross-entropy loss.
                ``-100`` is ignored.
            inference_n_loops: Override ``max_loops`` at inference time.

        Returns:
            :class:`ZAR1Output` with ``logits``, optional ``loss``, ``aux_loss``,
            ``ponder_cost``, and number of recurrent steps taken.
        """
        x = self.embed(input_ids)

        for block in self.prelude:
            x = block(x, attention_mask)

        # Recurrent core via ACT halting.
        def step_fn(h: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.recurrent_block(h, t, attention_mask)

        max_loops = inference_n_loops if inference_n_loops is not None else self.config.max_loops
        x, ponder_cost, n_loops, aux_loss_sum = self.act(x, step_fn, max_steps=max_loops)

        for block in self.coda:
            x = block(x, attention_mask)

        x = self.final_norm(x)
        logits = self._project_logits(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce + aux_loss_sum + ponder_cost

        return ZAR1Output(
            logits=logits,
            loss=loss,
            aux_loss=aux_loss_sum,
            ponder_cost=ponder_cost,
            n_loops=n_loops,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        n_loops: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Naive autoregressive sampling (no KV cache; suitable for demos)."""
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            ctx = out[:, -self.config.max_seq_len :]
            logits = self.forward(ctx, inference_n_loops=n_loops).logits[:, -1, :]
            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            out = torch.cat([out, next_tok], dim=-1)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return out

    def apply_spectral_constraints(self) -> torch.Tensor:
        """Apply the spectral-radius constraint on the recurrent injection."""
        return self.recurrent_block.apply_spectral_constraint()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
