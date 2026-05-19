from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.rope import apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = build_rope_cache(seq_len, self.head_dim, self.config.rope_base, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:  # pragma: no cover - modern torch should expose SDPA.
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = attn @ v

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        return self.out_proj(attn)
