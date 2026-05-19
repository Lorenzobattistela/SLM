from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    bias: bool = False
    norm_eps: float = 1.0e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelConfig":
        config = cls(**payload)
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if config.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        return config
