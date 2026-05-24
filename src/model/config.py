from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple_of must be positive")
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    name: str = "decoder_only_transformer"
    attention: str = "gqa"
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    bias: bool = False
    norm_eps: float = 1.0e-5
    flash_attention: bool = True
    flash_attention_fallback: bool = True
    target_parameters: int | None = None
    acceptable_min_parameters: int | None = None
    acceptable_max_parameters: int | None = None
    ffn_multiplier: float | None = None
    multiple_of: int | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def max_seq_len(self) -> int:
        return self.context_length

    @property
    def num_attention_heads(self) -> int:
        return self.n_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.n_kv_heads

    @property
    def num_kv_heads(self) -> int:
        return self.n_kv_heads

    @property
    def use_flash_attention(self) -> bool:
        return self.flash_attention

    @property
    def rope_theta(self) -> float:
        return self.rope_base

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        context_length = payload.get("context_length", payload.get("max_seq_len"))
        n_heads = payload.get("num_attention_heads", payload.get("n_heads"))
        n_kv_heads = payload.get(
            "num_key_value_heads",
            payload.get("num_kv_heads", payload.get("n_kv_heads")),
        )
        rope_base = payload.get("rope_base", payload.get("rope_theta", 10000.0))
        name = payload.get("name", payload.get("architecture", "decoder_only_transformer"))
        attention = str(payload.get("attention", "gqa"))
        flash_attention = payload.get("flash_attention", payload.get("use_flash_attention", True))

        ffn_multiplier = payload.get("ffn_multiplier")
        multiple_of = payload.get("multiple_of")
        ffn_dim = payload.get("ffn_dim")
        if ffn_dim is None:
            if ffn_multiplier is None:
                raise ValueError("model.ffn_multiplier is required when ffn_dim is not set")
            if multiple_of is None:
                raise ValueError("model.multiple_of is required when ffn_dim is not set")
            ffn_dim = _round_up_to_multiple(
                int(payload["d_model"] * float(ffn_multiplier)),
                int(multiple_of),
            )

        config = cls(
            vocab_size=int(payload["vocab_size"]),
            context_length=int(context_length),
            d_model=int(payload["d_model"]),
            n_layers=int(payload["n_layers"]),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads),
            ffn_dim=int(ffn_dim),
            name=str(name),
            attention=attention,
            dropout=float(payload.get("dropout", 0.0)),
            rope_base=float(rope_base),
            tie_embeddings=bool(payload.get("tie_embeddings", True)),
            bias=bool(payload.get("bias", False)),
            norm_eps=float(payload.get("norm_eps", 1.0e-5)),
            flash_attention=bool(flash_attention),
            flash_attention_fallback=bool(payload.get("flash_attention_fallback", True)),
            target_parameters=payload.get("target_parameters"),
            acceptable_min_parameters=payload.get("acceptable_min_parameters"),
            acceptable_max_parameters=payload.get("acceptable_max_parameters"),
            ffn_multiplier=float(ffn_multiplier) if ffn_multiplier is not None else None,
            multiple_of=int(multiple_of) if multiple_of is not None else None,
        )
        if config.attention != "gqa":
            raise ValueError("attention must be 'gqa'")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by num_attention_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if config.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if config.flash_attention and config.head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by 8 when flash_attention=true")
        return config
