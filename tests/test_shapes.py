from __future__ import annotations

import torch

from src.model.config import ModelConfig
from src.model.transformer import TransformerLM


def test_transformer_forward_shapes() -> None:
    config = ModelConfig(
        name="tiny-test",
        vocab_size=128,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=192,
    )
    model = TransformerLM(config)
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, config.context_length, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
