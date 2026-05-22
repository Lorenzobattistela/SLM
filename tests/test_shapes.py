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


def test_model_config_reads_task_yaml_shape() -> None:
    config = ModelConfig.from_dict(
        {
            "architecture": "decoder_only_transformer",
            "vocab_size": 128,
            "max_seq_len": 16,
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 4,
            "num_kv_heads": 1,
            "ffn_multiplier": 4,
            "multiple_of": 64,
            "norm_eps": 1.0e-5,
            "rope_theta": 10000.0,
            "dropout": 0.0,
            "tie_embeddings": True,
            "use_flash_attention": False,
            "flash_attention_fallback": True,
        }
    )

    assert config.context_length == 16
    assert config.n_kv_heads == 1
    assert config.ffn_dim == 256
    assert config.rope_base == 10000.0

    model = TransformerLM(config)
    assert model.lm_head.weight is model.tok_embeddings.weight
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    logits, _ = model(inputs)
    assert logits.shape == (2, config.context_length, config.vocab_size)
