from __future__ import annotations

from pathlib import Path

from src.config.schema import validate_run_config
from src.tokenizer.superbpe_tokenizer import ensure_pretrained_superbpe_artifacts


def _base_config(tmp_path: Path) -> dict:
    return {
        "project": {
            "name": "test",
            "seed": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
        "dataset": {
            "name": "dataset",
            "split": "train",
            "text_column": "text",
            "streaming": True,
            "target_train_tokens": 10,
            "validation_tokens": 2,
            "cache_dir": str(tmp_path / "cache"),
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "tokenize_num_workers": 1,
        },
        "tokenizer": {
            "type": "superbpe",
            "vocab_size": 200005,
            "pretrained": {
                "name": "superbpe-200k-t180k",
                "base_url": "https://example.test/tokenizer",
                "files": ["tokenizer.json"],
            },
            "special_tokens": {
                "pad_token": "<|padding|>",
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            },
            "save_dir": str(tmp_path / "tokenizer"),
            "append_eos": True,
        },
        "model": {
            "architecture": "decoder_only_transformer",
            "target_parameters": 1,
            "acceptable_min_parameters": 1,
            "acceptable_max_parameters": 2,
            "positional_encoding": "rope",
            "attention": "mqa",
            "use_flash_attention": True,
            "flash_attention_fallback": True,
            "activation": "swiglu",
            "normalization": "rmsnorm",
            "vocab_size": 200005,
            "max_seq_len": 8,
            "n_layers": 1,
            "d_model": 8,
            "n_heads": 1,
            "num_kv_heads": 1,
            "ffn_multiplier": 2,
            "multiple_of": 8,
            "norm_eps": 1.0e-5,
            "rope_theta": 10000.0,
            "dropout": 0.0,
            "tie_embeddings": True,
        },
        "training": {
            "distributed": {
                "enabled": False,
                "backend": "gloo",
                "strategy": "ddp",
                "num_gpus": 1,
            },
            "optimizer": {"name": "adamw"},
            "scheduler": {"name": "cosine"},
            "checkpointing": {"save_dir": str(tmp_path / "checkpoints")},
        },
        "evaluation": {},
        "logging": {},
        "plots": {"output_dir": str(tmp_path / "plots")},
    }


def test_pretrained_tokenizer_config_does_not_require_training_fields(tmp_path) -> None:
    config = _base_config(tmp_path)

    validate_run_config(config)


def test_pretrained_tokenizer_artifacts_are_downloaded_once(monkeypatch, tmp_path) -> None:
    tokenizer_cfg = _base_config(tmp_path)["tokenizer"]
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, url: str) -> None:
            self.url = url
            self._sent = False

        def read(self, size: int = -1) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return b"{}"

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(url: str, timeout: int):
        calls.append(f"{url} timeout={timeout}")
        return FakeResponse(url)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    ensure_pretrained_superbpe_artifacts(tokenizer_cfg)
    ensure_pretrained_superbpe_artifacts(tokenizer_cfg)

    assert calls == ["https://example.test/tokenizer/tokenizer.json timeout=120"]
    assert (Path(tokenizer_cfg["save_dir"]) / "tokenizer.json").exists()
    assert (Path(tokenizer_cfg["save_dir"]) / "tokenizer_metadata.json").exists()
