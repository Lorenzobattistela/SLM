from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.loader import resolve_project_path
from src.tokenizer.byte_bpe_tokenizer import load_byte_bpe_tokenizer
from src.tokenizer.superbpe_tokenizer import (
    load_superbpe_tokenizer,
    tokenizer_artifact_path,
)


def tokenizer_exists(tokenizer_cfg: dict[str, Any]) -> bool:
    if tokenizer_cfg.get("type") == "byte_bpe":
        return True
    return tokenizer_artifact_path(tokenizer_cfg).exists()


def require_tokenizer(tokenizer_cfg: dict[str, Any]):
    tokenizer_type = tokenizer_cfg.get("type")
    if tokenizer_type == "byte_bpe":
        return load_byte_bpe_tokenizer(tokenizer_cfg)
    if tokenizer_type == "superbpe":
        return load_superbpe_tokenizer(tokenizer_cfg)
    raise ValueError(f"Unsupported tokenizer.type={tokenizer_type!r}")


def load_tokenizer(tokenizer_cfg: dict[str, Any]):
    return require_tokenizer(tokenizer_cfg)


def tokenizer_output_dir(tokenizer_cfg: dict[str, Any]) -> Path:
    return resolve_project_path(tokenizer_cfg["save_dir"])
