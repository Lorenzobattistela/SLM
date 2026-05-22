from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.loader import resolve_project_path
from src.tokenizer.superbpe_tokenizer import (
    load_superbpe_tokenizer,
    tokenizer_artifact_path,
)


def tokenizer_exists(tokenizer_cfg: dict[str, Any]) -> bool:
    return tokenizer_artifact_path(tokenizer_cfg).exists()


def require_tokenizer(tokenizer_cfg: dict[str, Any]):
    return load_superbpe_tokenizer(tokenizer_cfg)


def tokenizer_output_dir(tokenizer_cfg: dict[str, Any]) -> Path:
    return resolve_project_path(tokenizer_cfg["save_dir"])
