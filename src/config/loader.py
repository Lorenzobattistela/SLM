from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import ensure_output_directories, validate_run_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_legacy_run_config(run_path: Path, run_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(run_cfg)
    model_path = resolve_project_path(run_cfg["model_config"])
    data_path = resolve_project_path(run_cfg["data_config"])
    merged["model"] = load_yaml(model_path)
    merged["data"] = load_yaml(data_path)
    merged["_meta"] = {
        "run_config_path": _relative_path(run_path),
        "model_config_path": _relative_path(model_path),
        "data_config_path": _relative_path(data_path),
    }
    return merged


def load_run_config(path: str | Path, *, create_dirs: bool = True) -> dict[str, Any]:
    run_path = resolve_project_path(path)
    run_cfg = load_yaml(run_path)

    if "model_config" in run_cfg and "data_config" in run_cfg:
        return _load_legacy_run_config(run_path, run_cfg)

    config = copy.deepcopy(run_cfg)
    validate_run_config(config)
    if create_dirs:
        ensure_output_directories(config)
    config["_meta"] = {"run_config_path": _relative_path(run_path)}
    return config
