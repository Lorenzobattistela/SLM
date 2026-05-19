from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def load_run_config(path: str | Path) -> dict[str, Any]:
    run_path = resolve_project_path(path)
    run_cfg = load_yaml(run_path)
    merged = copy.deepcopy(run_cfg)
    model_path = resolve_project_path(run_cfg["model_config"])
    data_path = resolve_project_path(run_cfg["data_config"])
    merged["model"] = load_yaml(model_path)
    merged["data"] = load_yaml(data_path)
    merged["_meta"] = {
        "run_config_path": str(run_path.relative_to(PROJECT_ROOT)),
        "model_config_path": str(model_path.relative_to(PROJECT_ROOT)),
        "data_config_path": str(data_path.relative_to(PROJECT_ROOT)),
    }
    return merged
