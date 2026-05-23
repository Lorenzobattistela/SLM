from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import resolve_project_path


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    config: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "config": config,
        "rng_state": capture_rng_state(),
    }
    if extra_state:
        payload["extra_state"] = extra_state
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def rotate_checkpoints(save_dir: str | Path, *, keep_last_n: int) -> None:
    keep_last_n = int(keep_last_n)
    if keep_last_n <= 0:
        return
    target_dir = Path(save_dir)
    checkpoints = sorted(target_dir.glob("step_*.pt"))
    for old_path in checkpoints[:-keep_last_n]:
        old_path.unlink(missing_ok=True)


def find_checkpoint(config: dict[str, Any], explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = resolve_project_path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpointing_cfg = config["training"]["checkpointing"]
    resume_from = checkpointing_cfg.get("resume_from")
    if resume_from:
        path = resolve_project_path(resume_from)
        if not path.exists():
            raise FileNotFoundError(f"Configured checkpoint does not exist: {path}")
        return path

    save_dir = resolve_project_path(checkpointing_cfg["save_dir"])
    for name in ("latest.pt", "final.pt"):
        candidate = save_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found in {save_dir}. Train first or pass --checkpoint."
    )
