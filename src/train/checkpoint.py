from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    tokens_seen: int,
    config: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if extra_state:
        payload["extra_state"] = extra_state
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
