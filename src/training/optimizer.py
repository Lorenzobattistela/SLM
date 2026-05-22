from __future__ import annotations

from typing import Any

import torch


def build_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    name = str(optimizer_cfg.get("name", "")).lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name!r}. Expected 'adamw'.")

    decay_params = []
    no_decay_params = []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "norm" not in param_name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(optimizer_cfg["weight_decay"])},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
    )
