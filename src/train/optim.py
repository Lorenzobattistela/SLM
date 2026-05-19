from __future__ import annotations

import math

import torch


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "norm" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(train_cfg["weight_decay"])},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(train_cfg["lr"]),
        betas=tuple(float(value) for value in train_cfg["betas"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict):
    max_steps = int(train_cfg["max_steps"])
    warmup_steps = int(train_cfg["warmup_steps"])
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg["min_lr"])
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
