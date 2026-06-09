from __future__ import annotations

import math
from typing import Any

import torch


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr = float(min_lr)
        self.step_index = 0
        self._last_lr = [self.lr_at_step(0) for _ in self.optimizer.param_groups]
        self.step(0)

    def lr_at_step(self, completed_steps: int) -> float:
        update_index = max(1, int(completed_steps) + 1)
        if self.warmup_steps > 0 and update_index <= self.warmup_steps:
            return self.base_lr * update_index / self.warmup_steps

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = (update_index - self.warmup_steps) / decay_steps
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self, completed_steps: int | None = None) -> None:
        if completed_steps is None:
            self.step_index += 1
        else:
            self.step_index = int(completed_steps)
        lr = self.lr_at_step(self.step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._last_lr = [lr for _ in self.optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_lr = float(state_dict["base_lr"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.total_steps = int(state_dict["total_steps"])
        self.min_lr = float(state_dict["min_lr"])
        self.step(int(state_dict.get("step_index", 0)))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    *,
    total_steps: int,
) -> WarmupCosineScheduler:
    scheduler_cfg = training_cfg["scheduler"]
    name = str(scheduler_cfg.get("name", "")).lower()
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name!r}. Expected 'cosine'.")
    return WarmupCosineScheduler(
        optimizer,
        base_lr=float(training_cfg["optimizer"]["learning_rate"]),
        warmup_steps=int(scheduler_cfg["warmup_steps"]),
        total_steps=total_steps,
        min_lr=float(scheduler_cfg["min_lr"]),
    )
