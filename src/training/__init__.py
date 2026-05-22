from __future__ import annotations

__all__ = [
    "find_checkpoint",
    "load_checkpoint",
    "run_training",
    "save_checkpoint",
]


def __getattr__(name: str):
    if name == "run_training":
        from src.training.trainer import run_training

        return run_training
    if name in {"find_checkpoint", "load_checkpoint", "save_checkpoint"}:
        from src.training import checkpointing

        return getattr(checkpointing, name)
    raise AttributeError(name)
