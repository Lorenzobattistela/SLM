from __future__ import annotations

import logging
import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedState:
    requested: bool
    enabled: bool
    backend: str
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed(training_cfg: dict[str, Any]) -> DistributedState:
    dist_cfg = training_cfg.get("distributed", {})
    requested = bool(dist_cfg.get("enabled", False))
    configured_backend = str(dist_cfg.get("backend", "nccl"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not requested and world_size <= 1:
        return DistributedState(requested=False, enabled=False, backend=configured_backend)

    if world_size <= 1:
        LOGGER.warning(
            "DDP is enabled in the YAML config, but WORLD_SIZE=1. "
            "Run with torchrun for distributed training; continuing in single-process mode."
        )
        return DistributedState(requested=requested, enabled=False, backend=configured_backend)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    if configured_backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError(
            "training.distributed.backend is 'nccl', but CUDA is not available. "
            "Run on a CUDA machine or use a CPU debug config with backend 'gloo'."
        )

    expected_world_size = int(dist_cfg.get("num_gpus", world_size))
    if configured_backend == "nccl" and torch.cuda.device_count() < expected_world_size:
        raise RuntimeError(
            "NCCL DDP was requested with "
            f"num_gpus={expected_world_size}, but only {torch.cuda.device_count()} CUDA "
            "device(s) are visible. Run with two visible GPUs or adjust the debug config."
        )
    if expected_world_size != world_size:
        LOGGER.warning(
            "Configured num_gpus=%s but torchrun WORLD_SIZE=%s.",
            expected_world_size,
            world_size,
        )

    if configured_backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=configured_backend)

    return DistributedState(
        requested=requested,
        enabled=True,
        backend=configured_backend,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def select_training_device(state: DistributedState) -> torch.device:
    if state.enabled and state.backend != "nccl":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", state.local_rank if state.enabled else 0)
    return torch.device("cpu")


def barrier(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        if state.backend == "nccl":
            dist.barrier(device_ids=[state.local_rank])
        else:
            dist.barrier()


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _read_status(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _wait_for_main_status(
    *,
    status_path: Path,
    ready_paths: Sequence[Path],
    description: str,
    poll_seconds: float,
) -> None:
    while True:
        payload = _read_status(status_path)
        if payload is not None:
            state = payload.get("state")
            if state == "failed":
                error_type = payload.get("error_type", "Exception")
                message = payload.get("message", "")
                raise RuntimeError(
                    f"{description} failed on the main process: {error_type}: {message}"
                )
            if state == "ready" and all(path.exists() for path in ready_paths):
                return
        time.sleep(poll_seconds)


def run_on_main_process_first(
    state: DistributedState,
    *,
    action: Callable[[], Any],
    status_path: Path,
    ready_paths: Sequence[Path],
    description: str,
    poll_seconds: float = 5.0,
) -> Any | None:
    """Run long file-producing work on rank 0 while other ranks poll the filesystem."""
    if not state.enabled:
        if state.is_main_process:
            return action()
        return None

    status_path = Path(status_path)
    ready_paths = tuple(Path(path) for path in ready_paths)

    barrier(state)
    if state.is_main_process:
        _write_status(status_path, {"state": "started", "description": description})
    barrier(state)

    result: Any | None = None
    if state.is_main_process:
        try:
            result = action()
            missing_paths = [str(path) for path in ready_paths if not path.exists()]
            if missing_paths:
                raise RuntimeError(
                    f"{description} did not produce expected file(s): "
                    + ", ".join(missing_paths)
                )
        except Exception as exc:
            _write_status(
                status_path,
                {
                    "state": "failed",
                    "description": description,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        _write_status(status_path, {"state": "ready", "description": description})
    else:
        _wait_for_main_status(
            status_path=status_path,
            ready_paths=ready_paths,
            description=description,
            poll_seconds=poll_seconds,
        )

    barrier(state)
    return result


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.destroy_process_group()
