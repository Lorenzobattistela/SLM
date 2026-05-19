from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedState:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed(requested: bool) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    should_init = requested or world_size > 1
    if not should_init:
        return DistributedState(enabled=False)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")
    if dist.is_initialized():
        return DistributedState(
            enabled=True,
            rank=dist.get_rank(),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=dist.get_world_size(),
        )

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistributedState(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def barrier(state: DistributedState) -> None:
    if state.enabled:
        dist.barrier()


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.destroy_process_group()
