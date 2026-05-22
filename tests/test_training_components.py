from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from src.training.scheduler import WarmupCosineScheduler

RUN_ALL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
RUN_ALL_SPEC = importlib.util.spec_from_file_location("run_all", RUN_ALL_PATH)
assert RUN_ALL_SPEC is not None
run_all = importlib.util.module_from_spec(RUN_ALL_SPEC)
assert RUN_ALL_SPEC.loader is not None
RUN_ALL_SPEC.loader.exec_module(run_all)
should_launch_ddp = run_all.should_launch_ddp


def test_warmup_cosine_scheduler_respects_warmup_and_min_lr() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=1.0,
        warmup_steps=2,
        total_steps=6,
        min_lr=0.1,
    )

    assert scheduler.get_last_lr() == [0.5]
    scheduler.step(1)
    assert scheduler.get_last_lr() == [1.0]
    scheduler.step(6)
    assert scheduler.get_last_lr() == [0.1]


def test_run_all_guides_when_nccl_needs_more_gpus_than_visible() -> None:
    ddp_cfg = {"backend": "nccl", "num_gpus": 2}

    assert not should_launch_ddp(
        ddp_cfg,
        cuda_available=True,
        cuda_device_count=1,
    )
    assert should_launch_ddp(
        ddp_cfg,
        cuda_available=True,
        cuda_device_count=2,
    )


def test_run_all_allows_non_nccl_debug_ddp_without_cuda() -> None:
    ddp_cfg = {"backend": "gloo", "num_gpus": 2}

    assert should_launch_ddp(
        ddp_cfg,
        cuda_available=False,
        cuda_device_count=0,
    )
