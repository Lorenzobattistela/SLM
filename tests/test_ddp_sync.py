from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training import ddp
from src.training.ddp import DistributedState


def _distributed_state(*, rank: int = 0, backend: str = "gloo") -> DistributedState:
    return DistributedState(
        requested=True,
        enabled=True,
        backend=backend,
        rank=rank,
        local_rank=rank,
        world_size=2,
    )


def test_barrier_passes_local_device_id_for_nccl(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(ddp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp.dist, "barrier", lambda **kwargs: calls.append(kwargs))

    ddp.barrier(_distributed_state(rank=1, backend="nccl"))

    assert calls == [{"device_ids": [1]}]


def test_run_on_main_process_first_marks_ready_after_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ddp, "barrier", lambda state: None)
    status_path = tmp_path / "package.status.json"
    output_path = tmp_path / "package.bin"

    def action() -> None:
        output_path.write_bytes(b"tokens")

    ddp.run_on_main_process_first(
        _distributed_state(rank=0),
        action=action,
        status_path=status_path,
        ready_paths=[output_path],
        description="Prepare package",
    )

    assert json.loads(status_path.read_text(encoding="utf-8"))["state"] == "ready"


def test_run_on_main_process_first_records_main_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ddp, "barrier", lambda state: None)
    status_path = tmp_path / "package.status.json"

    def action() -> None:
        raise ValueError("download timed out")

    with pytest.raises(ValueError, match="download timed out"):
        ddp.run_on_main_process_first(
            _distributed_state(rank=0),
            action=action,
            status_path=status_path,
            ready_paths=[tmp_path / "package.bin"],
            description="Prepare package",
        )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["state"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert payload["message"] == "download timed out"


def test_run_on_main_process_first_non_main_raises_status_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "package.status.json"
    action_called = False
    barrier_calls = 0

    def fake_barrier(state: DistributedState) -> None:
        nonlocal barrier_calls
        barrier_calls += 1
        if barrier_calls == 2:
            status_path.write_text(
                json.dumps(
                    {
                        "state": "failed",
                        "error_type": "RuntimeError",
                        "message": "HTTP 408",
                    }
                ),
                encoding="utf-8",
            )

    def action() -> None:
        nonlocal action_called
        action_called = True

    monkeypatch.setattr(ddp, "barrier", fake_barrier)

    with pytest.raises(RuntimeError, match="HTTP 408"):
        ddp.run_on_main_process_first(
            _distributed_state(rank=1),
            action=action,
            status_path=status_path,
            ready_paths=[tmp_path / "package.bin"],
            description="Prepare package",
            poll_seconds=0.0,
        )

    assert not action_called


def test_run_on_main_process_first_non_main_waits_for_ready_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "package.status.json"
    output_path = tmp_path / "package.bin"
    barrier_calls = 0

    def fake_barrier(state: DistributedState) -> None:
        nonlocal barrier_calls
        barrier_calls += 1
        if barrier_calls == 2:
            output_path.write_bytes(b"tokens")
            status_path.write_text(json.dumps({"state": "ready"}), encoding="utf-8")

    monkeypatch.setattr(ddp, "barrier", fake_barrier)

    ddp.run_on_main_process_first(
        _distributed_state(rank=1),
        action=lambda: pytest.fail("non-main rank should not run the action"),
        status_path=status_path,
        ready_paths=[output_path],
        description="Prepare package",
        poll_seconds=0.0,
    )

    assert barrier_calls == 3
