from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.training.ddp import DistributedState

MID_TRAIN_PATH = Path(__file__).resolve().parents[1] / "mid-training" / "scripts" / "mid_train.py"
MID_TRAIN_SPEC = importlib.util.spec_from_file_location("mid_train", MID_TRAIN_PATH)
assert MID_TRAIN_SPEC is not None
mid_train = importlib.util.module_from_spec(MID_TRAIN_SPEC)
assert MID_TRAIN_SPEC.loader is not None
MID_TRAIN_SPEC.loader.exec_module(mid_train)


def test_mid_training_resume_from_loads_pretrained_weights_only(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "gpt_bpe_latest.pt"
    config = {"training": {"checkpointing": {"resume_from": "last_pre_trains/gpt_bpe_latest.pt"}}}
    model = torch.nn.Linear(2, 2)
    device = torch.device("cpu")
    state = DistributedState(requested=False, enabled=False, backend="gloo")
    load_calls = []

    def fake_find_checkpoint(config_arg, explicit_path):
        assert config_arg is config
        assert explicit_path == "last_pre_trains/gpt_bpe_latest.pt"
        return checkpoint_path

    def fake_load_checkpoint(path, **kwargs):
        load_calls.append((path, kwargs))
        return {"model": {}}

    monkeypatch.setattr(mid_train, "find_checkpoint", fake_find_checkpoint)
    monkeypatch.setattr(mid_train, "load_checkpoint", fake_load_checkpoint)

    loaded_path = mid_train._initialize_from_pretraining_checkpoint(
        config=config,
        training_cfg=config["training"],
        model=model,
        device=device,
        state=state,
    )

    assert loaded_path == checkpoint_path
    assert len(load_calls) == 1
    path, kwargs = load_calls[0]
    assert path == checkpoint_path
    assert kwargs["model"] is model
    assert kwargs["optimizer"] is None
    assert kwargs["scheduler"] is None
    assert kwargs["map_location"] == device
    assert kwargs["restore_rng"] is False


def test_mid_training_resume_from_missing_checkpoint_is_fatal(monkeypatch) -> None:
    config = {"training": {"checkpointing": {"resume_from": "last_pre_trains/gpt_bpe_latest.pt"}}}
    model = torch.nn.Linear(2, 2)
    state = DistributedState(requested=False, enabled=False, backend="gloo")

    def fake_find_checkpoint(config_arg, explicit_path):
        raise FileNotFoundError(f"Checkpoint not found: {explicit_path}")

    monkeypatch.setattr(mid_train, "find_checkpoint", fake_find_checkpoint)

    with pytest.raises(FileNotFoundError, match="last_pre_trains/gpt_bpe_latest.pt"):
        mid_train._initialize_from_pretraining_checkpoint(
            config=config,
            training_cfg=config["training"],
            model=model,
            device=torch.device("cpu"),
            state=state,
        )
