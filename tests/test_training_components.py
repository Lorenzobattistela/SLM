from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import torch

from src.data.token_dataset import TokenBinWriter, write_metadata
from src.training.scheduler import WarmupCosineScheduler
from src.training.trainer import run_training

RUN_ALL_PATH = Path(__file__).resolve().parents[1] / "pre-train" / "scripts" / "run_all.py"
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


def _write_tokens(path: Path, *, vocab_size: int, count: int) -> None:
    with TokenBinWriter(path, vocab_size=vocab_size, target_tokens=count) as writer:
        writer.write((index % vocab_size for index in range(count)))


def _tiny_training_config(tmp_path: Path) -> dict:
    processed_dir = tmp_path / "processed"
    vocab_size = 64
    _write_tokens(processed_dir / "train_tokens.bin", vocab_size=vocab_size, count=80)
    _write_tokens(processed_dir / "val_tokens.bin", vocab_size=vocab_size, count=80)
    write_metadata(
        processed_dir / "metadata.json",
        {
            "storage_dtype": "uint16",
            "train_tokens": 80,
            "validation_tokens": 80,
            "vocab_size": vocab_size,
        },
    )

    return {
        "project": {
            "name": "tiny_training_test",
            "seed": 7,
            "output_dir": str(tmp_path / "outputs"),
        },
        "dataset": {
            "processed_dir": str(processed_dir),
        },
        "tokenizer": {
            "vocab_size": vocab_size,
        },
        "model": {
            "architecture": "decoder_only_transformer",
            "vocab_size": vocab_size,
            "max_seq_len": 8,
            "attention": "gqa",
            "n_layers": 1,
            "d_model": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "ffn_multiplier": 2,
            "multiple_of": 8,
            "norm_eps": 1.0e-5,
            "rope_theta": 10000.0,
            "dropout": 0.0,
            "tie_embeddings": True,
            "flash_attention": False,
            "flash_attention_fallback": True,
        },
        "training": {
            "distributed": {
                "enabled": False,
                "backend": "gloo",
                "strategy": "ddp",
                "num_gpus": 1,
            },
            "precision": "bf16",
            "compile_model": False,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "max_tokens": None,
            "optimizer": {
                "name": "adamw",
                "learning_rate": 1.0e-3,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 1,
                "min_lr": 1.0e-5,
            },
            "gradient_clipping": {
                "enabled": True,
                "max_norm": 1.0,
            },
            "checkpointing": {
                "save_dir": str(tmp_path / "checkpoints"),
                "save_every_steps": 100,
                "keep_last_n": 2,
                "resume_from": None,
            },
        },
        "evaluation": {
            "enabled": True,
            "eval_every_steps": 100,
            "eval_steps": 1,
        },
        "logging": {
            "log_every_steps": 100,
        },
    }


def test_training_logs_short_run_metrics_and_resumes_checkpoint(tmp_path) -> None:
    config = _tiny_training_config(tmp_path)

    run_training(config)

    metrics_path = tmp_path / "outputs" / "logs" / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train_metrics = [payload for payload in metrics if "train_loss" in payload]
    validation_metrics = [payload for payload in metrics if "perplexity" in payload]

    assert train_metrics
    assert train_metrics[-1]["step"] == 1
    assert train_metrics[-1]["tokens_seen"] == 16
    assert "learning_rate" in train_metrics[-1]
    assert "gradient_norm" in train_metrics[-1]
    assert validation_metrics
    assert validation_metrics[-1]["step"] == 1
    assert validation_metrics[-1]["tokens_seen"] == 16
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "final.pt").exists()

    resumed_config = copy.deepcopy(config)
    resumed_config["training"]["max_steps"] = 2
    resumed_config["training"]["checkpointing"]["resume_from"] = str(
        tmp_path / "checkpoints" / "latest.pt"
    )

    run_training(resumed_config)

    latest = torch.load(
        tmp_path / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert latest["step"] == 2
    assert latest["tokens_seen"] == 32
