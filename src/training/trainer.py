from __future__ import annotations

import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from src.config import resolve_project_path
from src.data.token_dataset import TokenBinDataset, token_dtype_for_vocab
from src.evaluation.evaluator import evaluate_model
from src.model import ModelConfig, TransformerLM
from src.training.checkpointing import (
    find_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    unwrap_model,
)
from src.training.ddp import (
    DistributedState,
    barrier,
    cleanup_distributed,
    init_distributed,
    select_training_device,
)
from src.training.metrics import append_metrics
from src.training.optimizer import build_optimizer
from src.training.precision import autocast_for_precision, resolve_precision
from src.training.scheduler import build_scheduler
from src.utils import set_seed

LOGGER = logging.getLogger(__name__)


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_datasets(
    config: dict[str, Any],
    model_cfg: ModelConfig,
) -> tuple[TokenBinDataset, TokenBinDataset]:
    dataset_cfg = config["dataset"]
    processed_dir = resolve_project_path(dataset_cfg["processed_dir"])
    metadata = _load_metadata(processed_dir)
    dtype = metadata.get("storage_dtype") or token_dtype_for_vocab(
        int(config["tokenizer"]["vocab_size"])
    )
    common = {
        "block_size": model_cfg.context_length,
        "vocab_size": int(config["tokenizer"]["vocab_size"]),
        "dtype": str(dtype),
    }
    return (
        TokenBinDataset(processed_dir / "train_tokens.bin", **common),
        TokenBinDataset(processed_dir / "val_tokens.bin", **common),
    )


def _compute_total_steps(training_cfg: dict[str, Any], model_cfg: ModelConfig, world_size: int) -> int:
    configured_max_steps = training_cfg.get("max_steps")
    step_limit = int(configured_max_steps) if configured_max_steps is not None else None
    max_tokens = training_cfg.get("max_tokens")
    token_limit = int(max_tokens) if max_tokens is not None else None
    tokens_per_step = (
        int(training_cfg["micro_batch_size"])
        * int(training_cfg["gradient_accumulation_steps"])
        * model_cfg.context_length
        * max(1, int(world_size))
    )
    token_steps = math.ceil(token_limit / tokens_per_step) if token_limit is not None else None

    if step_limit is not None and token_steps is not None:
        return max(1, min(step_limit, token_steps))
    if step_limit is not None:
        return max(1, step_limit)
    if token_steps is not None:
        return max(1, token_steps)
    raise ValueError("Configure training.max_steps, training.max_tokens, or both.")


def _should_continue(step: int, tokens_seen: int, training_cfg: dict[str, Any]) -> bool:
    max_steps = training_cfg.get("max_steps")
    max_tokens = training_cfg.get("max_tokens")
    if max_steps is not None and step >= int(max_steps):
        return False
    if max_tokens is not None and tokens_seen >= int(max_tokens):
        return False
    return True


def _log_main(state: DistributedState, message: str, *args: Any) -> None:
    if state.is_main_process:
        LOGGER.info(message, *args)


def _save_training_checkpoint(
    *,
    config: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    rng: np.random.Generator,
) -> None:
    checkpoint_cfg = config["training"]["checkpointing"]
    save_dir = resolve_project_path(checkpoint_cfg["save_dir"])
    save_checkpoint(
        save_dir / f"step_{step:08d}.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        tokens_seen=tokens_seen,
        config=config,
        extra_state={"data_rng_state": rng.bit_generator.state},
    )
    save_checkpoint(
        save_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        tokens_seen=tokens_seen,
        config=config,
        extra_state={"data_rng_state": rng.bit_generator.state},
    )
    rotate_checkpoints(save_dir, keep_last_n=int(checkpoint_cfg["keep_last_n"]))


def _run_training(config: dict[str, Any], state: DistributedState) -> None:
    training_cfg = config["training"]
    device = select_training_device(state)
    precision = resolve_precision(str(training_cfg.get("precision", "fp32")), device)
    set_seed(int(config["project"]["seed"]) + state.rank)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = ModelConfig.from_dict(config["model"])
    train_dataset, val_dataset = _build_datasets(config, model_cfg)
    model = TransformerLM(model_cfg).to(device)

    if bool(training_cfg.get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    if state.enabled:
        ddp_device_ids = [state.local_rank] if device.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=ddp_device_ids)

    optimizer = build_optimizer(unwrap_model(model), training_cfg["optimizer"])
    total_steps = _compute_total_steps(training_cfg, model_cfg, state.world_size)
    scheduler = build_scheduler(optimizer, training_cfg, total_steps=total_steps)

    step = 0
    tokens_seen = 0
    resume_data_rng_state = None
    resume_from = training_cfg["checkpointing"].get("resume_from")
    if resume_from:
        checkpoint = load_checkpoint(
            find_checkpoint(config, resume_from),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        step = int(checkpoint["step"])
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        resume_data_rng_state = checkpoint.get("extra_state", {}).get("data_rng_state")
        scheduler.step(step)
        _log_main(state, "Resumed checkpoint from step=%s tokens_seen=%s", step, tokens_seen)

    output_dir = resolve_project_path(config["project"]["output_dir"])
    metrics_path = output_dir / "logs" / "metrics.jsonl"
    rng = np.random.default_rng(int(config["project"]["seed"]) + state.rank + step)
    if resume_data_rng_state is not None:
        rng.bit_generator.state = resume_data_rng_state

    micro_batch_size = int(training_cfg["micro_batch_size"])
    grad_accum_steps = int(training_cfg["gradient_accumulation_steps"])
    tokens_per_step = micro_batch_size * grad_accum_steps * model_cfg.context_length * state.world_size
    effective_batch = micro_batch_size * grad_accum_steps * state.world_size

    _log_main(
        state,
        "Starting training project=%s device=%s precision=%s ddp=%s world_size=%s "
        "parameters=%s effective_batch=%s tokens_per_step=%s total_steps=%s",
        config["project"]["name"],
        device,
        precision,
        state.enabled,
        state.world_size,
        f"{unwrap_model(model).num_parameters():,}",
        effective_batch,
        tokens_per_step,
        total_steps,
    )

    last_log_time = time.perf_counter()
    try:
        while _should_continue(step, tokens_seen, training_cfg):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            grad_norm = 0.0
            step_started = time.perf_counter()

            for micro_step in range(grad_accum_steps):
                sync_context = (
                    model.no_sync()
                    if state.enabled and micro_step < grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    inputs, targets = train_dataset.sample_batch(
                        micro_batch_size,
                        device,
                        rng,
                        rank=state.rank,
                        world_size=state.world_size,
                    )
                    with autocast_for_precision(device, precision):
                        _, loss = model(inputs, targets)
                    accumulated_loss += float(loss.item())
                    (loss / grad_accum_steps).backward()

            if bool(training_cfg["gradient_clipping"]["enabled"]):
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(),
                    float(training_cfg["gradient_clipping"]["max_norm"]),
                )
                grad_norm = float(grad_norm_tensor.item())

            optimizer.step()
            step += 1
            tokens_seen += tokens_per_step
            scheduler.step(step)

            if state.is_main_process and step % int(config["logging"]["log_every_steps"]) == 0:
                elapsed = max(1.0e-9, time.perf_counter() - last_log_time)
                step_elapsed = max(1.0e-9, time.perf_counter() - step_started)
                last_log_time = time.perf_counter()
                train_loss = accumulated_loss / grad_accum_steps
                lr = float(scheduler.get_last_lr()[0])
                metrics = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss,
                    "learning_rate": lr,
                    "gradient_norm": grad_norm,
                    "tokens_per_second": tokens_per_step / step_elapsed,
                    "samples_per_second": effective_batch / step_elapsed,
                    "log_window_tokens_per_second": (
                        tokens_per_step * int(config["logging"]["log_every_steps"]) / elapsed
                    ),
                }
                append_metrics(metrics_path, metrics)
                LOGGER.info(
                    "step=%s loss=%.4f lr=%.6e tokens=%s grad_norm=%.4f tok/s=%.1f",
                    step,
                    train_loss,
                    lr,
                    tokens_seen,
                    grad_norm,
                    metrics["tokens_per_second"],
                )

            if (
                bool(config["evaluation"].get("enabled", True))
                and step % int(config["evaluation"]["eval_every_steps"]) == 0
            ):
                if state.is_main_process:
                    eval_stats = evaluate_model(
                        unwrap_model(model),
                        val_dataset,
                        batch_size=micro_batch_size,
                        device=device,
                        precision=precision,
                        max_batches=int(config["evaluation"]["eval_steps"]),
                    )
                    append_metrics(
                        metrics_path,
                        {
                            "step": step,
                            "tokens_seen": tokens_seen,
                            "validation_loss": eval_stats.loss,
                            "val_loss": eval_stats.loss,
                            "perplexity": eval_stats.perplexity,
                        },
                    )
                    LOGGER.info(
                        "step=%s validation_loss=%.4f perplexity=%.4f",
                        step,
                        eval_stats.loss,
                        eval_stats.perplexity,
                    )
                barrier(state)

            if step % int(training_cfg["checkpointing"]["save_every_steps"]) == 0:
                if state.is_main_process:
                    _save_training_checkpoint(
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        tokens_seen=tokens_seen,
                        rng=rng,
                    )
                    LOGGER.info("Saved checkpoint at step=%s", step)
                barrier(state)

        if state.is_main_process:
            save_dir = resolve_project_path(training_cfg["checkpointing"]["save_dir"])
            save_checkpoint(
                save_dir / "final.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                tokens_seen=tokens_seen,
                config=config,
                extra_state={"data_rng_state": rng.bit_generator.state},
            )
            save_checkpoint(
                save_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                tokens_seen=tokens_seen,
                config=config,
                extra_state={"data_rng_state": rng.bit_generator.state},
            )
            LOGGER.info("Training complete at step=%s tokens_seen=%s", step, tokens_seen)
        barrier(state)
    finally:
        cleanup_distributed(state)


def run_training(config: dict[str, Any]) -> None:
    state = init_distributed(config["training"])
    try:
        _run_training(config, state)
    finally:
        cleanup_distributed(state)
