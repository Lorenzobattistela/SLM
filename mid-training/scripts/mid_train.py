from __future__ import annotations
# ruff: noqa: E402

import argparse
import gc
import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


# Set PROJECT_ROOT to ensure correct imports
def _find_project_root(start: Path) -> Path:
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import load_yaml, resolve_project_path
from src.data.mid_train_dataset import (
    MidTrainStreamPreparer,
    delete_mid_train_package,
    mid_train_package_path,
)
from src.data.streaming_mix import source_metadata
from src.data.token_dataset import TokenBinDataset, token_dtype_for_vocab
from src.evaluation.evaluator import evaluate_model
from src.model import ModelConfig, TransformerLM
from src.model.attention import configure_attention_optimization
from src.training.checkpointing import (
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
from src.training.metrics import append_metrics, perplexity_from_loss
from src.training.optimizer import build_optimizer
from src.training.precision import autocast_for_precision, resolve_precision
from src.training.scheduler import build_scheduler
from src.utils import set_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mid-train the Small Language Model (SLM).")
    parser.add_argument(
        "--run-config",
        type=str,
        required=True,
        help="Path to the mid-training YAML configuration file.",
    )
    return parser.parse_args()


def load_mid_train_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)

    # Resolve paths and ensure all directories exist
    output_dir = resolve_project_path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = resolve_project_path(config["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    save_dir = resolve_project_path(config["training"]["checkpointing"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    if config.get("plots", {}).get("enabled", True):
        plots_dir = resolve_project_path(config["plots"]["output_dir"])
        plots_dir.mkdir(parents=True, exist_ok=True)

    return config


def _build_validation_dataset(
    config: dict[str, Any],
    model_cfg: ModelConfig,
) -> TokenBinDataset:
    dataset_cfg = config["dataset"]
    processed_dir = resolve_project_path(dataset_cfg["processed_dir"])

    metadata_path = processed_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            try:
                metadata = json.load(handle)
            except json.JSONDecodeError:
                metadata = {}
        dtype = metadata.get("storage_dtype")
    else:
        dtype = None

    if dtype is None:
        dtype = token_dtype_for_vocab(int(config["tokenizer"]["vocab_size"]))

    common = {
        "block_size": model_cfg.context_length,
        "vocab_size": int(config["tokenizer"]["vocab_size"]),
        "dtype": str(dtype),
    }
    return TokenBinDataset(processed_dir / "val_tokens.bin", **common)


def _streaming_package_tokens(
    config: dict[str, Any],
    model_cfg: ModelConfig,
    tokens_per_step: int,
) -> int:
    dataset_cfg = config["dataset"]
    configured_tokens = dataset_cfg.get("streaming_package_tokens")
    if configured_tokens is not None:
        package_tokens = int(configured_tokens)
    else:
        steps_per_package = int(dataset_cfg.get("streaming_steps_per_package", 32))
        package_tokens = tokens_per_step * max(1, steps_per_package)
    return max(package_tokens, tokens_per_step, model_cfg.context_length + 1)


def _compute_total_steps(
    training_cfg: dict[str, Any], model_cfg: ModelConfig, world_size: int
) -> int:
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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


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


def _run_validation(
    *,
    state: DistributedState,
    model: torch.nn.Module,
    val_dataset: TokenBinDataset,
    config: dict[str, Any],
    metrics_path: Path,
    batch_size: int,
    device: torch.device,
    precision: str,
    step: int,
    tokens_seen: int,
) -> None:
    if state.is_main_process:
        eval_stats = evaluate_model(
            unwrap_model(model),
            val_dataset,
            batch_size=batch_size,
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
                "validation_perplexity": eval_stats.perplexity,
            },
        )
        LOGGER.info(
            "step=%s validation_loss=%.4f perplexity=%.4f",
            step,
            eval_stats.loss,
            eval_stats.perplexity,
        )
    barrier(state)


def _run_mid_training(config: dict[str, Any], state: DistributedState) -> None:
    training_cfg = config["training"]
    device = select_training_device(state)
    precision = resolve_precision(str(training_cfg.get("precision", "fp32")), device)
    set_seed(int(config["project"]["seed"]) + state.rank)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = ModelConfig.from_dict(config["model"])
    micro_batch_size = int(training_cfg["micro_batch_size"])
    grad_accum_steps = int(training_cfg["gradient_accumulation_steps"])
    tokens_per_step = (
        micro_batch_size * grad_accum_steps * model_cfg.context_length * state.world_size
    )
    effective_batch = micro_batch_size * grad_accum_steps * state.world_size
    streaming_package_tokens = _streaming_package_tokens(config, model_cfg, tokens_per_step)
    target_train_tokens = int(config["dataset"]["target_train_tokens"])

    data_preparer: MidTrainStreamPreparer | None = None
    if state.is_main_process:
        data_preparer = MidTrainStreamPreparer(config)
        data_preparer.prepare_validation()
    barrier(state)

    val_dataset = _build_validation_dataset(config, model_cfg)
    model = TransformerLM(model_cfg).to(device)
    attention_info = configure_attention_optimization(
        model,
        config=model_cfg,
        device=device,
        precision=precision,
        batch_size=micro_batch_size,
        seq_len=model_cfg.context_length,
    )

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

    # Load weights from pre-training checkpoint (loading weights only, discarding optimizer state)
    resume_from = training_cfg["checkpointing"].get("resume_from")
    if resume_from:
        checkpoint_path = resolve_project_path(resume_from)
        if checkpoint_path.exists():
            load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=None,
                scheduler=None,
                map_location=device,
                restore_rng=False,
            )
            _log_main(
                state,
                "Initialized weights from pre-training checkpoint: %s (optimizer/scheduler state discarded)",
                checkpoint_path,
            )
        else:
            _log_main(
                state,
                "Pre-training checkpoint not found at %s. Starting with random weights.",
                checkpoint_path,
            )

    output_dir = resolve_project_path(config["project"]["output_dir"])
    metrics_path = output_dir / "logs" / "metrics.jsonl"
    rng = np.random.default_rng(int(config["project"]["seed"]) + state.rank + step)

    max_tokens = training_cfg.get("max_tokens")

    _log_main(
        state,
        "Starting mid-training project=%s device=%s precision=%s ddp=%s world_size=%s "
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

    if state.is_main_process:
        training_metadata = {
            "event": "training_start",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project": config["project"]["name"],
            "precision": precision,
            "device": str(device),
            "ddp": state.enabled,
            "world_size": state.world_size,
            "parameters": unwrap_model(model).num_parameters(),
            "model": {
                "architecture": model_cfg.name,
                "attention": model_cfg.attention,
                "flash_attention": model_cfg.flash_attention,
                "num_attention_heads": model_cfg.n_heads,
                "num_key_value_heads": model_cfg.n_kv_heads,
                "head_dim": model_cfg.head_dim,
                "layers": model_cfg.n_layers,
                "d_model": model_cfg.d_model,
                "context_length": model_cfg.context_length,
            },
            "attention_optimization": attention_info.to_dict(),
            "training": {
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "effective_batch_size": effective_batch,
                "tokens_per_step": tokens_per_step,
                "total_steps": total_steps,
                "max_tokens": max_tokens,
            },
            "dataset": {
                "streaming": True,
                "target_train_tokens": target_train_tokens,
                "streaming_package_tokens": streaming_package_tokens,
                "validation_tokens": val_dataset.num_tokens,
                "source_signature": (
                    data_preparer.source_signature if data_preparer is not None else None
                ),
                "sources": (
                    source_metadata(data_preparer.sources) if data_preparer is not None else []
                ),
            },
        }
        _write_json(output_dir / "logs" / "training_metadata.json", training_metadata)
        append_metrics(
            metrics_path,
            {
                "event": "training_start",
                "step": step,
                "tokens_seen": tokens_seen,
                "total_steps": total_steps,
                "train_dataset_tokens": target_train_tokens,
                "streaming_package_tokens": streaming_package_tokens,
                "validation_dataset_tokens": val_dataset.num_tokens,
                "effective_batch_size": effective_batch,
                "tokens_per_step": tokens_per_step,
                "attention_backend": attention_info.backend,
                "attention_flash_available": attention_info.flash_available,
                "attention_enable_gqa": attention_info.enable_gqa,
            },
        )

    train_dtype = token_dtype_for_vocab(int(config["tokenizer"]["vocab_size"]))
    train_dataset: TokenBinDataset | None = None
    current_package_index = -1
    current_package_tokens = 0
    package_steps = 0
    package_steps_used = 0

    def close_current_train_package(*, delete_package: bool) -> None:
        nonlocal train_dataset
        if train_dataset is None:
            return
        train_dataset.close()
        train_dataset = None
        gc.collect()
        barrier(state)
        if delete_package and state.is_main_process and current_package_index >= 0:
            delete_mid_train_package(config, current_package_index)
        barrier(state)

    def open_next_train_package() -> None:
        nonlocal current_package_index
        nonlocal current_package_tokens
        nonlocal package_steps
        nonlocal package_steps_used
        nonlocal train_dataset

        close_current_train_package(delete_package=True)
        current_package_index += 1
        if state.is_main_process:
            if data_preparer is None:
                raise RuntimeError("Main process is missing the mid-training stream preparer.")
            data_preparer.write_train_package(current_package_index, streaming_package_tokens)
        barrier(state)

        train_dataset = TokenBinDataset(
            mid_train_package_path(config, current_package_index),
            block_size=model_cfg.context_length,
            vocab_size=int(config["tokenizer"]["vocab_size"]),
            dtype=train_dtype,
        )
        current_package_tokens = train_dataset.num_tokens
        package_steps = max(1, math.ceil(current_package_tokens / tokens_per_step))
        package_steps_used = 0
        _log_main(
            state,
            "Opened streaming mid-training package index=%s tokens=%s planned_steps=%s",
            current_package_index,
            current_package_tokens,
            package_steps,
        )

    last_log_time = time.perf_counter()
    last_eval_step: int | None = None
    try:
        while _should_continue(step, tokens_seen, training_cfg):
            if train_dataset is None or package_steps_used >= package_steps:
                open_next_train_package()
            if train_dataset is None:
                raise RuntimeError("Mid-training package was not opened.")

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

            if "gradient_clipping" in training_cfg and bool(
                training_cfg["gradient_clipping"]["enabled"]
            ):
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(),
                    float(training_cfg["gradient_clipping"]["max_norm"]),
                )
                grad_norm = float(grad_norm_tensor.item())

            optimizer.step()
            step += 1
            package_steps_used += 1
            tokens_seen += tokens_per_step
            scheduler.step(step)
            stop_after_step = not _should_continue(step, tokens_seen, training_cfg)

            if state.is_main_process and (
                step % int(config["logging"]["log_every_steps"]) == 0 or stop_after_step
            ):
                elapsed = max(1.0e-9, time.perf_counter() - last_log_time)
                step_elapsed = max(1.0e-9, time.perf_counter() - step_started)
                last_log_time = time.perf_counter()
                train_loss = accumulated_loss / grad_accum_steps
                train_perplexity = perplexity_from_loss(train_loss)
                lr = float(scheduler.get_last_lr()[0])
                token_progress = (
                    min(1.0, tokens_seen / int(max_tokens)) if max_tokens is not None else None
                )
                metrics = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss,
                    "train_perplexity": train_perplexity,
                    "learning_rate": lr,
                    "gradient_norm": grad_norm,
                    "effective_batch_size": effective_batch,
                    "tokens_per_step": tokens_per_step,
                    "step_time_seconds": step_elapsed,
                    "tokens_per_second": tokens_per_step / step_elapsed,
                    "samples_per_second": effective_batch / step_elapsed,
                    "epoch_equivalent": tokens_seen / max(1, target_train_tokens),
                    "streaming_package_index": current_package_index,
                    "streaming_package_tokens": current_package_tokens,
                    "streaming_package_step": package_steps_used,
                    "streaming_package_steps": package_steps,
                    "log_window_tokens_per_second": (
                        tokens_per_step * int(config["logging"]["log_every_steps"]) / elapsed
                    ),
                }
                if token_progress is not None:
                    metrics["token_progress"] = token_progress
                    metrics["token_progress_percent"] = 100.0 * token_progress
                if device.type == "cuda":
                    metrics["gpu_memory_allocated_gb"] = (
                        torch.cuda.memory_allocated(device) / 1024**3
                    )
                    metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3
                    metrics["gpu_memory_max_allocated_gb"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                append_metrics(metrics_path, metrics)
                progress_pct = 100.0 * min(1.0, step / max(1, total_steps))
                LOGGER.info(
                    "step=%s/%s progress=%.2f%% loss=%.4f train_ppl=%.2f lr=%.6e "
                    "tokens=%s epoch_equiv=%.4f grad_norm=%.4f tok/s=%.1f",
                    step,
                    total_steps,
                    progress_pct,
                    train_loss,
                    train_perplexity,
                    lr,
                    tokens_seen,
                    metrics["epoch_equivalent"],
                    grad_norm,
                    metrics["tokens_per_second"],
                )

            if (
                bool(config["evaluation"].get("enabled", True))
                and step % int(config["evaluation"]["eval_every_steps"]) == 0
            ):
                _run_validation(
                    state=state,
                    model=model,
                    val_dataset=val_dataset,
                    config=config,
                    metrics_path=metrics_path,
                    batch_size=micro_batch_size,
                    device=device,
                    precision=precision,
                    step=step,
                    tokens_seen=tokens_seen,
                )
                last_eval_step = step

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

        if bool(config["evaluation"].get("enabled", True)) and step > 0 and last_eval_step != step:
            _run_validation(
                state=state,
                model=model,
                val_dataset=val_dataset,
                config=config,
                metrics_path=metrics_path,
                batch_size=micro_batch_size,
                device=device,
                precision=precision,
                step=step,
                tokens_seen=tokens_seen,
            )

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
            LOGGER.info("Mid-training complete at step=%s tokens_seen=%s", step, tokens_seen)

            # Generate plots if enabled
            if bool(config.get("plots", {}).get("enabled", True)):
                try:
                    from src.plotting import generate_training_plots, load_jsonl_metrics

                    plots_output_dir = resolve_project_path(config["plots"]["output_dir"])
                    plot_names = config["plots"].get("generate")
                    metrics = load_jsonl_metrics(metrics_path)
                    generate_training_plots(metrics, plots_output_dir, plot_names=plot_names)
                    LOGGER.info("Saved training plots to %s", plots_output_dir)
                except Exception as e:
                    LOGGER.exception("Failed to generate plots automatically: %s", e)

        close_current_train_package(delete_package=True)
        barrier(state)
    finally:
        if train_dataset is not None:
            train_dataset.close()
        val_dataset.close()
        cleanup_distributed(state)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_mid_train_config(args.run_config)

    state = init_distributed(config["training"])
    try:
        _run_mid_training(config, state)
    finally:
        cleanup_distributed(state)


if __name__ == "__main__":
    main()
