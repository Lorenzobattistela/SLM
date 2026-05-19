from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config import resolve_project_path
from src.data.dataset import TokenShardDataset
from src.data.tokenizer import get_tokenizer
from src.eval.perplexity import evaluate_perplexity
from src.inference.generate import generate
from src.model.config import ModelConfig
from src.model.transformer import TransformerLM
from src.train.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from src.train.ddp import DistributedState, barrier, cleanup_distributed, init_distributed
from src.train.optim import build_optimizer, build_scheduler
from src.utils import append_jsonl, autocast_context, select_device, set_seed, write_json


def _print_main(state: DistributedState, message: str) -> None:
    if state.is_main_process:
        print(message, flush=True)


def _save_config_snapshot(output_dir: Path, config: dict) -> None:
    with (output_dir / "resolved_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _sample_text(model: torch.nn.Module, tokenizer, train_cfg: dict, device: torch.device) -> str:
    prompt = train_cfg["sample_prompt"]
    prompt_ids = tokenizer.encode(prompt)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    outputs = generate(
        model=unwrap_model(model),
        input_ids=inputs,
        max_new_tokens=int(train_cfg["sample_max_new_tokens"]),
        temperature=float(train_cfg["sample_temperature"]),
        top_k=int(train_cfg["sample_top_k"]) if train_cfg["sample_top_k"] else None,
        eos_token_id=tokenizer.eot_token_id,
    )
    return tokenizer.decode(outputs[0].tolist())


def run_pretraining(config: dict) -> None:
    train_cfg = config["train"]
    model_cfg = ModelConfig.from_dict(config["model"])
    dist_state = init_distributed(bool(train_cfg.get("ddp", False)))
    device = select_device(train_cfg.get("device", "auto"))
    if dist_state.enabled and torch.cuda.is_available():
        device = torch.device("cuda", dist_state.local_rank)

    set_seed(int(train_cfg["seed"]) + dist_state.rank)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = resolve_project_path(train_cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if dist_state.is_main_process:
        _save_config_snapshot(output_dir, config)

    train_root = resolve_project_path(config["data"]["storage"]["output_dir"]) / "train"
    val_root = resolve_project_path(config["data"]["storage"]["output_dir"]) / "val"
    train_data = TokenShardDataset(train_root, block_size=model_cfg.context_length)
    val_data = TokenShardDataset(val_root, block_size=model_cfg.context_length)

    model = TransformerLM(model_cfg).to(device)
    raw_model = model

    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    if dist_state.enabled:
        if device.type == "cuda":
            model = DDP(model, device_ids=[dist_state.local_rank])
        else:
            model = DDP(model)

    optimizer = build_optimizer(unwrap_model(model), train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    start_step = 0
    tokens_seen = 0
    resume_path = train_cfg.get("resume_from")
    if resume_path:
        checkpoint = load_checkpoint(
            resolve_project_path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_step = int(checkpoint["step"])
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        _print_main(dist_state, f"Resumed from step {start_step}")

    tokenizer = get_tokenizer(config["data"]["tokenizer"]["name"]) if dist_state.is_main_process else None
    metrics_path = output_dir / "metrics.jsonl"
    samples_path = output_dir / "samples.jsonl"
    rng = np.random.default_rng(int(train_cfg["seed"]) + dist_state.rank)

    _print_main(
        dist_state,
        (
            f"Starting run `{config['name']}` on {device}. "
            f"Parameters: {unwrap_model(model).num_parameters():,}"
        ),
    )

    for step in range(start_step + 1, int(train_cfg["max_steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(int(train_cfg["grad_accum_steps"])):
            inputs, targets = train_data.sample_batch(int(train_cfg["batch_size"]), device, rng)
            sync_context = (
                model.no_sync()
                if dist_state.enabled and micro_step < int(train_cfg["grad_accum_steps"]) - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_context(device, train_cfg["precision"]):
                    _, loss = model(inputs, targets)
                step_loss += float(loss.item())
                (loss / int(train_cfg["grad_accum_steps"])).backward()

        grad_clip = float(train_cfg.get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        tokens_seen += (
            int(train_cfg["batch_size"])
            * int(train_cfg["grad_accum_steps"])
            * model_cfg.context_length
            * dist_state.world_size
        )

        if dist_state.is_main_process and step % int(train_cfg["log_interval"]) == 0:
            train_loss = step_loss / int(train_cfg["grad_accum_steps"])
            current_lr = float(scheduler.get_last_lr()[0])
            append_jsonl(
                metrics_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss,
                    "lr": current_lr,
                },
            )
            print(
                f"step={step} train_loss={train_loss:.4f} lr={current_lr:.6e} "
                f"tokens={tokens_seen}",
                flush=True,
            )

        if step % int(train_cfg["eval_interval"]) == 0:
            if dist_state.is_main_process:
                val_loss, val_ppl = evaluate_perplexity(
                    unwrap_model(model),
                    val_data,
                    batch_size=int(train_cfg["eval_batch_size"]),
                    device=device,
                    precision=train_cfg["precision"],
                    max_batches=int(train_cfg["eval_batches"]),
                )
                append_jsonl(
                    metrics_path,
                    {
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                    },
                )
                print(
                    f"step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}",
                    flush=True,
                )

                if train_cfg.get("sample_every_eval", False) and tokenizer is not None:
                    sample_text = _sample_text(model, tokenizer, train_cfg, device)
                    append_jsonl(
                        samples_path,
                        {
                            "step": step,
                            "tokens_seen": tokens_seen,
                            "text": sample_text,
                        },
                    )
            barrier(dist_state)

        if dist_state.is_main_process and step % int(train_cfg["save_interval"]) == 0:
            extra_state = {"device": str(device), "raw_model_device": str(next(raw_model.parameters()).device)}
            save_checkpoint(
                checkpoint_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                tokens_seen=tokens_seen,
                config=config,
                extra_state=extra_state,
            )
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                tokens_seen=tokens_seen,
                config=config,
                extra_state=extra_state,
            )
            write_json(output_dir / "latest.json", {"step": step, "tokens_seen": tokens_seen})
        if step % int(train_cfg["save_interval"]) == 0:
            barrier(dist_state)

    barrier(dist_state)
    if dist_state.is_main_process:
        save_checkpoint(
            checkpoint_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=int(train_cfg["max_steps"]),
            tokens_seen=tokens_seen,
            config=config,
        )
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=int(train_cfg["max_steps"]),
            tokens_seen=tokens_seen,
            config=config,
        )

    barrier(dist_state)
    cleanup_distributed(dist_state)
