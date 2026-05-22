from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.config import resolve_project_path
from src.data.token_dataset import TokenBinDataset, token_dtype_for_vocab
from src.model import ModelConfig, TransformerLM
from src.training.checkpointing import find_checkpoint, load_checkpoint
from src.training.metrics import LossStats, perplexity_from_loss
from src.training.precision import autocast_for_precision, resolve_precision

LOGGER = logging.getLogger(__name__)


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    import json

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_validation_dataset(config: dict[str, Any], model_cfg: ModelConfig) -> TokenBinDataset:
    dataset_cfg = config["dataset"]
    processed_dir = resolve_project_path(dataset_cfg["processed_dir"])
    metadata = _load_metadata(processed_dir)
    dtype = metadata.get("storage_dtype") or token_dtype_for_vocab(
        int(config["tokenizer"]["vocab_size"])
    )
    val_path = processed_dir / "val_tokens.bin"
    return TokenBinDataset(
        val_path,
        block_size=model_cfg.context_length,
        vocab_size=int(config["tokenizer"]["vocab_size"]),
        dtype=str(dtype),
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: TokenBinDataset,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
    max_batches: int | None,
) -> LossStats:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in dataset.iter_batches(
        batch_size,
        device,
        max_batches=max_batches,
    ):
        with autocast_for_precision(device, precision):
            _, loss = model(inputs, targets)
        batch_tokens = int(inputs.numel())
        total_loss += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens

    if total_tokens == 0:
        raise ValueError("Validation dataset produced zero evaluated tokens.")

    mean_loss = total_loss / total_tokens
    return LossStats(loss=mean_loss, perplexity=perplexity_from_loss(mean_loss))


def run_evaluation(config: dict[str, Any], *, checkpoint_path: str | Path | None = None) -> LossStats:
    model_cfg = ModelConfig.from_dict(config["model"])
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    precision = resolve_precision(str(config["training"].get("precision", "fp32")), device)

    model = TransformerLM(model_cfg).to(device)
    checkpoint = find_checkpoint(config, checkpoint_path)
    load_checkpoint(
        checkpoint,
        model=model,
        map_location=device,
        restore_rng=False,
    )

    dataset = build_validation_dataset(config, model_cfg)
    batch_size = int(config["training"].get("micro_batch_size", 1))
    max_batches = int(config["evaluation"].get("eval_steps", 0)) or None
    stats = evaluate_model(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        precision=precision,
        max_batches=max_batches,
    )
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Validation loss: %.6f", stats.loss)
    LOGGER.info("Perplexity: %.6f", stats.perplexity)
    return stats
