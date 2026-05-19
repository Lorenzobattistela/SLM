from __future__ import annotations

import math

import torch

from src.data.dataset import TokenShardDataset
from src.utils import autocast_context


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    dataset: TokenShardDataset,
    batch_size: int,
    device: torch.device,
    precision: str,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in dataset.iter_eval_batches(batch_size, device, max_batches=max_batches):
        with autocast_context(device, precision):
            _, loss = model(inputs, targets)
        batch_tokens = inputs.numel()
        total_loss += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens

    if total_tokens == 0:
        raise ValueError("Validation dataset produced zero tokens.")

    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss)
