from __future__ import annotations

import logging
from contextlib import nullcontext

import torch

LOGGER = logging.getLogger(__name__)


def resolve_precision(requested_precision: str, device: torch.device) -> str:
    requested = requested_precision.lower()
    if requested in {"fp32", "float32"}:
        return "fp32"
    if requested in {"bf16", "bfloat16"}:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        LOGGER.warning(
            "Requested bf16 precision is not available on %s; using fp32.",
            device,
        )
        return "fp32"
    if requested in {"fp16", "float16"}:
        if device.type == "cuda":
            return "fp16"
        LOGGER.warning(
            "Requested fp16 precision is only supported for CUDA training here; using fp32."
        )
        return "fp32"
    raise ValueError(f"Unsupported training.precision: {requested_precision!r}")


def autocast_for_precision(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
