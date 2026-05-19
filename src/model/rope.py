from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs)[None, None, :, :]
    sin = torch.sin(freqs)[None, None, :, :]
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    sin = sin.repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    return (x * cos) + (_rotate_half(x) * sin)
