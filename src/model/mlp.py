from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
