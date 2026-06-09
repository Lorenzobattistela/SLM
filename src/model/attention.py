from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.rope import apply_rope, build_rope_cache

logger = logging.getLogger(__name__)
_LOGGED_ATTENTION_PATHS: set[str] = set()

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - depends on the installed PyTorch version.
    SDPBackend = None
    sdpa_kernel = None


ATTENTION_BACKEND_MANUAL = "manual"
ATTENTION_BACKEND_SDPA_AUTO = "sdpa_auto"
ATTENTION_BACKEND_SDPA_FLASH = "sdpa_flash"


@dataclass(frozen=True)
class AttentionOptimizationInfo:
    backend: str
    flash_requested: bool
    flash_available: bool
    enable_gqa: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _log_attention_path_once(path: str, message: str, *, warning: bool = False) -> None:
    if path in _LOGGED_ATTENTION_PATHS:
        return
    _LOGGED_ATTENTION_PATHS.add(path)
    log = logger.warning if warning else logger.info
    log(message)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.enable_gqa = self.group_size > 1
        self.attention_backend = self._select_attention_path(config)

    @staticmethod
    def _select_attention_path(config: ModelConfig) -> str:
        sdpa_available = hasattr(F, "scaled_dot_product_attention")
        if config.flash_attention and sdpa_available:
            _log_attention_path_once(
                "sdpa",
                "Configured PyTorch scaled_dot_product_attention for causal GQA attention; "
                "training will probe Flash SDPA on CUDA startup.",
            )
            return ATTENTION_BACKEND_SDPA_AUTO
        if config.flash_attention and not config.flash_attention_fallback:
            raise RuntimeError(
                "flash_attention=true but PyTorch scaled_dot_product_attention "
                "is unavailable and flash_attention_fallback=false"
            )
        if config.flash_attention:
            _log_attention_path_once(
                "manual_fallback",
                "PyTorch scaled_dot_product_attention is unavailable; "
                "falling back to manual causal GQA attention.",
                warning=True,
            )
        else:
            _log_attention_path_once(
                "manual_configured",
                "Using manual causal GQA attention because flash_attention=false.",
            )
        return ATTENTION_BACKEND_MANUAL

    def set_attention_backend(self, backend: str) -> None:
        if backend not in {
            ATTENTION_BACKEND_MANUAL,
            ATTENTION_BACKEND_SDPA_AUTO,
            ATTENTION_BACKEND_SDPA_FLASH,
        }:
            raise ValueError(f"Unsupported attention backend: {backend}")
        self.attention_backend = backend

    def _sdpa_context(self):
        if self.attention_backend != ATTENTION_BACKEND_SDPA_FLASH:
            return nullcontext()
        if sdpa_kernel is None or SDPBackend is None:
            raise RuntimeError("SDPA Flash backend forcing requires torch.nn.attention.sdpa_kernel")
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        dropout_p = self.config.dropout if self.training else 0.0
        with self._sdpa_context():
            if self.enable_gqa:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    is_causal=True,
                    enable_gqa=True,
                )
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = build_rope_cache(seq_len, self.head_dim, self.config.rope_base, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        sdpa_supported = not (self.enable_gqa and x.device.type != "cuda")
        if self.attention_backend != ATTENTION_BACKEND_MANUAL and sdpa_supported:
            attn = self._sdpa_attention(q, k, v)
        else:
            if self.enable_gqa:
                k = k.repeat_interleave(self.group_size, dim=1)
                v = v.repeat_interleave(self.group_size, dim=1)
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = attn @ v

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        return self.out_proj(attn)


def _dtype_for_attention_probe(device: torch.device, precision: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _probe_flash_sdpa(
    *,
    config: ModelConfig,
    device: torch.device,
    precision: str,
    batch_size: int,
    seq_len: int,
) -> tuple[bool, str]:
    if device.type != "cuda":
        return False, f"device={device.type} is not CUDA"
    if not hasattr(F, "scaled_dot_product_attention"):
        return False, "torch.nn.functional.scaled_dot_product_attention is unavailable"
    if sdpa_kernel is None or SDPBackend is None:
        return False, "torch.nn.attention.sdpa_kernel is unavailable"

    dtype = _dtype_for_attention_probe(device, precision)
    enable_gqa = config.n_heads != config.n_kv_heads
    try:
        with torch.no_grad():
            q = torch.empty(
                int(batch_size),
                config.n_heads,
                int(seq_len),
                config.head_dim,
                device=device,
                dtype=dtype,
            )
            k = torch.empty(
                int(batch_size),
                config.n_kv_heads,
                int(seq_len),
                config.head_dim,
                device=device,
                dtype=dtype,
            )
            v = torch.empty(
                int(batch_size),
                config.n_kv_heads,
                int(seq_len),
                config.head_dim,
                device=device,
                dtype=dtype,
            )
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                if enable_gqa:
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=float(config.dropout),
                        is_causal=True,
                        enable_gqa=True,
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=float(config.dropout),
                        is_causal=True,
                    )
            torch.cuda.synchronize(device)
            del out, q, k, v
    except Exception as exc:  # noqa: BLE001 - the backend reports support via runtime errors.
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"Flash SDPA probe passed dtype={dtype} enable_gqa={enable_gqa}"


def configure_attention_optimization(
    model: nn.Module,
    *,
    config: ModelConfig,
    device: torch.device,
    precision: str,
    batch_size: int,
    seq_len: int,
) -> AttentionOptimizationInfo:
    modules = [module for module in model.modules() if isinstance(module, CausalSelfAttention)]
    enable_gqa = config.n_heads != config.n_kv_heads
    sdpa_available = hasattr(F, "scaled_dot_product_attention")

    if not config.flash_attention:
        for module in modules:
            module.set_attention_backend(ATTENTION_BACKEND_MANUAL)
        return AttentionOptimizationInfo(
            backend=ATTENTION_BACKEND_MANUAL,
            flash_requested=False,
            flash_available=False,
            enable_gqa=enable_gqa,
            detail="flash_attention=false; using manual causal GQA attention",
        )

    if not sdpa_available:
        if not config.flash_attention_fallback:
            raise RuntimeError(
                "flash_attention=true but scaled_dot_product_attention is unavailable "
                "and flash_attention_fallback=false"
            )
        for module in modules:
            module.set_attention_backend(ATTENTION_BACKEND_MANUAL)
        return AttentionOptimizationInfo(
            backend=ATTENTION_BACKEND_MANUAL,
            flash_requested=True,
            flash_available=False,
            enable_gqa=enable_gqa,
            detail="scaled_dot_product_attention unavailable; using manual causal GQA attention",
        )

    if enable_gqa and device.type != "cuda":
        if not config.flash_attention_fallback:
            raise RuntimeError(
                "flash_attention=true with GQA requires CUDA for PyTorch SDPA and "
                "flash_attention_fallback=false"
            )
        for module in modules:
            module.set_attention_backend(ATTENTION_BACKEND_MANUAL)
        return AttentionOptimizationInfo(
            backend=ATTENTION_BACKEND_MANUAL,
            flash_requested=True,
            flash_available=False,
            enable_gqa=enable_gqa,
            detail=f"device={device.type}; using manual causal GQA attention",
        )

    flash_available, detail = _probe_flash_sdpa(
        config=config,
        device=device,
        precision=precision,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if flash_available:
        for module in modules:
            module.set_attention_backend(ATTENTION_BACKEND_SDPA_FLASH)
        return AttentionOptimizationInfo(
            backend=ATTENTION_BACKEND_SDPA_FLASH,
            flash_requested=True,
            flash_available=True,
            enable_gqa=enable_gqa,
            detail=detail,
        )

    if not config.flash_attention_fallback:
        raise RuntimeError(
            "flash_attention=true but Flash SDPA is unavailable for the configured "
            f"attention shape and flash_attention_fallback=false: {detail}"
        )

    for module in modules:
        module.set_attention_backend(ATTENTION_BACKEND_SDPA_AUTO)
    return AttentionOptimizationInfo(
        backend=ATTENTION_BACKEND_SDPA_AUTO,
        flash_requested=True,
        flash_available=False,
        enable_gqa=enable_gqa,
        detail=f"Flash SDPA probe failed; using PyTorch SDPA fallback: {detail}",
    )
