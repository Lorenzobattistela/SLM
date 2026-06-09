from __future__ import annotations

import json
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def token_dtype_for_vocab(vocab_size: int) -> str:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return "uint16" if vocab_size <= 65_535 else "uint32"


def _array_typecode(dtype: str) -> str:
    if dtype == "uint16":
        return "H"
    if dtype == "uint32":
        return "I"
    raise ValueError(f"Unsupported token dtype: {dtype}")


def _numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "uint16":
        return np.dtype(np.uint16)
    if dtype == "uint32":
        return np.dtype(np.uint32)
    raise ValueError(f"Unsupported token dtype: {dtype}")


def token_file_token_count(path: str | Path, dtype: str) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    itemsize = _numpy_dtype(dtype).itemsize
    size_bytes = target.stat().st_size
    if size_bytes % itemsize != 0:
        raise ValueError(
            f"Token file {target} has size {size_bytes} bytes, which is not aligned "
            f"to dtype {dtype}."
        )
    return size_bytes // itemsize


@dataclass
class TokenWriteResult:
    written: int
    skipped: int


class TokenBinWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        vocab_size: int,
        target_tokens: int,
        append: bool = False,
        initial_tokens: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.target_tokens = target_tokens
        self.dtype = token_dtype_for_vocab(vocab_size)
        self._typecode = _array_typecode(self.dtype)
        if append:
            self.tokens_written = (
                token_file_token_count(self.path, self.dtype)
                if initial_tokens is None
                else int(initial_tokens)
            )
            mode = "ab"
        else:
            self.tokens_written = 0
            mode = "wb"
        self._handle = self.path.open(mode)

    @property
    def remaining(self) -> int:
        return max(0, self.target_tokens - self.tokens_written)

    @property
    def complete(self) -> bool:
        return self.tokens_written >= self.target_tokens

    def write(self, token_ids: Iterable[int]) -> TokenWriteResult:
        ids = list(token_ids)
        if not ids or self.complete:
            return TokenWriteResult(written=0, skipped=len(ids))

        writable = ids[: self.remaining]
        max_token_id = max(writable)
        if max_token_id >= 2 ** (16 if self.dtype == "uint16" else 32):
            raise ValueError(
                f"Token id {max_token_id} does not fit in configured storage dtype {self.dtype}"
            )

        values = array(self._typecode, writable)
        values.tofile(self._handle)
        self.tokens_written += len(values)
        return TokenWriteResult(written=len(values), skipped=max(0, len(ids) - len(values)))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TokenBinWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


class TokenBinDataset:
    """Memory-mapped token dataset backed by the .bin files from tokenize_dataset.py."""

    def __init__(
        self,
        path: str | Path,
        *,
        block_size: int,
        vocab_size: int,
        dtype: str | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Token file not found: {self.path}. Run scripts/tokenize_dataset.py first."
            )
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.dtype = dtype or token_dtype_for_vocab(self.vocab_size)
        self.tokens = np.memmap(self.path, dtype=_numpy_dtype(self.dtype), mode="r")
        self.num_tokens = int(self.tokens.shape[0])
        self.num_positions = max(0, self.num_tokens - self.block_size)
        if self.num_positions <= 0:
            raise ValueError(
                f"Token file {self.path} has {self.num_tokens} tokens, which is too small "
                f"for block_size={self.block_size}."
            )

    def __len__(self) -> int:
        return self.num_positions

    def _rank_position_count(self, *, rank: int, world_size: int) -> int:
        if rank >= self.num_positions:
            return 0
        return ((self.num_positions - 1 - rank) // world_size) + 1

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        rng: np.random.Generator,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = int(rank)
        world_size = max(1, int(world_size))
        rank_positions = self._rank_position_count(rank=rank, world_size=world_size)
        if rank_positions <= 0:
            raise ValueError(
                f"Rank {rank} has no sample positions in {self.path}; "
                f"num_positions={self.num_positions}, world_size={world_size}."
            )

        offsets = rng.integers(0, rank_positions, size=int(batch_size), endpoint=False)
        starts = rank + offsets * world_size
        windows = np.stack(
            [
                np.asarray(
                    self.tokens[int(start) : int(start) + self.block_size + 1],
                    dtype=np.int64,
                )
                for start in starts
            ]
        )
        inputs = torch.tensor(windows[:, :-1], dtype=torch.long, device=device)
        targets = torch.tensor(windows[:, 1:], dtype=torch.long, device=device)
        return inputs, targets

    def iter_batches(
        self,
        batch_size: int,
        device: torch.device,
        *,
        max_batches: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        emitted = 0
        rank = int(rank)
        world_size = max(1, int(world_size))
        batch_windows: list[np.ndarray] = []

        for start in range(rank, self.num_positions, self.block_size * world_size):
            window = np.asarray(
                self.tokens[start : start + self.block_size + 1],
                dtype=np.int64,
            )
            if window.shape[0] != self.block_size + 1:
                continue
            batch_windows.append(window)
            if len(batch_windows) == batch_size:
                stacked = np.stack(batch_windows)
                yield (
                    torch.tensor(stacked[:, :-1], dtype=torch.long, device=device),
                    torch.tensor(stacked[:, 1:], dtype=torch.long, device=device),
                )
                emitted += 1
                if max_batches is not None and emitted >= max_batches:
                    return
                batch_windows.clear()

        if batch_windows:
            stacked = np.stack(batch_windows)
            yield (
                torch.tensor(stacked[:, :-1], dtype=torch.long, device=device),
                torch.tensor(stacked[:, 1:], dtype=torch.long, device=device),
            )
