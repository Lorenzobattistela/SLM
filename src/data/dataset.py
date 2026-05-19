from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.shards import discover_shards


class TokenShardDataset:
    def __init__(self, root_dir: str | Path, block_size: int) -> None:
        self.root_dir = Path(root_dir)
        self.block_size = block_size
        self.shard_paths = discover_shards(self.root_dir)
        if not self.shard_paths:
            raise FileNotFoundError(f"No token shards found under {self.root_dir}")
        self.shards = [np.load(path, mmap_mode="r") for path in self.shard_paths]
        self.usable_lengths = np.asarray(
            [max(0, len(shard) - block_size - 1) for shard in self.shards],
            dtype=np.int64,
        )
        self.total_positions = int(self.usable_lengths.sum())
        if self.total_positions <= 0:
            raise ValueError(
                f"Prepared data in {self.root_dir} is too small for block size {self.block_size}"
            )
        self.sample_weights = self.usable_lengths / self.total_positions

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shard_indices = rng.choice(
            len(self.shards),
            size=batch_size,
            replace=True,
            p=self.sample_weights,
        )
        inputs = []
        targets = []
        for shard_index in shard_indices:
            shard = self.shards[int(shard_index)]
            max_start = len(shard) - self.block_size - 1
            start = int(rng.integers(0, max_start + 1))
            window = np.asarray(shard[start : start + self.block_size + 1], dtype=np.int64)
            inputs.append(window[:-1])
            targets.append(window[1:])
        x = torch.tensor(np.stack(inputs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(targets), dtype=torch.long, device=device)
        return x, y

    def iter_eval_batches(
        self,
        batch_size: int,
        device: torch.device,
        max_batches: int | None = None,
    ):
        emitted = 0
        batch_inputs: list[np.ndarray] = []
        batch_targets: list[np.ndarray] = []

        for shard in self.shards:
            limit = len(shard) - self.block_size - 1
            for start in range(0, limit + 1, self.block_size):
                window = np.asarray(shard[start : start + self.block_size + 1], dtype=np.int64)
                batch_inputs.append(window[:-1])
                batch_targets.append(window[1:])
                if len(batch_inputs) == batch_size:
                    x = torch.tensor(np.stack(batch_inputs), dtype=torch.long, device=device)
                    y = torch.tensor(np.stack(batch_targets), dtype=torch.long, device=device)
                    yield x, y
                    emitted += 1
                    if max_batches is not None and emitted >= max_batches:
                        return
                    batch_inputs.clear()
                    batch_targets.clear()

        if batch_inputs:
            x = torch.tensor(np.stack(batch_inputs), dtype=torch.long, device=device)
            y = torch.tensor(np.stack(batch_targets), dtype=torch.long, device=device)
            yield x, y
