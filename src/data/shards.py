from __future__ import annotations

from pathlib import Path

import numpy as np


def write_token_shard(
    output_dir: str | Path,
    prefix: str,
    shard_index: int,
    token_ids: np.ndarray,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shard_path = target_dir / f"{prefix}-{shard_index:06d}.npy"
    np.save(shard_path, token_ids)
    return shard_path


def discover_shards(root_dir: str | Path) -> list[Path]:
    return sorted(Path(root_dir).glob("*.npy"))
