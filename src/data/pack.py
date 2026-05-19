from __future__ import annotations

from array import array
from pathlib import Path

import numpy as np

from src.data.shards import write_token_shard


class TokenPacker:
    def __init__(self, output_dir: str | Path, prefix: str, shard_tokens: int) -> None:
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.buffer = array("I")
        self.shard_index = 0
        self.total_tokens = 0
        self.total_documents = 0
        self.shard_paths: list[str] = []

    def add_document(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        if max(token_ids) > np.iinfo(np.uint16).max:
            raise ValueError("Token id exceeds uint16 storage range. Increase shard dtype first.")
        self.total_documents += 1
        self.total_tokens += len(token_ids)
        self.buffer.extend(token_ids)
        self._flush_full_shards()

    def _flush_full_shards(self) -> None:
        while len(self.buffer) >= self.shard_tokens:
            chunk = np.asarray(self.buffer[: self.shard_tokens], dtype=np.uint16)
            shard_path = write_token_shard(self.output_dir, self.prefix, self.shard_index, chunk)
            self.shard_paths.append(str(shard_path))
            del self.buffer[: self.shard_tokens]
            self.shard_index += 1

    def close(self) -> None:
        if self.buffer:
            chunk = np.asarray(self.buffer, dtype=np.uint16)
            shard_path = write_token_shard(self.output_dir, self.prefix, self.shard_index, chunk)
            self.shard_paths.append(str(shard_path))
            self.buffer = array("I")
            self.shard_index += 1
