from __future__ import annotations

import numpy as np

from src.data.pack import TokenPacker


def test_token_packer_writes_expected_token_count(tmp_path) -> None:
    packer = TokenPacker(tmp_path, prefix="train", shard_tokens=4)
    packer.add_document([1, 2, 3])
    packer.add_document([4, 5, 6, 7])
    packer.close()

    shards = sorted(tmp_path.glob("*.npy"))
    assert len(shards) == 2
    total_tokens = sum(np.load(path).shape[0] for path in shards)
    assert total_tokens == 7
