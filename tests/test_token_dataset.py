from __future__ import annotations

from array import array

import numpy as np
import torch

from src.data.token_dataset import TokenBinDataset, TokenBinWriter, token_dtype_for_vocab


def test_token_dtype_for_vocab_selects_compact_storage() -> None:
    assert token_dtype_for_vocab(50_000) == "uint16"
    assert token_dtype_for_vocab(100_000) == "uint32"


def test_token_bin_writer_respects_target_token_limit(tmp_path) -> None:
    output_path = tmp_path / "tokens.bin"
    with TokenBinWriter(output_path, vocab_size=50_000, target_tokens=5) as writer:
        first = writer.write([1, 2, 3])
        second = writer.write([4, 5, 6, 7])

    assert first.written == 3
    assert first.skipped == 0
    assert second.written == 2
    assert second.skipped == 2

    values = array("H")
    with output_path.open("rb") as handle:
        values.fromfile(handle, 5)

    assert list(values) == [1, 2, 3, 4, 5]


def test_token_bin_dataset_samples_shifted_windows(tmp_path) -> None:
    output_path = tmp_path / "tokens.bin"
    with TokenBinWriter(output_path, vocab_size=128, target_tokens=10) as writer:
        writer.write(range(10))

    dataset = TokenBinDataset(output_path, block_size=4, vocab_size=128)
    inputs, targets = dataset.sample_batch(
        2,
        torch.device("cpu"),
        np.random.default_rng(0),
    )

    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
