from __future__ import annotations

import json
from array import array
from itertools import islice
from pathlib import Path
from typing import Any

from src.data import mid_train_dataset
from src.data.mid_train_dataset import (
    MidTrainStreamPreparer,
    delete_mid_train_package,
    mid_train_package_path,
)
from src.data.streaming_mix import StreamingSource, iter_permuted_source_names


class FakeTokenizer:
    vocab_size = 256
    special_token_ids = {"bos_token": 1, "eos_token": 2}

    def encode(self, text: str) -> list[int]:
        return [(ord(char) % 200) + 3 for char in text]


class FakeByteTokenizer(FakeTokenizer):
    special_token_ids = {"eos_token": 2, "eot_token": 2}


def _config(tmp_path: Path, *, validation_ratio: float = 0.0) -> dict[str, Any]:
    return {
        "project": {"seed": 123},
        "dataset": {
            "smoltalk": "fake/smoltalk",
            "gsm8k": "fake/gsm8k",
            "mix_ratio": {"smoltalk": 1.0, "gsm8k": 0.0},
            "target_train_tokens": 64,
            "validation_ratio": validation_ratio,
            "cache_dir": str(tmp_path / "cache"),
            "processed_dir": str(tmp_path / "processed"),
            "stream_shuffle_buffer_size": 0,
        },
        "tokenizer": {
            "type": "superbpe",
            "vocab_size": 256,
            "save_dir": "fake-tokenizer",
        },
        "model": {"max_seq_len": 8},
    }


def _install_fake_dependencies(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    samples = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "two plus two"},
                {"role": "assistant", "content": "four"},
            ]
        },
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return list(samples)

    monkeypatch.setattr(mid_train_dataset, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(mid_train_dataset, "load_tokenizer", lambda _: FakeTokenizer())
    return calls


def test_mid_train_package_streams_limited_tokens_and_deletes(monkeypatch, tmp_path) -> None:
    calls = _install_fake_dependencies(monkeypatch)
    config = _config(tmp_path, validation_ratio=0.0)
    preparer = MidTrainStreamPreparer(config)

    package = preparer.write_train_package(package_index=0, target_tokens=32)

    assert package.path == mid_train_package_path(config, 0)
    assert package.tokens_written == 32
    assert package.path.exists()
    assert package.metadata_path.exists()
    assert calls
    assert all(call["kwargs"]["streaming"] is True for call in calls)

    values = array("H")
    with package.path.open("rb") as handle:
        values.fromfile(handle, 32)
    assert len(values) == 32

    delete_mid_train_package(config, 0)
    assert not package.path.exists()
    assert not package.metadata_path.exists()


def test_mid_train_supports_tokenizer_without_bos(monkeypatch, tmp_path) -> None:
    _install_fake_dependencies(monkeypatch)
    monkeypatch.setattr(mid_train_dataset, "load_tokenizer", lambda _: FakeByteTokenizer())
    config = _config(tmp_path, validation_ratio=0.0)

    preparer = MidTrainStreamPreparer(config)
    package = preparer.write_train_package(package_index=0, target_tokens=16)

    values = array("H")
    with package.path.open("rb") as handle:
        values.fromfile(handle, 16)
    assert values[0] != FakeByteTokenizer.special_token_ids["eos_token"]


def test_mid_train_validation_is_streamed(monkeypatch, tmp_path) -> None:
    calls = _install_fake_dependencies(monkeypatch)
    config = _config(tmp_path, validation_ratio=1.0)
    preparer = MidTrainStreamPreparer(config)

    val_path = preparer.prepare_validation()

    assert val_path.exists()
    assert val_path.name == "val_tokens.bin"
    assert calls
    assert all(call["kwargs"]["streaming"] is True for call in calls)


def test_mid_train_preflight_groups_splits_and_records_sources(monkeypatch, tmp_path) -> None:
    calls = _install_fake_dependencies(monkeypatch)
    config = _config(tmp_path, validation_ratio=0.0)
    config["dataset"]["target_train_tokens"] = 128
    config["dataset"]["source_mixture_window_size"] = 8
    config["dataset"]["sources"] = [
        {
            "name": "source_a",
            "id": "fake/smoltalk2",
            "config_name": "SFT",
            "split": "split_a",
            "target_tokens": 64,
            "format": "chat",
        },
        {
            "name": "source_b",
            "id": "fake/smoltalk2",
            "config_name": "SFT",
            "split": "split_b",
            "target_tokens": 64,
            "format": "chat",
        },
    ]

    preparer = MidTrainStreamPreparer(config)
    package = preparer.write_train_package(package_index=0, target_tokens=128)

    grouped_calls = [
        call
        for call in calls
        if call["args"] == ("fake/smoltalk2", "SFT")
        and call["kwargs"]["split"] == ["split_a", "split_b"]
    ]
    assert grouped_calls

    with package.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["dataset_name"] == "mid_train_weighted_streaming_mix"
    assert {source["name"] for source in metadata["sources"]} == {"source_a", "source_b"}
    assert set(metadata["source_stats"]) == {"source_a", "source_b"}


def test_source_order_is_permuted_inside_weight_window() -> None:
    sources = [
        StreamingSource("large", "fake/large", None, "train", 3, "chat"),
        StreamingSource("small", "fake/small", None, "train", 1, "chat"),
    ]

    names = list(
        islice(iter_permuted_source_names(sources, seed=7, window_size=4), 8)
    )

    assert names[:4].count("large") == 3
    assert names[:4].count("small") == 1
    assert names[:4] != ["large", "large", "large", "small"]
