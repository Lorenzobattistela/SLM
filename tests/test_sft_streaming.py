from __future__ import annotations

from array import array
from pathlib import Path
from typing import Any

from src.finetuning import sft_dataset
from src.finetuning.sft_dataset import (
    SFTBinDataset,
    SFTStreamPreparer,
    delete_sft_package,
    sft_package_labels_path,
    sft_package_tokens_path,
    tokenize_sft_conversation,
)


class FakeTokenizer:
    vocab_size = 256
    special_token_ids = {"bos_token": 1, "eos_token": 2}

    def encode(self, text: str) -> list[int]:
        return [(ord(char) % 200) + 3 for char in text]


class FailsOnceIterable:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples
        self.failed = False

    def __iter__(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError("408 Client Error: Request Time-out for url")
        yield from self.samples


def _config(tmp_path: Path, *, validation_ratio: float = 0.0) -> dict[str, Any]:
    return {
        "project": {"seed": 123},
        "dataset": {
            "sft_dataset": "fake/smoltalk",
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
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "two plus two"},
                {"role": "assistant", "content": "four"},
            ]
        },
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return list(samples)

    monkeypatch.setattr(sft_dataset, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(sft_dataset, "load_tokenizer", lambda _: FakeTokenizer())
    return calls


def test_sft_package_streams_limited_tokens_labels_and_deletes(monkeypatch, tmp_path) -> None:
    calls = _install_fake_dependencies(monkeypatch)
    config = _config(tmp_path, validation_ratio=0.0)
    preparer = SFTStreamPreparer(config)

    package = preparer.write_train_package(package_index=0, target_tokens=32)

    assert package.tokens_path == sft_package_tokens_path(config, 0)
    assert package.labels_path == sft_package_labels_path(config, 0)
    assert package.tokens_written == 32
    assert package.tokens_path.exists()
    assert package.labels_path.exists()
    assert package.metadata_path.exists()
    assert calls
    assert all(call["kwargs"]["streaming"] is True for call in calls)

    token_values = array("H")
    label_values = array("i")
    with package.tokens_path.open("rb") as handle:
        token_values.fromfile(handle, 32)
    with package.labels_path.open("rb") as handle:
        label_values.fromfile(handle, 32)
    assert len(token_values) == 32
    assert len(label_values) == 32

    dataset = SFTBinDataset(package.tokens_path, package.labels_path, block_size=8, vocab_size=256)
    dataset.close()
    delete_sft_package(config, 0)
    assert not package.tokens_path.exists()
    assert not package.labels_path.exists()
    assert not package.metadata_path.exists()


def test_sft_validation_is_streamed(monkeypatch, tmp_path) -> None:
    calls = _install_fake_dependencies(monkeypatch)
    config = _config(tmp_path, validation_ratio=1.0)
    preparer = SFTStreamPreparer(config)

    val_path = preparer.prepare_validation()

    assert val_path.exists()
    assert val_path.name == "val_tokens.bin"
    assert (val_path.parent / "val_labels.bin").exists()
    assert calls
    assert all(call["kwargs"]["streaming"] is True for call in calls)


def test_sft_source_validation_retries_transient_stream_error(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, Any]] = []
    samples = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if len(calls) == 2:
            return FailsOnceIterable(samples)
        return list(samples)

    monkeypatch.setattr(sft_dataset, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(sft_dataset, "load_tokenizer", lambda _: FakeTokenizer())

    config = _config(tmp_path, validation_ratio=0.0)
    config["dataset"]["stream_error_retry_backoff_seconds"] = 0

    SFTStreamPreparer(config)

    assert len(calls) >= 3


def test_tokenize_sft_conversation_supports_missing_bos() -> None:
    tokens, labels = tokenize_sft_conversation(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
        FakeTokenizer(),
        bos_id=None,
        eos_id=2,
    )

    assert tokens[0] != 1
    assert tokens[-1] == 2
    assert labels[-1] == 2
    assert len(tokens) == len(labels)
