from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

from src.data.token_dataset import TokenBinWriter, write_metadata

TOKENIZE_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "pre-train" / "scripts" / "tokenize_dataset.py"
)
TOKENIZE_DATASET_SPEC = importlib.util.spec_from_file_location(
    "tokenize_dataset",
    TOKENIZE_DATASET_PATH,
)
assert TOKENIZE_DATASET_SPEC is not None
tokenize_dataset = importlib.util.module_from_spec(TOKENIZE_DATASET_SPEC)
assert TOKENIZE_DATASET_SPEC.loader is not None
sys.modules[TOKENIZE_DATASET_SPEC.name] = tokenize_dataset
TOKENIZE_DATASET_SPEC.loader.exec_module(tokenize_dataset)


def _requires_fork_process_pool() -> None:
    if mp.get_start_method() != "fork":
        pytest.skip("parallel monkeypatch test requires fork-based multiprocessing")


class FakeTokenizer:
    vocab_size = 64

    def encode(self, text: str, *, add_eos: bool) -> list[int]:
        token = len(text) % self.vocab_size
        ids = [max(1, token)]
        if add_eos:
            ids.append(2)
        return ids


def _dataset_cfg(tmp_path: Path) -> dict:
    return {
        "name": "fake-dataset",
        "split": "train",
        "text_column": "text",
        "streaming": True,
        "processed_dir": str(tmp_path / "processed"),
    }


def _tokenizer_cfg(tmp_path: Path) -> dict:
    return {
        "type": "superbpe",
        "save_dir": str(tmp_path / "tokenizer"),
        "append_eos": True,
    }


def _metadata(
    *,
    dataset_cfg: dict,
    tokenizer_cfg: dict,
    train_tokens: int,
    validation_tokens: int,
    target_train_tokens: int,
    target_validation_tokens: int,
    tokenize_workers: int,
) -> dict:
    return {
        "dataset_name": dataset_cfg["name"],
        "split": dataset_cfg["split"],
        "text_column": dataset_cfg["text_column"],
        "streaming": dataset_cfg["streaming"],
        "tokenizer_type": tokenizer_cfg["type"],
        "tokenizer_dir": str(Path(tokenizer_cfg["save_dir"])),
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "target_train_tokens": target_train_tokens,
        "target_validation_tokens": target_validation_tokens,
        "vocab_size": FakeTokenizer.vocab_size,
        "storage_dtype": "uint16",
        "train_tokens_path": "train_tokens.bin",
        "validation_tokens_path": "val_tokens.bin",
        "samples_seen": 10,
        "samples_tokenized": 10,
        "skipped_empty": 0,
        "append_eos": True,
        "validation_ratio": 0.25,
        "validation_salt": "salt",
        "tokenize_num_workers": tokenize_workers,
    }


def test_existing_tokenization_state_marks_complete_artifacts(tmp_path) -> None:
    dataset_cfg = _dataset_cfg(tmp_path)
    tokenizer_cfg = _tokenizer_cfg(tmp_path)
    processed_dir = Path(dataset_cfg["processed_dir"])
    train_path = processed_dir / "train_tokens.bin"
    val_path = processed_dir / "val_tokens.bin"
    metadata_path = processed_dir / "metadata.json"

    with TokenBinWriter(train_path, vocab_size=FakeTokenizer.vocab_size, target_tokens=8) as writer:
        writer.write(range(8))
    with TokenBinWriter(val_path, vocab_size=FakeTokenizer.vocab_size, target_tokens=4) as writer:
        writer.write(range(4))
    write_metadata(
        metadata_path,
        _metadata(
            dataset_cfg=dataset_cfg,
            tokenizer_cfg=tokenizer_cfg,
            train_tokens=8,
            validation_tokens=4,
            target_train_tokens=8,
            target_validation_tokens=4,
            tokenize_workers=2,
        ),
    )

    state = tokenize_dataset._existing_tokenization_state(
        metadata_path=metadata_path,
        train_path=train_path,
        val_path=val_path,
        dataset_cfg=dataset_cfg,
        tokenizer_cfg=tokenizer_cfg,
        tokenizer_vocab_size=FakeTokenizer.vocab_size,
        target_train_tokens=8,
        target_validation_tokens=4,
        validation_ratio=0.25,
        validation_salt="salt",
        tokenize_workers=2,
        append_eos=True,
    )

    assert state["complete"]
    assert not state["resumable"]


def test_parallel_tokenization_returns_aggregated_stats(monkeypatch, tmp_path) -> None:
    _requires_fork_process_pool()
    dataset_cfg = _dataset_cfg(tmp_path)
    tokenizer_cfg = _tokenizer_cfg(tmp_path)
    texts = [
        "train one",
        "val one",
        "train two",
        "val two",
        "train three",
        "train four",
        "val three",
        "train five",
    ]

    def iter_texts(_dataset_cfg, *, shard_index=None, num_shards=None):
        for index, text in enumerate(texts):
            if shard_index is None or index % num_shards == shard_index:
                yield text

    monkeypatch.setattr(tokenize_dataset, "load_tokenizer", lambda _cfg: FakeTokenizer())
    monkeypatch.setattr(tokenize_dataset, "iter_dataset_texts", iter_texts)
    monkeypatch.setattr(
        tokenize_dataset,
        "is_validation_text",
        lambda text, _ratio, _salt: text.startswith("val"),
    )

    processed_dir = Path(dataset_cfg["processed_dir"])
    stats = tokenize_dataset.tokenize_dataset_parallel(
        dataset_cfg=dataset_cfg,
        tokenizer_cfg=tokenizer_cfg,
        processed_dir=processed_dir,
        train_path=processed_dir / "train_tokens.bin",
        val_path=processed_dir / "val_tokens.bin",
        target_train_tokens=6,
        target_validation_tokens=4,
        validation_ratio=0.25,
        validation_salt="salt",
        num_workers=2,
        resume_metadata=None,
        append=False,
        logger=tokenize_dataset.logging.getLogger("test"),
    )

    assert stats["train_tokens"] == 6
    assert stats["validation_tokens"] == 4
    assert stats["samples_seen"] > 0
    assert stats["samples_tokenized"] > 0
    assert stats["tokenize_num_workers"] == 2
    assert len(stats["shards"]) == 2
