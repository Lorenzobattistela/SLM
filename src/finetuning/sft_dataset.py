from __future__ import annotations

import json
import logging
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from datasets import load_dataset
from src.config.loader import resolve_project_path
from src.data.streaming_mix import (
    StreamingSource,
    belongs_to_partition,
    belongs_to_source_range,
    iter_streaming_source_with_retries,
    iter_permuted_source_names,
    parse_streaming_sources,
    preload_streaming_sources,
    source_metadata,
    source_signature,
    stream_retry_options,
    validate_source_token_budget,
)
from src.data.token_dataset import TokenBinWriter, write_metadata
from src.tokenizer import load_tokenizer

LOGGER = logging.getLogger(__name__)
_DATASET_NAME = "sft_weighted_streaming_mix"


@dataclass(frozen=True)
class SFTPackage:
    index: int
    tokens_path: Path
    labels_path: Path
    metadata_path: Path
    tokens_written: int
    dtype: str


def sft_package_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["dataset"]["processed_dir"]) / "streaming_train_packages"


def sft_package_tokens_path(config: dict[str, Any], package_index: int) -> Path:
    return sft_package_dir(config) / f"train_package_{package_index:06d}_tokens.bin"


def sft_package_labels_path(config: dict[str, Any], package_index: int) -> Path:
    return sft_package_dir(config) / f"train_package_{package_index:06d}_labels.bin"


def sft_package_metadata_path(config: dict[str, Any], package_index: int) -> Path:
    return sft_package_dir(config) / f"train_package_{package_index:06d}.metadata.json"


def delete_sft_package(config: dict[str, Any], package_index: int) -> None:
    for path in (
        sft_package_tokens_path(config, package_index),
        sft_package_labels_path(config, package_index),
        sft_package_metadata_path(config, package_index),
    ):
        if path.exists():
            path.unlink()
            LOGGER.info("Deleted processed SFT package: %s", path)


class LabelsBinWriter:
    """Binary file writer for target labels storing signed 32-bit integers."""

    def __init__(self, path: str | Path, target_tokens: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.target_tokens = int(target_tokens)
        self.tokens_written = 0
        self.dtype = "int32"
        self._handle = self.path.open("wb")

    @property
    def remaining(self) -> int:
        return max(0, self.target_tokens - self.tokens_written)

    @property
    def complete(self) -> bool:
        return self.tokens_written >= self.target_tokens

    def write(self, labels: list[int]) -> None:
        if not labels or self.complete:
            return
        writable = labels[: self.remaining]
        values = array("i", writable)  # signed 32-bit integers
        values.tofile(self._handle)
        self.tokens_written += len(values)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> LabelsBinWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class SFTBinDataset:
    """Memory-mapped dataset containing tokens (inputs) and target labels (prompt-masked)."""

    def __init__(
        self,
        tokens_path: str | Path,
        labels_path: str | Path,
        *,
        block_size: int,
        vocab_size: int,
        tokens_dtype: str = "uint16",
        labels_dtype: str = "int32",
    ) -> None:
        self.tokens_path = Path(tokens_path)
        self.labels_path = Path(labels_path)

        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens bin file not found: {self.tokens_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels bin file not found: {self.labels_path}")

        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.tokens_dtype = tokens_dtype
        self.labels_dtype = labels_dtype

        self.tokens = np.memmap(self.tokens_path, dtype=self.tokens_dtype, mode="r")
        self.labels = np.memmap(self.labels_path, dtype=self.labels_dtype, mode="r")

        if len(self.tokens) != len(self.labels):
            raise ValueError(
                f"Length mismatch: tokens ({len(self.tokens)}) vs labels ({len(self.labels)})"
            )

        self.num_tokens = int(self.tokens.shape[0])
        self.num_positions = max(0, self.num_tokens - self.block_size)

        if self.num_positions <= 0:
            raise ValueError(
                f"Token file {self.tokens_path} has {self.num_tokens} tokens, which is too small "
                f"for block_size={self.block_size}."
            )

    def close(self) -> None:
        for values in (self.tokens, self.labels):
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> SFTBinDataset:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

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
                f"Rank {rank} has no sample positions; "
                f"num_positions={self.num_positions}, world_size={world_size}."
            )

        offsets = rng.integers(0, rank_positions, size=int(batch_size), endpoint=False)
        starts = rank + offsets * world_size

        input_windows = []
        target_windows = []
        for start in starts:
            input_windows.append(
                np.asarray(self.tokens[int(start) : int(start) + self.block_size], dtype=np.int64)
            )
            target_windows.append(
                np.asarray(self.labels[int(start) : int(start) + self.block_size], dtype=np.int64)
            )

        inputs = torch.tensor(np.stack(input_windows), dtype=torch.long, device=device)
        targets = torch.tensor(np.stack(target_windows), dtype=torch.long, device=device)
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
        batch_inputs = []
        batch_targets = []

        for start in range(rank, self.num_positions, self.block_size * world_size):
            inp_window = np.asarray(self.tokens[start : start + self.block_size], dtype=np.int64)
            tar_window = np.asarray(self.labels[start : start + self.block_size], dtype=np.int64)

            if inp_window.shape[0] != self.block_size:
                continue
            batch_inputs.append(inp_window)
            batch_targets.append(tar_window)

            if len(batch_inputs) == batch_size:
                yield (
                    torch.tensor(np.stack(batch_inputs), dtype=torch.long, device=device),
                    torch.tensor(np.stack(batch_targets), dtype=torch.long, device=device),
                )
                emitted += 1
                if max_batches is not None and emitted >= max_batches:
                    return
                batch_inputs.clear()
                batch_targets.clear()

        if batch_inputs:
            yield (
                torch.tensor(np.stack(batch_inputs), dtype=torch.long, device=device),
                torch.tensor(np.stack(batch_targets), dtype=torch.long, device=device),
            )


def tokenize_sft_conversation(
    messages: list[dict[str, str]], tokenizer: Any, bos_id: int | None, eos_id: int
) -> tuple[list[int], list[int]]:
    """Tokenize conversation with loss masking on user/system prompts."""
    role_map = {"user": "User", "assistant": "Assistant", "system": "System"}

    tokens = []
    labels = []

    if bos_id is not None:
        tokens.append(bos_id)
        labels.append(-100)

    for i, msg in enumerate(messages):
        role = role_map.get(msg["role"], msg["role"].capitalize())
        content = msg["content"]

        prefix = "\n" if i > 0 else ""
        turn_text = f"{prefix}{role}: {content}"
        turn_tokens = tokenizer.encode(turn_text)

        if msg["role"] == "assistant":
            # Assistant response: compute loss
            tokens.extend(turn_tokens)
            labels.extend(turn_tokens)
        else:
            # User or system prompt: mask loss
            tokens.extend(turn_tokens)
            labels.extend([-100] * len(turn_tokens))

    # Append eos_id, calculated in loss
    tokens.append(eos_id)
    labels.append(eos_id)

    return tokens, labels


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def _messages_from_sample(
    sample: dict[str, Any],
    source: StreamingSource,
) -> list[dict[str, str]]:
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = str(message.get("content", "")).strip()
            if role and content:
                normalized.append({"role": role, "content": content})
        return normalized

    if source.format == "gsm8k" or {"question", "answer"}.issubset(sample):
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        if question and answer:
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

    text = str(sample.get("text", "")).strip()
    if text:
        return [{"role": "assistant", "content": text}]
    return []


def _legacy_sources(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    subset = dataset_cfg.get("sft_subset", "all")
    return [
        {
            "name": "sft",
            "id": dataset_cfg.get("sft_dataset", "HuggingFaceTB/smoltalk"),
            "config_name": subset,
            "split": dataset_cfg.get("split", "train"),
            "target_tokens": int(dataset_cfg["target_train_tokens"]),
            "format": "chat",
        }
    ]


class SFTStreamPreparer:
    """Streams, prompt-masks, and writes only the next SFT package to disk."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dataset_cfg = config["dataset"]
        self.seed = int(config["project"].get("seed", 42))
        self.validation_ratio = float(self.dataset_cfg.get("validation_ratio", 0.05))
        self.cache_dir = resolve_project_path(self.dataset_cfg.get("cache_dir", "./data/cache"))
        self.shuffle_buffer_size = int(self.dataset_cfg.get("stream_shuffle_buffer_size", 10_000))
        self.mixture_window_size = int(self.dataset_cfg.get("source_mixture_window_size", 1_024))
        self.stream_error_retries, self.stream_error_retry_backoff_seconds = stream_retry_options(
            self.dataset_cfg
        )
        self.sources = parse_streaming_sources(
            self.dataset_cfg,
            fallback_sources=_legacy_sources(self.dataset_cfg),
        )
        validate_source_token_budget(
            self.sources,
            target_train_tokens=int(self.dataset_cfg["target_train_tokens"]),
        )
        self.source_by_name = {source.name: source for source in self.sources}
        self.source_signature = source_signature(self.sources)
        if bool(self.dataset_cfg.get("validate_sources_on_start", True)):
            self.validate_sources()

        self.tokenizer = load_tokenizer(config["tokenizer"])
        self.bos_id = self.tokenizer.special_token_ids.get("bos_token")
        self.eos_id = self.tokenizer.special_token_ids.get("eos_token")
        if self.eos_id is None:
            raise ValueError(
                f"Tokenizer is missing eos_token: {self.tokenizer.special_token_ids}"
            )
        if self.bos_id is None:
            LOGGER.info(
                "Tokenizer does not expose bos_token; SFT sequences will use eos only."
            )

        self._train_pairs = self._tokenized_partition("train", self.seed + 101)

    def validate_sources(self) -> None:
        preload_streaming_sources(
            self.sources,
            cache_dir=self.cache_dir,
            load_dataset_fn=load_dataset,
            max_retries=self.stream_error_retries,
            retry_backoff_seconds=self.stream_error_retry_backoff_seconds,
        )
        for index, source in enumerate(self.sources):
            dataset = iter_streaming_source_with_retries(
                source,
                cache_dir=self.cache_dir,
                seed=self.seed + 17 + index,
                shuffle_buffer_size=self.shuffle_buffer_size,
                load_dataset_fn=load_dataset,
                max_retries=self.stream_error_retries,
                retry_backoff_seconds=self.stream_error_retry_backoff_seconds,
                operation="validating SFT stream",
            )
            sample = next(
                sample
                for sample in dataset
                if isinstance(sample, dict) and belongs_to_source_range(sample, source)
            )
            messages = _messages_from_sample(sample, source)
            if not messages:
                raise ValueError(
                    f"SFT source {source.name!r} loaded but did not produce messages."
                )
            LOGGER.info(
                "Validated SFT source=%s dataset=%s config=%s split=%s",
                source.name,
                source.dataset_id,
                source.config_name,
                source.split,
            )

    def _source_stream(
        self,
        source: StreamingSource,
        partition: str,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        epoch = 0
        while True:
            emitted = False
            dataset = iter_streaming_source_with_retries(
                source,
                cache_dir=self.cache_dir,
                seed=seed + epoch,
                shuffle_buffer_size=self.shuffle_buffer_size,
                load_dataset_fn=load_dataset,
                max_retries=self.stream_error_retries,
                retry_backoff_seconds=self.stream_error_retry_backoff_seconds,
                operation=f"streaming SFT {partition} data",
            )
            for sample in dataset:
                if not isinstance(sample, dict):
                    continue
                if not belongs_to_source_range(sample, source):
                    continue
                if not belongs_to_partition(
                    sample,
                    partition=partition,
                    validation_ratio=self.validation_ratio,
                ):
                    continue
                emitted = True
                yield sample

            if not emitted:
                raise ValueError(
                    f"No {partition} samples were produced by SFT source {source.name!r}. "
                    "Adjust dataset.validation_ratio, source sample_range, or source configuration."
                )
            epoch += 1

    def _tokenized_partition(
        self,
        partition: str,
        seed: int,
    ) -> Iterator[tuple[str, list[int], list[int]]]:
        source_streams = {
            source.name: self._source_stream(source, partition, seed + 17 * (index + 1))
            for index, source in enumerate(self.sources)
        }
        source_names = iter_permuted_source_names(
            self.sources,
            seed=seed,
            window_size=self.mixture_window_size,
        )
        while True:
            source_name = next(source_names)
            source = self.source_by_name[source_name]
            sample = next(source_streams[source_name])
            messages = _messages_from_sample(sample, source)
            if not messages:
                continue
            tokens, labels = tokenize_sft_conversation(
                messages,
                self.tokenizer,
                self.bos_id,
                self.eos_id,
            )
            if len(tokens) <= 1 or len(tokens) != len(labels):
                continue
            yield source.name, tokens[:-1], labels[1:]

    def _write_pairs(
        self,
        *,
        token_label_pairs: Iterator[tuple[str, list[int], list[int]]],
        tokens_path: Path,
        labels_path: Path,
        target_tokens: int,
    ) -> tuple[int, str, dict[str, dict[str, int]]]:
        source_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"samples": 0, "tokens": 0}
        )
        with (
            TokenBinWriter(
                tokens_path,
                vocab_size=self.tokenizer.vocab_size,
                target_tokens=target_tokens,
            ) as tokens_writer,
            LabelsBinWriter(labels_path, target_tokens=target_tokens) as labels_writer,
        ):
            for source_name, tokens, labels in token_label_pairs:
                result = tokens_writer.write(tokens)
                labels_writer.write(labels)
                if result.written > 0:
                    source_stats[source_name]["samples"] += 1
                    source_stats[source_name]["tokens"] += result.written
                if tokens_writer.complete:
                    break

            if tokens_writer.tokens_written != labels_writer.tokens_written:
                raise ValueError(
                    "SFT token/label package length mismatch: "
                    f"tokens={tokens_writer.tokens_written}, labels={labels_writer.tokens_written}"
                )
            return tokens_writer.tokens_written, tokens_writer.dtype, dict(source_stats)

    def prepare_validation(self, *, force: bool = False) -> Path:
        processed_dir = resolve_project_path(self.dataset_cfg["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = processed_dir / "metadata.json"
        val_tokens_path = processed_dir / "val_tokens.bin"
        val_labels_path = processed_dir / "val_labels.bin"

        target_train_tokens = int(self.dataset_cfg["target_train_tokens"])
        target_validation_tokens = int(target_train_tokens * self.validation_ratio)
        if target_validation_tokens <= 0:
            raise ValueError("SFT validation requires dataset.validation_ratio > 0.")

        metadata = _load_metadata(processed_dir)
        if (
            not force
            and val_tokens_path.exists()
            and val_labels_path.exists()
            and metadata.get("dataset_name") == _DATASET_NAME
            and metadata.get("source_signature") == self.source_signature
            and metadata.get("target_validation_tokens") == target_validation_tokens
            and metadata.get("tokenizer_type") == self.config["tokenizer"]["type"]
        ):
            LOGGER.info("Reusing streaming SFT validation data at %s", processed_dir)
            return val_tokens_path

        LOGGER.info("Streaming SFT validation data to %s", processed_dir)
        actual_val_tokens, storage_dtype, source_stats = self._write_pairs(
            token_label_pairs=self._tokenized_partition("validation", self.seed + 307),
            tokens_path=val_tokens_path,
            labels_path=val_labels_path,
            target_tokens=target_validation_tokens,
        )

        metadata = {
            "dataset_name": _DATASET_NAME,
            "streaming": True,
            "source_signature": self.source_signature,
            "sources": source_metadata(self.sources),
            "validation_source_stats": source_stats,
            "tokenizer_type": self.config["tokenizer"]["type"],
            "tokenizer_dir": str(Path(self.config["tokenizer"]["save_dir"])),
            "target_train_tokens": target_train_tokens,
            "target_validation_tokens": target_validation_tokens,
            "validation_tokens": actual_val_tokens,
            "validation_ratio": self.validation_ratio,
            "vocab_size": self.tokenizer.vocab_size,
            "storage_dtype": storage_dtype,
            "validation_tokens_path": str(val_tokens_path),
            "validation_labels_path": str(val_labels_path),
            "train_package_dir": str(sft_package_dir(self.config)),
        }
        write_metadata(metadata_path, metadata)
        return val_tokens_path

    def write_train_package(self, package_index: int, target_tokens: int) -> SFTPackage:
        package_dir = sft_package_dir(self.config)
        package_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = sft_package_tokens_path(self.config, package_index)
        labels_path = sft_package_labels_path(self.config, package_index)
        metadata_path = sft_package_metadata_path(self.config, package_index)
        for path in (tokens_path, labels_path, metadata_path):
            if path.exists():
                path.unlink()

        LOGGER.info(
            "Streaming SFT package %s to %s with target_tokens=%s",
            package_index,
            package_dir,
            target_tokens,
        )
        tokens_written, storage_dtype, source_stats = self._write_pairs(
            token_label_pairs=self._train_pairs,
            tokens_path=tokens_path,
            labels_path=labels_path,
            target_tokens=int(target_tokens),
        )

        metadata = {
            "dataset_name": _DATASET_NAME,
            "streaming": True,
            "source_signature": self.source_signature,
            "sources": source_metadata(self.sources),
            "source_stats": source_stats,
            "package_index": int(package_index),
            "package_tokens": tokens_written,
            "target_package_tokens": int(target_tokens),
            "tokenizer_type": self.config["tokenizer"]["type"],
            "tokenizer_dir": str(Path(self.config["tokenizer"]["save_dir"])),
            "vocab_size": self.tokenizer.vocab_size,
            "storage_dtype": storage_dtype,
            "tokens_path": str(tokens_path),
            "labels_path": str(labels_path),
        }
        write_metadata(metadata_path, metadata)
        return SFTPackage(
            index=int(package_index),
            tokens_path=tokens_path,
            labels_path=labels_path,
            metadata_path=metadata_path,
            tokens_written=tokens_written,
            dtype=storage_dtype,
        )


def prepare_sft_data(config: dict[str, Any], force: bool = False) -> None:
    """Prepare reusable validation data and the first streaming SFT train package."""

    preparer = SFTStreamPreparer(config)
    preparer.prepare_validation(force=force)
    target_tokens = int(
        config["dataset"].get(
            "streaming_package_tokens",
            config["dataset"].get("target_train_tokens", 0),
        )
    )
    if target_tokens > 0:
        preparer.write_train_package(0, target_tokens)
