from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
_DATASET_NAME = "mid_train_weighted_streaming_mix"


@dataclass(frozen=True)
class MidTrainPackage:
    index: int
    path: Path
    metadata_path: Path
    tokens_written: int
    dtype: str


def mid_train_package_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["dataset"]["processed_dir"]) / "streaming_train_packages"


def mid_train_package_path(config: dict[str, Any], package_index: int) -> Path:
    return mid_train_package_dir(config) / f"train_package_{package_index:06d}.bin"


def mid_train_package_metadata_path(config: dict[str, Any], package_index: int) -> Path:
    return mid_train_package_path(config, package_index).with_suffix(".metadata.json")


def delete_mid_train_package(config: dict[str, Any], package_index: int) -> None:
    for path in (
        mid_train_package_path(config, package_index),
        mid_train_package_metadata_path(config, package_index),
    ):
        if path.exists():
            path.unlink()
            LOGGER.info("Deleted processed mid-training package: %s", path)


def _format_conversation(sample: dict[str, Any]) -> str:
    role_map = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
    }
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return ""

    formatted_turns: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        formatted_role = role_map.get(role, role.capitalize() or "User")
        formatted_turns.append(f"{formatted_role}: {content}")
    return "\n".join(formatted_turns)


def _format_gsm8k(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    if not question or not answer:
        return ""
    return f"User: {question}\nAssistant: {answer}"


def _format_sample(sample: dict[str, Any], source: StreamingSource) -> str:
    if source.format in {"chat", "conversation"} or "messages" in sample:
        return _format_conversation(sample)
    if source.format == "gsm8k" or {"question", "answer"}.issubset(sample):
        return _format_gsm8k(sample)
    text = str(sample.get("text", "")).strip()
    return text


def _legacy_sources(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    target_train_tokens = int(dataset_cfg["target_train_tokens"])
    mix_ratio = dataset_cfg.get("mix_ratio", {"smoltalk": 0.7, "gsm8k": 0.3})
    smoltalk_weight = float(mix_ratio.get("smoltalk", 0.7))
    gsm8k_weight = float(mix_ratio.get("gsm8k", 0.3))
    total_weight = smoltalk_weight + gsm8k_weight
    if total_weight <= 0:
        raise ValueError("dataset.mix_ratio must assign positive weight to at least one source.")

    sources: list[dict[str, Any]] = []
    if smoltalk_weight > 0:
        sources.append(
            {
                "name": "smoltalk",
                "id": dataset_cfg.get("smoltalk", "HuggingFaceTB/smoltalk"),
                "config_name": dataset_cfg.get("smoltalk_subset", "all"),
                "split": dataset_cfg.get("split", "train"),
                "target_tokens": round(target_train_tokens * smoltalk_weight / total_weight),
                "format": "chat",
            }
        )
    if gsm8k_weight > 0:
        sources.append(
            {
                "name": "gsm8k",
                "id": dataset_cfg.get("gsm8k", "openai/gsm8k"),
                "config_name": dataset_cfg.get("gsm8k_subset", "main"),
                "split": dataset_cfg.get("split", "train"),
                "target_tokens": round(target_train_tokens * gsm8k_weight / total_weight),
                "format": "gsm8k",
            }
        )

    delta = target_train_tokens - sum(int(source["target_tokens"]) for source in sources)
    if sources and delta:
        sources[0]["target_tokens"] = int(sources[0]["target_tokens"]) + delta
    return sources


class MidTrainStreamPreparer:
    """Streams, tokenizes, and writes only the next mid-training package to disk."""

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
                "Tokenizer does not expose bos_token; mid-training sequences will use eos only."
            )

        self._train_tokens = self._mixed_token_stream("train", self.seed + 101)

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
                operation="validating mid-training stream",
            )
            sample = next(
                sample
                for sample in dataset
                if isinstance(sample, dict) and belongs_to_source_range(sample, source)
            )
            text = _format_sample(sample, source)
            if not text:
                raise ValueError(
                    f"Source {source.name!r} loaded but did not produce formatable text."
                )
            LOGGER.info(
                "Validated mid-training source=%s dataset=%s config=%s split=%s",
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
                operation=f"streaming mid-training {partition} data",
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
                    f"No {partition} samples were produced by source={source.name!r}. "
                    "Adjust dataset.validation_ratio, source sample_range, or source configuration."
                )
            epoch += 1

    def _tokenized_source(
        self,
        source: StreamingSource,
        partition: str,
        seed: int,
    ) -> Iterator[tuple[str, list[int]]]:
        for sample in self._source_stream(source, partition, seed):
            text = _format_sample(sample, source)
            if not text:
                continue
            token_ids = self.tokenizer.encode(text) + [self.eos_id]
            if self.bos_id is not None:
                token_ids.insert(0, self.bos_id)
            if len(token_ids) > 1:
                yield source.name, token_ids

    def _mixed_token_stream(self, partition: str, seed: int) -> Iterator[tuple[str, list[int]]]:
        source_streams = {
            source.name: self._tokenized_source(source, partition, seed + 17 * (index + 1))
            for index, source in enumerate(self.sources)
        }
        source_names = iter_permuted_source_names(
            self.sources,
            seed=seed,
            window_size=self.mixture_window_size,
        )

        while True:
            source_name = next(source_names)
            yield next(source_streams[source_name])

    def prepare_validation(self, *, force: bool = False) -> Path:
        processed_dir = resolve_project_path(self.dataset_cfg["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = processed_dir / "metadata.json"
        val_path = processed_dir / "val_tokens.bin"

        target_train_tokens = int(self.dataset_cfg["target_train_tokens"])
        target_validation_tokens = int(target_train_tokens * self.validation_ratio)
        if target_validation_tokens <= 0:
            raise ValueError("Mid-training validation requires dataset.validation_ratio > 0.")

        existing_metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                try:
                    existing_metadata = json.load(handle)
                except json.JSONDecodeError:
                    existing_metadata = {}

        if (
            not force
            and val_path.exists()
            and existing_metadata.get("dataset_name") == _DATASET_NAME
            and existing_metadata.get("source_signature") == self.source_signature
            and existing_metadata.get("target_validation_tokens") == target_validation_tokens
            and existing_metadata.get("tokenizer_type") == self.config["tokenizer"]["type"]
        ):
            LOGGER.info("Reusing streaming validation tokens at %s", val_path)
            return val_path

        LOGGER.info("Streaming validation data to %s", val_path)
        validation_stream = self._mixed_token_stream("validation", self.seed + 307)
        source_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"samples": 0, "tokens": 0}
        )
        with TokenBinWriter(
            val_path,
            vocab_size=self.tokenizer.vocab_size,
            target_tokens=target_validation_tokens,
        ) as writer:
            for source_name, token_ids in validation_stream:
                result = writer.write(token_ids)
                if result.written > 0:
                    source_stats[source_name]["samples"] += 1
                    source_stats[source_name]["tokens"] += result.written
                if writer.complete:
                    break

        metadata = {
            "dataset_name": _DATASET_NAME,
            "streaming": True,
            "source_signature": self.source_signature,
            "sources": source_metadata(self.sources),
            "validation_source_stats": dict(source_stats),
            "tokenizer_type": self.config["tokenizer"]["type"],
            "tokenizer_dir": str(Path(self.config["tokenizer"]["save_dir"])),
            "target_train_tokens": target_train_tokens,
            "target_validation_tokens": target_validation_tokens,
            "validation_tokens": writer.tokens_written,
            "validation_ratio": self.validation_ratio,
            "vocab_size": self.tokenizer.vocab_size,
            "storage_dtype": writer.dtype,
            "validation_tokens_path": str(val_path),
            "train_package_dir": str(mid_train_package_dir(self.config)),
        }
        write_metadata(metadata_path, metadata)
        return val_path

    def write_train_package(self, package_index: int, target_tokens: int) -> MidTrainPackage:
        package_dir = mid_train_package_dir(self.config)
        package_dir.mkdir(parents=True, exist_ok=True)
        path = mid_train_package_path(self.config, package_index)
        metadata_path = mid_train_package_metadata_path(self.config, package_index)
        if path.exists():
            path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

        LOGGER.info(
            "Streaming mid-training package %s to %s with target_tokens=%s",
            package_index,
            path,
            target_tokens,
        )
        with TokenBinWriter(
            path,
            vocab_size=self.tokenizer.vocab_size,
            target_tokens=int(target_tokens),
        ) as writer:
            source_stats: dict[str, dict[str, int]] = defaultdict(
                lambda: {"samples": 0, "tokens": 0}
            )
            for source_name, token_ids in self._train_tokens:
                result = writer.write(token_ids)
                if result.written > 0:
                    source_stats[source_name]["samples"] += 1
                    source_stats[source_name]["tokens"] += result.written
                if writer.complete:
                    break

        metadata = {
            "dataset_name": _DATASET_NAME,
            "streaming": True,
            "source_signature": self.source_signature,
            "sources": source_metadata(self.sources),
            "source_stats": dict(source_stats),
            "package_index": int(package_index),
            "package_tokens": writer.tokens_written,
            "target_package_tokens": int(target_tokens),
            "tokenizer_type": self.config["tokenizer"]["type"],
            "tokenizer_dir": str(Path(self.config["tokenizer"]["save_dir"])),
            "vocab_size": self.tokenizer.vocab_size,
            "storage_dtype": writer.dtype,
            "tokens_path": str(path),
        }
        write_metadata(metadata_path, metadata)
        return MidTrainPackage(
            index=int(package_index),
            path=path,
            metadata_path=metadata_path,
            tokens_written=writer.tokens_written,
            dtype=writer.dtype,
        )


def prepare_mid_train_data(config: dict[str, Any], force: bool = False) -> None:
    """Prepare reusable validation data and the first streaming train package.

    The training script keeps a persistent :class:`MidTrainStreamPreparer` and creates
    more packages as needed. This wrapper exists for callers that only need to warm up
    the processed directory without materializing the full mid-training corpus.
    """

    preparer = MidTrainStreamPreparer(config)
    preparer.prepare_validation(force=force)
    target_tokens = int(
        config["dataset"].get(
            "streaming_package_tokens",
            config["dataset"].get("target_train_tokens", 0),
        )
    )
    if target_tokens > 0:
        preparer.write_train_package(0, target_tokens)
