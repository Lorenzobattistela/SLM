from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

LOGGER = logging.getLogger(__name__)
_HASH_DENOMINATOR = 2**64


@dataclass(frozen=True)
class StreamingSource:
    name: str
    dataset_id: str
    config_name: str | None
    split: str
    target_tokens: int
    format: str
    sample_range: tuple[float, float] = (0.0, 1.0)
    sample_range_key: str | None = None

    @property
    def range_key(self) -> str:
        if self.sample_range_key:
            return self.sample_range_key
        return f"{self.dataset_id}:{self.config_name or ''}:{self.split}"


def normalise_dataset_id(dataset_id: str) -> str:
    return "openai/gsm8k" if dataset_id == "gsm8k" else dataset_id


def _read_sample_range(raw: dict[str, Any], source_name: str) -> tuple[float, float]:
    value = raw.get("sample_range", (0.0, 1.0))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"dataset.sources[{source_name}].sample_range must contain two numbers.")
    start = float(value[0])
    end = float(value[1])
    if start < 0.0 or end > 1.0 or start >= end:
        raise ValueError(
            f"dataset.sources[{source_name}].sample_range must satisfy 0 <= start < end <= 1."
        )
    return start, end


def parse_streaming_sources(
    dataset_cfg: dict[str, Any],
    *,
    fallback_sources: Iterable[dict[str, Any]],
) -> list[StreamingSource]:
    raw_sources = dataset_cfg.get("sources")
    if raw_sources is None:
        raw_sources = list(fallback_sources)
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("dataset.sources must be a non-empty list.")

    sources: list[StreamingSource] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"dataset.sources[{index}] must be a mapping.")

        source_name = str(raw_source.get("name") or f"source_{index:02d}")
        dataset_id = raw_source.get("id", raw_source.get("dataset_id", raw_source.get("dataset")))
        if not dataset_id:
            raise ValueError(f"dataset.sources[{source_name}] is missing id/dataset_id.")

        target_tokens = int(raw_source.get("target_tokens", raw_source.get("weight", 0)))
        if target_tokens <= 0:
            continue

        config_name = raw_source.get("config_name", raw_source.get("subset"))
        split = str(raw_source.get("split", dataset_cfg.get("split", "train")))
        sample_format = str(raw_source.get("format", raw_source.get("sample_format", "chat")))
        sources.append(
            StreamingSource(
                name=source_name,
                dataset_id=normalise_dataset_id(str(dataset_id)),
                config_name=None if config_name is None else str(config_name),
                split=split,
                target_tokens=target_tokens,
                format=sample_format,
                sample_range=_read_sample_range(raw_source, source_name),
                sample_range_key=(
                    None
                    if raw_source.get("sample_range_key") is None
                    else str(raw_source["sample_range_key"])
                ),
            )
        )

    if not sources:
        raise ValueError("dataset.sources must include at least one source with target_tokens > 0.")
    return sources


def validate_source_token_budget(
    sources: list[StreamingSource],
    *,
    target_train_tokens: int,
    tolerance: int | None = None,
) -> None:
    total = sum(source.target_tokens for source in sources)
    allowed_delta = tolerance if tolerance is not None else max(1, int(target_train_tokens * 0.001))
    if abs(total - int(target_train_tokens)) > allowed_delta:
        raise ValueError(
            "dataset.sources target_tokens must sum to dataset.target_train_tokens: "
            f"sources={total}, target_train_tokens={target_train_tokens}."
        )


def source_signature(sources: list[StreamingSource]) -> str:
    payload = [
        {
            "name": source.name,
            "dataset_id": source.dataset_id,
            "config_name": source.config_name,
            "split": source.split,
            "target_tokens": source.target_tokens,
            "format": source.format,
            "sample_range": list(source.sample_range),
            "sample_range_key": source.sample_range_key,
        }
        for source in sources
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=12).hexdigest()


def source_metadata(sources: list[StreamingSource]) -> list[dict[str, Any]]:
    return [
        {
            "name": source.name,
            "dataset_id": source.dataset_id,
            "config_name": source.config_name,
            "split": source.split,
            "target_tokens": source.target_tokens,
            "weight": source.target_tokens / sum(item.target_tokens for item in sources),
            "format": source.format,
            "sample_range": list(source.sample_range),
            "sample_range_key": source.sample_range_key,
        }
        for source in sources
    ]


def load_streaming_source(
    source: StreamingSource,
    *,
    cache_dir: Path,
    seed: int,
    shuffle_buffer_size: int,
    load_dataset_fn: Callable[..., Any],
):
    kwargs: dict[str, Any] = {
        "split": source.split,
        "cache_dir": str(cache_dir),
        "streaming": True,
    }
    if source.config_name is None:
        dataset = load_dataset_fn(source.dataset_id, **kwargs)
    else:
        dataset = load_dataset_fn(source.dataset_id, source.config_name, **kwargs)

    if shuffle_buffer_size > 1 and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return dataset


def preload_streaming_sources(
    sources: list[StreamingSource],
    *,
    cache_dir: Path,
    load_dataset_fn: Callable[..., Any],
) -> None:
    groups: dict[tuple[str, str | None], list[StreamingSource]] = {}
    for source in sources:
        groups.setdefault((source.dataset_id, source.config_name), []).append(source)

    for (dataset_id, config_name), grouped_sources in groups.items():
        splits = list(dict.fromkeys(source.split for source in grouped_sources))
        split_arg: str | list[str] = splits if len(splits) > 1 else splits[0]
        kwargs: dict[str, Any] = {
            "split": split_arg,
            "cache_dir": str(cache_dir),
            "streaming": True,
        }
        LOGGER.info(
            "Preloading dataset=%s config=%s split=%s",
            dataset_id,
            config_name,
            split_arg,
        )
        try:
            if config_name is None:
                load_dataset_fn(dataset_id, **kwargs)
            else:
                load_dataset_fn(dataset_id, config_name, **kwargs)
        except TypeError as exc:
            if not isinstance(split_arg, list) or "unhashable type: 'list'" not in str(exc):
                raise
            LOGGER.warning(
                "datasets.load_dataset does not support split lists with streaming in this "
                "environment; preloading %s splits individually.",
                len(split_arg),
            )
            for split in split_arg:
                split_kwargs = dict(kwargs)
                split_kwargs["split"] = split
                if config_name is None:
                    load_dataset_fn(dataset_id, **split_kwargs)
                else:
                    load_dataset_fn(dataset_id, config_name, **split_kwargs)


def _sample_key(sample: dict[str, Any]) -> str:
    if "messages" in sample:
        return json.dumps(sample["messages"], sort_keys=True, ensure_ascii=True)
    fields = [sample.get("question"), sample.get("answer"), sample.get("text")]
    if any(field is not None for field in fields):
        return "\n".join(str(field) for field in fields if field is not None)
    return json.dumps(sample, sort_keys=True, ensure_ascii=True)


def sample_bucket(sample: dict[str, Any], *, salt: str) -> float:
    payload = f"{salt}\n{_sample_key(sample)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") / _HASH_DENOMINATOR


def belongs_to_source_range(sample: dict[str, Any], source: StreamingSource) -> bool:
    start, end = source.sample_range
    if start <= 0.0 and end >= 1.0:
        return True
    bucket = sample_bucket(sample, salt=f"source-range:{source.range_key}")
    return start <= bucket < end


def belongs_to_partition(
    sample: dict[str, Any],
    *,
    partition: str,
    validation_ratio: float,
) -> bool:
    if validation_ratio <= 0:
        return partition == "train"
    if validation_ratio >= 1:
        return partition == "validation"

    is_validation = sample_bucket(sample, salt="validation") < validation_ratio
    return is_validation if partition == "validation" else not is_validation


def iter_permuted_source_names(
    sources: list[StreamingSource],
    *,
    seed: int,
    window_size: int,
) -> Iterator[str]:
    if len(sources) == 1:
        while True:
            yield sources[0].name

    rng = random.Random(seed)
    window_size = max(len(sources), int(window_size))
    total_weight = sum(source.target_tokens for source in sources)
    raw_counts = [source.target_tokens * window_size / total_weight for source in sources]
    counts = [max(1, math.floor(count)) for count in raw_counts]

    while sum(counts) < window_size:
        fractions = [raw - math.floor(raw) for raw in raw_counts]
        best_index = max(range(len(sources)), key=lambda index: (fractions[index], rng.random()))
        counts[best_index] += 1
    while sum(counts) > window_size:
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            break
        worst_index = min(
            candidates,
            key=lambda index: (raw_counts[index] - counts[index], rng.random()),
        )
        counts[worst_index] -= 1

    template: list[str] = []
    for source, count in zip(sources, counts):
        template.extend([source.name] * count)

    while True:
        names = list(template)
        rng.shuffle(names)
        yield from names
