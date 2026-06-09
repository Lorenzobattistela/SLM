from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from typing import Any

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - dependency is checked at runtime.
    load_dataset = None


def _infer_text_field(sample: dict[str, Any], preferred: str | None = None) -> str:
    if preferred and preferred in sample:
        return preferred
    for field_name in ("text", "content", "raw_content", "document"):
        value = sample.get(field_name)
        if isinstance(value, str):
            return field_name
    raise KeyError(
        "Could not infer a text field from the dataset sample. "
        "Set `dataset.text_field` in the data config."
    )


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _quality_key(sample: dict[str, Any]) -> tuple[float, float]:
    score = _as_float(sample.get("score"))
    language_score = _as_float(sample.get("language_score"))
    return (
        float("-inf") if score is None else score,
        float("-inf") if language_score is None else language_score,
    )


def _passes_filters(sample: dict[str, Any], filters: dict[str, Any]) -> bool:
    language = filters.get("language")
    if language is not None and sample.get("language") != language:
        return False

    language_score_min = filters.get("language_score_min")
    if language_score_min is not None:
        language_score = _as_float(sample.get("language_score"))
        if language_score is None or language_score < float(language_score_min):
            return False

    score_min = filters.get("score_min")
    if score_min is not None:
        score = _as_float(sample.get("score"))
        if score is None or score < float(score_min):
            return False

    token_count_min = filters.get("token_count_min")
    token_count_max = filters.get("token_count_max")
    if token_count_min is not None or token_count_max is not None:
        token_count = _as_int(sample.get("token_count"))
        if token_count is None:
            return False
        if token_count_min is not None and token_count < int(token_count_min):
            return False
        if token_count_max is not None and token_count > int(token_count_max):
            return False

    return True


def _prioritized_samples(
    samples: Iterable[dict[str, Any]],
    *,
    buffer_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    buffer: list[dict[str, Any]] = []
    rng = random.Random(seed)

    def flush() -> Iterator[dict[str, Any]]:
        buffer.sort(key=_quality_key, reverse=True)
        if shuffle:
            # Preserve score/language_score priority and only shuffle exact ties.
            group: list[dict[str, Any]] = []
            previous_key: tuple[float, float] | None = None
            for item in buffer:
                key = _quality_key(item)
                if previous_key is not None and key != previous_key:
                    rng.shuffle(group)
                    yield from group
                    group = []
                group.append(item)
                previous_key = key
            if group:
                rng.shuffle(group)
                yield from group
        else:
            yield from buffer

    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= buffer_size:
            yield from flush()
            buffer.clear()

    if buffer:
        yield from flush()


def _load_dataset(dataset_cfg: dict[str, Any]):
    if load_dataset is None:
        raise RuntimeError("datasets is not installed. Run `pip install -e .` first.")

    dataset_id = dataset_cfg.get("id", dataset_cfg.get("name"))
    if dataset_id is None:
        raise KeyError("Missing dataset id. Set `dataset.id` or `dataset.name` in the config.")

    return load_dataset(
        dataset_id,
        name=dataset_cfg.get("config_name", dataset_cfg.get("name"))
        if "id" in dataset_cfg
        else dataset_cfg.get("config_name"),
        split=dataset_cfg.get("split", "train"),
        revision=dataset_cfg.get("revision"),
        streaming=dataset_cfg.get("streaming", True),
        cache_dir=dataset_cfg.get("cache_dir"),
    )


def iter_dataset_samples(
    dataset_cfg: dict[str, Any],
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> Iterator[dict[str, Any]]:
    dataset = _load_dataset(dataset_cfg)
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("Both shard_index and num_shards must be set for dataset sharding.")
        if num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("shard_index must be in [0, num_shards).")
        if hasattr(dataset, "shard"):
            dataset = dataset.shard(num_shards=num_shards, index=shard_index)
        else:
            dataset = (
                sample
                for index, sample in enumerate(dataset)
                if index % num_shards == shard_index
            )

    filters = dict(dataset_cfg.get("filters") or {})
    selection = dict(dataset_cfg.get("selection") or {})
    if hasattr(dataset, "shuffle") and selection.get("shuffle_before_filter", False):
        dataset = dataset.shuffle(
            buffer_size=int(selection.get("shuffle_buffer_size", 10_000)),
            seed=int(selection.get("seed", 42)),
        )

    samples = (sample for sample in dataset if _passes_filters(sample, filters))
    prioritize_quality = bool(selection.get("prioritize_quality", bool(filters)))
    if prioritize_quality:
        samples = _prioritized_samples(
            samples,
            buffer_size=int(selection.get("quality_buffer_size", 10_000)),
            shuffle=bool(selection.get("shuffle", False)),
            seed=int(selection.get("seed", 42)),
        )
    elif selection.get("shuffle", False) and hasattr(dataset, "shuffle"):
        samples = dataset.shuffle(
            buffer_size=int(selection.get("shuffle_buffer_size", 10_000)),
            seed=int(selection.get("seed", 42)),
        )
        samples = (sample for sample in samples if _passes_filters(sample, filters))

    yield from samples


def iter_dataset_texts(
    dataset_cfg: dict[str, Any],
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> Iterator[str]:
    text_field = dataset_cfg.get("text_field")
    if text_field is None:
        text_field = dataset_cfg.get("text_column")

    for sample in iter_dataset_samples(
        dataset_cfg,
        shard_index=shard_index,
        num_shards=num_shards,
    ):
        if text_field is None:
            text_field = _infer_text_field(sample, dataset_cfg.get("text_column"))
        text = sample.get(text_field)
        if isinstance(text, str):
            yield text
