from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.hf_stream import iter_dataset_texts as iter_filtered_dataset_texts

LOGGER = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - dependency is checked at runtime.
    load_dataset = None


def load_configured_dataset(dataset_cfg: dict[str, Any]):
    if load_dataset is None:
        raise RuntimeError(
            "The `datasets` package is required to load FineWeb-Edu. "
            "Install project dependencies with `pip install -e .`."
        )

    dataset_name = dataset_cfg["name"]
    dataset_config_name = dataset_cfg.get("config_name")
    split = dataset_cfg.get("split", "train")
    streaming = bool(dataset_cfg.get("streaming", True))
    cache_dir = dataset_cfg.get("cache_dir")

    LOGGER.info(
        "Loading dataset=%s split=%s streaming=%s cache_dir=%s",
        dataset_name,
        split,
        streaming,
        cache_dir,
    )
    return load_dataset(
        dataset_name,
        name=dataset_config_name,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def iter_dataset_texts(
    dataset_cfg: dict[str, Any],
    *,
    max_samples: int | None = None,
) -> Iterator[str]:
    for sample_index, text in enumerate(iter_filtered_dataset_texts(dataset_cfg)):
        if max_samples is not None and sample_index >= max_samples:
            break
        if isinstance(text, str):
            yield text


@dataclass
class CorpusWriteStats:
    files: list[Path]
    samples_seen: int
    samples_written: int
    bytes_written: int


def write_training_corpus(
    dataset_cfg: dict[str, Any],
    output_dir: str | Path,
    *,
    max_samples: int,
    chunk_samples: int = 100_000,
) -> CorpusWriteStats:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    samples_seen = 0
    samples_written = 0
    bytes_written = 0
    chunk_index = -1
    handle = None

    try:
        for text in iter_dataset_texts(dataset_cfg, max_samples=max_samples):
            samples_seen += 1
            stripped = " ".join(text.strip().splitlines())
            if not stripped:
                continue

            if handle is None or samples_written % chunk_samples == 0:
                if handle is not None:
                    handle.close()
                chunk_index += 1
                chunk_path = target_dir / f"fineweb_edu_tokenizer_train_{chunk_index:05d}.txt"
                files.append(chunk_path)
                handle = chunk_path.open("w", encoding="utf-8")

            handle.write(stripped)
            handle.write("\n")
            samples_written += 1
            bytes_written += len(stripped.encode("utf-8")) + 1
    finally:
        if handle is not None:
            handle.close()

    return CorpusWriteStats(
        files=files,
        samples_seen=samples_seen,
        samples_written=samples_written,
        bytes_written=bytes_written,
    )
