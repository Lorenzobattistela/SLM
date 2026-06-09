from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
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
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> Iterator[str]:
    texts = iter_filtered_dataset_texts(
        dataset_cfg,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    for sample_index, text in enumerate(texts):
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


def _write_training_corpus_single_worker(
    dataset_cfg: dict[str, Any],
    output_dir: str | Path,
    *,
    max_samples: int,
    chunk_samples: int = 100_000,
    shard_index: int | None = None,
    num_shards: int | None = None,
    filename_prefix: str = "fineweb_edu_tokenizer_train",
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
        for text in iter_dataset_texts(
            dataset_cfg,
            max_samples=max_samples,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            samples_seen += 1
            stripped = " ".join(text.strip().splitlines())
            if not stripped:
                continue

            if handle is None or samples_written % chunk_samples == 0:
                if handle is not None:
                    handle.close()
                chunk_index += 1
                chunk_path = target_dir / f"{filename_prefix}_{chunk_index:05d}.txt"
                files.append(chunk_path)
                handle = chunk_path.open("w", encoding="utf-8")

            handle.write(stripped)
            handle.write("\n")
            samples_written += 1
            bytes_written += len(stripped.encode("utf-8")) + 1

            if samples_written == 1 or samples_written % chunk_samples == 0:
                LOGGER.info(
                    "Tokenizer corpus progress: samples_seen=%s samples_written=%s/%s "
                    "chunks=%s bytes=%s",
                    samples_seen,
                    samples_written,
                    max_samples,
                    len(files),
                    bytes_written,
                )
    finally:
        if handle is not None:
            handle.close()

    return CorpusWriteStats(
        files=files,
        samples_seen=samples_seen,
        samples_written=samples_written,
        bytes_written=bytes_written,
    )


def _write_training_corpus_worker(args: tuple[dict[str, Any], str, int, int, int, int]):
    dataset_cfg, output_dir, max_samples, chunk_samples, worker_index, num_workers = args
    worker_dir = Path(output_dir) / f"worker_{worker_index:03d}"
    return _write_training_corpus_single_worker(
        dataset_cfg,
        worker_dir,
        max_samples=max_samples,
        chunk_samples=chunk_samples,
        shard_index=worker_index,
        num_shards=num_workers,
        filename_prefix=f"fineweb_edu_tokenizer_train_worker_{worker_index:03d}",
    )


def write_training_corpus(
    dataset_cfg: dict[str, Any],
    output_dir: str | Path,
    *,
    max_samples: int,
    chunk_samples: int = 100_000,
    num_workers: int = 1,
) -> CorpusWriteStats:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    if num_workers <= 1:
        return _write_training_corpus_single_worker(
            dataset_cfg,
            output_dir,
            max_samples=max_samples,
            chunk_samples=chunk_samples,
        )

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    num_workers = min(num_workers, max_samples)
    worker_limit = ceil(max_samples / num_workers)
    LOGGER.info(
        "Writing tokenizer corpus with %s workers: max_samples=%s worker_limit=%s",
        num_workers,
        max_samples,
        worker_limit,
    )

    stats_by_worker: dict[int, CorpusWriteStats] = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _write_training_corpus_worker,
                (
                    dataset_cfg,
                    str(target_dir),
                    min(worker_limit, max_samples - worker_index * worker_limit),
                    chunk_samples,
                    worker_index,
                    num_workers,
                ),
            ): worker_index
            for worker_index in range(num_workers)
            if max_samples - worker_index * worker_limit > 0
        }
        for future in as_completed(futures):
            worker_index = futures[future]
            stats = future.result()
            stats_by_worker[worker_index] = stats
            LOGGER.info(
                "Tokenizer corpus worker %s done: samples_seen=%s samples_written=%s "
                "chunks=%s bytes=%s",
                worker_index,
                stats.samples_seen,
                stats.samples_written,
                len(stats.files),
                stats.bytes_written,
            )

    files: list[Path] = []
    samples_seen = 0
    samples_written = 0
    bytes_written = 0
    for worker_index in sorted(stats_by_worker):
        stats = stats_by_worker[worker_index]
        files.extend(stats.files)
        samples_seen += stats.samples_seen
        samples_written += stats.samples_written
        bytes_written += stats.bytes_written

    return CorpusWriteStats(
        files=files,
        samples_seen=samples_seen,
        samples_written=samples_written,
        bytes_written=bytes_written,
    )
