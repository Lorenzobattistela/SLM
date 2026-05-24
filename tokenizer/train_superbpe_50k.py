from __future__ import annotations

import argparse
import logging
import os
import resource
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import resolve_project_path
from src.data.fineweb_edu import write_training_corpus
from src.tokenizer import (
    SuperBPEError,
    load_superbpe_tokenizer,
    remove_existing_tokenizer,
    tokenizer_exists,
    train_superbpe_tokenizer,
    validate_superbpe_backend,
)

DEFAULT_CONFIG = Path(__file__).with_name("superbpe_50k_olmo_p99.yml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 50K-vocabulary SuperBPE tokenizer from a streaming dataset."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Tokenizer training YAML. Defaults to tokenizer/superbpe_50k_olmo_p99.yml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing tokenizer artifacts before training.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Override tokenizer.train_samples from the YAML.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override tokenizer.corpus_num_workers from the YAML.",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=None,
        help="Override tokenizer.max_memory_gb. Use 0 to disable the process memory cap.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_training_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "dataset" not in config or "tokenizer" not in config:
        raise SuperBPEError("Config must contain top-level dataset and tokenizer sections.")
    return config


def configured_corpus_workers(tokenizer_cfg: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return max(1, int(override))
    env_value = os.environ.get("TOKENIZER_CORPUS_WORKERS")
    if env_value is not None:
        return max(1, int(env_value))
    return max(1, int(tokenizer_cfg.get("corpus_num_workers", 1)))


def configured_memory_limit_gb(tokenizer_cfg: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return max(0.0, float(override))
    env_value = os.environ.get("TOKENIZER_MEMORY_LIMIT_GB")
    if env_value is not None:
        return max(0.0, float(env_value))
    return max(0.0, float(tokenizer_cfg.get("max_memory_gb", 0)))


def apply_process_memory_limit(limit_gb: float, logger: logging.Logger) -> None:
    if limit_gb <= 0:
        logger.info("Tokenizer process memory cap: disabled")
        return

    limit_bytes = int(limit_gb * 1024**3)
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = current_hard
    if current_hard != resource.RLIM_INFINITY:
        limit_bytes = min(limit_bytes, current_hard)
    else:
        new_hard = limit_bytes
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, new_hard))
    logger.info("Tokenizer process memory cap: %.2f GiB", limit_bytes / 1024**3)


def _count_corpus_file(args: tuple[Path, dict[str, Any], bool]) -> tuple[int, int]:
    corpus_file, tokenizer_cfg, append_eos = args
    tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
    return _count_corpus_tokens_file(corpus_file, tokenizer, append_eos=append_eos)


def _count_corpus_tokens_file(corpus_file: Path, tokenizer, *, append_eos: bool) -> tuple[int, int]:
    samples = 0
    tokens = 0
    with corpus_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            samples += 1
            tokens += len(tokenizer.encode(text, add_eos=append_eos))
    return samples, tokens


def count_corpus_tokens(
    corpus_files: list[Path],
    tokenizer_cfg: dict[str, Any],
    *,
    num_workers: int,
) -> tuple[int, int]:
    append_eos = bool(tokenizer_cfg.get("append_eos", True))
    if num_workers <= 1 or len(corpus_files) <= 1:
        samples = 0
        tokens = 0
        tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
        for corpus_file in corpus_files:
            file_samples, file_tokens = _count_corpus_tokens_file(
                corpus_file,
                tokenizer,
                append_eos=append_eos,
            )
            samples += file_samples
            tokens += file_tokens
        return samples, tokens

    workers = min(num_workers, len(corpus_files))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        counts = list(
            executor.map(
                _count_corpus_file,
                [(path, tokenizer_cfg, append_eos) for path in corpus_files],
            )
        )
    samples = 0
    tokens = 0
    for file_samples, file_tokens in counts:
        samples += file_samples
        tokens += file_tokens
    return samples, tokens


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_training_config(args.config)
    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]
    logger = logging.getLogger("train_superbpe_50k")

    if int(tokenizer_cfg["vocab_size"]) != 50000:
        raise SuperBPEError("This trainer is intended for tokenizer.vocab_size: 50000.")

    if args.train_samples is not None:
        tokenizer_cfg["train_samples"] = int(args.train_samples)

    memory_limit_gb = configured_memory_limit_gb(tokenizer_cfg, args.memory_limit_gb)
    apply_process_memory_limit(memory_limit_gb, logger)

    logger.info("Dataset name: %s", dataset_cfg["name"])
    logger.info("Dataset split: %s", dataset_cfg.get("split", "train"))
    logger.info("Dataset text column: %s", dataset_cfg.get("text_column", "auto"))
    logger.info("Dataset streaming: %s", dataset_cfg.get("streaming", True))
    logger.info("Tokenizer vocab size: %s", tokenizer_cfg["vocab_size"])
    logger.info("SuperBPE transition point: %s", tokenizer_cfg["superbpe_stage1_vocab_size"])
    logger.info("Tokenizer output dir: %s", resolve_project_path(tokenizer_cfg["save_dir"]))
    logger.info("Tokenizer configured train samples: %s", tokenizer_cfg.get("train_samples"))

    validate_superbpe_backend(tokenizer_cfg)

    if tokenizer_exists(tokenizer_cfg) and not args.force:
        tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
        logger.info("Loaded existing SuperBPE tokenizer; pass --force to retrain.")
        logger.info("Vocab size: %s", tokenizer.vocab_size)
        return

    if args.force:
        remove_existing_tokenizer(tokenizer_cfg["save_dir"])

    train_samples = int(tokenizer_cfg.get("train_samples", 0))
    if train_samples <= 0:
        raise SuperBPEError("tokenizer.train_samples must be a positive integer.")

    save_dir = resolve_project_path(tokenizer_cfg["save_dir"])
    corpus_dir = save_dir / "training_corpus"
    corpus_workers = configured_corpus_workers(tokenizer_cfg, args.workers)
    logger.info("Tokenizer corpus workers: %s", corpus_workers)

    corpus_stats = write_training_corpus(
        dataset_cfg,
        corpus_dir,
        max_samples=train_samples,
        chunk_samples=int(tokenizer_cfg.get("corpus_chunk_samples", 100000)),
        num_workers=corpus_workers,
    )
    logger.info("Samples processed: %s", corpus_stats.samples_seen)
    logger.info("Samples written for tokenizer training: %s", corpus_stats.samples_written)
    logger.info("Training corpus bytes: %s", corpus_stats.bytes_written)
    logger.info("Training corpus files: %s", [str(Path(path)) for path in corpus_stats.files])

    metadata = train_superbpe_tokenizer(
        tokenizer_cfg=tokenizer_cfg,
        corpus_files=corpus_stats.files,
        save_dir=save_dir,
    )
    logger.info("Tokenizer artifacts saved to: %s", save_dir)
    logger.info("Tokenizer metadata: %s", save_dir / "tokenizer_metadata.json")
    logger.info("Vocab size: %s", metadata["vocab_size"])

    counted_samples, counted_tokens = count_corpus_tokens(
        corpus_stats.files,
        tokenizer_cfg,
        num_workers=corpus_workers,
    )
    logger.info("Samples tokenized for count: %s", counted_samples)
    logger.info("Tokens processed: %s", counted_tokens)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, SuperBPEError) as exc:
        raise SystemExit(str(exc)) from None
