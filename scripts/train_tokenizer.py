from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args, resolve_project_path
from src.data.fineweb_edu import write_training_corpus
from src.tokenizer import (
    SuperBPEError,
    load_superbpe_tokenizer,
    remove_existing_tokenizer,
    tokenizer_exists,
    train_superbpe_tokenizer,
    validate_superbpe_backend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or load the configured SuperBPE tokenizer.")
    add_run_config_argument(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing tokenizer artifacts before training.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _count_corpus_file(args: tuple[Path, dict, bool]) -> tuple[int, int]:
    corpus_file, tokenizer_cfg, append_eos = args
    tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
    return _count_corpus_tokens_file(corpus_file, tokenizer, append_eos=append_eos)


def count_corpus_tokens(
    corpus_files: list[Path],
    tokenizer_cfg: dict,
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


def configured_corpus_workers(tokenizer_cfg: dict) -> int:
    env_value = os.environ.get("TOKENIZER_CORPUS_WORKERS")
    if env_value is not None:
        return max(1, int(env_value))
    return max(1, int(tokenizer_cfg.get("corpus_num_workers", 1)))


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config_from_args(args)
    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]
    logger = logging.getLogger("train_tokenizer")

    logger.info("Dataset name: %s", dataset_cfg["name"])
    logger.info("Dataset split: %s", dataset_cfg.get("split", "train"))
    logger.info("Dataset text column: %s", dataset_cfg["text_column"])
    logger.info("Dataset streaming: %s", dataset_cfg.get("streaming", True))
    logger.info("Tokenizer type: %s", tokenizer_cfg["type"])
    logger.info("Tokenizer output dir: %s", resolve_project_path(tokenizer_cfg["save_dir"]))

    try:
        validate_superbpe_backend(tokenizer_cfg)

        if tokenizer_cfg.get("pretrained"):
            tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
            logger.info("Loaded pretrained SuperBPE tokenizer.")
            logger.info("Vocab size: %s", tokenizer.vocab_size)
            logger.info("Samples processed: 0")
            logger.info("Tokens processed: 0")
            return

        if tokenizer_exists(tokenizer_cfg) and not args.force:
            tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
            logger.info("Loaded existing SuperBPE tokenizer.")
            logger.info("Vocab size: %s", tokenizer.vocab_size)
            logger.info("Samples processed: 0")
            logger.info("Tokens processed: 0")
            return

        if args.force:
            remove_existing_tokenizer(tokenizer_cfg["save_dir"])

        train_samples = int(tokenizer_cfg.get("train_samples", 0))
        if train_samples <= 0:
            raise SuperBPEError("tokenizer.train_samples must be a positive integer.")

        save_dir = resolve_project_path(tokenizer_cfg["save_dir"])
        corpus_dir = save_dir / "training_corpus"
        corpus_workers = configured_corpus_workers(tokenizer_cfg)
        logger.info("Tokenizer corpus workers: %s", corpus_workers)
        corpus_stats = write_training_corpus(
            dataset_cfg,
            corpus_dir,
            max_samples=train_samples,
            chunk_samples=int(tokenizer_cfg.get("corpus_chunk_samples", 100_000)),
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
    except (RuntimeError, OSError, SuperBPEError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
