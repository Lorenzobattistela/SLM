from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import sys
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


def count_corpus_tokens(corpus_files: list[Path], tokenizer) -> tuple[int, int]:
    samples = 0
    tokens = 0
    for corpus_file in corpus_files:
        with corpus_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                samples += 1
                tokens += len(tokenizer.encode(text, add_eos=True))
    return samples, tokens


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
        corpus_stats = write_training_corpus(
            dataset_cfg,
            corpus_dir,
            max_samples=train_samples,
            chunk_samples=int(tokenizer_cfg.get("corpus_chunk_samples", 100_000)),
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
        tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
        counted_samples, counted_tokens = count_corpus_tokens(corpus_stats.files, tokenizer)
        logger.info("Samples tokenized for count: %s", counted_samples)
        logger.info("Tokens processed: %s", counted_tokens)
    except (RuntimeError, OSError, SuperBPEError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
