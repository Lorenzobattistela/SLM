from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args, resolve_project_path
from src.data.fineweb_edu import iter_dataset_texts
from src.data.token_dataset import TokenBinWriter, write_metadata
from src.tokenizer import SuperBPEError, load_superbpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu into train/val token bins.")
    add_run_config_argument(parser)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config_from_args(args)
    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]
    logger = logging.getLogger("tokenize_dataset")

    processed_dir = resolve_project_path(dataset_cfg["processed_dir"])
    train_path = processed_dir / "train_tokens.bin"
    val_path = processed_dir / "val_tokens.bin"
    metadata_path = processed_dir / "metadata.json"

    target_train_tokens = int(dataset_cfg["target_train_tokens"])
    target_validation_tokens = int(dataset_cfg["validation_tokens"])

    logger.info("Dataset name: %s", dataset_cfg["name"])
    logger.info("Dataset split: %s", dataset_cfg.get("split", "train"))
    logger.info("Dataset text column: %s", dataset_cfg["text_column"])
    logger.info("Dataset streaming: %s", dataset_cfg.get("streaming", True))
    logger.info("Tokenizer type: %s", tokenizer_cfg["type"])
    logger.info("Tokenizer dir: %s", resolve_project_path(tokenizer_cfg["save_dir"]))
    logger.info("Target train tokens: %s", target_train_tokens)
    logger.info("Target validation tokens: %s", target_validation_tokens)
    logger.info("Output train path: %s", train_path)
    logger.info("Output validation path: %s", val_path)

    try:
        tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
        append_eos = bool(tokenizer_cfg.get("append_eos", True))

        samples_seen = 0
        samples_tokenized = 0
        skipped_empty = 0

        with TokenBinWriter(
            val_path,
            vocab_size=tokenizer.vocab_size,
            target_tokens=target_validation_tokens,
        ) as val_writer, TokenBinWriter(
            train_path,
            vocab_size=tokenizer.vocab_size,
            target_tokens=target_train_tokens,
        ) as train_writer:
            for text in iter_dataset_texts(dataset_cfg):
                samples_seen += 1
                stripped = text.strip()
                if not stripped:
                    skipped_empty += 1
                    continue

                token_ids = tokenizer.encode(stripped, add_eos=append_eos)
                if not token_ids:
                    skipped_empty += 1
                    continue
                samples_tokenized += 1

                if not val_writer.complete:
                    result = val_writer.write(token_ids)
                    token_ids = token_ids[result.written :]

                if token_ids and not train_writer.complete:
                    train_writer.write(token_ids)

                if samples_seen % 1_000 == 0:
                    logger.info(
                        "Samples processed=%s train_tokens=%s validation_tokens=%s",
                        samples_seen,
                        train_writer.tokens_written,
                        val_writer.tokens_written,
                    )

                if train_writer.complete and val_writer.complete:
                    break

            metadata = {
                "dataset_name": dataset_cfg["name"],
                "split": dataset_cfg.get("split", "train"),
                "text_column": dataset_cfg["text_column"],
                "streaming": bool(dataset_cfg.get("streaming", True)),
                "tokenizer_type": tokenizer_cfg["type"],
                "tokenizer_dir": str(Path(tokenizer_cfg["save_dir"])),
                "train_tokens": train_writer.tokens_written,
                "validation_tokens": val_writer.tokens_written,
                "target_train_tokens": target_train_tokens,
                "target_validation_tokens": target_validation_tokens,
                "vocab_size": tokenizer.vocab_size,
                "storage_dtype": train_writer.dtype,
                "train_tokens_path": str(train_path),
                "validation_tokens_path": str(val_path),
                "samples_seen": samples_seen,
                "samples_tokenized": samples_tokenized,
                "skipped_empty": skipped_empty,
                "append_eos": append_eos,
            }

        write_metadata(metadata_path, metadata)

        logger.info("Samples processed: %s", samples_seen)
        logger.info("Samples tokenized: %s", samples_tokenized)
        total_tokens = metadata["train_tokens"] + metadata["validation_tokens"]
        logger.info("Tokens processed: %s", total_tokens)
        logger.info("Actual train tokens: %s", metadata["train_tokens"])
        logger.info("Actual validation tokens: %s", metadata["validation_tokens"])
        logger.info("Metadata path: %s", metadata_path)

        if metadata["train_tokens"] < target_train_tokens:
            logger.warning(
                "Stopped before target_train_tokens because the dataset iterator ended: %s < %s",
                metadata["train_tokens"],
                target_train_tokens,
            )
        if metadata["validation_tokens"] < target_validation_tokens:
            logger.warning(
                "Stopped before validation_tokens because the dataset iterator ended: %s < %s",
                metadata["validation_tokens"],
                target_validation_tokens,
            )
    except (RuntimeError, OSError, SuperBPEError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None


def _clean_process_exit(code: int) -> None:
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            _clean_process_exit(1)
        _clean_process_exit(int(exc.code or 0))
    _clean_process_exit(0)
