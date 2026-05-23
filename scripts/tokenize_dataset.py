from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import os
import sys
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args, resolve_project_path
from src.data.fineweb_edu import iter_dataset_texts
from src.data.split import is_validation_text
from src.data.token_dataset import TokenBinWriter, token_dtype_for_vocab, write_metadata
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


def validation_ratio_for_targets(dataset_cfg: dict, train_tokens: int, validation_tokens: int) -> float:
    configured_ratio = dataset_cfg.get("validation_ratio")
    if configured_ratio is not None:
        ratio = float(configured_ratio)
    else:
        ratio = validation_tokens / max(1, train_tokens + validation_tokens)
    if not 0.0 < ratio < 1.0:
        raise ValueError("dataset.validation_ratio must be between 0 and 1 when configured.")
    return ratio


def configured_tokenize_workers(dataset_cfg: dict) -> int:
    env_value = os.environ.get("TOKENIZE_DATASET_WORKERS")
    if env_value is not None:
        return max(1, int(env_value))
    return max(1, int(dataset_cfg.get("tokenize_num_workers", 1)))


def _tokenize_shard(args: tuple[dict, dict, str, int, int, int, int, float, str]) -> dict:
    (
        dataset_cfg,
        tokenizer_cfg,
        shard_dir,
        worker_index,
        num_workers,
        target_train_tokens,
        target_validation_tokens,
        validation_ratio,
        validation_salt,
    ) = args
    tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
    append_eos = bool(tokenizer_cfg.get("append_eos", True))
    output_dir = Path(shard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"train_tokens_worker_{worker_index:03d}.bin"
    val_path = output_dir / f"val_tokens_worker_{worker_index:03d}.bin"

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
        for text in iter_dataset_texts(
            dataset_cfg,
            shard_index=worker_index,
            num_shards=num_workers,
        ):
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

            send_to_validation = (
                not val_writer.complete
                and is_validation_text(stripped, validation_ratio, validation_salt)
            )
            if send_to_validation:
                val_writer.write(token_ids)
            elif not train_writer.complete:
                train_writer.write(token_ids)
            elif not val_writer.complete:
                val_writer.write(token_ids)

            if train_writer.complete and val_writer.complete:
                break

        return {
            "worker_index": worker_index,
            "train_path": str(train_path),
            "validation_path": str(val_path),
            "train_tokens": train_writer.tokens_written,
            "validation_tokens": val_writer.tokens_written,
            "samples_seen": samples_seen,
            "samples_tokenized": samples_tokenized,
            "skipped_empty": skipped_empty,
        }


def _copy_token_file(
    source_path: Path,
    writer: TokenBinWriter,
    *,
    dtype: str,
    chunk_tokens: int = 1_000_000,
) -> None:
    typecode = "H" if dtype == "uint16" else "I"
    with source_path.open("rb") as handle:
        while not writer.complete:
            values = array(typecode)
            try:
                values.fromfile(handle, chunk_tokens)
            except EOFError:
                pass
            if not values:
                break
            writer.write(values)
            if len(values) < chunk_tokens:
                break


def tokenize_dataset_parallel(
    *,
    dataset_cfg: dict,
    tokenizer_cfg: dict,
    processed_dir: Path,
    train_path: Path,
    val_path: Path,
    target_train_tokens: int,
    target_validation_tokens: int,
    validation_ratio: float,
    validation_salt: str,
    num_workers: int,
    logger: logging.Logger,
) -> dict:
    tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
    shard_dir = processed_dir / "tokenize_shards"
    local_train_target = ceil(target_train_tokens / num_workers)
    local_validation_target = ceil(target_validation_tokens / num_workers)
    logger.info(
        "Tokenizing dataset with %s workers: shard_train_target=%s shard_validation_target=%s",
        num_workers,
        local_train_target,
        local_validation_target,
    )

    shard_stats: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _tokenize_shard,
                (
                    dataset_cfg,
                    tokenizer_cfg,
                    str(shard_dir),
                    worker_index,
                    num_workers,
                    local_train_target,
                    local_validation_target,
                    validation_ratio,
                    validation_salt,
                ),
            ): worker_index
            for worker_index in range(num_workers)
        }
        for future in as_completed(futures):
            worker_index = futures[future]
            stats = future.result()
            shard_stats[worker_index] = stats
            logger.info(
                "Tokenize worker %s done: samples=%s train_tokens=%s validation_tokens=%s",
                worker_index,
                stats["samples_seen"],
                stats["train_tokens"],
                stats["validation_tokens"],
            )

    dtype = token_dtype_for_vocab(tokenizer.vocab_size)
    with TokenBinWriter(
        val_path,
        vocab_size=tokenizer.vocab_size,
        target_tokens=target_validation_tokens,
    ) as val_writer, TokenBinWriter(
        train_path,
        vocab_size=tokenizer.vocab_size,
        target_tokens=target_train_tokens,
    ) as train_writer:
        for worker_index in sorted(shard_stats):
            _copy_token_file(
                Path(shard_stats[worker_index]["train_path"]),
                train_writer,
                dtype=dtype,
            )
            _copy_token_file(
                Path(shard_stats[worker_index]["validation_path"]),
                val_writer,
                dtype=dtype,
            )

        return {
            "train_tokens": train_writer.tokens_written,
            "validation_tokens": val_writer.tokens_written,
            "storage_dtype": train_writer.dtype,
            "samples_seen": sum(stats["samples_seen"] for stats in shard_stats.values()),
            "samples_tokenized": sum(
                stats["samples_tokenized"] for stats in shard_stats.values()
            ),
            "skipped_empty": sum(stats["skipped_empty"] for stats in shard_stats.values()),
            "tokenize_num_workers": num_workers,
            "shards": [shard_stats[index] for index in sorted(shard_stats)],
        }


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
    validation_ratio = validation_ratio_for_targets(
        dataset_cfg,
        target_train_tokens,
        target_validation_tokens,
    )
    validation_salt = str(dataset_cfg.get("validation_salt", "slm-pretrain-v1"))

    logger.info("Dataset name: %s", dataset_cfg["name"])
    logger.info("Dataset split: %s", dataset_cfg.get("split", "train"))
    logger.info("Dataset text column: %s", dataset_cfg["text_column"])
    logger.info("Dataset streaming: %s", dataset_cfg.get("streaming", True))
    logger.info("Tokenizer type: %s", tokenizer_cfg["type"])
    logger.info("Tokenizer dir: %s", resolve_project_path(tokenizer_cfg["save_dir"]))
    logger.info("Target train tokens: %s", target_train_tokens)
    logger.info("Target validation tokens: %s", target_validation_tokens)
    logger.info("Validation split ratio: %.8f", validation_ratio)
    logger.info("Validation split salt: %s", validation_salt)
    logger.info("Output train path: %s", train_path)
    logger.info("Output validation path: %s", val_path)
    tokenize_workers = configured_tokenize_workers(dataset_cfg)
    logger.info("Dataset tokenization workers: %s", tokenize_workers)

    try:
        tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
        append_eos = bool(tokenizer_cfg.get("append_eos", True))

        if tokenize_workers > 1:
            tokenization_stats = tokenize_dataset_parallel(
                dataset_cfg=dataset_cfg,
                tokenizer_cfg=tokenizer_cfg,
                processed_dir=processed_dir,
                train_path=train_path,
                val_path=val_path,
                target_train_tokens=target_train_tokens,
                target_validation_tokens=target_validation_tokens,
                validation_ratio=validation_ratio,
                validation_salt=validation_salt,
                num_workers=tokenize_workers,
                logger=logger,
            )
        else:
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

                    send_to_validation = (
                        not val_writer.complete
                        and is_validation_text(stripped, validation_ratio, validation_salt)
                    )
                    if send_to_validation:
                        val_writer.write(token_ids)
                    elif not train_writer.complete:
                        train_writer.write(token_ids)
                    elif not val_writer.complete:
                        val_writer.write(token_ids)

                    if samples_seen % 1_000 == 0:
                        logger.info(
                            "Samples processed=%s train_tokens=%s validation_tokens=%s",
                            samples_seen,
                            train_writer.tokens_written,
                            val_writer.tokens_written,
                        )

                    if train_writer.complete and val_writer.complete:
                        break

                tokenization_stats = {
                    "train_tokens": train_writer.tokens_written,
                    "validation_tokens": val_writer.tokens_written,
                    "storage_dtype": train_writer.dtype,
                    "samples_seen": samples_seen,
                    "samples_tokenized": samples_tokenized,
                    "skipped_empty": skipped_empty,
                    "tokenize_num_workers": tokenize_workers,
                }

        metadata = {
            "dataset_name": dataset_cfg["name"],
            "split": dataset_cfg.get("split", "train"),
            "text_column": dataset_cfg["text_column"],
            "streaming": bool(dataset_cfg.get("streaming", True)),
            "tokenizer_type": tokenizer_cfg["type"],
            "tokenizer_dir": str(Path(tokenizer_cfg["save_dir"])),
            "train_tokens": tokenization_stats["train_tokens"],
            "validation_tokens": tokenization_stats["validation_tokens"],
            "target_train_tokens": target_train_tokens,
            "target_validation_tokens": target_validation_tokens,
            "vocab_size": tokenizer.vocab_size,
            "storage_dtype": tokenization_stats["storage_dtype"],
            "train_tokens_path": str(train_path),
            "validation_tokens_path": str(val_path),
            "samples_seen": tokenization_stats["samples_seen"],
            "samples_tokenized": tokenization_stats["samples_tokenized"],
            "skipped_empty": tokenization_stats["skipped_empty"],
            "append_eos": append_eos,
            "validation_ratio": validation_ratio,
            "validation_salt": validation_salt,
            "tokenize_num_workers": tokenization_stats["tokenize_num_workers"],
        }
        if "shards" in tokenization_stats:
            metadata["shards"] = tokenization_stats["shards"]

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
