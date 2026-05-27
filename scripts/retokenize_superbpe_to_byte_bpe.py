from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import logging
import shutil
import sys
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args, resolve_project_path
from src.data.token_dataset import token_dtype_for_vocab, token_file_token_count, write_metadata
from src.tokenizer import load_byte_bpe_tokenizer, load_superbpe_tokenizer

LOGGER = logging.getLogger("retokenize_superbpe_to_byte_bpe")


@dataclass
class RetokenizeStats:
    source_tokens: int = 0
    target_tokens: int = 0
    documents: int = 0
    empty_documents: int = 0
    partial_source_documents: int = 0


class TokenIdStreamWriter:
    def __init__(self, path: str | Path, *, vocab_size: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.dtype = token_dtype_for_vocab(vocab_size)
        self._typecode = "H" if self.dtype == "uint16" else "I"
        self.tokens_written = 0
        self._handle = self.path.open("wb")

    def write(self, token_ids: Iterable[int]) -> int:
        ids = list(token_ids)
        if not ids:
            return 0
        max_token_id = max(ids)
        if max_token_id >= 2 ** (16 if self.dtype == "uint16" else 32):
            raise ValueError(
                f"Token id {max_token_id} does not fit in configured storage dtype {self.dtype}"
            )
        values = array(self._typecode, ids)
        values.tofile(self._handle)
        self.tokens_written += len(values)
        return len(values)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TokenIdStreamWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode existing SuperBPE token .bin files back to text, then re-tokenize with "
            "a ready byte-level BPE tokenizer into a separate processed directory."
        )
    )
    add_run_config_argument(parser)
    parser.add_argument(
        "--source-processed-dir",
        type=str,
        default=None,
        help="Directory containing SuperBPE train_tokens.bin, val_tokens.bin, and metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_byte_bpe_gpt2",
        help="Directory where byte-level BPE train_tokens.bin, val_tokens.bin, and metadata.json are written.",
    )
    parser.add_argument(
        "--byte-bpe-name",
        type=str,
        default="gpt2",
        help="Ready tiktoken byte-level BPE encoding to use. Default: gpt2.",
    )
    parser.add_argument(
        "--append-eot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the target tokenizer EOT token after each reconstructed source document.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1_000_000,
        help="Number of source tokens to read per file chunk.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional smoke-test limit per split. Omit for the full conversion.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _array_typecode(dtype: str) -> str:
    if dtype == "uint16":
        return "H"
    if dtype == "uint32":
        return "I"
    raise ValueError(f"Unsupported token dtype: {dtype}")


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_token_ids(path: Path, *, dtype: str, chunk_tokens: int) -> Iterable[int]:
    typecode = _array_typecode(dtype)
    with path.open("rb") as handle:
        while True:
            values = array(typecode)
            try:
                values.fromfile(handle, chunk_tokens)
            except EOFError:
                pass
            if not values:
                break
            yield from values
            if len(values) < chunk_tokens:
                break


def _retokenize_document(
    *,
    source_tokenizer,
    target_tokenizer,
    source_document_ids: list[int],
    writer: TokenIdStreamWriter,
    append_eot: bool,
    stats: RetokenizeStats,
    partial: bool,
) -> None:
    if not source_document_ids:
        stats.empty_documents += 1
        if partial:
            stats.partial_source_documents += 1
        return

    text = source_tokenizer.decode(source_document_ids)
    target_ids = target_tokenizer.encode(text, add_eos=append_eot)
    writer.write(target_ids)
    stats.documents += 1
    stats.target_tokens = writer.tokens_written
    if partial:
        stats.partial_source_documents += 1


def retokenize_file(
    *,
    source_path: Path,
    target_path: Path,
    source_dtype: str,
    source_tokenizer,
    target_tokenizer,
    source_eos_token_id: int,
    append_eot: bool,
    chunk_tokens: int = 1_000_000,
    max_documents: int | None = None,
) -> RetokenizeStats:
    stats = RetokenizeStats()
    source_document_ids: list[int] = []
    with TokenIdStreamWriter(target_path, vocab_size=target_tokenizer.vocab_size) as writer:
        for token_id in _iter_token_ids(source_path, dtype=source_dtype, chunk_tokens=chunk_tokens):
            stats.source_tokens += 1
            if int(token_id) == source_eos_token_id:
                _retokenize_document(
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer,
                    source_document_ids=source_document_ids,
                    writer=writer,
                    append_eot=append_eot,
                    stats=stats,
                    partial=False,
                )
                source_document_ids = []
                if max_documents is not None and stats.documents >= max_documents:
                    break
            else:
                source_document_ids.append(int(token_id))

        if source_document_ids and (max_documents is None or stats.documents < max_documents):
            _retokenize_document(
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                source_document_ids=source_document_ids,
                writer=writer,
                append_eot=append_eot,
                stats=stats,
                partial=bool(source_document_ids),
            )

        stats.target_tokens = writer.tokens_written
    return stats


def _prepare_output_dir(output_dir: Path, source_processed_dir: Path, overwrite: bool) -> None:
    if output_dir == source_processed_dir:
        raise ValueError("Output directory must be different from the source processed directory.")
    if output_dir.exists():
        existing_entries = list(output_dir.iterdir())
        if existing_entries and not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config_from_args(args)
    dataset_cfg = config["dataset"]
    configured_source_tokenizer_cfg = config["tokenizer"]
    if configured_source_tokenizer_cfg.get("type") != "superbpe":
        raise SystemExit("The source run config must use tokenizer.type: 'superbpe'.")

    source_processed_dir = resolve_project_path(
        args.source_processed_dir or dataset_cfg["processed_dir"]
    )
    output_dir = resolve_project_path(args.output_dir)
    _prepare_output_dir(output_dir, source_processed_dir, overwrite=args.overwrite)

    source_metadata = _load_metadata(source_processed_dir / "metadata.json")
    source_tokenizer_cfg = dict(configured_source_tokenizer_cfg)
    if source_metadata.get("tokenizer_type") == "superbpe":
        if source_metadata.get("tokenizer_dir"):
            source_tokenizer_cfg["save_dir"] = source_metadata["tokenizer_dir"]
        if source_metadata.get("vocab_size"):
            source_tokenizer_cfg["vocab_size"] = int(source_metadata["vocab_size"])

    source_vocab_size = int(source_metadata.get("vocab_size", source_tokenizer_cfg["vocab_size"]))
    source_dtype = str(source_metadata.get("storage_dtype") or token_dtype_for_vocab(source_vocab_size))
    source_train_path = source_processed_dir / "train_tokens.bin"
    source_val_path = source_processed_dir / "val_tokens.bin"
    if not source_train_path.exists() or not source_val_path.exists():
        raise FileNotFoundError(
            f"Missing SuperBPE token files in {source_processed_dir}. Expected train_tokens.bin "
            "and val_tokens.bin."
        )

    source_tokenizer = load_superbpe_tokenizer(source_tokenizer_cfg)
    source_eos_token_id = source_tokenizer.eos_token_id
    if source_eos_token_id is None:
        raise ValueError("Source SuperBPE tokenizer does not expose eos_token_id.")

    target_tokenizer_cfg = {
        "type": "byte_bpe",
        "name": args.byte_bpe_name,
    }
    target_tokenizer = load_byte_bpe_tokenizer(target_tokenizer_cfg)
    LOGGER.info("Source processed dir: %s", source_processed_dir)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Source dtype: %s", source_dtype)
    LOGGER.info(
        "Target tokenizer: byte_bpe/%s vocab_size=%s eot_token_id=%s append_eot=%s",
        target_tokenizer.name,
        target_tokenizer.vocab_size,
        target_tokenizer.eos_token_id,
        args.append_eot,
    )

    train_stats = retokenize_file(
        source_path=source_train_path,
        target_path=output_dir / "train_tokens.bin",
        source_dtype=source_dtype,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_eos_token_id=source_eos_token_id,
        append_eot=args.append_eot,
        chunk_tokens=args.chunk_tokens,
        max_documents=args.max_documents,
    )
    LOGGER.info("Train stats: %s", train_stats)

    val_stats = retokenize_file(
        source_path=source_val_path,
        target_path=output_dir / "val_tokens.bin",
        source_dtype=source_dtype,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_eos_token_id=source_eos_token_id,
        append_eot=args.append_eot,
        chunk_tokens=args.chunk_tokens,
        max_documents=args.max_documents,
    )
    LOGGER.info("Validation stats: %s", val_stats)

    target_dtype = token_dtype_for_vocab(target_tokenizer.vocab_size)
    metadata = {
        "dataset_name": source_metadata.get("dataset_name", dataset_cfg["name"]),
        "split": source_metadata.get("split", dataset_cfg.get("split", "train")),
        "text_column": source_metadata.get("text_column", dataset_cfg["text_column"]),
        "streaming": bool(source_metadata.get("streaming", dataset_cfg.get("streaming", True))),
        "tokenizer_type": "byte_bpe",
        "tokenizer_name": target_tokenizer.name,
        "vocab_size": target_tokenizer.vocab_size,
        "storage_dtype": target_dtype,
        "train_tokens": train_stats.target_tokens,
        "validation_tokens": val_stats.target_tokens,
        "train_tokens_path": str(output_dir / "train_tokens.bin"),
        "validation_tokens_path": str(output_dir / "val_tokens.bin"),
        "append_eot": bool(args.append_eot),
        "eot_token_id": target_tokenizer.eos_token_id,
        "source": {
            "tokenizer_type": "superbpe",
            "tokenizer_dir": str(Path(source_tokenizer_cfg["save_dir"])),
            "vocab_size": source_vocab_size,
            "storage_dtype": source_dtype,
            "processed_dir": str(source_processed_dir),
            "source_eos_token_id": source_eos_token_id,
            "train_tokens": token_file_token_count(source_train_path, source_dtype),
            "validation_tokens": token_file_token_count(source_val_path, source_dtype),
            "metadata": source_metadata,
        },
        "retokenization": {
            "path": "superbpe_bin_to_text_to_byte_bpe",
            "max_documents_per_split": args.max_documents,
            "train": asdict(train_stats),
            "validation": asdict(val_stats),
        },
    }
    write_metadata(output_dir / "metadata.json", metadata)
    LOGGER.info("Wrote byte-level BPE metadata: %s", output_dir / "metadata.json")


if __name__ == "__main__":
    main()
