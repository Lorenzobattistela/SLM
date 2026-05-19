from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.config import load_yaml, resolve_project_path
from src.data.hf_stream import iter_dataset_texts
from src.data.pack import TokenPacker
from src.data.split import is_validation_text
from src.data.tokenizer import get_tokenizer
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, tokenize and shard pretraining data.")
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to a data config under configs/data/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated shards for this preset.",
    )
    return parser.parse_args()


def _clean_existing_outputs(output_root: Path, manifest_path: Path) -> None:
    for split_dir in (output_root / "train", output_root / "val"):
        if split_dir.exists():
            for shard_path in split_dir.glob("*.npy"):
                shard_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_project_path(args.data_config))
    tokenizer = get_tokenizer(config["tokenizer"]["name"])

    output_root = resolve_project_path(config["storage"]["output_dir"])
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    manifest_path = resolve_project_path(config["storage"]["manifest_path"])

    existing_shards = list(train_dir.glob("*.npy")) + list(val_dir.glob("*.npy"))
    if existing_shards and not args.force:
        raise FileExistsError(
            f"Found existing shards under {output_root}. Re-run with --force to overwrite them."
        )
    if existing_shards and args.force:
        _clean_existing_outputs(output_root, manifest_path)

    train_packer = TokenPacker(train_dir, prefix="train", shard_tokens=int(config["storage"]["shard_tokens"]))
    val_packer = TokenPacker(val_dir, prefix="val", shard_tokens=int(config["storage"]["shard_tokens"]))

    target_train_tokens = int(config["sampling"]["target_train_tokens"])
    target_val_tokens = int(config["sampling"]["target_val_tokens"])
    max_documents = config["sampling"].get("max_documents")
    max_documents = None if max_documents is None else int(max_documents)
    min_document_tokens = int(config["sampling"].get("min_document_tokens", 1))
    append_eot = bool(config["tokenizer"].get("append_eot", True))
    validation_ratio = float(config["split"]["validation_ratio"])
    salt = str(config["split"]["salt"])

    documents_seen = 0
    documents_kept = 0
    skipped_empty = 0
    skipped_short = 0

    progress = tqdm(desc=config["name"], unit="docs")

    for text in iter_dataset_texts(config["dataset"]):
        if max_documents is not None and documents_seen >= max_documents:
            break
        documents_seen += 1
        progress.update(1)

        stripped = text.strip()
        if not stripped:
            skipped_empty += 1
            continue

        token_ids = tokenizer.encode(stripped)
        if len(token_ids) < min_document_tokens:
            skipped_short += 1
            continue

        if append_eot:
            token_ids.append(tokenizer.eot_token_id)

        send_to_val = is_validation_text(stripped, validation_ratio, salt)
        if send_to_val and val_packer.total_tokens >= target_val_tokens:
            send_to_val = False
        if not send_to_val and train_packer.total_tokens >= target_train_tokens:
            if val_packer.total_tokens < target_val_tokens:
                send_to_val = True
            else:
                break

        if send_to_val:
            val_packer.add_document(token_ids)
        else:
            train_packer.add_document(token_ids)
        documents_kept += 1

        progress.set_postfix(
            train_tokens=train_packer.total_tokens,
            val_tokens=val_packer.total_tokens,
        )

        if (
            train_packer.total_tokens >= target_train_tokens
            and val_packer.total_tokens >= target_val_tokens
        ):
            break

    progress.close()
    train_packer.close()
    val_packer.close()

    manifest = {
        "name": config["name"],
        "dataset": config["dataset"],
        "tokenizer": {
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "eot_token_id": tokenizer.eot_token_id,
        },
        "sampling": config["sampling"],
        "split": config["split"],
        "storage": {
            "output_dir": str(Path(config["storage"]["output_dir"])),
            "train_shards": train_packer.shard_paths,
            "val_shards": val_packer.shard_paths,
            "manifest_path": str(Path(config["storage"]["manifest_path"])),
        },
        "stats": {
            "documents_seen": documents_seen,
            "documents_kept": documents_kept,
            "skipped_empty": skipped_empty,
            "skipped_short": skipped_short,
            "train_tokens": train_packer.total_tokens,
            "val_tokens": val_packer.total_tokens,
            "train_documents": train_packer.total_documents,
            "val_documents": val_packer.total_documents,
        },
    }
    write_json(manifest_path, manifest)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
