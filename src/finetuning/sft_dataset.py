from __future__ import annotations

import json
import logging
import random
from array import array
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from src.config.loader import resolve_project_path
from src.tokenizer import load_tokenizer
from src.data.token_dataset import TokenBinWriter, write_metadata

LOGGER = logging.getLogger(__name__)


class LabelsBinWriter:
    """Binary file writer for target labels storing signed 32-bit integers."""

    def __init__(self, path: str | Path, target_tokens: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.target_tokens = int(target_tokens)
        self.tokens_written = 0
        self.dtype = "int32"
        self._handle = self.path.open("wb")

    @property
    def remaining(self) -> int:
        return max(0, self.target_tokens - self.tokens_written)

    @property
    def complete(self) -> bool:
        return self.tokens_written >= self.target_tokens

    def write(self, labels: list[int]) -> None:
        if not labels or self.complete:
            return
        writable = labels[: self.remaining]
        values = array("i", writable)  # signed 32-bit integers
        values.tofile(self._handle)
        self.tokens_written += len(values)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> LabelsBinWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class SFTBinDataset:
    """Memory-mapped dataset containing tokens (inputs) and target labels (prompt-masked)."""

    def __init__(
        self,
        tokens_path: str | Path,
        labels_path: str | Path,
        *,
        block_size: int,
        vocab_size: int,
        tokens_dtype: str = "uint16",
        labels_dtype: str = "int32",
    ) -> None:
        self.tokens_path = Path(tokens_path)
        self.labels_path = Path(labels_path)

        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens bin file not found: {self.tokens_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels bin file not found: {self.labels_path}")

        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.tokens_dtype = tokens_dtype
        self.labels_dtype = labels_dtype

        self.tokens = np.memmap(self.tokens_path, dtype=self.tokens_dtype, mode="r")
        self.labels = np.memmap(self.labels_path, dtype=self.labels_dtype, mode="r")

        if len(self.tokens) != len(self.labels):
            raise ValueError(
                f"Length mismatch: tokens ({len(self.tokens)}) vs labels ({len(self.labels)})"
            )

        self.num_tokens = int(self.tokens.shape[0])
        self.num_positions = max(0, self.num_tokens - self.block_size)

        if self.num_positions <= 0:
            raise ValueError(
                f"Token file {self.tokens_path} has {self.num_tokens} tokens, which is too small "
                f"for block_size={self.block_size}."
            )

    def __len__(self) -> int:
        return self.num_positions

    def _rank_position_count(self, *, rank: int, world_size: int) -> int:
        if rank >= self.num_positions:
            return 0
        return ((self.num_positions - 1 - rank) // world_size) + 1

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        rng: np.random.Generator,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = int(rank)
        world_size = max(1, int(world_size))
        rank_positions = self._rank_position_count(rank=rank, world_size=world_size)
        if rank_positions <= 0:
            raise ValueError(
                f"Rank {rank} has no sample positions; "
                f"num_positions={self.num_positions}, world_size={world_size}."
            )

        offsets = rng.integers(0, rank_positions, size=int(batch_size), endpoint=False)
        starts = rank + offsets * world_size

        input_windows = []
        target_windows = []
        for start in starts:
            input_windows.append(
                np.asarray(self.tokens[int(start) : int(start) + self.block_size], dtype=np.int64)
            )
            target_windows.append(
                np.asarray(self.labels[int(start) : int(start) + self.block_size], dtype=np.int64)
            )

        inputs = torch.tensor(np.stack(input_windows), dtype=torch.long, device=device)
        targets = torch.tensor(np.stack(target_windows), dtype=torch.long, device=device)
        return inputs, targets

    def iter_batches(
        self,
        batch_size: int,
        device: torch.device,
        *,
        max_batches: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        emitted = 0
        rank = int(rank)
        world_size = max(1, int(world_size))
        batch_inputs = []
        batch_targets = []

        for start in range(rank, self.num_positions, self.block_size * world_size):
            inp_window = np.asarray(self.tokens[start : start + self.block_size], dtype=np.int64)
            tar_window = np.asarray(self.labels[start : start + self.block_size], dtype=np.int64)

            if inp_window.shape[0] != self.block_size:
                continue
            batch_inputs.append(inp_window)
            batch_targets.append(tar_window)

            if len(batch_inputs) == batch_size:
                yield (
                    torch.tensor(np.stack(batch_inputs), dtype=torch.long, device=device),
                    torch.tensor(np.stack(batch_targets), dtype=torch.long, device=device),
                )
                emitted += 1
                if max_batches is not None and emitted >= max_batches:
                    return
                batch_inputs.clear()
                batch_targets.clear()

        if batch_inputs:
            yield (
                torch.tensor(np.stack(batch_inputs), dtype=torch.long, device=device),
                torch.tensor(np.stack(batch_targets), dtype=torch.long, device=device),
            )


def tokenize_sft_conversation(
    messages: list[dict[str, str]], tokenizer: Any, bos_id: int, eos_id: int
) -> tuple[list[int], list[int]]:
    """Tokenize conversation with loss masking on user/system prompts."""
    role_map = {"user": "User", "assistant": "Assistant", "system": "System"}

    tokens = []
    labels = []

    # Prepend bos_id, masked in targets
    tokens.append(bos_id)
    labels.append(-100)

    for i, msg in enumerate(messages):
        role = role_map.get(msg["role"], msg["role"].capitalize())
        content = msg["content"]

        prefix = "\n" if i > 0 else ""
        turn_text = f"{prefix}{role}: {content}"
        turn_tokens = tokenizer.encode(turn_text)

        if msg["role"] == "assistant":
            # Assistant response: compute loss
            tokens.extend(turn_tokens)
            labels.extend(turn_tokens)
        else:
            # User or system prompt: mask loss
            tokens.extend(turn_tokens)
            labels.extend([-100] * len(turn_tokens))

    # Append eos_id, calculated in loss
    tokens.append(eos_id)
    labels.append(eos_id)

    return tokens, labels


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def prepare_sft_data(config: dict[str, Any], force: bool = False) -> None:
    dataset_cfg = config["dataset"]
    processed_dir = resolve_project_path(dataset_cfg["processed_dir"])
    metadata_path = processed_dir / "metadata.json"

    # Verify if we can reuse existing files
    metadata = _load_metadata(processed_dir)
    target_train_tokens = int(dataset_cfg["target_train_tokens"])
    validation_ratio = float(dataset_cfg.get("validation_ratio", 0.05))
    target_validation_tokens = int(target_train_tokens * validation_ratio)

    train_tokens_path = processed_dir / "train_tokens.bin"
    train_labels_path = processed_dir / "train_labels.bin"
    val_tokens_path = processed_dir / "val_tokens.bin"
    val_labels_path = processed_dir / "val_labels.bin"

    if (
        not force
        and metadata_path.exists()
        and train_tokens_path.exists()
        and train_labels_path.exists()
        and val_tokens_path.exists()
        and val_labels_path.exists()
    ):
        if (
            metadata.get("target_train_tokens") == target_train_tokens
            and metadata.get("validation_ratio") == validation_ratio
            and metadata.get("tokenizer_type") == config["tokenizer"]["type"]
        ):
            LOGGER.info("Reusing existing tokenized SFT datasets at %s", processed_dir)
            return

    LOGGER.info("Tokenizing SFT dataset with prompt loss masking...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["tokenizer"])
    bos_id = tokenizer.special_token_ids.get("bos_token")
    eos_id = tokenizer.special_token_ids.get("eos_token")

    if bos_id is None or eos_id is None:
        raise ValueError(
            f"Tokenizer is missing bos_token or eos_token: {tokenizer.special_token_ids}"
        )

    # Load SmolTalk SFT dataset
    sft_dataset_name = dataset_cfg.get("sft_dataset", "HuggingFaceTB/smoltalk")
    cache_dir = resolve_project_path(dataset_cfg.get("cache_dir", "./data/cache"))

    LOGGER.info("Loading SFT dataset: %s", sft_dataset_name)
    sft_ds = load_dataset(sft_dataset_name, "all", split="train", cache_dir=str(cache_dir))

    seed = int(config["project"].get("seed", 42))
    rng = random.Random(seed)
    indices = list(range(len(sft_ds)))
    rng.shuffle(indices)
    val_size = int(len(sft_ds) * validation_ratio)
    train_indices = indices[:-val_size] if val_size > 0 else indices
    val_indices = indices[-val_size:] if val_size > 0 else []

    sft_train = sft_ds.select(train_indices)
    sft_val = sft_ds.select(val_indices) if val_size > 0 else []

    def process_and_write(dataset_part, tokens_path, labels_path, target_tokens, part_name):
        tokens_writer = TokenBinWriter(
            tokens_path, vocab_size=tokenizer.vocab_size, target_tokens=target_tokens
        )
        labels_writer = LabelsBinWriter(labels_path, target_tokens=target_tokens)

        part_indices = list(range(len(dataset_part)))
        random.Random(seed + 3).shuffle(part_indices)

        idx = 0
        try:
            with tokens_writer, labels_writer:
                while not tokens_writer.complete:
                    if idx >= len(part_indices):
                        idx = 0
                        random.Random(seed + idx).shuffle(part_indices)

                    sample = dataset_part[part_indices[idx]]
                    idx += 1

                    messages = sample["messages"]
                    tokens, labels = tokenize_sft_conversation(messages, tokenizer, bos_id, eos_id)

                    # Write inputs (tokens[:-1]) and labels[1:] (alignment shift)
                    tokens_writer.write(tokens[:-1])
                    labels_writer.write(labels[1:])
        except Exception as e:
            LOGGER.exception("Error writing SFT binaries: %s", e)
            raise

        return tokens_writer.tokens_written, tokens_writer.dtype

    actual_train_tokens, storage_dtype = process_and_write(
        sft_train, train_tokens_path, train_labels_path, target_train_tokens, "train"
    )

    actual_val_tokens = 0
    if len(sft_val) > 0 and target_validation_tokens > 0:
        actual_val_tokens, _ = process_and_write(
            sft_val, val_tokens_path, val_labels_path, target_validation_tokens, "val"
        )

    # Write SFT metadata
    metadata = {
        "dataset_name": "sft_smoltalk",
        "tokenizer_type": config["tokenizer"]["type"],
        "tokenizer_dir": str(Path(config["tokenizer"]["save_dir"])),
        "train_tokens": actual_train_tokens,
        "validation_tokens": actual_val_tokens,
        "target_train_tokens": target_train_tokens,
        "target_validation_tokens": target_validation_tokens,
        "vocab_size": tokenizer.vocab_size,
        "storage_dtype": storage_dtype,
        "validation_ratio": validation_ratio,
    }
    write_metadata(metadata_path, metadata)
    LOGGER.info(
        "Finished tokenizing SFT dataset. Train tokens: %s, Val tokens: %s",
        actual_train_tokens,
        actual_val_tokens,
    )
