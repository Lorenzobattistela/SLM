from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from src.config.loader import resolve_project_path
from src.tokenizer import load_tokenizer
from src.data.token_dataset import TokenBinWriter, token_dtype_for_vocab, write_metadata

LOGGER = logging.getLogger(__name__)


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def prepare_mid_train_data(config: dict[str, Any], force: bool = False) -> None:
    dataset_cfg = config["dataset"]
    processed_dir = resolve_project_path(dataset_cfg["processed_dir"])
    metadata_path = processed_dir / "metadata.json"

    # Check if we can reuse existing tokenized files
    metadata = _load_metadata(processed_dir)
    target_train_tokens = int(dataset_cfg["target_train_tokens"])
    validation_ratio = float(dataset_cfg.get("validation_ratio", 0.05))
    target_validation_tokens = int(target_train_tokens * validation_ratio)

    train_path = processed_dir / "train_tokens.bin"
    val_path = processed_dir / "val_tokens.bin"

    if not force and metadata_path.exists() and train_path.exists() and val_path.exists():
        if (
            metadata.get("target_train_tokens") == target_train_tokens
            and metadata.get("validation_ratio") == validation_ratio
            and metadata.get("tokenizer_type") == config["tokenizer"]["type"]
        ):
            LOGGER.info("Reusing existing tokenized datasets at %s", processed_dir)
            return

    LOGGER.info("Tokenizing and mixing datasets for mid-training...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["tokenizer"])
    # Fetch special tokens from tokenizer configuration or default to mapped special tokens
    bos_id = tokenizer.special_token_ids.get("bos_token")
    eos_id = tokenizer.special_token_ids.get("eos_token")

    if bos_id is None or eos_id is None:
        raise ValueError(
            f"Tokenizer is missing bos_token or eos_token: {tokenizer.special_token_ids}"
        )

    # Load SmolTalk and GSM8K
    smoltalk_name = dataset_cfg.get("smoltalk", "HuggingFaceTB/smoltalk")
    gsm8k_name = dataset_cfg.get("gsm8k", "gsm8k")
    if gsm8k_name == "gsm8k":
        gsm8k_name = "openai/gsm8k"
    cache_dir = resolve_project_path(dataset_cfg.get("cache_dir", "./data/cache"))

    LOGGER.info("Loading SmolTalk dataset: %s", smoltalk_name)
    # Smoltalk is composed of multiple subsets, "all" config loads the full composition
    smoltalk_ds = load_dataset(smoltalk_name, "all", split="train", cache_dir=str(cache_dir))

    LOGGER.info("Loading GSM8K dataset: %s", gsm8k_name)
    # GSM8K requires "main" config
    gsm8k_ds = load_dataset(gsm8k_name, "main", split="train", cache_dir=str(cache_dir))

    seed = int(config["project"].get("seed", 42))

    # Deterministic split helper
    def split_dataset(ds, val_ratio, shuffle_seed):
        rng = random.Random(shuffle_seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        val_size = int(len(ds) * val_ratio)
        train_indices = indices[:-val_size] if val_size > 0 else indices
        val_indices = indices[-val_size:] if val_size > 0 else []
        return ds.select(train_indices), ds.select(val_indices) if val_size > 0 else []

    smoltalk_train, smoltalk_val = split_dataset(smoltalk_ds, validation_ratio, seed)
    gsm8k_train, gsm8k_val = split_dataset(gsm8k_ds, validation_ratio, seed + 1)

    mix_ratio = dataset_cfg.get("mix_ratio", {"smoltalk": 0.7, "gsm8k": 0.3})

    role_map = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System"
    }

    def sample_generator(smoltalk_part, gsm8k_part, gen_seed):
        rng = random.Random(gen_seed)
        smol_indices = list(range(len(smoltalk_part)))
        gsm_indices = list(range(len(gsm8k_part)))

        rng.shuffle(smol_indices)
        rng.shuffle(gsm_indices)

        smol_idx = 0
        gsm_idx = 0

        while True:
            r = rng.random()
            if r < mix_ratio["smoltalk"]:
                # Draw from SmolTalk
                idx = smol_indices[smol_idx]
                smol_idx += 1
                if smol_idx >= len(smol_indices):
                    smol_idx = 0
                    rng.shuffle(smol_indices)

                sample = smoltalk_part[idx]
                messages = sample["messages"]
                formatted_turns = []
                for msg in messages:
                    role = role_map.get(msg["role"], msg["role"].capitalize())
                    content = msg["content"]
                    formatted_turns.append(f"{role}: {content}")
                text = "\n".join(formatted_turns)
                # Wrap with BOS and EOS IDs
                token_ids = [bos_id] + tokenizer.encode(text) + [eos_id]
                yield token_ids
            else:
                # Draw from GSM8K
                idx = gsm_indices[gsm_idx]
                gsm_idx += 1
                if gsm_idx >= len(gsm_indices):
                    gsm_idx = 0
                    rng.shuffle(gsm_indices)

                sample = gsm8k_part[idx]
                question = sample["question"]
                answer = sample["answer"]
                text = f"User: {question}\nAssistant: {answer}"
                # Wrap with BOS and EOS IDs
                token_ids = [bos_id] + tokenizer.encode(text) + [eos_id]
                yield token_ids

    # Write training tokens
    LOGGER.info("Writing train tokens to %s...", train_path)
    train_gen = sample_generator(smoltalk_train, gsm8k_train, seed)
    with TokenBinWriter(
        train_path, vocab_size=tokenizer.vocab_size, target_tokens=target_train_tokens
    ) as train_writer:
        for token_ids in train_gen:
            train_writer.write(token_ids)
            if train_writer.complete:
                break

    # Write validation tokens
    LOGGER.info("Writing validation tokens to %s...", val_path)
    val_gen = (
        sample_generator(smoltalk_val, gsm8k_val, seed + 2)
        if len(smoltalk_val) > 0 and len(gsm8k_val) > 0
        else None
    )

    actual_val_tokens = 0
    if val_gen is not None and target_validation_tokens > 0:
        with TokenBinWriter(
            val_path, vocab_size=tokenizer.vocab_size, target_tokens=target_validation_tokens
        ) as val_writer:
            for token_ids in val_gen:
                val_writer.write(token_ids)
                if val_writer.complete:
                    break
        actual_val_tokens = val_writer.tokens_written

    # Write metadata json
    metadata = {
        "dataset_name": "smoltalk_gsm8k_mix",
        "tokenizer_type": config["tokenizer"]["type"],
        "tokenizer_dir": str(Path(config["tokenizer"]["save_dir"])),
        "train_tokens": train_writer.tokens_written,
        "validation_tokens": actual_val_tokens,
        "target_train_tokens": target_train_tokens,
        "target_validation_tokens": target_validation_tokens,
        "vocab_size": tokenizer.vocab_size,
        "storage_dtype": train_writer.dtype,
        "train_tokens_path": str(train_path),
        "validation_tokens_path": str(val_path),
        "validation_ratio": validation_ratio,
    }
    write_metadata(metadata_path, metadata)
    LOGGER.info(
        "Finished tokenizing and mixing. Train tokens: %s, Val tokens: %s",
        train_writer.tokens_written,
        actual_val_tokens,
    )
