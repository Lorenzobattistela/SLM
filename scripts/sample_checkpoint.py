from __future__ import annotations
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import torch


def _find_project_root(start: Path) -> Path:
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_run_config, resolve_project_path
from src.data.tokenizer import get_tokenizer
from src.inference.generate import generate
from src.model.config import ModelConfig
from src.model.transformer import TransformerLM
from src.tokenizer import load_tokenizer
from src.train.checkpoint import load_checkpoint as load_legacy_checkpoint
from src.training.checkpointing import (
    find_checkpoint,
    load_checkpoint as load_training_checkpoint,
)
from src.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a saved checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to the configured latest/final checkpoint for modern configs.",
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default=None,
        help="Run config to use. Omit only when loading a checkpoint with embedded config.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to complete.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff.")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximum generated tokens.")
    return parser.parse_args()


def _is_modern_config(config: dict) -> bool:
    return "training" in config and "tokenizer" in config and "model" in config


def _load_config(args: argparse.Namespace, checkpoint_path: Path | None) -> dict:
    if args.run_config:
        return load_run_config(args.run_config)
    if checkpoint_path is None:
        raise SystemExit("Either --run-config or --checkpoint is required.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint["config"]


def _default_prompt(config: dict) -> str:
    if _is_modern_config(config):
        return "Scientific progress depends on"
    return str(config.get("train", {}).get("sample_prompt", "Scientific progress depends on"))


def _load_tokenizer(config: dict):
    if _is_modern_config(config):
        return load_tokenizer(config["tokenizer"])
    return get_tokenizer(config["data"]["tokenizer"]["name"])


def _eos_token_id(tokenizer) -> int | None:
    return getattr(tokenizer, "eos_token_id", getattr(tokenizer, "eot_token_id", None))


def _load_model_checkpoint(
    checkpoint_path: Path,
    model: TransformerLM,
    config: dict,
    device: torch.device,
) -> None:
    if _is_modern_config(config):
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            map_location=device,
            restore_rng=False,
        )
    else:
        load_legacy_checkpoint(checkpoint_path, model=model, map_location=device)


def main() -> None:
    args = parse_args()
    device = select_device("auto")
    checkpoint_path = resolve_project_path(args.checkpoint) if args.checkpoint else None

    config = _load_config(args, checkpoint_path)
    if checkpoint_path is None:
        if not _is_modern_config(config):
            raise SystemExit("--checkpoint is required for legacy configs.")
        checkpoint_path = find_checkpoint(config)

    model_cfg = ModelConfig.from_dict(config["model"])
    tokenizer = _load_tokenizer(config)
    prompt = args.prompt or _default_prompt(config)

    model = TransformerLM(model_cfg).to(device)
    _load_model_checkpoint(checkpoint_path, model, config, device)
    model.eval()

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        eos_token_id=_eos_token_id(tokenizer),
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
