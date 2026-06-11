from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finetuning.sft_trainer import load_sft_config, run_sft_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) training script.")
    parser.add_argument(
        "--run-config",
        type=str,
        required=True,
        help="Path to SFT run config.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_sft_config(args.run_config)
    run_sft_training(config)


if __name__ == "__main__":
    main()
