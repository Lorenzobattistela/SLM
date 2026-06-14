from __future__ import annotations

import argparse

from src.config import load_run_config
from src.train.trainer import run_pretraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decoder-only SLM pretraining.")
    parser.add_argument(
        "--run-config",
        type=str,
        required=True,
        help="Path to a run config under pre-train/configs/run/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = load_run_config(args.run_config)
    run_pretraining(run_config)


if __name__ == "__main__":
    main()
