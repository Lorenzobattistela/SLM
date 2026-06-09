from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args
from src.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the configured decoder-only LM.")
    add_run_config_argument(parser)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config_from_args(args)
    run_training(config)


if __name__ == "__main__":
    main()
