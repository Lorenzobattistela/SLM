from __future__ import annotations

import argparse
from typing import Any

from src.config.loader import load_run_config
from src.config.schema import ConfigError


def add_run_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-config",
        type=str,
        required=True,
        help="Path to a YAML run config.",
    )


def load_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return load_run_config(args.run_config)
    except ConfigError as exc:
        raise SystemExit(f"Config error in {args.run_config}: {exc}") from None
