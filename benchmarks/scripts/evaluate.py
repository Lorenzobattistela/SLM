from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args
from src.evaluation import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a configured LM checkpoint.")
    add_run_config_argument(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to resume_from, latest.pt, or final.pt.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_config_from_args(args)
    stats = run_evaluation(config, checkpoint_path=args.checkpoint)
    print(f"validation_loss: {stats.loss:.6f}")
    print(f"perplexity: {stats.perplexity:.6f}")


if __name__ == "__main__":
    main()
