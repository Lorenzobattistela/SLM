from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args, resolve_project_path
from src.plotting import generate_training_plots, load_jsonl_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training plots from run metrics.")
    add_run_config_argument(parser)
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Optional metrics JSONL path. Defaults to <project.output_dir>/logs/metrics.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_config_from_args(args)
    plots_cfg = config["plots"]

    if not bool(plots_cfg.get("enabled", True)):
        logging.info("Plotting is disabled by plots.enabled=false.")
        return

    metrics_path = (
        resolve_project_path(args.metrics)
        if args.metrics
        else resolve_project_path(config["project"]["output_dir"]) / "logs" / "metrics.jsonl"
    )
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    output_dir = resolve_project_path(plots_cfg["output_dir"])
    plot_names = plots_cfg.get("generate")

    metrics = load_jsonl_metrics(metrics_path)
    results = generate_training_plots(metrics, output_dir, plot_names=plot_names)

    for result in results:
        if result.skipped:
            print(f"Skipped {result.name}: {result.reason}")
        else:
            print(f"Wrote {result.name}: {result.path} ({result.points} points)")


if __name__ == "__main__":
    main()
