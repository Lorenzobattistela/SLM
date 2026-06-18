from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
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
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help=(
            "Optional training metadata JSON path. Defaults to "
            "<project.output_dir>/logs/training_metadata.json when present."
        ),
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
    metadata_path = (
        resolve_project_path(args.metadata)
        if args.metadata
        else resolve_project_path(config["project"]["output_dir"])
        / "logs"
        / "training_metadata.json"
    )
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        logging.info(
            "Loaded training metadata: parameters=%s attention=%s backend=%s",
            metadata.get("parameters", "unknown"),
            metadata.get("model", {}).get("attention", "unknown"),
            metadata.get("attention_optimization", {}).get("backend", "unknown"),
        )
    else:
        logging.info("Training metadata file not found, plotting metrics only: %s", metadata_path)

    metrics = load_jsonl_metrics(metrics_path)
    results = generate_training_plots(metrics, output_dir, plot_names=plot_names)

    for result in results:
        if result.skipped:
            print(f"Skipped {result.name}: {result.reason}")
        else:
            print(f"Wrote {result.name}: {result.path} ({result.points} points)")


if __name__ == "__main__":
    main()
