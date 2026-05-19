from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - dependency is checked at runtime.
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train and validation loss from metrics.jsonl.")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Optional output PNG path")
    return parser.parse_args()


def main() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Run `pip install -e .` first.")

    args = parse_args()
    metrics_path = Path(args.metrics)
    output_path = Path(args.output) if args.output else metrics_path.with_suffix(".png")

    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "train_loss" in payload:
                train_steps.append(payload["step"])
                train_losses.append(payload["train_loss"])
            if "val_loss" in payload:
                val_steps.append(payload["step"])
                val_losses.append(payload["val_loss"])

    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label="train_loss")
    if val_steps:
        plt.plot(val_steps, val_losses, label="val_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
