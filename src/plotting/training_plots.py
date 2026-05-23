from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - dependency is checked at runtime.
    plt = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlotSpec:
    name: str
    keys: tuple[str, ...]
    filename: str
    title: str
    ylabel: str
    optional: bool = False


@dataclass(frozen=True)
class PlotResult:
    name: str
    path: Path | None
    points: int
    skipped: bool
    reason: str | None = None


PLOT_SPECS: dict[str, PlotSpec] = {
    "train_loss": PlotSpec(
        name="train_loss",
        keys=("train_loss",),
        filename="train_loss.png",
        title="Training Loss",
        ylabel="Loss",
    ),
    "validation_loss": PlotSpec(
        name="validation_loss",
        keys=("val_loss", "validation_loss"),
        filename="validation_loss.png",
        title="Validation Loss",
        ylabel="Loss",
    ),
    "perplexity": PlotSpec(
        name="perplexity",
        keys=("perplexity",),
        filename="perplexity.png",
        title="Validation Perplexity",
        ylabel="Perplexity",
    ),
    "learning_rate": PlotSpec(
        name="learning_rate",
        keys=("learning_rate", "lr"),
        filename="learning_rate.png",
        title="Learning Rate",
        ylabel="Learning rate",
    ),
    "tokens_seen": PlotSpec(
        name="tokens_seen",
        keys=("tokens_seen",),
        filename="tokens_seen.png",
        title="Tokens Seen",
        ylabel="Tokens",
    ),
    "gradient_norm": PlotSpec(
        name="gradient_norm",
        keys=("gradient_norm", "grad_norm"),
        filename="gradient_norm.png",
        title="Gradient Norm",
        ylabel="Norm",
        optional=True,
    ),
}


def load_jsonl_metrics(path: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(path)
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in metrics file {metrics_path} at line {line_number}: {exc}"
                ) from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _series(metrics: Iterable[dict[str, Any]], spec: PlotSpec) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []
    for index, payload in enumerate(metrics, start=1):
        value = None
        for key in spec.keys:
            value = _coerce_float(payload.get(key))
            if value is not None:
                break
        if value is None:
            continue
        step = _coerce_float(payload.get("step"))
        steps.append(float(index if step is None else step))
        values.append(value)
    return steps, values


def _plot_series(steps: list[float], values: list[float], spec: PlotSpec, output_dir: Path) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Run `pip install -e .` first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec.filename

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, linewidth=1.8)
    plt.xlabel("Step")
    plt.ylabel(spec.ylabel)
    plt.title(spec.title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_training_plots(
    metrics: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    plot_names: Iterable[str] | None = None,
) -> list[PlotResult]:
    rows = list(metrics)
    names = list(plot_names) if plot_names is not None else list(PLOT_SPECS)
    results: list[PlotResult] = []
    target_dir = Path(output_dir)

    for name in names:
        spec = PLOT_SPECS.get(name)
        if spec is None:
            LOGGER.warning("Unknown plot name in config, skipping: %s", name)
            results.append(
                PlotResult(name=name, path=None, points=0, skipped=True, reason="unknown plot")
            )
            continue

        steps, values = _series(rows, spec)
        if not values:
            level = logging.WARNING if spec.optional else logging.WARNING
            LOGGER.log(level, "Metric unavailable for %s; skipping %s", spec.keys, spec.filename)
            results.append(
                PlotResult(
                    name=name,
                    path=None,
                    points=0,
                    skipped=True,
                    reason="metric unavailable",
                )
            )
            continue

        output_path = _plot_series(steps, values, spec, target_dir)
        results.append(
            PlotResult(name=name, path=output_path, points=len(values), skipped=False)
        )

    return results
