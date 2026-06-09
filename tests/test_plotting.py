from __future__ import annotations

import json
from pathlib import Path

from src.plotting import generate_training_plots, load_jsonl_metrics


def test_generate_training_plots_skips_missing_optional_metric(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "step": 1,
            "tokens_seen": 128,
            "train_loss": 4.2,
            "train_perplexity": 66.7,
            "learning_rate": 1.0e-4,
            "tokens_per_second": 1000.0,
            "step_time_seconds": 0.25,
            "token_progress_percent": 50.0,
            "epoch_equivalent": 0.5,
        },
        {
            "step": 2,
            "tokens_seen": 256,
            "validation_loss": 4.0,
            "validation_perplexity": 54.6,
            "learning_rate": 9.0e-5,
        },
    ]
    metrics_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    metrics = load_jsonl_metrics(metrics_path)
    results = generate_training_plots(
        metrics,
        tmp_path / "plots",
        plot_names=[
            "train_loss",
            "validation_loss",
            "perplexity",
            "train_perplexity",
            "learning_rate",
            "tokens_seen",
            "gradient_norm",
            "tokens_per_second",
            "step_time_seconds",
            "token_progress",
            "epoch_equivalent",
        ],
    )

    written = {result.name: result for result in results if not result.skipped}
    skipped = {result.name: result for result in results if result.skipped}

    assert (tmp_path / "plots" / "train_loss.png").exists()
    assert (tmp_path / "plots" / "validation_loss.png").exists()
    assert (tmp_path / "plots" / "perplexity.png").exists()
    assert (tmp_path / "plots" / "train_perplexity.png").exists()
    assert (tmp_path / "plots" / "learning_rate.png").exists()
    assert (tmp_path / "plots" / "tokens_seen.png").exists()
    assert (tmp_path / "plots" / "tokens_per_second.png").exists()
    assert (tmp_path / "plots" / "step_time_seconds.png").exists()
    assert (tmp_path / "plots" / "token_progress.png").exists()
    assert (tmp_path / "plots" / "epoch_equivalent.png").exists()
    assert written["train_loss"].points == 1
    assert skipped["gradient_norm"].reason == "metric unavailable"
