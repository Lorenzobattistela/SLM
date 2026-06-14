from __future__ import annotations

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

import torch  # noqa: E402

from src.config import ConfigError, add_run_config_argument, load_run_config  # noqa: E402
from src.model import ModelConfig, TransformerLM, describe_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the configured language model and count its parameters."
    )
    add_run_config_argument(parser)
    return parser.parse_args()


def _fmt(value: int | None) -> str:
    if value is None:
        return "not configured"
    return f"{value:,}"


def _build_model(model_config: ModelConfig) -> TransformerLM:
    with torch.device("meta"):
        return TransformerLM(model_config)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        config = load_run_config(args.run_config, create_dirs=False)
        model_config = ModelConfig.from_dict(config["model"])
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Config error in {args.run_config}: {exc}") from None

    model = _build_model(model_config)
    report = describe_parameters(model)

    min_parameters = model_config.acceptable_min_parameters
    max_parameters = model_config.acceptable_max_parameters

    print(f"Loaded config: {args.run_config}")
    print("Architecture: decoder-only Transformer")
    print(
        "Dimensions: "
        f"attention={model_config.attention}, "
        f"layers={model_config.n_layers}, "
        f"d_model={model_config.d_model}, "
        f"num_attention_heads={model_config.n_heads}, "
        f"num_key_value_heads={model_config.n_kv_heads}, "
        f"head_dim={model_config.head_dim}, "
        f"ffn_dim={model_config.ffn_dim}, "
        f"max_seq_len={model_config.context_length}"
    )
    print(f"Total parameters: {_fmt(report.total)}")
    print(f"Trainable parameters: {_fmt(report.trainable)}")
    print(f"Target parameters: {_fmt(model_config.target_parameters)}")
    print(f"Acceptable range: {_fmt(min_parameters)} - {_fmt(max_parameters)}")
    print("Module estimates:")
    print(f"  Embedding parameters: {_fmt(report.embedding)}")
    print(f"  Attention parameters: {_fmt(report.attention)}")
    print(f"  FFN parameters: {_fmt(report.ffn)}")
    lm_head_note = " (tied to token embedding)" if report.tied_embeddings else ""
    print(f"  LM head parameters: {_fmt(report.lm_head)}{lm_head_note}")
    print(f"  Norm parameters: {_fmt(report.norm)}")
    if report.other:
        print(f"  Other parameters: {_fmt(report.other)}")

    if min_parameters is None or max_parameters is None:
        print("Status: WARNING (acceptable parameter range is not configured)")
        return

    if min_parameters <= report.total <= max_parameters:
        print("Status: OK")
        return

    print("Status: OUT_OF_RANGE")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
