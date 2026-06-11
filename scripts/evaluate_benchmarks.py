from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.config.loader import load_yaml, resolve_project_path
from src.tokenizer import load_tokenizer
from src.model import ModelConfig, TransformerLM
from src.training.checkpointing import load_checkpoint
from src.evaluation.benchmarks import run_all_benchmarks

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on downstream benchmarks.")
    parser.add_argument(
        "--run-config",
        type=str,
        required=True,
        help="Path to evaluation YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint. If omitted, evaluates a randomly initialized model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override the maximum number of samples to evaluate per dataset (e.g. for fast debugging).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_yaml(args.run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # Initialize tokenizer
    tokenizer = load_tokenizer(config["tokenizer"])

    # Load model config and instantiate model
    # To evaluate a checkpoint, we should extract the model architecture config from the checkpoint,
    # or fallback to a standard config if we're evaluating a custom model size.
    # We will use the model configuration from configs/mid_train_200m.yml or similar as fallback.
    # Let's search if model key is in config. If not, we look at the checkpoint config.
    checkpoint_path = resolve_project_path(args.checkpoint) if args.checkpoint else None
    
    model_cfg_dict = None
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "config" in checkpoint and "model" in checkpoint["config"]:
                model_cfg_dict = checkpoint["config"]["model"]
                LOGGER.info("Loaded model architecture config from checkpoint.")
        except Exception as e:
            LOGGER.warning("Could not read config from checkpoint: %s. Falling back to default architecture.", e)

    if model_cfg_dict is None:
        # Load from config or create a fallback 200m architecture
        model_cfg_dict = config.get("model")
        if model_cfg_dict is None:
            # Fallback to standard 200M architecture
            model_cfg_dict = {
                "architecture": "decoder_only_transformer",
                "positional_encoding": "rope",
                "attention": "gqa",
                "flash_attention": False,
                "activation": "swiglu",
                "normalization": "rmsnorm",
                "vocab_size": 50000,
                "max_seq_len": 2048,
                "n_layers": 12,
                "d_model": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "ffn_multiplier": 3.125,
                "multiple_of": 64,
                "norm_eps": 1.0e-5,
                "rope_theta": 10000.0,
                "dropout": 0.0,
                "tie_embeddings": True,
            }

    model_cfg = ModelConfig.from_dict(model_cfg_dict)
    model = TransformerLM(model_cfg).to(device)

    # Load checkpoint weights
    if checkpoint_path and checkpoint_path.exists():
        load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            map_location=device,
            restore_rng=False,
        )
        LOGGER.info("Successfully loaded checkpoint weights from %s", checkpoint_path)
    else:
        LOGGER.warning("No checkpoint loaded. Running evaluation on randomly initialized weights.")

    # Read evaluation parameters
    eval_cfg = config.get("evaluation", {})
    benchmarks_list = eval_cfg.get("benchmarks", ["perplexity"])
    ppl_dataset = eval_cfg.get("ppl_dataset", "HuggingFaceFW/fineweb-edu")
    
    limit = args.limit if args.limit is not None else eval_cfg.get("limit", 10)

    # Run evaluations
    LOGGER.info("Running evaluation suite with limit=%s...", limit)
    results = run_all_benchmarks(
        model=model,
        tokenizer=tokenizer,
        benchmarks_list=benchmarks_list,
        ppl_dataset=ppl_dataset,
        device=device,
        limit=limit,
    )

    # Print results
    print("\n================ EVALUATION RESULTS ================")
    for k, v in results.items():
        if k == "perplexity":
            print(f"{k.capitalize()}: {v:.4f}")
        else:
            print(f"{k.replace('_', ' ').title()}: {v * 100:.2f}%")
    print("====================================================\n")

    # Save results to JSON file
    output_dir = resolve_project_path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "evaluation_results.json"
    
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    LOGGER.info("Saved evaluation results to %s", output_file)


if __name__ == "__main__":
    main()
