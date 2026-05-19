from __future__ import annotations

import argparse

import numpy as np
import torch

from src.config import load_run_config, resolve_project_path
from src.data.dataset import TokenShardDataset
from src.model.config import ModelConfig
from src.model.transformer import TransformerLM
from src.train.optim import build_optimizer
from src.utils import autocast_context, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a single batch to smoke test the trainer.")
    parser.add_argument("--run-config", type=str, required=True, help="Run config to use.")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.run_config)
    train_cfg = config["train"]
    model_cfg = ModelConfig.from_dict(config["model"])
    device = select_device(train_cfg.get("device", "auto"))

    set_seed(int(train_cfg["seed"]))
    train_data = TokenShardDataset(
        resolve_project_path(config["data"]["storage"]["output_dir"]) / "train",
        block_size=model_cfg.context_length,
    )
    rng = np.random.default_rng(int(train_cfg["seed"]))
    inputs, targets = train_data.sample_batch(int(train_cfg["batch_size"]), device, rng)

    model = TransformerLM(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_cfg["precision"]):
            _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"step={step} loss={loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
