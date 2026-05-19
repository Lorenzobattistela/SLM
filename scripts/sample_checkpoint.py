from __future__ import annotations

import argparse

import torch

from src.config import load_run_config, resolve_project_path
from src.data.tokenizer import get_tokenizer
from src.inference.generate import generate
from src.model.config import ModelConfig
from src.model.transformer import TransformerLM
from src.train.checkpoint import load_checkpoint
from src.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--run-config", type=str, default=None, help="Optional run config override.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to complete.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff.")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximum generated tokens.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device("auto")
    checkpoint_path = resolve_project_path(args.checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.run_config is not None:
        config = load_run_config(args.run_config)
    else:
        config = checkpoint["config"]

    model_cfg = ModelConfig.from_dict(config["model"])
    tokenizer = get_tokenizer(config["data"]["tokenizer"]["name"])
    prompt = args.prompt or config["train"]["sample_prompt"]

    model = TransformerLM(model_cfg).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eot_token_id,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
