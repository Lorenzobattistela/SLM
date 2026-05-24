# FineWeb-Edu Language Model

This project builds a decoder-only language model with a FineWeb-Edu
pretraining pipeline. The main run is configured through YAML and is intended
for a Linux machine with 2 NVIDIA GPUs.

## Objective

- Train a decoder-only Transformer for next-token prediction.
- Use the pretrained SuperBPE 200K tokenizer with transition point t=180K
  without silently falling back to standard BPE.
- Pretrain on FineWeb-Edu with a 4B token target.
- Support full-pipeline and independent-step execution.
- Preserve training plots and write practical project documentation under `docs/`.

## Architecture

The model uses RoPE positional encoding, Multi-Query Attention, Flash Attention when available with a safe fallback, SwiGLU feed-forward layers, RMSNorm, and a tied language-modeling head when enabled. With the pretrained 200K tokenizer, the configured parameter count is 343,420,800.

Check the parameter count with:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

## Dataset And Tokens

The main dataset is `HuggingFaceFW/fineweb-edu`. The token target is:

```text
Configured target = 4B training tokens
```

The main config writes processed token files to `data/processed/train_tokens.bin`, `data/processed/val_tokens.bin`, and `data/processed/metadata.json`.

## Environment

```bash
uv venv --python /usr/bin/python3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Install the SuperBPE backend before tokenization commands:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```

Use Python 3.10-3.12 for this project. The SuperBPE tokenizer fork depends on
PyO3 0.21, which does not build against Python 3.13+.

## YAML Config Usage

The main config is `configs/train_200m_fineweb_edu.yml`. All modern scripts use:

```bash
--run-config configs/train_200m_fineweb_edu.yml
```

It contains `project`, `dataset`, `tokenizer`, `model`, `training`, `evaluation`, `logging`, and `plots` sections.

## Full Pipeline

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

`run_all.py` tokenizes data with the configured pretrained tokenizer, checks parameters, launches DDP when CUDA/NCCL is available, evaluates, generates plots, and prints a text completion from the latest checkpoint. If DDP cannot be launched safely, it prints the manual command.

## Independent Steps

```bash
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/plot_training.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/sample_checkpoint.py --run-config configs/train_200m_fineweb_edu.yml --checkpoint checkpoints/llm_200m_fineweb_edu/latest.pt --prompt "Scientific progress depends on"
```

The DDP training command for the 2-GPU run is:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

## Metrics And Plots

Training metrics are written to:

```text
outputs/llm_200m_fineweb_edu/logs/metrics.jsonl
```

Plots are written to:

```text
outputs/llm_200m_fineweb_edu/plots/
```

Generate plots with:

```bash
python scripts/plot_training.py --run-config configs/train_200m_fineweb_edu.yml
```

The legacy `scripts/plot_train_loss.py` script is preserved for older metrics workflows.

## Text Generation

Generate qualitative completions from a trained checkpoint with:

```bash
python scripts/sample_checkpoint.py --run-config configs/train_200m_fineweb_edu.yml --checkpoint checkpoints/llm_200m_fineweb_edu/latest.pt --prompt "Scientific progress depends on"
```

## Documentation

See `docs/` for practical notes:

- `docs/architecture.md`
- `docs/dataset.md`
- `docs/tokenizer.md`
- `docs/training.md`
- `docs/distributed_training.md`
- `docs/configs.md`
- `docs/running.md`
- `docs/how-to-run.md`

## Notes

- Do not commit full datasets, checkpoints, or generated run artifacts.
- Use the debug config for small local checks.
- Do not claim a full training run is complete unless the training command has actually been run to completion.
