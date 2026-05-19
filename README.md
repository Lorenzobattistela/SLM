# Small Language Model - Pretraining Scaffold

This repository contains the first-stage pretraining pipeline for the Deep Learning II SLM project. The initial focus is a decoder-only Transformer trained with next-token prediction, plus the tooling needed to validate the pipeline locally on tiny datasets before launching larger runs on a Linux machine with NVIDIA GPUs.

## Current Scope

- Streaming data preparation from Hugging Face datasets
- Deterministic train/validation split
- GPT-2 BPE tokenization with packed token shards
- Decoder-only Transformer with RoPE, GQA, SwiGLU and RMSNorm
- Pretraining loop with checkpointing, validation perplexity and text sampling
- Tiny/debug/pilot/full presets for local and remote execution

## Environment

Create a virtual environment and install dependencies:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

MacBook runs should stay on `tiny` or short `debug` checks. Real pretraining should run on the Linux box with CUDA.

## Config Layout

- `configs/model/`: architecture presets
- `configs/data/`: dataset/token-budget presets
- `configs/run/`: runnable training presets that compose model + data + optimizer settings

## Suggested Workflow

1. Prepare a tiny local dataset:

```bash
python3 scripts/prepare_pretrain_data.py --data-config configs/data/fineweb_edu_tiny.yaml
```

2. Overfit a single batch to validate the pipeline:

```bash
python3 scripts/smoke_overfit_batch.py --run-config configs/run/pretrain_local_tiny.yaml
```

3. Run a tiny local training session:

```bash
python3 -m src.train.pretrain --run-config configs/run/pretrain_local_tiny.yaml
```

4. Sample from the latest checkpoint:

```bash
python3 scripts/sample_checkpoint.py \
  --checkpoint runs/pretrain_local_tiny/checkpoints/latest.pt \
  --prompt "Language models are useful because"
```

5. Plot training loss:

```bash
python3 scripts/plot_train_loss.py --metrics runs/pretrain_local_tiny/metrics.jsonl
```

## Remote GPU Runs

Once the Linux machine is ready and the dataset has been prepared there, use:

```bash
bash scripts/launch_pretrain_ddp.sh configs/run/pretrain_remote_full_2gpu.yaml
```

This launches `torchrun` with the run config already set up for a 2-GPU pretraining job.

## Notes

- Generated artifacts live under `data/processed/` and `runs/`.
- The full dataset should not be committed.
- Checkpoints should be uploaded externally for the course delivery.
- For the first milestone, GPT-2 BPE is the safest tokenizer choice. If needed, a custom tokenizer can be added later without rewriting the trainer.
