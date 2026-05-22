# Small Language Model - Pretraining Scaffold

This repository contains the first-stage pretraining pipeline for the Deep Learning II SLM project. The initial focus is a decoder-only Transformer trained with next-token prediction, plus the tooling needed to validate the pipeline locally on tiny datasets before launching larger runs on a Linux machine with NVIDIA GPUs.

## Current Scope

- Streaming data preparation from Hugging Face datasets
- Deterministic train/validation split
- SuperBPE tokenization for FineWeb-Edu with chunked token files
- Decoder-only Transformer with RoPE, MQA, Flash Attention fallback, SwiGLU and RMSNorm
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

SuperBPE uses the official SuperBPE tokenizer backend, which is a custom fork of
Hugging Face `tokenizers`. Install it in a dedicated environment before running
the tokenizer scripts:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```

The pipeline refuses to silently fall back to standard BPE. If the backend is
missing, `scripts/train_tokenizer.py` and `scripts/tokenize_dataset.py` stop with
a clear error.

## Config Layout

- `configs/train_200m_fineweb_edu.yml`: YAML config for the 200M FineWeb-Edu run
- `configs/model/`: architecture presets
- `configs/data/`: dataset/token-budget presets
- `configs/run/`: runnable training presets that compose model + data + optimizer settings

The 200M FineWeb-Edu flow uses the shared `--run-config` pattern:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

The main FineWeb-Edu config targets 4B training tokens and 10M validation tokens.
Tokenized outputs are written to `data/processed/train_tokens.bin`,
`data/processed/val_tokens.bin`, and `data/processed/metadata.json`.
The parameter counter builds the configured model and must report `Status: OK`
inside the configured 195M-205M acceptable range before training.
Training logs are written as JSONL metrics under
`outputs/llm_200m_fineweb_edu/logs/metrics.jsonl`, and checkpoints are saved
under `checkpoints/llm_200m_fineweb_edu/`.

`scripts/run_all.py` prepares the tokenizer/data, checks the parameter count,
and launches the configured DDP command when CUDA/NCCL is available. On a machine
where DDP cannot be launched safely, it prints the exact `torchrun` command to
run manually.

For a small end-to-end check without changing the main 4B-token config, use:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu_debug.yml --force
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu_debug.yml
```

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
- The 200M FineWeb-Edu path is configured for SuperBPE. Legacy tiny/debug
  pretraining presets may still use the older GPT-2 tokenizer until those flows
  are migrated.
