# How To Run

This guide is the runbook for the approximately 192M SLM FineWeb-Edu pipeline. The main entry
point is the unified YAML config:

```text
configs/train_200m_fineweb_edu.yml
```

All modern scripts read that file with `--run-config`.

## Environment

Create and activate a virtual environment, then install the project:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Install the SuperBPE tokenizer backend before tokenizer or data-preparation
commands:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```

Use Python 3.10-3.12 for this environment. The SuperBPE tokenizer fork depends
on PyO3 0.21, which rejects Python 3.13+ during the Rust build.

For the main DDP run, use a Linux machine with two visible NVIDIA GPUs and a
CUDA-enabled PyTorch build. Check visibility with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Full Pipeline

Run all steps with:

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

`run_all.py` trains or loads the tokenizer, tokenizes FineWeb-Edu, checks the
parameter count, launches DDP when CUDA/NCCL is available, evaluates,
generates plots, and prints a text completion from the latest checkpoint. To prepare everything but print the DDP command instead of
launching it from the script, use:

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml --skip-ddp
```

## Independent Steps

Run each stage explicitly when debugging or resuming a partial run:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/plot_training.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/sample_checkpoint.py --run-config configs/train_200m_fineweb_edu.yml --checkpoint checkpoints/llm_200m_fineweb_edu/latest.pt --prompt "Scientific progress depends on"
```

Retrain the tokenizer from scratch with:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml --force
```

## DDP With 2 GPUs

The required two-GPU training command is:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

Important DDP settings:

- `--nproc_per_node=2` must match `training.distributed.num_gpus: 2`.
- `training.distributed.backend: "nccl"` expects NVIDIA GPUs and CUDA.
- `torchrun` provides `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`; the training code
  uses those values to initialize PyTorch DistributedDataParallel.
- Rank 0 writes metrics and checkpoints. All ranks participate in training,
  validation barriers, and checkpoint barriers.
- If NCCL startup fails, run with `NCCL_DEBUG=INFO` for more diagnostics.

Example with two selected GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

For local CPU or single-GPU checks, use `configs/train_200m_fineweb_edu_debug.yml`
and reduce the distributed settings further if needed. The main config is
intentionally a two-GPU NCCL config.

## Project Settings

Project settings identify the run and centralize output paths.

| Key | Meaning |
| --- | --- |
| `project.name` | Human-readable run name used in output naming. |
| `project.seed` | Base random seed. Training adds the DDP rank to this seed per process. |
| `project.output_dir` | Main output directory for logs and other run outputs. |
| `project.docs_dir` | Documentation directory created by config loading when present. |

## FineWeb-Edu Configuration

FineWeb-Edu data is configured in the `dataset` section.

| Key | Meaning |
| --- | --- |
| `dataset.name` | Hugging Face dataset id. The main run uses `HuggingFaceFW/fineweb-edu`. |
| `dataset.split` | Dataset split to stream, usually `train`. |
| `dataset.text_column` | Column containing text samples. FineWeb-Edu uses `text`. |
| `dataset.streaming` | Whether to stream from Hugging Face instead of downloading the full dataset first. |
| `dataset.target_train_tokens` | Number of training tokens to write to `train_tokens.bin`; the main target is 4B. |
| `dataset.validation_tokens` | Number of validation tokens to write to `val_tokens.bin`. |
| `dataset.validation_salt` | Salt for the deterministic hash split used to choose validation documents. |
| `dataset.cache_dir` | Hugging Face datasets cache directory. |
| `dataset.raw_dir` | Reserved raw-data directory for the project. |
| `dataset.processed_dir` | Output directory for `train_tokens.bin`, `val_tokens.bin`, and `metadata.json`. |
| `dataset.tokenize_num_workers` | Number of parallel processes used to tokenize dataset shards. Can be overridden with `TOKENIZE_DATASET_WORKERS`. |
| `dataset.config_name` | Optional Hugging Face dataset configuration name, if a dataset variant needs one. |

`scripts/train_tokenizer.py` streams up to `tokenizer.train_samples` documents
for the tokenizer corpus. `scripts/tokenize_dataset.py` streams FineWeb-Edu
again, assigns documents to validation with a deterministic hash split, then
writes training and validation tokens until the configured targets are reached.

## Tokenizer Hyperparameters

Tokenizer settings live in the `tokenizer` section.

| Key | Meaning |
| --- | --- |
| `tokenizer.type` | Must be `superbpe`; the config validator rejects other values. |
| `tokenizer.vocab_size` | Final tokenizer vocabulary size. This should match `model.vocab_size`. |
| `tokenizer.min_frequency` | Minimum pair/token frequency used during tokenizer training. |
| `tokenizer.special_tokens.pad_token` | Padding token string. |
| `tokenizer.special_tokens.bos_token` | Beginning-of-sequence token string. |
| `tokenizer.special_tokens.eos_token` | End-of-sequence token string. |
| `tokenizer.special_tokens.unk_token` | Unknown token string passed to the BPE model. |
| `tokenizer.save_dir` | Directory for `tokenizer.json`, vocab/merge files, corpus chunks, and metadata. |
| `tokenizer.train_samples` | Maximum number of FineWeb-Edu samples used to train the tokenizer. |
| `tokenizer.corpus_chunk_samples` | Number of tokenizer-training samples per written corpus chunk file. |
| `tokenizer.corpus_num_workers` | Number of parallel processes used to stream/filter/write tokenizer corpus shards. Can be overridden with `TOKENIZER_CORPUS_WORKERS`. |
| `tokenizer.append_eos` | Whether dataset tokenization appends the configured EOS token to each sample. |
| `tokenizer.superbpe_stage1_vocab_size` | Vocabulary size for SuperBPE stage 1. Must be positive and no larger than `tokenizer.vocab_size`. |
| `tokenizer.superbpe_num_inherit_merges` | Number of stage-1 merges copied into the final SuperBPE stage. |
| `tokenizer.superbpe_stage1_regex` | Optional override for the first-stage SuperBPE pre-tokenization regex. |
| `tokenizer.superbpe_stage2_regex` | Optional override for the second-stage SuperBPE pre-tokenization regex. |
| `tokenizer.allow_unverified_superbpe_backend` | Allows an unverified `tokenizers` package only for explicit development experiments. Keep `false` for final runs. |
| `tokenizer.do_not_fallback_to_standard_bpe_silently` | Documentation/config guard that records the requirement not to replace SuperBPE with standard BPE silently. |

The tokenizer code refuses to continue if the official SuperBPE backend cannot
be verified and `allow_unverified_superbpe_backend` is `false`.

## Model Hyperparameters

Model settings live in the `model` section and define the SLM architecture.

| Key | Meaning |
| --- | --- |
| `model.architecture` | Must be `decoder_only_transformer`. |
| `model.target_parameters` | Intended model size target. |
| `model.acceptable_min_parameters` | Lower bound accepted by the parameter-count check. |
| `model.acceptable_max_parameters` | Upper bound accepted by the parameter-count check. |
| `model.positional_encoding` | Must be `rope`. |
| `model.attention` | Must be `mqa` for multi-query attention. |
| `model.use_flash_attention` | Enables Flash Attention when the installed PyTorch/backend supports it. |
| `model.flash_attention_fallback` | Allows fallback attention when Flash Attention is unavailable. |
| `model.activation` | Must be `swiglu`. |
| `model.normalization` | Must be `rmsnorm`. |
| `model.vocab_size` | Model vocabulary size. Keep aligned with `tokenizer.vocab_size`. |
| `model.max_seq_len` | Context length/block size used for training batches. |
| `model.n_layers` | Number of Transformer blocks. |
| `model.d_model` | Hidden width. Must be divisible by `model.n_heads`. |
| `model.n_heads` | Number of query attention heads. |
| `model.num_kv_heads` | Number of key/value heads. `model.n_heads` must be divisible by this value. |
| `model.ffn_multiplier` | Multiplier used to derive the feed-forward hidden size from `d_model`. |
| `model.multiple_of` | Rounds the derived feed-forward hidden size up to this multiple. |
| `model.norm_eps` | RMSNorm epsilon. |
| `model.rope_theta` | RoPE base frequency. |
| `model.dropout` | Dropout probability. |
| `model.tie_embeddings` | Whether the token embedding and LM head weights are tied. |

Check parameter count before a long run:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

## Training Hyperparameters

Training settings live in the `training` section.

| Key | Meaning |
| --- | --- |
| `training.distributed.enabled` | Requests distributed training. With `WORLD_SIZE=1`, the code warns and continues single-process. |
| `training.distributed.backend` | Distributed backend. Use `nccl` for the main NVIDIA GPU run. |
| `training.distributed.strategy` | Must be `ddp`. |
| `training.distributed.num_gpus` | Expected GPU/process count for `torchrun`; main run uses `2`. |
| `training.precision` | Requested precision: `bf16`, `fp16`, or `fp32`. Unsupported lower precision falls back to `fp32`. |
| `training.compile_model` | Enables `torch.compile` when available. |
| `training.micro_batch_size` | Per-process micro-batch size. |
| `training.gradient_accumulation_steps` | Number of micro-steps accumulated before each optimizer step. |
| `training.max_steps` | Optional hard cap on optimizer steps. `null` means token limit controls the run. |
| `training.max_tokens` | Optional token budget. Total steps are derived from this when `max_steps` is `null`. |
| `training.optimizer.name` | Must be `adamw`. |
| `training.optimizer.learning_rate` | AdamW base learning rate and scheduler peak LR. |
| `training.optimizer.betas` | AdamW beta values. |
| `training.optimizer.eps` | AdamW epsilon. |
| `training.optimizer.weight_decay` | AdamW weight decay. |
| `training.scheduler.name` | Must be `cosine`. |
| `training.scheduler.warmup_steps` | Linear warmup steps before cosine decay. |
| `training.scheduler.min_lr` | Minimum learning rate at the end of cosine decay. |
| `training.gradient_clipping.enabled` | Enables gradient norm clipping. |
| `training.gradient_clipping.max_norm` | Maximum gradient norm when clipping is enabled. |
| `training.checkpointing.save_dir` | Directory for `latest.pt`, `final.pt`, and periodic step checkpoints. |
| `training.checkpointing.save_every_steps` | Checkpoint interval in optimizer steps. |
| `training.checkpointing.keep_last_n` | Number of periodic `step_*.pt` checkpoints to keep. |
| `training.checkpointing.resume_from` | Checkpoint path or selector consumed by the checkpoint loader; `null` starts fresh. |

Effective batch and token accounting:

```text
effective_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
tokens_per_step = effective_batch_size * model.max_seq_len
```

With the main config and two GPUs:

```text
effective_batch_size = 8 * 16 * 2 = 256 sequences
tokens_per_step = 256 * 2048 = 524288 tokens
```

If both `training.max_steps` and `training.max_tokens` are set, training stops
at the smaller derived limit.

## Evaluation, Logging, And Plots

Related runtime settings are configured outside the `training` section.

| Key | Meaning |
| --- | --- |
| `evaluation.enabled` | Enables validation during and after training. |
| `evaluation.eval_every_steps` | Validation interval in optimizer steps. |
| `evaluation.eval_steps` | Number of validation batches per evaluation. |
| `evaluation.metrics` | Metric names expected from evaluation, currently loss and perplexity. |
| `logging.log_every_steps` | Training metrics logging interval. |
| `logging.use_tensorboard` | Enables TensorBoard directory creation. |
| `logging.tensorboard_dir` | TensorBoard output directory. |
| `logging.use_wandb` | Records whether W&B should be used; current scripts do not initialize W&B. |
| `logging.wandb_project` | W&B project name if W&B integration is added/enabled. |
| `plots.enabled` | Records whether plot generation is part of the run. |
| `plots.output_dir` | Directory for generated plot images. |
| `plots.keep_existing_plots` | Records whether existing plot files should be preserved by plotting workflows. |
| `plots.generate` | Plot names to generate, such as train loss, validation loss, perplexity, learning rate, tokens seen, and gradient norm. |

Metrics are appended to:

```text
outputs/llm_200m_fineweb_edu/logs/metrics.jsonl
```

Plots are written to:

```text
outputs/llm_200m_fineweb_edu/plots/
```

## Outputs To Expect

After a complete run, expect these paths:

```text
artifacts/tokenizer/tokenizer.json
artifacts/tokenizer/tokenizer_metadata.json
data/processed/train_tokens.bin
data/processed/val_tokens.bin
data/processed/metadata.json
checkpoints/llm_200m_fineweb_edu/latest.pt
checkpoints/llm_200m_fineweb_edu/final.pt
outputs/llm_200m_fineweb_edu/logs/metrics.jsonl
outputs/llm_200m_fineweb_edu/plots/
```

Do not commit full datasets, checkpoints, tokenizer corpora, or generated run
artifacts.
