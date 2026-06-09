# Task 04: Training Loop with DDP, AdamW, Warmup, and Cosine Decay

## Objective

Implement the training pipeline.

Read `goal.md` completely and use the outputs from Tasks 01, 02, and 03.  
Execute only this task.

The model and tokenizer should already exist before this task starts.

---

## Required Training Features

Implement training with:

- DDP using 2 GPUs
- NCCL backend
- AdamW optimizer
- LR warmup
- LR cosine decay
- Gradient accumulation
- Gradient clipping
- bf16 support when available
- Checkpoint saving
- Checkpoint resume
- Validation loop
- Token counting
- Logging of training metrics

---

## Required Command

The main DDP training command must be:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## Required Source Structure

Create or adapt:

```text
src/training/
src/evaluation/
```

Suggested files:

```text
src/training/__init__.py
src/training/trainer.py
src/training/ddp.py
src/training/optimizer.py
src/training/scheduler.py
src/training/checkpointing.py
src/training/metrics.py

src/evaluation/__init__.py
src/evaluation/evaluator.py
```

Use existing project conventions if they are already clean.

---

## Required Scripts

Create or adapt:

```text
scripts/train.py
scripts/evaluate.py
scripts/run_all.py
```

The independent commands must work:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
```

The full pipeline command must also exist:

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

If `run_all.py` cannot safely launch DDP, it must print the exact command to run training manually.

---

## DDP Requirements

Training must use config:

```yaml
training:
  distributed:
    enabled: true
    backend: "nccl"
    strategy: "ddp"
    num_gpus: 2
```

Requirements:

- Initialize process group correctly
- Set local rank
- Move model to correct GPU
- Wrap model with DDP
- Use distributed sampler or equivalent sharding
- Only rank 0 writes logs, checkpoints, and plots
- Cleanly destroy process group on exit

---

## Optimizer

Use AdamW from config:

```yaml
training:
  optimizer:
    name: "adamw"
    learning_rate: 3.0e-4
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 0.1
```

Do not hardcode optimizer settings.

---

## Scheduler

Implement:

- Linear warmup
- Cosine decay after warmup
- Minimum learning rate

From config:

```yaml
training:
  scheduler:
    name: "cosine"
    warmup_steps: 2000
    min_lr: 3.0e-5
```

The scheduler should work with token-based or step-based stopping.

---

## Training Stop Condition

The training loop must track tokens seen.

Stop when either:

```yaml
training:
  max_tokens: 4000000000
```

or:

```yaml
training:
  max_steps
```

is reached.

If both are configured, stop at whichever comes first.

---

## Loss

Use causal language modeling loss:

- Input: tokens except last token
- Target: tokens shifted by one position
- Cross entropy loss

---

## Precision

Support:

```yaml
training:
  precision: "bf16"
```

If bf16 is unavailable, fall back safely and log the chosen precision.

---

## Gradient Accumulation

Use:

```yaml
training:
  micro_batch_size: 8
  gradient_accumulation_steps: 16
```

Effective batch size should be logged.

---

## Gradient Clipping

Use config:

```yaml
training:
  gradient_clipping:
    enabled: true
    max_norm: 1.0
```

---

## Checkpointing

Use config:

```yaml
training:
  checkpointing:
    save_dir: "checkpoints/llm_200m_fineweb_edu"
    save_every_steps: 1000
    keep_last_n: 5
    resume_from: null
```

Checkpoint must include:

- Model state
- Optimizer state
- Scheduler state
- Step
- Tokens seen
- Config
- RNG state when practical

Resume must restore training state.

---

## Logging

Log at least:

- Step
- Epoch or token progress
- Tokens seen
- Training loss
- Validation loss
- Perplexity
- Learning rate
- Gradient norm
- Samples per second or tokens per second when possible

Only rank 0 should write logs.

---

## Evaluation

Create:

```bash
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
```

It must:

- Load model checkpoint
- Load validation tokens
- Compute loss
- Compute perplexity
- Print results clearly

---

## Full Pipeline Runner

Create:

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

It should run or guide:

1. Tokenizer training or loading
2. Dataset tokenization
3. Parameter count check
4. Training
5. Evaluation
6. Plot generation

If it cannot run DDP internally, it must print:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## README Update

Update the README with:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## Testing

For local testing, use a smaller config or override values:

```yaml
dataset:
  target_train_tokens: 1000000
  validation_tokens: 100000

training:
  max_steps: 20
```

Do not change the default full 4B-token config unless creating a separate debug config.

---

## Acceptance Criteria

This task is complete when:

- DDP training works with 2 GPUs
- AdamW is configured from YAML
- Warmup and cosine decay work
- Gradient accumulation works
- Gradient clipping works
- Checkpoint save and resume work
- Evaluation works
- Token-based stopping works
- Logs include loss, LR, tokens seen, and perplexity
- The full pipeline runner exists
- No plotting rewrite is done in this task unless needed for logging compatibility
