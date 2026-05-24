# Task 06: Documentation and README Update

## Objective

Create practical project documentation and update the README.

Read `goal.md` completely and use the final implementation from Tasks 01 through 05.  
Execute only this task.

Do not rewrite model, tokenizer, training, or plotting logic unless documentation reveals a small broken command that must be fixed.

---

## Required Docs Directory

Create:

```text
docs/
```

Required files:

```text
docs/architecture.md
docs/dataset.md
docs/tokenizer.md
docs/training.md
docs/distributed_training.md
docs/configs.md
docs/running.md
```

Each file should be concise, practical, and useful for someone running the project.

---

## docs/architecture.md

Explain:

- Decoder-only Transformer architecture
- RoPE
- GQA
- Flash Attention
- SwiGLU
- RMSNorm
- LM head
- Parameter target
- How to run the parameter counter

Include:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## docs/dataset.md

Explain:

- FineWeb-Edu dataset
- Why it is suitable for this experiment
- 200M parameters times 20 tokens per parameter
- 4B token target
- Streaming or chunked processing
- Train and validation split/token handling
- Output files under `data/processed/`

Include:

```bash
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## docs/tokenizer.md

Explain:

- SuperBPE requirement
- Why SuperBPE is different from regular BPE
- How tokenizer training works
- Where tokenizer artifacts are saved
- How missing dependencies are handled

Include:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## docs/training.md

Explain:

- AdamW
- Learning rate warmup
- Cosine decay
- Gradient accumulation
- Gradient clipping
- bf16
- Checkpoints
- Resume behavior
- Metrics logging

Include:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## docs/distributed_training.md

Explain:

- DDP
- Why `torchrun` is used
- Why `nproc_per_node=2`
- NCCL backend
- Rank 0 logging/checkpointing
- Common GPU troubleshooting notes

Include:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## docs/configs.md

Explain:

- YAML config structure
- `--run-config`
- Main sections:
  - `project`
  - `dataset`
  - `tokenizer`
  - `model`
  - `training`
  - `evaluation`
  - `logging`
  - `plots`
- How to create a smaller debug config

Include example:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## docs/running.md

Explain both execution modes.

### Full Pipeline

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

If DDP must be launched separately, document the command printed by `run_all.py`.

### Independent Steps

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/plot_training.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## README Update

Update the root README with:

1. Project objective
2. Architecture summary
3. Dataset summary
4. Parameter target
5. Token target
6. YAML config usage
7. Full pipeline command
8. Independent step commands
9. DDP command for 2 GPUs
10. Plot output location
11. Documentation directory reference

The README must clearly include:

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

and:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## Final Validation Checklist

At the end of this task, report:

- Docs created
- README updated
- Commands documented
- Any missing implementation detail found during documentation

Do not claim that training was completed unless it was actually run.

---

## Acceptance Criteria

This task is complete when:

- `docs/` exists
- All required Markdown files exist
- README is updated
- Full pipeline command is documented
- Independent commands are documented
- DDP command is documented
- Plot output location is documented
- The documentation matches the actual implemented scripts and paths
