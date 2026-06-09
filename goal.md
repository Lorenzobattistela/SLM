# Goal: Build and Train a 200M Parameter Language Model

## Main Objective

Refactor and implement this project to train a decoder-only language model with approximately **200 million parameters** using a modern Transformer architecture.

Do not get attached to the current files or implementation details.  
Use the existing repository only as a reference for the **directory organization style**.

The final project must support:

1. Tokenization using **SuperBPE**
2. Pretraining on **FineWeb-Edu**
3. A decoder-only Transformer architecture with:
   - RoPE positional encoding
   - Grouped-Query Attention, also called GQA
   - Flash Attention
   - SwiGLU feed-forward layers
   - RMSNorm
   - AdamW optimizer
4. Distributed training using **DDP with 2 GPUs**
5. Learning rate warmup
6. Learning rate cosine decay
7. Existing training plots must be preserved
8. All important hyperparameters must be stored in YAML
9. Two usable execution flows:
   - One full pipeline command that tokenizes and trains
   - Separate commands to run each step independently
10. A `docs/` directory with practical Markdown documentation

---

## Recommended Codex Execution Strategy

Do not ask Codex to implement everything in one single pass.

Use this file as the main project goal, then execute the tasks one by one:

```text
tasks/
  01_config.md
  02_tokenizer.md
  03_model.md
  04_training.md
  05_plots.md
  06_docs.md
```

Recommended prompt to Codex:

```text
Read goal.md completely, but execute only tasks/01_config.md.
Do not implement tokenizer, model, or training yet.
At the end, show the changed files and explain how to test this step.
```

Then continue with each task sequentially.

---

## Dataset Requirement

Use the **FineWeb-Edu** dataset from Hugging Face.

Training target:

```text
Target parameters: 200M
Chinchilla rule: 20 tokens per parameter
Required training tokens: 200M * 20 = 4B tokens
```

FineWeb-Edu contains far more than 4 billion tokens, so the dataset is large enough for this experiment.

The implementation must allow limiting training to approximately **4 billion tokens** through the YAML config.

The data pipeline should support streaming or chunked processing because the dataset is large.

---

## Desired Model Architecture

Implement or refactor the model as a decoder-only Transformer language model.

Required architecture:

```text
Positional Encoder: RoPE
Attention mechanism: GQA with Flash Attention
FFN activation: SwiGLU
Normalization: RMSNorm
Training optimizer: AdamW
Tokenizer: SuperBPE
```

### Architecture Details

The model should include:

- Token embedding
- Multiple Transformer decoder blocks
- RMSNorm before attention
- RMSNorm before FFN
- Causal self-attention
- RoPE applied to query and key tensors
- Grouped-Query Attention:
  - Multiple query heads
  - Shared or reduced key/value heads
  - Configurable `num_key_value_heads`
- Flash Attention when available
- Safe attention fallback if Flash Attention is unavailable
- SwiGLU MLP block
- Final RMSNorm
- Language modeling head
- Optional weight tying between token embedding and LM head

The model must include a parameter counting utility.

The final model should target:

```text
target_parameters: 200_000_000
acceptable_range: 195_000_000 to 205_000_000
```

If the exact architecture does not reach this range, adjust model dimensions through config values, not hardcoded constants.

---

## Suggested Initial Model Configuration

The following is a starting point only. Codex must validate this with a parameter counter and adjust if needed.

```yaml
model:
  architecture: "decoder_only_transformer"
  target_parameters: 200000000
  acceptable_min_parameters: 195000000
  acceptable_max_parameters: 205000000

  positional_encoding: "rope"
  attention: "gqa"
  flash_attention: true
  flash_attention_fallback: true

  activation: "swiglu"
  normalization: "rmsnorm"

  vocab_size: 50000
  max_seq_len: 2048
  n_layers: 12
  d_model: 768
  num_attention_heads: 16
  num_key_value_heads: 4
  ffn_multiplier: 4
  multiple_of: 256
  norm_eps: 1.0e-5
  rope_theta: 10000.0
  dropout: 0.0
  tie_embeddings: true
```

Important:

- Do not silently replace SuperBPE with standard BPE.
- If SuperBPE installation or integration requires a custom dependency, document it clearly.
- If Flash Attention is unavailable, implement a safe fallback and clearly log that fallback attention is being used.

---

## Configuration Requirement

All training, tokenization, dataset, model, optimizer, scheduler, logging, checkpointing, and plotting hyperparameters must be stored in a YAML config file.

Follow the README style using:

```bash
--run-config
```

Create:

```text
configs/train_200m_fineweb_edu.yml
```

The project should be runnable like this:

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

And also step-by-step:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## Required YAML Configuration

Create and adapt this file:

```yaml
project:
  name: "llm_200m_fineweb_edu"
  seed: 42
  output_dir: "outputs/llm_200m_fineweb_edu"
  docs_dir: "docs"

dataset:
  name: "HuggingFaceFW/fineweb-edu"
  split: "train"
  text_column: "text"
  streaming: true
  target_train_tokens: 4000000000
  validation_tokens: 10000000
  cache_dir: "data/cache"
  raw_dir: "data/raw"
  processed_dir: "data/processed"

tokenizer:
  type: "superbpe"
  vocab_size: 50000
  min_frequency: 2
  special_tokens:
    pad_token: "<pad>"
    bos_token: "<bos>"
    eos_token: "<eos>"
    unk_token: "<unk>"
  save_dir: "artifacts/tokenizer"
  train_samples: 2000000
  do_not_fallback_to_standard_bpe_silently: true

model:
  architecture: "decoder_only_transformer"
  target_parameters: 200000000
  acceptable_min_parameters: 195000000
  acceptable_max_parameters: 205000000

  positional_encoding: "rope"
  attention: "gqa"
  flash_attention: true
  flash_attention_fallback: true

  activation: "swiglu"
  normalization: "rmsnorm"

  vocab_size: 50000
  max_seq_len: 2048
  n_layers: 12
  d_model: 768
  num_attention_heads: 16
  num_key_value_heads: 4
  ffn_multiplier: 4
  multiple_of: 256
  norm_eps: 1.0e-5
  rope_theta: 10000.0
  dropout: 0.0
  tie_embeddings: true

training:
  distributed:
    enabled: true
    backend: "nccl"
    strategy: "ddp"
    num_gpus: 2

  precision: "bf16"
  compile_model: false

  micro_batch_size: 8
  gradient_accumulation_steps: 16
  max_steps: null
  max_tokens: 4000000000

  optimizer:
    name: "adamw"
    learning_rate: 3.0e-4
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 0.1

  scheduler:
    name: "cosine"
    warmup_steps: 2000
    min_lr: 3.0e-5

  gradient_clipping:
    enabled: true
    max_norm: 1.0

  checkpointing:
    save_dir: "checkpoints/llm_200m_fineweb_edu"
    save_every_steps: 1000
    keep_last_n: 5
    resume_from: null

evaluation:
  enabled: true
  eval_every_steps: 500
  eval_steps: 100
  metrics:
    - "loss"
    - "perplexity"

logging:
  log_every_steps: 10
  use_tensorboard: true
  tensorboard_dir: "outputs/llm_200m_fineweb_edu/tensorboard"
  use_wandb: false
  wandb_project: "llm_200m_fineweb_edu"

plots:
  enabled: true
  output_dir: "outputs/llm_200m_fineweb_edu/plots"
  keep_existing_plots: true
  generate:
    - "train_loss"
    - "validation_loss"
    - "perplexity"
    - "learning_rate"
    - "tokens_seen"
    - "gradient_norm"
```

---

## Directory Structure

Keep the repository clean.

Use or adapt this structure:

```text
.
├── configs/
│   └── train_200m_fineweb_edu.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── artifacts/
│   └── tokenizer/
│
├── checkpoints/
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── tokenizer.md
│   ├── training.md
│   ├── distributed_training.md
│   ├── configs.md
│   └── running.md
│
├── outputs/
│   └── llm_200m_fineweb_edu/
│       ├── plots/
│       ├── logs/
│       └── tensorboard/
│
├── scripts/
│   ├── run_all.py
│   ├── train_tokenizer.py
│   ├── tokenize_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── count_parameters.py
│
└── src/
    ├── config/
    ├── data/
    ├── tokenizer/
    ├── model/
    ├── training/
    ├── evaluation/
    ├── plotting/
    └── utils/
```

Adapt this to the existing repository if needed, but do not mix everything into a single script.

---

## Required Implementation Areas

The project must include:

1. YAML config loader
2. SuperBPE tokenizer pipeline
3. FineWeb-Edu loading and token limiting
4. Decoder-only Transformer model
5. Parameter counting script
6. DDP training loop
7. AdamW optimizer
8. Warmup and cosine LR scheduler
9. Checkpoint save and resume
10. Evaluation loop
11. Training plots
12. Full pipeline runner
13. Independent step scripts
14. Markdown documentation
15. README update

---

## Full Pipeline Command

The final full pipeline command should be:

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

If `run_all.py` cannot safely launch DDP internally, it must print this command clearly:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## Independent Step Commands

The project must also support:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## Acceptance Criteria

The task is complete only when:

- A YAML config exists for the 200M FineWeb-Edu run
- The model architecture implements RoPE, GQA, Flash Attention, SwiGLU, and RMSNorm
- AdamW is configured through YAML
- LR warmup and cosine decay are configured through YAML
- SuperBPE tokenizer support exists
- FineWeb-Edu loading exists
- Training token target is configurable and set to 4B
- DDP training works with 2 GPUs
- A full pipeline command exists
- Independent step commands exist
- Existing training plots are preserved
- New plots are saved correctly
- A docs directory exists with Markdown documentation
- The README explains the new workflow
- The parameter counter confirms that the model is close to 200M parameters
- No important hyperparameters are hardcoded inside scripts

---

## Important Notes for Codex

Prefer clean, maintainable code over quick hacks.

Do not overfit the implementation to the current files.  
Use the current project structure only as guidance for directory organization.

If something is missing or incompatible, implement the missing piece properly.

If a dependency such as SuperBPE or Flash Attention cannot be installed automatically, document the exact issue and provide a clear fallback or installation instruction.

Do not remove existing useful functionality, especially training plots.

The final project should be runnable, documented, and easy to modify through YAML configs.
