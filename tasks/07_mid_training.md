# Task 07: Mid-Training on SmolTalk and GSM8K

## Objective

Implement the mid-training phase to specialize the Small Language Model (SLM) on conversational and reasoning tasks. The weights must be initialized from the final pre-training checkpoint (loading weights only, discarding optimizer states).

---

## Required Features

The mid-training pipeline must support:

1. **Dataset Mix**:
   - Load **SmolTalk** (`HuggingFaceTB/smoltalk`) for conversational quality.
   - Load **GSM8K** (`gsm8k`) for mathematical reasoning.
   - Allow blending these datasets in a configurable ratio (e.g., 70% SmolTalk / 30% GSM8K) inside the YAML configuration.

2. **Chat Templating**:
   - Format conversational and reasoning sequences into standard chat format:
     ```text
     <bos>User: {prompt}\nAssistant: {response}<eos>
     ```
   - Ensure the template matches the model tokenizer's special tokens.

3. **Sequence Packing**:
   - Concatenate multiple conversational sessions separated by `<eos>` tokens and pack them into uniform sequences matching the model's context length (e.g., 2048 tokens). This ensures efficient GPU utilization by minimizing padding tokens.

4. **Training settings**:
   - Initialize model weights from the pre-training checkpoint (weights only, discarding optimizer state).
   - Use a lower learning rate than pre-training (e.g., `1e-4` to `3e-4`) with a short warm-up.
   - Train for approximately 1 epoch (or about 100M to 300M tokens).
   - Track and log loss curves for both training and validation splits.

---

## Required Command

The mid-training script must be executable using DDP:

```bash
torchrun --standalone --nproc_per_node=2 scripts/mid_train.py --run-config configs/mid_train_200m.yml
```

---

## Required Source Structure

Create or adapt the following files:

```text
configs/mid_train_200m.yml
scripts/mid_train.py
src/data/mid_train_dataset.py
```

### Config File (`configs/mid_train_200m.yml`)
Add configuration parameters for the mid-training run:
```yaml
project:
  name: "llm_200m_mid_training"

dataset:
  smoltalk: "HuggingFaceTB/smoltalk"
  gsm8k: "gsm8k"
  mix_ratio:
    smoltalk: 0.7
    gsm8k: 0.3
  context_length: 2048

training:
  learning_rate: 1.5e-4
  epochs: 1
  optimizer: "adamw"
  # Add other typical training settings matching configs/train_200m_fineweb_edu.yml
```

### Dataset Loader (`src/data/mid_train_dataset.py`)
Implement dataset loading, formatting with chat templates, tokenization, and sequence packing.

### Training Script (`scripts/mid_train.py`)
Create a CLI script that:
1. Loads the configuration via `--run-config`.
2. Initializes the model and loads pretrained weights.
3. Prepares data loaders for mid-training data.
4. Executes the training loop with DDP.
5. Saves the final mid-training checkpoint.

---

## Testing

Verify mid-training functionality with a fast debug configuration:

```bash
python scripts/mid_train.py --run-config configs/mid_train_200m_debug.yml
```

Expected outcomes:
- Dataset is loaded and tokenized correctly.
- Pretrained weights are loaded successfully without loading old optimizer states.
- The model trains on the mixed dataset and saves checkpoints.

---

## Acceptance Criteria

This task is complete when:
- `scripts/mid_train.py` is fully functional.
- SmolTalk and GSM8K datasets are successfully loaded, formatted with chat templates, packed, and used for training.
- Checkpoints are saved successfully.
