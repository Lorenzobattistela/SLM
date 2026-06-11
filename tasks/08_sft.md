# Task 08: Supervised Fine-Tuning (SFT) with Loss Masking

## Objective

Implement the Supervised Fine-Tuning (SFT) phase to align the SLM to follow instructions and respond in a helpful, conversational format. This step initializes from the final mid-training checkpoint and implements prompt loss masking so the model is only evaluated and updated based on generating the assistant's response.

---

## Required Features

The SFT pipeline must support:

1. **Dataset Selection**:
   - Use **SmolTalk** (`HuggingFaceTB/smoltalk`) or a curated subset for instruction following.

2. **Prompt Loss Masking**:
   - Format conversational sequences into the chat template:
     ```text
     <bos>User: {prompt}\nAssistant: {response}<eos>
     ```
   - Implement loss masking: tokens belonging to the prompt (`User: {prompt}\nAssistant:`) must have their target label replaced with `-100` (PyTorch CrossEntropyLoss ignore index).
   - Only tokens belonging to the response (`{response}<eos>`) should retain their token IDs as targets for gradient updates.

3. **Training Configuration**:
   - Load model weights from the final checkpoint of Task 07 (mid-training).
   - Set a lower learning rate (e.g., `1e-5` to `3e-5`) with a short warm-up.
   - Train for 1 to 2 epochs, monitoring validation loss closely to avoid overfitting.
   - Save the final SFT checkpoint.

---

## Required Command

The SFT script must be executable using DDP:

```bash
torchrun --standalone --nproc_per_node=2 scripts/sft.py --run-config configs/sft_200m.yml
```

---

## Required Source Structure

Create or adapt the following files:

```text
configs/sft_200m.yml
scripts/sft.py
src/training/sft_trainer.py
```

### Config File (`configs/sft_200m.yml`)
Add SFT-specific configuration parameters:
```yaml
project:
  name: "llm_200m_sft"

dataset:
  sft_dataset: "HuggingFaceTB/smoltalk"
  context_length: 2048

training:
  learning_rate: 2e-5
  epochs: 2
  optimizer: "adamw"
  # Add other typical training settings
```

### Trainer script (`src/training/sft_trainer.py`)
Implement the training run for SFT, applying target masking on prompt tokens to ensure they are excluded from the loss calculation.

### CLI Script (`scripts/sft.py`)
A CLI script that runs the SFT training loop on the target model.

---

## Testing

Verify that loss masking behaves correctly. You can test SFT with a fast debug configuration:

```bash
python scripts/sft.py --run-config configs/sft_200m_debug.yml
```

Expected outcomes:
- The data loader produces targets where prompt tokens are set to `-100`.
- Model weights are initialized from the mid-training checkpoint.
- SFT completes and outputs checkpoints under `outputs/llm_200m_sft/`.

---

## Acceptance Criteria

This task is complete when:
- `scripts/sft.py` and `src/training/sft_trainer.py` are fully functional.
- Target masking is verified (prompt tokens do not contribute to loss computation).
- Model weights are initialized from the mid-training checkpoint.
- The model trains and saves checkpoints successfully.
