# Task 03: Model Architecture and Parameter Counting

## Objective

Implement the decoder-only Transformer model with the requested architecture.

Read `goal.md` completely and use the config system from Task 01.  
Execute only this task.

Do not implement the full training loop yet.

The main goal is to build the model and validate that it is close to **200 million parameters**.

---

## Required Architecture

Implement a decoder-only Transformer language model with:

```text
RoPE positional encoding
GQA attention
Flash Attention support
SwiGLU feed-forward network
RMSNorm
Causal language modeling head
AdamW is not required yet in this task
```

---

## Required Source Structure

Create or adapt:

```text
src/model/
```

Suggested files:

```text
src/model/__init__.py
src/model/transformer.py
src/model/attention.py
src/model/rope.py
src/model/norms.py
src/model/mlp.py
src/model/params.py
```

Use existing project conventions if they are already clean.

---

## Model Requirements

The model must read its architecture from YAML:

```yaml
model:
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

Do not hardcode model dimensions inside the model files.

---

## RMSNorm

Implement RMSNorm.

Expected behavior:

- Normalize by root mean square
- Learnable scale parameter
- Epsilon from config

---

## RoPE

Implement Rotary Positional Embeddings.

Requirements:

- Apply RoPE to query and key tensors
- Support configurable `rope_theta`
- Support sequence length up to `max_seq_len`
- Work with causal attention

---

## Grouped-Query Attention

Implement GQA.

Requirements:

- `num_attention_heads` controls query heads
- `num_key_value_heads` controls key/value heads
- If `num_key_value_heads == 1`, all query groups share a single key/value group
- Support repeating/broadcasting KV heads to match query heads
- Validate that `num_attention_heads % num_key_value_heads == 0`

---

## Flash Attention

Use Flash Attention when available.

Acceptable implementations:

1. PyTorch `torch.nn.functional.scaled_dot_product_attention` with causal mode
2. Flash Attention library if the project already uses it

Requirements:

- Prefer Flash Attention path when config says `flash_attention: true`
- Provide a safe fallback if unavailable and config allows fallback
- Log which attention path is being used
- Do not fail silently

---

## SwiGLU MLP

Implement SwiGLU.

Expected structure:

```text
gate = linear_gate(x)
up = linear_up(x)
hidden = silu(gate) * up
output = linear_down(hidden)
```

The hidden dimension should be derived from:

```yaml
ffn_multiplier: 4
multiple_of: 256
```

Round hidden dimensions to a multiple of `multiple_of`.

---

## Language Modeling Head

Implement:

- Final RMSNorm
- LM head
- Optional weight tying with token embedding

If `tie_embeddings: true`, tie token embedding weights and LM head weights.

---

## Parameter Counter

Fully implement:

```text
scripts/count_parameters.py
```

It must:

1. Load config
2. Build model
3. Count total parameters
4. Count trainable parameters
5. Print module-level estimates when possible:
   - Embedding parameters
   - Attention parameters
   - FFN parameters
   - LM head parameters
6. Check configured acceptable range
7. Fail or warn clearly if outside the acceptable range

Command:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

Expected output example:

```text
Total parameters: 201,234,432
Trainable parameters: 201,234,432
Target parameters: 200,000,000
Acceptable range: 195,000,000 - 205,000,000
Status: OK
```

---

## Optional Debug Script

If useful, create:

```text
scripts/inspect_model.py
```

This may print layer shapes and module names.

---

## README Update

Update the README with:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

Mention that the model must remain within the configured acceptable parameter range.

---

## Testing

The following command must work after this task:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

It must instantiate the model without running training.

---

## Acceptance Criteria

This task is complete when:

- The model implements RoPE
- The model implements GQA
- The model supports Flash Attention or a safe fallback
- The model implements SwiGLU
- The model implements RMSNorm
- The model reads dimensions from YAML
- The parameter counter works
- The model is close to 200M parameters or the config is adjusted to reach the acceptable range
- No dataset tokenization or training loop was implemented in this task
