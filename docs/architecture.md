# Architecture

The language model is a decoder-only Transformer for next-token language
modeling. The two trained 200M experiment configs keep the full project, dataset,
tokenizer, model, training, evaluation, logging, and plotting settings in one
YAML file:

- `pre-train/configs/train_200m_fineweb_edu.yml`
- `pre-train/configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml`

## Components

- Token embeddings map tokenizer ids into the model dimension.
- Decoder blocks use pre-norm RMSNorm before attention and before the
  feed-forward layer.
- RoPE applies rotary positional encoding to query and key tensors, avoiding
  learned absolute position embeddings.
- Attention is Grouped-Query Attention (GQA): query heads are grouped over a
  smaller number of key/value heads through `num_key_value_heads`.
- Flash Attention is requested through PyTorch scaled dot-product attention.
  Training probes the configured CUDA shape at startup, forces the Flash SDPA
  backend when supported, and logs the selected attention backend.
- The feed-forward network uses SwiGLU. Both 200M configs derive `ffn_dim` from
  `ffn_multiplier` and `multiple_of`.
- A final RMSNorm feeds the language-modeling head.
- The LM head is tied to the token embedding matrix in both trained configs, so
  it adds no separate parameters.

## Parameter Accounting

`scripts/count_parameters.py` builds the configured model on the PyTorch `meta`
device, so it can verify model sizes without allocating the full parameter
tensors. Run it against each trained 200M config:

```bash
python scripts/count_parameters.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config pre-train/configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml
```

The counter reports total and trainable parameters, configured target/range,
and module buckets. The module buckets are based on the actual instantiated
module names:

- `tok_embeddings.*`: embedding parameters.
- `*.attn.*`: attention projection parameters.
- `*.mlp.*`: SwiGLU feed-forward parameters.
- `lm_head.*`: untied LM head parameters. This is `0` when tied embeddings are
  enabled because PyTorch exposes the shared weight only once.
- names containing `norm`: RMSNorm weights.

For the current bias-free Transformer blocks, the main formulas are:

- Embedding: `vocab_size * d_model`.
- Attention per layer: `d_model * d_model` for `q_proj`, `d_model *
  (num_key_value_heads * head_dim)` each for `k_proj` and `v_proj`, and
  `d_model * d_model` for `o_proj`.
- FFN per layer: `d_model * ffn_dim` for `gate_proj`, `d_model * ffn_dim` for
  `up_proj`, and `ffn_dim * d_model` for `down_proj`.
- Norms: two RMSNorm weights per block plus one final RMSNorm.

## Trained Models

These counts were verified with `scripts/count_parameters.py` in this worktree.

| Config | Model source | Tokenizer/vocab | Layers | Width | Heads | KV heads | Head dim | FFN dim | Context | Final parameters | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pre-train/configs/train_200m_fineweb_edu.yml` | inline `model` section | SuperBPE, 50,000 | 12 | 1024 | 16 | 4 | 64 | 3200 | 2048 | 200,647,680 | OK |
| `pre-train/configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml` | inline `model` section | GPT-2 byte BPE, 50,257 | 12 | 1024 | 16 | 4 | 64 | 3200 | 2048 | 200,910,848 | OK |

## Module Estimates

The current module-level counts are:

| Config | Embedding | Attention | FFN | LM head | Norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pre-train/configs/train_200m_fineweb_edu.yml` | 51,200,000 | 31,457,280 | 117,964,800 | 0 | 25,600 |
| `pre-train/configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml` | 51,463,168 | 31,457,280 | 117,964,800 | 0 | 25,600 |

Run the counter after changing any architecture or tokenizer vocabulary setting.
`Status: OK` means the instantiated parameter count is inside the configured
acceptable range.
