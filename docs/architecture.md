# Architecture

The main model is a decoder-only Transformer for next-token language modeling. It is configured from `configs/train_200m_fineweb_edu.yml` and implemented under `src/model/`.

## Components

- Token embedding maps SuperBPE token ids into the model dimension.
- Decoder blocks use pre-norm RMSNorm before attention and before the feed-forward layer.
- RoPE applies rotary positional encoding to query and key tensors, avoiding learned absolute position embeddings.
- Attention is Multi-Query Attention: many query heads share a smaller number of key/value heads through `num_kv_heads`.
- Flash Attention is requested through PyTorch scaled dot-product attention when available, with a safe causal fallback when it is not.
- The feed-forward network uses SwiGLU with a rounded hidden dimension controlled by `ffn_multiplier` and `multiple_of`.
- A final RMSNorm feeds the language-modeling head.
- The LM head can be tied to the token embedding matrix through `model.tie_embeddings`.

The configured target is 199,416,000 parameters with the local SuperBPE 50K
vocabulary. Validate the current config with:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

The counter builds the configured model and prints the total, trainable count, module estimates, and `Status: OK` when the result is inside the configured range.
