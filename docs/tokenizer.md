# Tokenizer

The main run uses SuperBPE. The config records
`do_not_fallback_to_standard_bpe_silently: true` so this path does not silently
turn into standard BPE.

The pipeline uses the locally trained SuperBPE 50K tokenizer created by the
scripts in `tokenizer/`:

```text
artifacts/tokenizer_superbpe_50k_olmo_p99/
```

Train or refresh those artifacts with:

```bash
bash tokenizer/run_tokenizer.sh
```

Expected artifacts include `tokenizer.json`, `vocab.json`, `merges.txt`,
and `tokenizer_metadata.json`. The main run config sets
`tokenizer.train_if_missing: false`, so it fails clearly if this local tokenizer
has not been trained instead of downloading the upstream 200K tokenizer or
training a different tokenizer on FineWeb-Edu.

Install the SuperBPE backend before tokenization commands:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```

## Byte-Level BPE Comparison

For a byte-level BPE comparison from the exact SuperBPE tokenized corpus, use
the reconstruction/retokenization script:

```bash
python scripts/retokenize_superbpe_to_byte_bpe.py \
  --run-config configs/train_200m_fineweb_edu.yml \
  --output-dir data/processed_byte_bpe_gpt2
```

This reads `data/processed/train_tokens.bin` and `val_tokens.bin`, splits them
on the SuperBPE EOS token, decodes each segment back to text, then encodes that
text with the ready GPT-2 byte-level BPE tokenizer from `tiktoken`. GPT-2 is the
standard ready 50K-class byte-level BPE tokenizer; its actual vocabulary size is
`50257`, so use `model.vocab_size: 50257` for this comparison.

The matching training config is:

```bash
configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml
```
