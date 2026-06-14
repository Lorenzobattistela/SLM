# Dataset

The project uses `HuggingFaceFW/fineweb-edu`, a filtered educational subset of FineWeb. It is suitable for this experiment because it is large enough to support the target token budget while keeping the domain focused on higher-quality educational web text.

The configured training target is:

```text
Configured target = 4B training tokens
```

`pre-train/configs/train_200m_fineweb_edu.yml` sets `dataset.target_train_tokens` to `4000000000` and `dataset.validation_tokens` to `10000000`.

## Processing

The data scripts support streaming/chunked processing so the full dataset does not need to be held in memory. `pre-train/scripts/tokenize_dataset.py` streams text, loads the pretrained SuperBPE tokenizer, assigns documents to validation with a deterministic hash split, and writes train/validation token files until the configured targets are reached.

Tokenized outputs are written under `data/processed/`:

- `train_tokens.bin`
- `val_tokens.bin`
- `metadata.json`

Run tokenization with:

```bash
python pre-train/scripts/tokenize_dataset.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
```

## Byte-Level BPE Retokenization

To compare against a ready byte-level BPE tokenizer without streaming
FineWeb-Edu again, reconstruct text from the existing SuperBPE `.bin` files and
write a second processed dataset:

```bash
python scripts/retokenize_superbpe_to_byte_bpe.py \
  --run-config pre-train/configs/train_200m_fineweb_edu.yml \
  --output-dir data/processed_byte_bpe_gpt2
```

The output directory contains its own `train_tokens.bin`, `val_tokens.bin`, and
`metadata.json`. Train against it with:

```bash
python pre-train/scripts/train.py --run-config pre-train/configs/train_200m_fineweb_edu_byte_bpe_gpt2.yml
```
