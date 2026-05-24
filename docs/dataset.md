# Dataset

The project uses `HuggingFaceFW/fineweb-edu`, a filtered educational subset of FineWeb. It is suitable for this experiment because it is large enough to support the target token budget while keeping the domain focused on higher-quality educational web text.

The configured training target is:

```text
Configured target = 4B training tokens
```

`configs/train_200m_fineweb_edu.yml` sets `dataset.target_train_tokens` to `4000000000` and `dataset.validation_tokens` to `10000000`.

## Processing

The data scripts support streaming/chunked processing so the full dataset does not need to be held in memory. `scripts/tokenize_dataset.py` streams text, loads the pretrained SuperBPE tokenizer, assigns documents to validation with a deterministic hash split, and writes train/validation token files until the configured targets are reached.

Tokenized outputs are written under `data/processed/`:

- `train_tokens.bin`
- `val_tokens.bin`
- `metadata.json`

Run tokenization with:

```bash
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
```
