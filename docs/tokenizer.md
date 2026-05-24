# Tokenizer

The required tokenizer type is SuperBPE. The config enforces
`tokenizer.type: "superbpe"` and
`do_not_fallback_to_standard_bpe_silently: true`.

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
