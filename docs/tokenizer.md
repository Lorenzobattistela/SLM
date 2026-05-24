# Tokenizer

The required tokenizer type is SuperBPE. The config enforces
`tokenizer.type: "superbpe"` and
`do_not_fallback_to_standard_bpe_silently: true`.

The pipeline uses the upstream pretrained SuperBPE 200K tokenizer with
transition point `t=180K`:

```text
tokenizer_json/olmo2_p99_truncate_10G_180K_extend_200K_mw4_colon/
```

The tokenizer artifacts are downloaded on first use into:

```text
artifacts/tokenizer_superbpe_200k_t180k/
```

Expected artifacts include `tokenizer.json`, `vocab.json`, `merges.txt`,
`meta.json`, and `tokenizer_metadata.json`. The project no longer trains a
SuperBPE tokenizer on the local FineWeb-Edu corpus.

Install the SuperBPE backend before tokenization commands:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```
