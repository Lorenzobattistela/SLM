# Tokenizer

The required tokenizer type is SuperBPE. The config enforces `tokenizer.type: "superbpe"` and `do_not_fallback_to_standard_bpe_silently: true`.

SuperBPE is different from regular BPE because it uses the SuperBPE backend from the upstream project rather than silently substituting the standard Hugging Face BPE implementation. This matters for reproducibility: if the SuperBPE dependency is missing, the tokenizer scripts stop with an explicit error.

## Training

Tokenizer training reads FineWeb-Edu text through the configured dataset stream, writes corpus chunks, trains the SuperBPE tokenizer, then saves artifacts under:

```text
artifacts/tokenizer/
```

Expected artifacts include the tokenizer model files and `tokenizer_metadata.json`. Existing tokenizer artifacts are reused unless `--force` is passed.

Run:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
```

Install the SuperBPE backend before running tokenizer commands:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```
