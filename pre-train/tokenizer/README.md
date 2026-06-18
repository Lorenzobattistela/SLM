# SuperBPE 50K Training

This directory contains the project-local trainer for a 50,000-token SuperBPE
tokenizer.

## Source And Dataset

Implementation source: the code calls this repository's SuperBPE wrapper, which
uses the official PythonNut/SuperBPE `tokenizers` backend:

```text
https://github.com/PythonNut/superbpe
```

Training dataset configured here:

```text
UW/olmo-mix-1124-subset-p99
```

That dataset is the OLMo Mix p99 subset referenced by the official SuperBPE
training instructions. This config does not train on FineWeb-Edu by default.

## Defaults

```text
final vocabulary: 50000
transition point: 40000
training samples: 250000
process memory cap: 20 GiB
output directory: pre-train/artifacts/tokenizer_superbpe_50k_olmo_p99/
config: pre-train/tokenizer/superbpe_50k_olmo_p99.yml
```

The 40K transition follows the same pattern as the 200K/t=180K tokenizer: the
SuperBPE stage begins 10K tokens before the final vocabulary size.

## Usage

Install the official SuperBPE backend first:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git /tmp/superbpe
pip install -e /tmp/superbpe/tokenizers_superbpe/bindings/python/
```

Train with the default config:

```bash
bash pre-train/tokenizer/run_tokenizer.sh
```

Useful overrides:

```bash
TOKENIZER_CORPUS_WORKERS=4 bash pre-train/tokenizer/run_tokenizer.sh
TOKENIZER_TRAIN_SAMPLES=500000 bash pre-train/tokenizer/run_tokenizer.sh
TOKENIZER_MEMORY_LIMIT_GB=32 bash pre-train/tokenizer/run_tokenizer.sh
TOKENIZER_FORCE=1 bash pre-train/tokenizer/run_tokenizer.sh
RAYON_NUM_THREADS=4 bash pre-train/tokenizer/run_tokenizer.sh
```

On a SLURM cluster:

```bash
sbatch pre-train/tokenizer/run_tokenizer.sh
```

The runner keeps tokenizer training memory-conscious by default:

```text
TOKENIZER_MEMORY_LIMIT_GB=20
TOKENIZER_CORPUS_WORKERS=2
RAYON_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
```

`TOKENIZER_MEMORY_LIMIT_GB` is enforced inside Python with a process address-space
limit. If the SuperBPE backend needs more memory, it should fail instead of
growing until the machine swaps heavily. Raise this limit only when the node has
enough free RAM.

The runner parallelizes CPU work through `TOKENIZER_CORPUS_WORKERS` for corpus
writing/counting and exposes `RAYON_NUM_THREADS`/`TOKENIZERS_PARALLELISM` for
the Rust `tokenizers` backend. More workers can also increase peak RAM. The
current SuperBPE training path does not use GPU compute directly; `nvidia-smi`
is logged only as a visibility check.

Expected output files:

```text
pre-train/artifacts/tokenizer_superbpe_50k_olmo_p99/tokenizer.json
pre-train/artifacts/tokenizer_superbpe_50k_olmo_p99/vocab.json
pre-train/artifacts/tokenizer_superbpe_50k_olmo_p99/merges.txt
pre-train/artifacts/tokenizer_superbpe_50k_olmo_p99/tokenizer_metadata.json
```
