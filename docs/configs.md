# Configs

The main configuration file is:

```text
configs/train_200m_fineweb_edu.yml
```

Scripts read it with the shared `--run-config` flag:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

## Main Sections

- `project`: project name, seed, output directory, docs directory.
- `dataset`: FineWeb-Edu source, streaming settings, token targets, cache/raw/processed paths.
- `tokenizer`: SuperBPE settings, vocabulary size, special tokens, artifact path.
- `model`: decoder-only Transformer dimensions and architecture choices.
- `training`: DDP, precision, optimizer, scheduler, gradient accumulation, clipping, checkpoints.
- `evaluation`: validation schedule and metrics.
- `logging`: log interval and TensorBoard/W&B settings.
- `plots`: whether plotting is enabled, plot output directory, and plot names to generate.

## Smaller Debug Configs

Create a smaller debug config by copying the main YAML and reducing token counts, model dimensions, batch size, and training steps. Keep the same section names so all scripts can continue using `--run-config`.

The repository also contains `configs/train_200m_fineweb_edu_debug.yml` for small tokenizer/data checks.
