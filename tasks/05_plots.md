# Task 05: Preserve and Extend Training Plots

## Objective

Preserve the existing training plots and extend them if necessary.

Read `goal.md` completely and use the training logs from Task 04.  
Execute only this task.

Do not rewrite the model, tokenizer, or training pipeline unless required to expose metrics cleanly.

---

## Main Requirement

The repository already has training plots.

Do not remove them.

Keep all existing useful plots and make sure they still work after the refactor.

---

## Required Plot Outputs

At minimum, generate plots for:

- Training loss
- Validation loss
- Perplexity
- Learning rate
- Tokens seen
- Gradient norm, if available

Plots should be saved under:

```text
outputs/llm_200m_fineweb_edu/plots/
```

This path must come from YAML:

```yaml
plots:
  enabled: true
  output_dir: "outputs/llm_200m_fineweb_edu/plots"
  keep_existing_plots: true
```

---

## Required Source Structure

Create or adapt:

```text
src/plotting/
```

Suggested files:

```text
src/plotting/__init__.py
src/plotting/training_plots.py
```

If plotting already exists elsewhere, preserve the old files and adapt the new config flow to them.

---

## Required Script

Create or adapt:

```text
scripts/plot_training.py
```

Command:

```bash
python scripts/plot_training.py --run-config configs/train_200m_fineweb_edu.yml
```

This command must:

1. Load config
2. Read training logs or metrics
3. Generate configured plots
4. Save plots to the configured output directory

---

## Metrics Input

The plotting script may read from:

- JSONL logs
- CSV logs
- TensorBoard event files
- Existing repository log format

Prefer the existing project log format if it is already available.

If needed, update the training loop from Task 04 only to ensure it writes a clean metrics file.

Suggested metrics file:

```text
outputs/llm_200m_fineweb_edu/logs/metrics.jsonl
```

Each line may include:

```json
{
  "step": 100,
  "tokens_seen": 13107200,
  "train_loss": 3.41,
  "val_loss": 3.52,
  "perplexity": 33.8,
  "learning_rate": 0.00028,
  "gradient_norm": 0.94
}
```

---

## Plot Style

Keep plots simple and readable.

Do not overcomplicate style.

Each plot should:

- Have a clear title
- Have clear axis labels
- Save as `.png`
- Use filenames such as:
  - `train_loss.png`
  - `validation_loss.png`
  - `perplexity.png`
  - `learning_rate.png`
  - `tokens_seen.png`
  - `gradient_norm.png`

---

## README Update

Update the README with:

```bash
python scripts/plot_training.py --run-config configs/train_200m_fineweb_edu.yml
```

Also document the plot output location.

---

## Testing

Use fake or small training logs if necessary to test the plotting command.

The command must not fail if an optional metric is missing.  
For example, if `gradient_norm` is unavailable, it should print a warning and continue.

---

## Acceptance Criteria

This task is complete when:

- Existing training plots are preserved
- Plotting works with the YAML config
- Plots are saved to the configured output directory
- Required plots are generated when metrics exist
- Missing optional metrics do not crash the script
- README documents the plotting command
- Model, tokenizer, and training logic were not unnecessarily rewritten
