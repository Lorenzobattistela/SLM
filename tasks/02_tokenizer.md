# Task 02: SuperBPE Tokenizer and FineWeb-Edu Data Pipeline

## Objective

Implement the tokenizer and dataset preparation pipeline.

Read `goal.md` completely and use the config system from Task 01.  
Execute only this task.

Do not implement the model architecture or training loop yet.

---

## Required Features

The pipeline must support:

1. Loading FineWeb-Edu from Hugging Face
2. Training or loading a SuperBPE tokenizer
3. Tokenizing FineWeb-Edu
4. Respecting the token limit configured in YAML
5. Creating train and validation token files
6. Supporting streaming or chunked processing when possible
7. Logging token counts clearly

---

## Required Scripts

Create or adapt:

```text
scripts/train_tokenizer.py
scripts/tokenize_dataset.py
```

These commands must work:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
```

---

## Required Source Structure

Create or adapt:

```text
src/tokenizer/
src/data/
```

Suggested files:

```text
src/tokenizer/__init__.py
src/tokenizer/superbpe_tokenizer.py
src/tokenizer/io.py

src/data/__init__.py
src/data/fineweb_edu.py
src/data/token_dataset.py
```

Adapt names if the repository already has equivalent modules.

---

## SuperBPE Requirement

The tokenizer type must come from YAML:

```yaml
tokenizer:
  type: "superbpe"
```

Important:

- Do not silently replace SuperBPE with standard BPE.
- If the SuperBPE dependency is unavailable, raise a clear error explaining what is missing.
- If a development fallback is implemented, it must be explicitly controlled by config and must print a clear warning.
- The default behavior must not silently use a regular BPE tokenizer.

The tokenizer should support special tokens:

```yaml
special_tokens:
  pad_token: "<pad>"
  bos_token: "<bos>"
  eos_token: "<eos>"
  unk_token: "<unk>"
```

---

## Dataset Requirement

Use:

```yaml
dataset:
  name: "HuggingFaceFW/fineweb-edu"
  split: "train"
  text_column: "text"
  streaming: true
  target_train_tokens: 4000000000
  validation_tokens: 10000000
```

The pipeline must:

1. Load the dataset
2. Read examples from the configured split
3. Extract the configured text column
4. Train the tokenizer on a limited number of samples if configured
5. Tokenize text examples
6. Stop when `target_train_tokens` is reached
7. Save validation tokens separately according to `validation_tokens`

---

## Output Requirements

Tokenizer artifacts should be saved to:

```text
artifacts/tokenizer/
```

Tokenized data should be saved to:

```text
data/processed/
```

Suggested output files:

```text
data/processed/train_tokens.bin
data/processed/val_tokens.bin
data/processed/metadata.json
```

The metadata file should include:

```json
{
  "dataset_name": "HuggingFaceFW/fineweb-edu",
  "tokenizer_type": "superbpe",
  "train_tokens": 4000000000,
  "validation_tokens": 10000000,
  "vocab_size": 50000
}
```

Actual token counts should be recorded.

---

## Memory and Disk Safety

FineWeb-Edu is large.

The implementation should avoid loading the entire dataset into memory.

Prefer:

- Streaming
- Iterative processing
- Chunked writing
- Memory-mapped binary files if appropriate

---

## Logging

Both scripts must log:

- Dataset name
- Split
- Tokenizer type
- Number of samples processed
- Number of tokens processed
- Output paths
- Whether streaming was used

---

## README Update

Update the README with tokenizer and dataset preparation commands:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
```

Also document that the configured target is 4B training tokens.

---

## Testing

After this task, the following commands must run:

```bash
python scripts/train_tokenizer.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
```

For local testing, it is acceptable to temporarily reduce the token target in the YAML, for example:

```yaml
dataset:
  target_train_tokens: 1000000
  validation_tokens: 100000
```

Do not change the default target of the main config unless creating a separate debug config.

---

## Acceptance Criteria

This task is complete when:

- FineWeb-Edu can be loaded through config
- SuperBPE tokenizer training/loading is implemented
- Tokenized train and validation outputs are generated
- Token limits are respected
- Token counts are logged
- Tokenizer artifacts are saved
- No model or training loop implementation was added in this task
