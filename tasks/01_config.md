# Task 01: YAML Configuration and Config Loader

## Objective

Implement the configuration foundation of the project.

Read `goal.md` completely, but execute only this task.  
Do not implement the tokenizer, model, or training loop yet.

The goal of this step is to create a clean YAML-based configuration system that all later tasks will use.

---

## Required Files

Create or adapt the following files:

```text
configs/train_200m_fineweb_edu.yml
src/config/
scripts/count_parameters.py
```

If the repository already has a config system, adapt it instead of duplicating logic.

---

## YAML Config

Create:

```text
configs/train_200m_fineweb_edu.yml
```

The config must include sections for:

- `project`
- `dataset`
- `tokenizer`
- `model`
- `training`
- `evaluation`
- `logging`
- `plots`

Use the complete YAML structure described in `goal.md`.

Important values:

```yaml
dataset:
  name: "HuggingFaceFW/fineweb-edu"
  target_train_tokens: 4000000000

model:
  target_parameters: 200000000
  acceptable_min_parameters: 195000000
  acceptable_max_parameters: 205000000

training:
  distributed:
    enabled: true
    strategy: "ddp"
    num_gpus: 2

  optimizer:
    name: "adamw"

  scheduler:
    name: "cosine"
```

---

## Config Loader

Implement a reusable config loader under:

```text
src/config/
```

Suggested files:

```text
src/config/__init__.py
src/config/loader.py
src/config/schema.py
```

The loader must support:

```bash
--run-config configs/train_200m_fineweb_edu.yml
```

The loader must:

1. Read YAML files
2. Return an object or dictionary usable by scripts
3. Validate required sections
4. Validate important fields
5. Create output directories when necessary
6. Fail clearly when required config values are missing

Required validation:

- Dataset name exists
- Dataset text column exists
- Tokenizer type exists
- Model target parameter range exists
- Optimizer name is `adamw`
- Scheduler name is `cosine`
- DDP settings exist
- Output directories exist or can be created
- Plot output directory exists or can be created

---

## CLI Pattern

All future scripts must follow this pattern:

```bash
python scripts/some_script.py --run-config configs/train_200m_fineweb_edu.yml
```

Implement a reusable CLI helper if appropriate.

Suggested file:

```text
src/config/cli.py
```

---

## Temporary Parameter Counter Stub

Create or prepare:

```text
scripts/count_parameters.py
```

At this stage, it may be a stub that:

1. Loads the YAML config
2. Prints the model config
3. Prints a clear message saying the real parameter count will be implemented in Task 03

Do not implement the full model yet.

Example output:

```text
Loaded config: configs/train_200m_fineweb_edu.yml
Target parameters: 200000000
Acceptable range: 195000000 - 205000000
Model implementation is not available yet. Run Task 03 to enable real parameter counting.
```

---

## README Update

Update the README only enough to document the new config flow:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

Do not document model or training details yet, unless they already exist.

---

## Testing

After this task, the following command must work:

```bash
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
```

Expected result:

- Config loads successfully
- Required directories are created
- Target parameter range is printed
- No tokenizer or training is executed

---

## Acceptance Criteria

This task is complete when:

- `configs/train_200m_fineweb_edu.yml` exists
- Config loader exists under `src/config/`
- `--run-config` is supported
- Important config fields are validated
- Required output directories are created
- `scripts/count_parameters.py` can load and print config information
- No model, tokenizer, or training implementation was added in this task
