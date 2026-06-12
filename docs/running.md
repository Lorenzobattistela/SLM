# Running

There are two supported execution modes: the full pipeline and independent steps.

## Full Pipeline

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

`run_all.py` tokenizes the dataset with the configured local SuperBPE tokenizer, checks the model parameter count, launches DDP training when CUDA/NCCL is available, runs evaluation, generates plots from the metrics file, and prints a text completion from the latest checkpoint.

If DDP cannot be launched safely from the script, it prints the command to run manually:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
```

## Independent Steps

```bash
python scripts/tokenize_dataset.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/count_parameters.py --run-config configs/train_200m_fineweb_edu.yml
torchrun --standalone --nproc_per_node=2 scripts/train.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/evaluate.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/plot_training.py --run-config configs/train_200m_fineweb_edu.yml
python scripts/sample_checkpoint.py --run-config configs/train_200m_fineweb_edu.yml --checkpoint checkpoints/llm_200m_fineweb_edu/latest.pt --prompt "Scientific progress depends on"
```

Plots are written to:

```text
outputs/llm_200m_fineweb_edu/plots/
```

## Unified Training & Evaluation Pipeline (Mid-Training, SFT, & Benchmark Evaluation)

A unified bash/SLURM pipeline script, [run_pipeline.sh](file:///C:/Users/arthu/OneDrive/%C3%81rea%20de%20Trabalho/CDIA/profundo%202/SLM/run_pipeline.sh), automates mid-training, supervised fine-tuning (SFT), and downstream evaluation in a single command. It is compatible with both SLURM clusters and local environments.

### Execution Modes

#### 1. Full Run on SLURM Cluster (Recommended)
Submit the job to a SLURM queue:
```bash
sbatch run_pipeline.sh
```
This runs with standard configs:
- Mid-training on mixed SmolTalk + GSM8K (`configs/mid_train_200m.yml`)
- Fine-tuning SFT on SmolTalk (`configs/sft_200m.yml`)
- Zero-shot evaluations for perplexity, HellaSwag, ARC-Easy, PIQA, WinoGrande, and GSM8K (`configs/eval.yml`) on both final checkpoints

#### 2. Debug Run (Quick Test)
Run a lightweight, quick-termination run to verify the installation and dataset streaming:
```bash
# On a cluster via sbatch
sbatch run_pipeline.sh --debug

# Locally
./run_pipeline.sh --debug
```
Debug mode automatically swaps out standard configurations for their `_debug.yml` equivalents and caps downstream evaluations at 5 samples per dataset.

#### 3. Custom Starting Checkpoint
To resume mid-training from a custom pre-trained checkpoint path instead of the default location in `configs/mid_train_200m.yml`:
```bash
sbatch run_pipeline.sh --resume-from /path/to/pretrain/checkpoint.pt
```

### Options Reference
- `--debug`: Use debug configurations and minimal steps.
- `--full`: Use standard 200M configurations (default).
- `--resume-from <path>`: Override the pre-trained checkpoint path to load weights from.
- `--gpus <N>`: Override the number of processes/GPUs to run under torchrun (default: 2 in full, 1 in debug).
- `--python <path>`: Path to the python interpreter (default: `python3`).

### Automatic Checkpoint Detection & Directory Organization

The pipeline script automatically scans the `outputs/` folder for any `.pt` files (e.g. `outputs/pretrain_model_A.pt`):
1. For each `.pt` file found, it extracts its name (e.g., `pretrain_model_A`).
2. It creates a dedicated directory: `outputs/pretrain_model_A/`.
3. All subsequent steps (mid-training, SFT, and evaluations) are fully isolated within that folder.
4. If multiple `.pt` files are present in `outputs/`, the pipeline runs sequentially for each of them in one command.

### Outputs & Logs
For a starting checkpoint named `<model_name>`, outputs are structured as follows:
- **Mid-training Outputs**: `outputs/<model_name>/mid_training/` (checkpoints, metrics, plots, and `evaluation_results.json`)
- **SFT Outputs**: `outputs/<model_name>/sft/` (checkpoints, SFT metrics, plots, and `evaluation_results.json`)
- **Pipeline Console Logs**: `logs/pipeline-run-<ID>.log`


