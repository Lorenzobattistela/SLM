# Running

There are two supported execution modes: the full pipeline and independent steps.

## Full Pipeline

```bash
python scripts/run_all.py --run-config configs/train_200m_fineweb_edu.yml
```

`run_all.py` tokenizes the dataset with the configured pretrained SuperBPE tokenizer, checks the model parameter count, launches DDP training when CUDA/NCCL is available, runs evaluation, generates plots from the metrics file, and prints a text completion from the latest checkpoint.

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
