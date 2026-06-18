# Distributed Training

The main training path uses PyTorch DistributedDataParallel through `torchrun`.

```bash
torchrun --standalone --nproc_per_node=2 pre-train/scripts/train.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
```

## How It Works

- `torchrun` starts one process per GPU and provides the rank/world-size environment used by DDP.
- `--nproc_per_node=2` matches `training.distributed.num_gpus: 2` in the main YAML config.
- The configured backend is NCCL, which is the expected backend for NVIDIA GPU training.
- Rank 0 handles metrics logging and checkpoint writes so duplicate files are not produced.
- Barriers keep workers synchronized around validation and checkpoint boundaries.

## Troubleshooting

- Confirm CUDA is visible with `python -c "import torch; print(torch.cuda.device_count())"`.
- Use recent NVIDIA drivers and a PyTorch build compiled with CUDA.
- Set `NCCL_DEBUG=INFO` when diagnosing NCCL startup or communication failures.
- If `pre-train/scripts/run_all.py` cannot safely launch DDP, it prints the exact `torchrun` command to run manually.
- For local CPU checks, use the debug config rather than the main 2-GPU config.
