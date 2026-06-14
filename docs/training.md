# Training

Training is implemented by `pre-train/scripts/train.py` and `src/training/`. The main run uses AdamW, bf16 precision when supported, gradient accumulation, gradient clipping, warmup, cosine learning-rate decay, checkpointing, and JSONL metrics logging.

Run the configured 2-GPU job with:

```bash
torchrun --standalone --nproc_per_node=2 pre-train/scripts/train.py --run-config pre-train/configs/train_200m_fineweb_edu.yml
```

## Behavior

- AdamW is configured under `training.optimizer`.
- Warmup and cosine decay are configured under `training.scheduler`.
- `micro_batch_size` and `gradient_accumulation_steps` define the effective batch size.
- Gradient clipping uses `training.gradient_clipping.max_norm` when enabled.
- bf16 is requested through `training.precision`; unsupported devices fall back through the precision helper.
- Checkpoints are written to `checkpoints/llm_200m_fineweb_edu/`.
- `latest.pt` and `final.pt` are maintained, with periodic `step_*.pt` checkpoints rotated by `keep_last_n`.
- Resume is controlled by `training.checkpointing.resume_from`.
- Metrics are appended to `outputs/llm_200m_fineweb_edu/logs/metrics.jsonl`.
- Training run metadata is written to `outputs/llm_200m_fineweb_edu/logs/training_metadata.json`.
- At startup, training logs the selected attention optimization backend, including whether Flash SDPA is available for the configured GQA shape.

Metrics include training loss, training perplexity, validation loss, validation perplexity, learning rate, tokens seen, gradient norm, throughput, step time, token progress, epoch-equivalent progress, and CUDA memory usage when the values are available.
