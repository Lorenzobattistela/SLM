#!/usr/bin/env bash
set -euo pipefail

RUN_CONFIG="${1:-configs/run/pretrain_remote_full_2gpu.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  -m src.train.pretrain \
  --run-config "${RUN_CONFIG}"
