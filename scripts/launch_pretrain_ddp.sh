#!/usr/bin/env bash
set -euo pipefail

RUN_CONFIG="${1:-configs/train_200m_fineweb_edu.yml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  scripts/train.py \
  --run-config "${RUN_CONFIG}"
