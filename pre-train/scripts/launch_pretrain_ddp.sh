#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"

RUN_CONFIG="${1:-pre-train/configs/train_200m_fineweb_edu.yml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  pre-train/scripts/train.py \
  --run-config "${RUN_CONFIG}"
