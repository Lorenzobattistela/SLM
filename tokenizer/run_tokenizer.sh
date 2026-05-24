#!/usr/bin/env bash
#SBATCH --job-name=superbpe-50k
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/TOKENIZER-%j.log

# HOW TO RUN:
#   cd /caminho/do/projeto
#   mkdir -p logs
#   sbatch tokenizer/run_tokenizer.sh
#
# Local/manual:
#   bash tokenizer/run_tokenizer.sh
#
# Optional overrides:
#   TOKENIZER_CONFIG=tokenizer/superbpe_50k_olmo_p99.yml sbatch tokenizer/run_tokenizer.sh
#   TOKENIZER_CORPUS_WORKERS=4 sbatch tokenizer/run_tokenizer.sh
#   TOKENIZER_TRAIN_SAMPLES=500000 sbatch tokenizer/run_tokenizer.sh
#   TOKENIZER_MEMORY_LIMIT_GB=32 sbatch tokenizer/run_tokenizer.sh
#   TOKENIZER_FORCE=1 sbatch tokenizer/run_tokenizer.sh
#   RAYON_NUM_THREADS=4 sbatch tokenizer/run_tokenizer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-tokenizer/superbpe_50k_olmo_p99.yml}"
TOKENIZER_CORPUS_WORKERS="${TOKENIZER_CORPUS_WORKERS:-2}"
TOKENIZER_TRAIN_SAMPLES="${TOKENIZER_TRAIN_SAMPLES:-}"
TOKENIZER_MEMORY_LIMIT_GB="${TOKENIZER_MEMORY_LIMIT_GB:-20}"
TOKENIZER_FORCE="${TOKENIZER_FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
TOKENIZER_RUN_LOG="${TOKENIZER_RUN_LOG:-$PROJECT_DIR/logs/tokenizer-run-${RUN_ID}.log}"

cd "$PROJECT_DIR"
mkdir -p logs

exec > >(tee -a "$TOKENIZER_RUN_LOG") 2>&1

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  log "FAILED at line ${line_no} with exit code ${exit_code}"
  log "See full log: $TOKENIZER_RUN_LOG"
  exit "$exit_code"
}

trap 'on_error "$LINENO"' ERR

run_stage() {
  local stage_name="$1"
  shift
  log "START ${stage_name}"
  log "CMD $*"
  "$@"
  log "DONE ${stage_name}"
}

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -d ".venv312" ]; then
  # shellcheck disable=SC1091
  source .venv312/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZER_CORPUS_WORKERS
export TOKENIZER_MEMORY_LIMIT_GB
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"

log "Project dir: $PROJECT_DIR"
log "Tokenizer config: $TOKENIZER_CONFIG"
log "TOKENIZER_CORPUS_WORKERS: $TOKENIZER_CORPUS_WORKERS"
log "TOKENIZER_TRAIN_SAMPLES: ${TOKENIZER_TRAIN_SAMPLES:-from config}"
log "TOKENIZER_MEMORY_LIMIT_GB: $TOKENIZER_MEMORY_LIMIT_GB"
log "TOKENIZER_FORCE: $TOKENIZER_FORCE"
log "OMP_NUM_THREADS: $OMP_NUM_THREADS"
log "RAYON_NUM_THREADS: $RAYON_NUM_THREADS"
log "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
log "Python: $(command -v "$PYTHON_BIN")"
log "Tokenizer run log: $TOKENIZER_RUN_LOG"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU visibility check only; SuperBPE training path is CPU/tokenizers based."
  nvidia-smi || true
fi

TRAIN_CMD=(
  "$PYTHON_BIN"
  tokenizer/train_superbpe_50k.py
  --config "$TOKENIZER_CONFIG"
  --workers "$TOKENIZER_CORPUS_WORKERS"
  --memory-limit-gb "$TOKENIZER_MEMORY_LIMIT_GB"
)

if [ -n "$TOKENIZER_TRAIN_SAMPLES" ]; then
  TRAIN_CMD+=(--train-samples "$TOKENIZER_TRAIN_SAMPLES")
fi

if [ "$TOKENIZER_FORCE" = "1" ] || [ "$TOKENIZER_FORCE" = "true" ]; then
  TRAIN_CMD+=(--force)
fi

run_stage "SuperBPE 50K tokenizer training" "${TRAIN_CMD[@]}"

log "Tokenizer training finished."
