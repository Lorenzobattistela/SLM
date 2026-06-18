#!/usr/bin/env bash
#SBATCH --job-name=retok-byte-bpe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=24:00:00

# HOW TO RUN WITH SLURM:
#   cd /home/ap208/SLM
#   sbatch scripts/run_tokenizer.sh
#
# Local/manual:
#   bash scripts/run_tokenizer.sh
#
# Optional overrides:
#   RUN_CONFIG=pre-train/configs/train_200m_fineweb_edu.yml sbatch scripts/run_tokenizer.sh
#   OUTPUT_DIR=/home/ap208/slm-utils/data/processed_byte_bpe_gpt2 sbatch scripts/run_tokenizer.sh
#   SOURCE_PROCESSED_DIR=/home/ap208/slm-utils/data/processed sbatch scripts/run_tokenizer.sh
#   BYTE_BPE_NAME=gpt2 sbatch scripts/run_tokenizer.sh
#   RETOKENIZE_OVERWRITE=1 sbatch scripts/run_tokenizer.sh
#   MAX_DOCUMENTS=100 sbatch scripts/run_tokenizer.sh
#   PYTHON_BIN=.venv/bin/python sbatch scripts/run_tokenizer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

RUN_CONFIG="${RUN_CONFIG:-pre-train/configs/train_200m_fineweb_edu.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ap208/slm-utils/data/processed_byte_bpe_gpt2}"
SOURCE_PROCESSED_DIR="${SOURCE_PROCESSED_DIR:-}"
BYTE_BPE_NAME="${BYTE_BPE_NAME:-gpt2}"
RETOKENIZE_OVERWRITE="${RETOKENIZE_OVERWRITE:-0}"
MAX_DOCUMENTS="${MAX_DOCUMENTS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
RETOKENIZE_RUN_LOG="${RETOKENIZE_RUN_LOG:-$PROJECT_DIR/logs/retokenize-byte-bpe-${RUN_ID}.log}"

cd "$PROJECT_DIR"
mkdir -p logs

exec > >(tee -a "$RETOKENIZE_RUN_LOG") 2>&1

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
  log "See full log: $RETOKENIZE_RUN_LOG"
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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

log "Project dir: $PROJECT_DIR"
log "Run config: $RUN_CONFIG"
log "Output dir: $OUTPUT_DIR"
log "Source processed dir override: ${SOURCE_PROCESSED_DIR:-from run config}"
log "Byte BPE name: $BYTE_BPE_NAME"
log "RETOKENIZE_OVERWRITE: $RETOKENIZE_OVERWRITE"
log "MAX_DOCUMENTS: ${MAX_DOCUMENTS:-full dataset}"
log "OMP_NUM_THREADS: $OMP_NUM_THREADS"
log "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
log "Python: $(command -v "$PYTHON_BIN")"
log "Retokenize run log: $RETOKENIZE_RUN_LOG"

RETOKENIZE_CMD=(
  "$PYTHON_BIN"
  scripts/retokenize_superbpe_to_byte_bpe.py
  --run-config "$RUN_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --byte-bpe-name "$BYTE_BPE_NAME"
)

if [ -n "$SOURCE_PROCESSED_DIR" ]; then
  RETOKENIZE_CMD+=(--source-processed-dir "$SOURCE_PROCESSED_DIR")
fi

if [ "$RETOKENIZE_OVERWRITE" = "1" ] || [ "$RETOKENIZE_OVERWRITE" = "true" ]; then
  RETOKENIZE_CMD+=(--overwrite)
fi

if [ -n "$MAX_DOCUMENTS" ]; then
  RETOKENIZE_CMD+=(--max-documents "$MAX_DOCUMENTS")
fi

run_stage "SuperBPE bin to GPT-2 byte-level BPE retokenization" "${RETOKENIZE_CMD[@]}"

log "Retokenization finished."
