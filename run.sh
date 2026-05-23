#!/usr/bin/env bash
#SBATCH --job-name=slm-full-run
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/RUN-%j.log

# HOW TO RUN:
#   cd /caminho/do/projeto
#   mkdir -p logs
#   sbatch run.sh
#
# Optional overrides:
#   RUN_CONFIG=configs/train_200m_fineweb_edu_debug.yml sbatch run.sh
#   NPROC_PER_NODE=2 sbatch run.sh
#   LOG_EVERY_STEPS=5 sbatch run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

RUN_CONFIG="${RUN_CONFIG:-configs/train_200m_fineweb_edu.yml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
FULL_RUN_LOG="${FULL_RUN_LOG:-$PROJECT_DIR/logs/full-run-${RUN_ID}.log}"

cd "$PROJECT_DIR"
mkdir -p logs

exec > >(tee -a "$FULL_RUN_LOG") 2>&1

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
  log "See full log: $FULL_RUN_LOG"
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
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

log "Project dir: $PROJECT_DIR"
log "Run config: $RUN_CONFIG"
log "NPROC_PER_NODE: $NPROC_PER_NODE"
log "Python: $(command -v "$PYTHON_BIN")"
log "Full run log: $FULL_RUN_LOG"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

METRICS_PATH="$("$PYTHON_BIN" - "$RUN_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(Path(config["project"]["output_dir"]) / "logs" / "metrics.jsonl")
PY
)"

if [ -n "${LOG_EVERY_STEPS:-}" ]; then
  log "Overriding logging.log_every_steps to $LOG_EVERY_STEPS for this run"
  RUN_CONFIG_ORIGINAL="$RUN_CONFIG"
  RUN_CONFIG="$PROJECT_DIR/logs/run-config-${RUN_ID}.yml"
  "$PYTHON_BIN" - "$RUN_CONFIG_ORIGINAL" "$RUN_CONFIG" "$LOG_EVERY_STEPS" <<'PY'
from pathlib import Path
import sys
import yaml

source, target, log_every_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config.setdefault("logging", {})["log_every_steps"] = log_every_steps
Path(target).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
  log "Temporary run config: $RUN_CONFIG"
fi

LOG_EVERY_CONFIG="$("$PYTHON_BIN" - "$RUN_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config.get("logging", {}).get("log_every_steps", "unknown"))
PY
)"

log "Training step logs will be emitted every ${LOG_EVERY_CONFIG} optimizer steps"
log "Training metrics JSONL: $METRICS_PATH"

run_stage "tokenizer training/loading" \
  "$PYTHON_BIN" scripts/train_tokenizer.py --run-config "$RUN_CONFIG"

run_stage "dataset tokenization" \
  "$PYTHON_BIN" scripts/tokenize_dataset.py --run-config "$RUN_CONFIG"

run_stage "parameter count check" \
  "$PYTHON_BIN" scripts/count_parameters.py --run-config "$RUN_CONFIG"

run_stage "DDP pretraining on $NPROC_PER_NODE GPUs" \
  torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py \
  --run-config "$RUN_CONFIG"

run_stage "evaluation" \
  "$PYTHON_BIN" scripts/evaluate.py --run-config "$RUN_CONFIG"

run_stage "training plots" \
  "$PYTHON_BIN" scripts/plot_training.py --run-config "$RUN_CONFIG"

log "Full run finished."
