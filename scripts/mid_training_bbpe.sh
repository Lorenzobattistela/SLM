#!/usr/bin/env bash
#SBATCH --job-name=slm-mid-bbpe
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/mid-training-bbpe-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

cd "$PROJECT_DIR"
mkdir -p logs

RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
FULL_RUN_LOG="$PROJECT_DIR/logs/mid-training-bbpe-run-${RUN_ID}.log"
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
  log "ERROR: byte-BPE mid-training failed at line ${line_no} with exit code ${exit_code}."
  log "Refer to full log: $FULL_RUN_LOG"
  exit "$exit_code"
}

trap 'on_error "$LINENO"' ERR

RUN_MODE="full"
PROFILE="byte_bpe_gpt2"
BASE_CONFIG=""
RESUME_FROM=""
RUN_CONFIG=""
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      RUN_MODE="debug"
      shift
      ;;
    --full)
      RUN_MODE="full"
      shift
      ;;
    --config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --checkpoint|--resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --run-config)
      RUN_CONFIG="$2"
      shift 2
      ;;
    --gpus)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      log "ERROR: Unknown command-line option: $1"
      exit 1
      ;;
  esac
done

if [ -d ".venv" ]; then
  log "Activating virtual environment (.venv)..."
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [ "$RUN_MODE" = "debug" ]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
fi

default_mid_config() {
  case "$RUN_MODE" in
    full)
      echo "mid-training/configs/mid_train_200m_byte_bpe_gpt2.yml"
      ;;
    debug)
      echo "mid-training/configs/mid_train_200m_byte_bpe_gpt2_debug.yml"
      ;;
    *)
      log "ERROR: Unknown run mode '$RUN_MODE'."
      exit 1
      ;;
  esac
}

is_distributed() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

enabled = config.get("training", {}).get("distributed", {}).get("enabled", False)
print("true" if enabled else "false")
PY
}

config_resume_from() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

print(config.get("training", {}).get("checkpointing", {}).get("resume_from", "null"))
PY
}

require_file() {
  local path="$1"
  local description="$2"
  if [ -f "$path" ] || [ -f "$PROJECT_DIR/$path" ]; then
    return
  fi
  log "ERROR: ${description} not found: $path"
  exit 1
}

log "=============================================================================="
log "Starting GPT byte-level BPE mid-training Run ID: ${RUN_ID}"
log "Project Directory: $PROJECT_DIR"
log "Mode: $RUN_MODE"
log "Python Binary: $(command -v "$PYTHON_BIN" || echo "Not Found")"
log "=============================================================================="

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  log "WARNING: torch is not importable via '$PYTHON_BIN'. Ensure your environment is active."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [ -n "$RUN_CONFIG" ]; then
  require_file "$RUN_CONFIG" "mid-training run config"
  MID_RUN_CONFIG="$RUN_CONFIG"
  log "Using existing mid-training run config: $MID_RUN_CONFIG"
else
  MID_BASE_CONFIG="${BASE_CONFIG:-$(default_mid_config)}"
  require_file "$MID_BASE_CONFIG" "byte-BPE mid-training base config"

  if [ -z "$RESUME_FROM" ]; then
    RESUME_FROM="$(config_resume_from "$MID_BASE_CONFIG")"
  fi

  if [ "$RESUME_FROM" = "None" ] || [ "$RESUME_FROM" = "null" ]; then
    RESUME_FROM=""
  fi
  if [ -n "$RESUME_FROM" ]; then
    require_file "$RESUME_FROM" "pre-trained GPT byte-BPE checkpoint"
  fi

  MID_RUN_CONFIG="logs/mid_train_config_${PROFILE}_${RUN_ID}.yml"
  log "Base config: $MID_BASE_CONFIG"
  if [ -n "$RESUME_FROM" ]; then
    log "Starting checkpoint: $RESUME_FROM"
  else
    log "Starting checkpoint: random initialization"
  fi
  log "Generating mid-training config: $MID_RUN_CONFIG"

  "$PYTHON_BIN" - "$MID_BASE_CONFIG" "$MID_RUN_CONFIG" "$RESUME_FROM" "$PROFILE" <<'PY'
import sys
import yaml

source, target, resume_path, profile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(source, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

base_dir = f"outputs/{profile}/mid_training"
config["project"]["name"] = f"{profile}_mid_training"
config["project"]["output_dir"] = base_dir
config["dataset"]["processed_dir"] = f"{base_dir}/processed"
config["training"]["checkpointing"]["save_dir"] = f"{base_dir}/checkpoints"
config["training"]["checkpointing"]["resume_from"] = resume_path or None

if "plots" in config:
    config["plots"]["output_dir"] = f"{base_dir}/plots"

with open(target, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
fi

MID_DIST="$(is_distributed "$MID_RUN_CONFIG")"

if [ "$MID_DIST" = "true" ]; then
  log "Launching distributed mid-training via torch.distributed.run on $NPROC_PER_NODE process(es)..."
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    mid-training/scripts/mid_train.py \
    --run-config "$MID_RUN_CONFIG"
else
  log "Launching single-process mid-training..."
  "$PYTHON_BIN" mid-training/scripts/mid_train.py --run-config "$MID_RUN_CONFIG"
fi

MID_FINAL_CHECKPOINT="outputs/${PROFILE}/mid_training/checkpoints/final.pt"
if [ -f "$MID_FINAL_CHECKPOINT" ]; then
  log "Mid-training complete. Final checkpoint: $MID_FINAL_CHECKPOINT"
else
  log "Mid-training command completed. Final checkpoint was not found at default path: $MID_FINAL_CHECKPOINT"
  log "If you used --run-config with a custom save_dir, check that configured checkpoint directory."
fi

log "Full mid-training execution log file: $FULL_RUN_LOG"
