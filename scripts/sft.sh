#!/usr/bin/env bash
#SBATCH --job-name=slm-sft
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/sft-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

cd "$PROJECT_DIR"
mkdir -p logs

RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
FULL_RUN_LOG="$PROJECT_DIR/logs/sft-run-${RUN_ID}.log"
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
  log "ERROR: SFT failed at line ${line_no} with exit code ${exit_code}."
  log "Refer to full log: $FULL_RUN_LOG"
  exit "$exit_code"
}

trap 'on_error "$LINENO"' ERR

RUN_MODE="full"
PROFILE="superbpe"
BASE_CONFIG=""
CHECKPOINT=""
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
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --checkpoint|--resume-from)
      CHECKPOINT="$2"
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

profile_sft_config() {
  local profile="$1"
  case "${RUN_MODE}:${profile}" in
    full:superbpe)
      echo "supervised-fine-tuning/configs/sft_200m_superbpe.yml"
      ;;
    debug:superbpe)
      echo "supervised-fine-tuning/configs/sft_200m_superbpe_debug.yml"
      ;;
    full:byte_bpe_gpt2)
      echo "supervised-fine-tuning/configs/sft_200m_byte_bpe_gpt2.yml"
      ;;
    debug:byte_bpe_gpt2)
      echo "supervised-fine-tuning/configs/sft_200m_byte_bpe_gpt2_debug.yml"
      ;;
    *)
      log "ERROR: Unknown SFT profile '$profile'. Use superbpe or byte_bpe_gpt2."
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
log "Starting standalone SFT Run ID: ${RUN_ID}"
log "Project Directory: $PROJECT_DIR"
log "Profile: $PROFILE"
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
  require_file "$RUN_CONFIG" "SFT run config"
  SFT_RUN_CONFIG="$RUN_CONFIG"
  log "Using existing SFT run config: $SFT_RUN_CONFIG"
else
  SFT_BASE_CONFIG="${BASE_CONFIG:-$(profile_sft_config "$PROFILE")}"
  require_file "$SFT_BASE_CONFIG" "SFT base config"

  CHECKPOINT="${CHECKPOINT:-outputs/${PROFILE}/mid_training/checkpoints/final.pt}"
  require_file "$CHECKPOINT" "Mid-training checkpoint"

  SFT_RUN_CONFIG="logs/sft_config_${PROFILE}_${RUN_ID}.yml"
  log "Base config: $SFT_BASE_CONFIG"
  log "Mid-training checkpoint: $CHECKPOINT"
  log "Generating SFT config: $SFT_RUN_CONFIG"

  "$PYTHON_BIN" - "$SFT_BASE_CONFIG" "$SFT_RUN_CONFIG" "$CHECKPOINT" "$PROFILE" <<'PY'
import sys
import yaml

source, target, resume_path, profile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(source, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

base_dir = f"outputs/{profile}/sft"
config["project"]["name"] = f"{profile}_sft"
config["project"]["output_dir"] = base_dir
config["dataset"]["processed_dir"] = f"{base_dir}/processed"
config["training"]["checkpointing"]["save_dir"] = f"{base_dir}/checkpoints"
config["training"]["checkpointing"]["resume_from"] = resume_path

if "plots" in config:
    config["plots"]["output_dir"] = f"{base_dir}/plots"

with open(target, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
fi

SFT_DIST="$(is_distributed "$SFT_RUN_CONFIG")"

if [ "$SFT_DIST" = "true" ]; then
  log "Launching distributed SFT via torch.distributed.run on $NPROC_PER_NODE process(es)..."
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    supervised-fine-tuning/scripts/sft.py \
    --run-config "$SFT_RUN_CONFIG"
else
  log "Launching single-process SFT..."
  "$PYTHON_BIN" supervised-fine-tuning/scripts/sft.py --run-config "$SFT_RUN_CONFIG"
fi

SFT_FINAL_CHECKPOINT="outputs/${PROFILE}/sft/checkpoints/final.pt"
if [ -f "$SFT_FINAL_CHECKPOINT" ]; then
  log "SFT complete. Final checkpoint: $SFT_FINAL_CHECKPOINT"
else
  log "SFT command completed. Final checkpoint was not found at default path: $SFT_FINAL_CHECKPOINT"
  log "If you used --run-config with a custom save_dir, check that configured checkpoint directory."
fi

log "Full SFT execution log file: $FULL_RUN_LOG"
