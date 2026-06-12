#!/usr/bin/env bash
#SBATCH --job-name=slm-pipeline
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/pipeline-%j.log

# ==============================================================================
# SLM Pipeline: Mid-training, Supervised Fine-Tuning (SFT), and Evaluation
# ==============================================================================
#
# This script runs the entire training and evaluation pipeline for the 200M SLM.
# It can be submitted to SLURM or run directly on a local machine.
#
# It scans the `outputs/` folder for any `.pt` files.
# For each `.pt` file found (or specified), it:
#   1. Creates a new folder under `outputs/` named after the .pt file (without extension)
#   2. Performs Mid-training using the .pt file as the resume_from checkpoint,
#      saving checkpoints, metrics, and plots under `outputs/<model_name>/mid_training/`
#   3. Performs SFT training using the final mid-training checkpoint,
#      saving checkpoints, metrics, and plots under `outputs/<model_name>/sft/`
#   4. Evaluates both the mid-trained and SFT final checkpoints, saving zero-shot
#      results in `outputs/<model_name>/mid_training/evaluation_results.json` and
#      `outputs/<model_name>/sft/evaluation_results.json` respectively.
#
# Usage:
#   # Run the pipeline scanning outputs/ for .pt files on SLURM:
#   sbatch run_pipeline.sh
#
#   # Run in debug mode:
#   sbatch run_pipeline.sh --debug
#
#   # Run locally:
#   ./run_pipeline.sh --debug
#
#   # Specify a custom checkpoint path directly:
#   sbatch run_pipeline.sh --resume-from outputs/my_model.pt
# ==============================================================================

set -euo pipefail

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

cd "$PROJECT_DIR"
mkdir -p logs

# Setup unique run ID
RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
FULL_RUN_LOG="$PROJECT_DIR/logs/pipeline-run-${RUN_ID}.log"

# Redirect stdout and stderr to both console and log file
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
  log "ERROR: Pipeline failed at line ${line_no} with exit code ${exit_code}."
  log "Refer to full log: $FULL_RUN_LOG"
  exit "$exit_code"
}

trap 'on_error "$LINENO"' ERR

log "=============================================================================="
log "Starting SLM Pipeline Run ID: ${RUN_ID}"
log "Project Directory: $PROJECT_DIR"
log "=============================================================================="

# Parse command line arguments
RUN_MODE="full"
NPROC_PER_NODE=""
PYTHON_BIN="python3"
RESUME_FROM_PRETRAIN=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      RUN_MODE="debug"
      shift
      ;;
    --full)
      RUN_MODE="full"
      shift
      ;;
    --gpus)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM_PRETRAIN="$2"
      shift 2
      ;;
    *)
      log "ERROR: Unknown command-line option: $1"
      exit 1
      ;;
  esac
done

# Activate virtual environment if present
if [ -d ".venv" ]; then
  log "Activating virtual environment (.venv)..."
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Set python paths and logging variables
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

# Setup configs and parameters depending on run mode
if [ "$RUN_MODE" = "debug" ]; then
  log "Mode: DEBUG (Using fast configurations)"
  MID_CONFIG="configs/mid_train_200m_debug.yml"
  SFT_CONFIG="configs/sft_200m_debug.yml"
  EVAL_CONFIG="configs/eval.yml"
  EVAL_LIMIT="5"
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
else
  log "Mode: FULL (Standard 200M pipeline)"
  MID_CONFIG="configs/mid_train_200m.yml"
  SFT_CONFIG="configs/sft_200m.yml"
  EVAL_CONFIG="configs/eval.yml"
  EVAL_LIMIT="" # Uses default limit from eval.config (20)
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
fi

# Check Python env
log "Python Binary: $(command -v "$PYTHON_BIN" || echo "Not Found")"
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  log "WARNING: torch is not importable via '$PYTHON_BIN'. Ensure your environment is active."
fi

# Display hardware status if available
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

# Function to check if training configuration has DDP enabled
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

# --- Check outputs/ folder for .pt files ---
PT_FILES=()
shopt -s nullglob
for f in outputs/*.pt; do
  PT_FILES+=("$f")
done
shopt -u nullglob

# Determine target list of checkpoints to run
TARGET_CHECKPOINTS=()
if [ -n "$RESUME_FROM_PRETRAIN" ]; then
  # If explicitly passed via CLI, only run this one
  TARGET_CHECKPOINTS+=("$RESUME_FROM_PRETRAIN")
elif [ ${#PT_FILES[@]} -gt 0 ]; then
  # If there are .pt files in the outputs dir, run for all of them
  for f in "${PT_FILES[@]}"; do
    TARGET_CHECKPOINTS+=("$f")
  done
else
  # Default fallback if no .pt files are in outputs/
  # Extract resume_from path from midtraining config
  DEFAULT_RESUME="$("$PYTHON_BIN" - "$MID_CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config.get("training", {}).get("checkpointing", {}).get("resume_from", "null"))
PY
)"
  if [ "$DEFAULT_RESUME" != "None" ] && [ "$DEFAULT_RESUME" != "null" ]; then
    TARGET_CHECKPOINTS+=("$DEFAULT_RESUME")
  else
    # Scratch or debug run starting with random weights
    TARGET_CHECKPOINTS+=("random_init")
  fi
fi

log "------------------------------------------------------------------------------"
log "Target checkpoints to process: ${#TARGET_CHECKPOINTS[@]}"
for ckpt in "${TARGET_CHECKPOINTS[@]}"; do
  log "  - $ckpt"
done
log "------------------------------------------------------------------------------"

# Run the pipeline for each target checkpoint
for CKPT_PATH in "${TARGET_CHECKPOINTS[@]}"; do
  if [ "$CKPT_PATH" = "random_init" ]; then
    PT_NAME="scratch_run"
    PT_FILE=""
    log "Starting a run from scratch (random initialization)."
  else
    PT_NAME=$(basename "$CKPT_PATH" .pt)
    PT_FILE="$CKPT_PATH"
    log "Starting run for checkpoint: $PT_NAME (file: $PT_FILE)"
  fi

  log "=============================================================================="
  log "Processing Checkpoint: ${PT_NAME}"
  log "=============================================================================="

  # Create directory structure for this specific checkpoint
  RUN_OUTPUT_DIR="outputs/${PT_NAME}"
  log "Creating run output directory: ${RUN_OUTPUT_DIR}"
  mkdir -p "${RUN_OUTPUT_DIR}/mid_training"
  mkdir -p "${RUN_OUTPUT_DIR}/sft"

  # 1. Generate customized mid-training config dynamically
  MID_RUN_CONFIG="logs/mid_train_config_${PT_NAME}_${RUN_ID}.yml"
  log "Generating mid-training config: $MID_RUN_CONFIG"
  
  "$PYTHON_BIN" - "$MID_CONFIG" "$MID_RUN_CONFIG" "${PT_FILE:-}" "$PT_NAME" <<'PY'
import sys
import yaml

source, target, resume_path, pt_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(source, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

# Dynamic directories
base_dir = f"outputs/{pt_name}/mid_training"
config["project"]["output_dir"] = base_dir
config["dataset"]["processed_dir"] = f"{base_dir}/processed"
config["training"]["checkpointing"]["save_dir"] = f"{base_dir}/checkpoints"
if "plots" in config:
    config["plots"]["output_dir"] = f"{base_dir}/plots"

# Set resume path (if empty, set to null)
if resume_path:
    config["training"]["checkpointing"]["resume_from"] = resume_path
else:
    config["training"]["checkpointing"]["resume_from"] = None

with open(target, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

  # ==============================================================================
  # STAGE 1: Mid-Training
  # ==============================================================================
  log "Starting STAGE 1 (Mid-Training) for ${PT_NAME}..."
  
  MID_DIST=$(is_distributed "$MID_RUN_CONFIG")

  if [ "$MID_DIST" = "true" ]; then
    log "Launching distributed mid-training via torchrun on $NPROC_PER_NODE GPUs..."
    torchrun \
      --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      scripts/mid_train.py \
      --run-config "$MID_RUN_CONFIG"
  else
    log "Launching single-process mid-training..."
    "$PYTHON_BIN" scripts/mid_train.py --run-config "$MID_RUN_CONFIG"
  fi

  # Resolve final mid-training checkpoint path
  MID_SAVE_DIR="outputs/${PT_NAME}/mid_training"
  MID_FINAL_CHECKPOINT="${MID_SAVE_DIR}/checkpoints/final.pt"

  if [ ! -f "$MID_FINAL_CHECKPOINT" ]; then
    log "ERROR: Mid-training final checkpoint not found at $MID_FINAL_CHECKPOINT!"
    exit 1
  fi

  log "Mid-training complete for ${PT_NAME}. Checkpoint saved at: $MID_FINAL_CHECKPOINT"

  # ==============================================================================
  # STAGE 2: Supervised Fine-Tuning (SFT)
  # ==============================================================================
  log "Starting STAGE 2 (Supervised Fine-Tuning) for ${PT_NAME}..."
  
  SFT_RUN_CONFIG="logs/sft_config_${PT_NAME}_${RUN_ID}.yml"
  log "Generating SFT config: $SFT_RUN_CONFIG"

  "$PYTHON_BIN" - "$SFT_CONFIG" "$SFT_RUN_CONFIG" "$MID_FINAL_CHECKPOINT" "$PT_NAME" <<'PY'
import sys
import yaml

source, target, resume_path, pt_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(source, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

# Dynamic directories
base_dir = f"outputs/{pt_name}/sft"
config["project"]["output_dir"] = base_dir
config["dataset"]["processed_dir"] = f"{base_dir}/processed"
config["training"]["checkpointing"]["save_dir"] = f"{base_dir}/checkpoints"
if "plots" in config:
    config["plots"]["output_dir"] = f"{base_dir}/plots"

# Point to mid-training output checkpoint
config["training"]["checkpointing"]["resume_from"] = resume_path

with open(target, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

  SFT_DIST=$(is_distributed "$SFT_RUN_CONFIG")

  if [ "$SFT_DIST" = "true" ]; then
    log "Launching distributed SFT via torchrun on $NPROC_PER_NODE GPUs..."
    torchrun \
      --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      scripts/sft.py \
      --run-config "$SFT_RUN_CONFIG"
  else
    log "Launching single-process SFT..."
    "$PYTHON_BIN" scripts/sft.py --run-config "$SFT_RUN_CONFIG"
  fi

  # Resolve final SFT checkpoint path
  SFT_SAVE_DIR="outputs/${PT_NAME}/sft"
  SFT_FINAL_CHECKPOINT="${SFT_SAVE_DIR}/checkpoints/final.pt"

  if [ ! -f "$SFT_FINAL_CHECKPOINT" ]; then
    log "ERROR: SFT final checkpoint not found at $SFT_FINAL_CHECKPOINT!"
    exit 1
  fi

  log "SFT complete for ${PT_NAME}. Checkpoint saved at: $SFT_FINAL_CHECKPOINT"

  # ==============================================================================
  # STAGE 3: Evaluation Suite
  # ==============================================================================
  log "Starting STAGE 3 (Benchmark Evaluation) for ${PT_NAME}..."
  
  EVAL_ARGS=("--run-config" "$EVAL_CONFIG")
  if [ -n "$EVAL_LIMIT" ]; then
    EVAL_ARGS+=("--limit" "$EVAL_LIMIT")
  fi

  MID_EVAL_OUT="${MID_SAVE_DIR}/evaluation_results.json"
  log "Evaluating Mid-trained final checkpoint..."
  "$PYTHON_BIN" scripts/evaluate_benchmarks.py \
    "${EVAL_ARGS[@]}" \
    --checkpoint "$MID_FINAL_CHECKPOINT" \
    --output "$MID_EVAL_OUT"

  SFT_EVAL_OUT="${SFT_SAVE_DIR}/evaluation_results.json"
  log "Evaluating SFT-trained final checkpoint..."
  "$PYTHON_BIN" scripts/evaluate_benchmarks.py \
    "${EVAL_ARGS[@]}" \
    --checkpoint "$SFT_FINAL_CHECKPOINT" \
    --output "$SFT_EVAL_OUT"

  log "------------------------------------------------------------------------------"
  log "Checkpoint ${PT_NAME} Pipeline Completed Successfully!"
  log "  - Mid-trained checkpoint: $MID_FINAL_CHECKPOINT"
  log "  - SFT-trained checkpoint:  $SFT_FINAL_CHECKPOINT"
  log "  - Mid-trained evaluations: $MID_EVAL_OUT"
  log "  - SFT-trained evaluations:  $SFT_EVAL_OUT"
  log "------------------------------------------------------------------------------"
done

log "=============================================================================="
log "All target checkpoints processed successfully!"
log "Full pipeline execution log file: $FULL_RUN_LOG"
log "=============================================================================="
