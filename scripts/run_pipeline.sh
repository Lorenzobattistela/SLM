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
# This script runs the full training and evaluation pipeline for the 200M SLM.
# It can be submitted to SLURM or run directly on a local machine.
#
# By default it runs two tokenizer-specific profiles:
#   1. superbpe: starts from the SuperBPE pre-training checkpoint configured in
#      mid-training/configs/mid_train_200m_superbpe.yml.
#   2. byte_bpe_gpt2: starts from the GPT-2 byte-level BPE pre-training checkpoint
#      configured in mid-training/configs/mid_train_200m_byte_bpe_gpt2.yml.
#
# For each profile it:
#   1. Creates `outputs/<profile>/mid_training` and `outputs/<profile>/sft`.
#   2. Runs mid-training from that profile's own pre-trained checkpoint.
#   3. Runs SFT from that profile's final mid-training checkpoint.
#   4. Evaluates both final checkpoints.
#
# Usage:
#   # Run both tokenizer profiles on SLURM:
#   sbatch scripts/run_pipeline.sh
#
#   # Run in debug mode:
#   sbatch scripts/run_pipeline.sh --debug
#
#   # Run locally:
#   ./scripts/run_pipeline.sh --debug
#
#   # Run only one tokenizer profile:
#   sbatch scripts/run_pipeline.sh --profile byte_bpe_gpt2
#
#   # Override the starting checkpoint for a single selected profile:
#   sbatch scripts/run_pipeline.sh --profile superbpe --resume-from outputs/my_model.pt
# ==============================================================================

set -euo pipefail

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
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
DEFAULT_PROFILES=("superbpe" "byte_bpe_gpt2")
REQUESTED_PROFILES=()

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
    --profile)
      REQUESTED_PROFILES+=("$2")
      shift 2
      ;;
    --profiles)
      IFS=',' read -r -a PARSED_PROFILES <<< "$2"
      for profile in "${PARSED_PROFILES[@]}"; do
        REQUESTED_PROFILES+=("$profile")
      done
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

if [ ${#REQUESTED_PROFILES[@]} -eq 0 ]; then
  REQUESTED_PROFILES=("${DEFAULT_PROFILES[@]}")
fi

if [ -n "$RESUME_FROM_PRETRAIN" ] && [ ${#REQUESTED_PROFILES[@]} -ne 1 ]; then
  log "ERROR: --resume-from can only be used when exactly one --profile is selected."
  exit 1
fi

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
  EVAL_CONFIG="benchmarks/configs/eval.yml"
  EVAL_LIMIT="5"
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
else
  log "Mode: FULL (Standard 200M pipeline)"
  EVAL_CONFIG="benchmarks/configs/eval.yml"
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

profile_mid_config() {
  local profile="$1"
  case "${RUN_MODE}:${profile}" in
    full:superbpe)
      echo "mid-training/configs/mid_train_200m_superbpe.yml"
      ;;
    debug:superbpe)
      echo "mid-training/configs/mid_train_200m_superbpe_debug.yml"
      ;;
    full:byte_bpe_gpt2)
      echo "mid-training/configs/mid_train_200m_byte_bpe_gpt2.yml"
      ;;
    debug:byte_bpe_gpt2)
      echo "mid-training/configs/mid_train_200m_byte_bpe_gpt2_debug.yml"
      ;;
    *)
      log "ERROR: Unknown pipeline profile '$profile'. Use superbpe or byte_bpe_gpt2."
      exit 1
      ;;
  esac
}

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
      log "ERROR: Unknown pipeline profile '$profile'. Use superbpe or byte_bpe_gpt2."
      exit 1
      ;;
  esac
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

require_existing_checkpoint() {
  local checkpoint="$1"
  local profile="$2"
  if [ "$checkpoint" = "None" ] || [ "$checkpoint" = "null" ] || [ -z "$checkpoint" ]; then
    return
  fi
  if [ -f "$checkpoint" ] || [ -f "$PROJECT_DIR/$checkpoint" ]; then
    return
  fi
  log "ERROR: Pre-trained checkpoint for profile '$profile' was not found: $checkpoint"
  log "Set training.checkpointing.resume_from in the profile config or use --profile $profile --resume-from <checkpoint>."
  exit 1
}

log "------------------------------------------------------------------------------"
log "Profiles to process: ${#REQUESTED_PROFILES[@]}"
for profile in "${REQUESTED_PROFILES[@]}"; do
  log "  - $profile"
done
log "------------------------------------------------------------------------------"

# Run the pipeline for each tokenizer profile
for PROFILE in "${REQUESTED_PROFILES[@]}"; do
  MID_CONFIG="$(profile_mid_config "$PROFILE")"
  SFT_CONFIG="$(profile_sft_config "$PROFILE")"
  PT_NAME="$PROFILE"

  if [ -n "$RESUME_FROM_PRETRAIN" ]; then
    PT_FILE="$RESUME_FROM_PRETRAIN"
  else
    PT_FILE="$(config_resume_from "$MID_CONFIG")"
    if [ "$PT_FILE" = "None" ] || [ "$PT_FILE" = "null" ]; then
      PT_FILE=""
    fi
  fi

  require_existing_checkpoint "${PT_FILE:-null}" "$PROFILE"

  log "=============================================================================="
  log "Processing Profile: ${PROFILE}"
  log "Mid config: $MID_CONFIG"
  log "SFT config: $SFT_CONFIG"
  if [ -n "${PT_FILE:-}" ]; then
    log "Starting checkpoint: $PT_FILE"
  else
    log "Starting checkpoint: random initialization"
  fi
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
config["project"]["name"] = f"{pt_name}_mid_training"
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
      mid-training/scripts/mid_train.py \
      --run-config "$MID_RUN_CONFIG"
  else
    log "Launching single-process mid-training..."
    "$PYTHON_BIN" mid-training/scripts/mid_train.py --run-config "$MID_RUN_CONFIG"
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
config["project"]["name"] = f"{pt_name}_sft"
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
      supervised-fine-tuning/scripts/sft.py \
      --run-config "$SFT_RUN_CONFIG"
  else
    log "Launching single-process SFT..."
    "$PYTHON_BIN" supervised-fine-tuning/scripts/sft.py --run-config "$SFT_RUN_CONFIG"
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
  "$PYTHON_BIN" benchmarks/scripts/evaluate_benchmarks.py \
    "${EVAL_ARGS[@]}" \
    --checkpoint "$MID_FINAL_CHECKPOINT" \
    --output "$MID_EVAL_OUT"

  SFT_EVAL_OUT="${SFT_SAVE_DIR}/evaluation_results.json"
  log "Evaluating SFT-trained final checkpoint..."
  "$PYTHON_BIN" benchmarks/scripts/evaluate_benchmarks.py \
    "${EVAL_ARGS[@]}" \
    --checkpoint "$SFT_FINAL_CHECKPOINT" \
    --output "$SFT_EVAL_OUT"

  log "------------------------------------------------------------------------------"
  log "Profile ${PT_NAME} Pipeline Completed Successfully!"
  log "  - Mid-trained checkpoint: $MID_FINAL_CHECKPOINT"
  log "  - SFT-trained checkpoint:  $SFT_FINAL_CHECKPOINT"
  log "  - Mid-trained evaluations: $MID_EVAL_OUT"
  log "  - SFT-trained evaluations:  $SFT_EVAL_OUT"
  log "------------------------------------------------------------------------------"
done

log "=============================================================================="
log "All tokenizer profiles processed successfully!"
log "Full pipeline execution log file: $FULL_RUN_LOG"
log "=============================================================================="
