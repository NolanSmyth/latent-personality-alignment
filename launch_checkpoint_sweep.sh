#!/bin/bash
#SBATCH --gres=gpu:h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-3:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=checkpoint-sweep
#SBATCH --output=logs/slurm/checkpoint_sweep_%j.out
#SBATCH --error=logs/slurm/checkpoint_sweep_%j.err

# =============================================================================
# Checkpoint Sweep Script
# 
# Evaluates multiple checkpoints from a training run to find the "sweet spot"
# where safety improves but utility hasn't collapsed.
#
# Usage:
#   sbatch launch_checkpoint_sweep.sh [cache_dir] [start] [end] [step]
#
# Examples:
#   # Default: evaluate checkpoints 10-200 in steps of 10
#   sbatch launch_checkpoint_sweep.sh
#   
#   # Custom cache directory
#   sbatch launch_checkpoint_sweep.sh cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503
#   
#   # Custom range (checkpoints 20, 40, 60, ..., 100)
#   sbatch launch_checkpoint_sweep.sh cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503 20 100 20
# =============================================================================

# Default values
CACHE_DIR=${1:-"cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503"}
START=${2:-10}
END=${3:-200}
STEP=${4:-10}
MODEL_NAME=${5:-"Qwen/Qwen3-8B"}

# Optional: override with specific checkpoints via env var
# e.g., CHECKPOINTS="5 10 20 30 40 50" sbatch launch_checkpoint_sweep.sh <cache_dir>
# If not set, the script falls back to start/end/step range.
CHECKPOINTS=${CHECKPOINTS:-""}

# Derived paths
RUN_NAME=$(basename ${CACHE_DIR})
OUTPUT_FILE="diagnostics/checkpoint_sweep_results_${RUN_NAME}.csv"
PLOT_DIR="diagnostics/figures/checkpoint_sweep_${RUN_NAME}"

echo "=============================================="
echo "CHECKPOINT SWEEP CONFIGURATION"
echo "=============================================="
echo "Cache directory: ${CACHE_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Checkpoints: ${START} to ${END} (step ${STEP})"
echo "Output CSV: ${OUTPUT_FILE}"
echo "Plot directory: ${PLOT_DIR}"
echo "=============================================="

# Setup environment
module load cuda httpproxy

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

# Create log directory
mkdir -p logs/slurm

echo ""
echo "Starting checkpoint sweep at $(date)"
echo ""

# Run the sweep
if [ -n "${CHECKPOINTS}" ]; then
    echo "Using specific checkpoints: ${CHECKPOINTS}"
    time python diagnostics/checkpoint_sweep.py \
        --cache_dir ${CACHE_DIR} \
        --model_name ${MODEL_NAME} \
        --output_file ${OUTPUT_FILE} \
        --checkpoints ${CHECKPOINTS}
else
    time python diagnostics/checkpoint_sweep.py \
        --cache_dir ${CACHE_DIR} \
        --model_name ${MODEL_NAME} \
        --output_file ${OUTPUT_FILE} \
        --start ${START} \
        --end ${END} \
        --step ${STEP}
fi

SWEEP_EXIT_CODE=$?

if [ ${SWEEP_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Checkpoint sweep failed with exit code ${SWEEP_EXIT_CODE}"
    exit ${SWEEP_EXIT_CODE}
fi

echo ""
echo "=============================================="
echo "Checkpoint sweep completed. Generating plots..."
echo "=============================================="

# Generate plots
python diagnostics/plot_checkpoint_sweep.py \
    --input_file ${OUTPUT_FILE} \
    --output_dir ${PLOT_DIR} \
    --baseline_asr 0.40 \
    --baseline_mmlu 0.71

PLOT_EXIT_CODE=$?

if [ ${PLOT_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "WARNING: Plot generation failed with exit code ${PLOT_EXIT_CODE}"
    echo "Sweep results are still available in: ${OUTPUT_FILE}"
fi

echo ""
echo "=============================================="
echo "SWEEP COMPLETE"
echo "=============================================="
echo "Results CSV: ${OUTPUT_FILE}"
echo "Plots: ${PLOT_DIR}/"
echo "  - sweep_combined.png"
echo "  - sweep_tradeoff.png"
echo "  - sweep_frontier.png"
echo ""
echo "Finished at $(date)"
