#!/bin/bash
#SBATCH --gres=gpu:h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-6:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=sft-sweep-comparison
#SBATCH --output=logs/slurm/sft_sweep_comparison_%j.out
#SBATCH --error=logs/slurm/sft_sweep_comparison_%j.err

# =============================================================================
# SFT Sweep Comparison
#
# Runs checkpoint sweeps for both the with-SFT and without-SFT conditions
# of the toward-only adversary runs and generates comparison plots.
#
# Usage:
#   sbatch launch_sft_sweep_comparison.sh [timestamp]
#
# Default timestamp: 2026-02-23_22-29-45-846331
# (lpa-with-sft and lpa-without-sft, toward-only adversary, 100 steps)
# =============================================================================

TIMESTAMP=${1:-"2026-02-23_22-29-45-846331"}
MODEL_NAME="Qwen/Qwen3-8B"

CACHE_WITH="cache/lpa-with-sft_${TIMESTAMP}"
CACHE_WITHOUT="cache/lpa-without-sft_${TIMESTAMP}"

CSV_WITH="diagnostics/checkpoint_sweep_results_lpa-with-sft_${TIMESTAMP}.csv"
CSV_WITHOUT="diagnostics/checkpoint_sweep_results_lpa-without-sft_${TIMESTAMP}.csv"

OUTPUT_DIR="diagnostics/figures/sft_sweep_comparison_${TIMESTAMP}"

echo "=============================================="
echo "SFT SWEEP COMPARISON"
echo "Timestamp: ${TIMESTAMP}"
echo "With-SFT:    ${CACHE_WITH}"
echo "Without-SFT: ${CACHE_WITHOUT}"
echo "=============================================="

module load cuda httpproxy
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

mkdir -p logs/slurm

# Determine checkpoints present (every 5 steps up to 100)
CKPT_LIST=$(python -c "
import os
from pathlib import Path
cache = '${CACHE_WITH}'
ckpts = sorted(
    int(d.name.split('_')[1])
    for d in Path(cache).iterdir()
    if d.is_dir() and d.name.startswith('checkpoint_')
)
print(' '.join(map(str, ckpts)))
")
echo "Checkpoints found: ${CKPT_LIST}"

# -----------------------------------------------------------------------
# Sweep 1: with-SFT
# -----------------------------------------------------------------------
if [ -f "${CSV_WITH}" ]; then
    echo ""
    echo "Skipping with-SFT sweep — CSV already exists: ${CSV_WITH}"
else
    echo ""
    echo "=== Sweeping with-SFT checkpoints ==="
    time python diagnostics/checkpoint_sweep.py \
        --cache_dir ${CACHE_WITH} \
        --model_name ${MODEL_NAME} \
        --output_file ${CSV_WITH} \
        --checkpoints ${CKPT_LIST}
    echo "with-SFT sweep done."
fi

# -----------------------------------------------------------------------
# Sweep 2: without-SFT
# -----------------------------------------------------------------------
if [ -f "${CSV_WITHOUT}" ]; then
    echo ""
    echo "Skipping without-SFT sweep — CSV already exists: ${CSV_WITHOUT}"
else
    echo ""
    echo "=== Sweeping without-SFT checkpoints ==="
    time python diagnostics/checkpoint_sweep.py \
        --cache_dir ${CACHE_WITHOUT} \
        --model_name ${MODEL_NAME} \
        --output_file ${CSV_WITHOUT} \
        --checkpoints ${CKPT_LIST}
    echo "without-SFT sweep done."
fi

# -----------------------------------------------------------------------
# Plot comparison
# -----------------------------------------------------------------------
echo ""
echo "=== Generating comparison plots ==="
python diagnostics/plot_sft_sweep_comparison.py \
    --with_sft_csv    ${CSV_WITH} \
    --without_sft_csv ${CSV_WITHOUT} \
    --output_dir      ${OUTPUT_DIR}

echo ""
echo "=============================================="
echo "Done. Figures saved to: ${OUTPUT_DIR}"
echo "=============================================="
