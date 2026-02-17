#!/bin/bash
# Evaluate all checkpoints from a completed SFT recovery run.
# Usage: bash launch_sft_recovery_sweep.sh <MODEL> <PROJECT_NAME> <TIMESTAMP> [STEP_LIST]
#
# Example:
#   bash launch_sft_recovery_sweep.sh Qwen/Qwen3-8B lpa-sft-recovery_alpaca_checkpoint50 2026-02-17_14-00-00-000000
#   bash launch_sft_recovery_sweep.sh Qwen/Qwen3-8B lpa-sft-recovery_alpaca_checkpoint50 2026-02-17_14-00-00-000000 "50 100 200 500"

MODEL=${1:?Usage: $0 <MODEL> <PROJECT_NAME> <TIMESTAMP> [STEP_LIST]}
PROJECT_NAME=${2:?Usage: $0 <MODEL> <PROJECT_NAME> <TIMESTAMP> [STEP_LIST]}
TIMESTAMP=${3:?Usage: $0 <MODEL> <PROJECT_NAME> <TIMESTAMP> [STEP_LIST]}
STEP_LIST=${4:-"50 100 150 200 250 300 350 400 450 500"}

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment

echo "=== SFT Recovery Evaluation Sweep ==="
echo "Model: ${MODEL}"
echo "Project: ${PROJECT_NAME}"
echo "Timestamp: ${TIMESTAMP}"
echo "Steps: ${STEP_LIST}"
echo "======================================"

for STEP in ${STEP_LIST}; do
    CKPT_DIR="cache/${PROJECT_NAME}_${TIMESTAMP}/checkpoint_${STEP}"
    if [ -d "${CKPT_DIR}" ]; then
        echo "Submitting eval for checkpoint_${STEP}..."
        sbatch --job-name=eval-sft-${STEP} \
            --output=logs/slurm/%j-eval-sft-recovery-step${STEP}.out \
            --error=logs/slurm/%j-eval-sft-recovery-step${STEP}.err \
            launch_evaluation.sh ${MODEL} ${PROJECT_NAME} ${TIMESTAMP} ${STEP}
    else
        echo "SKIP: ${CKPT_DIR} not found"
    fi
done

# Final model (project root)
FINAL_DIR="cache/${PROJECT_NAME}_${TIMESTAMP}"
if [ -f "${FINAL_DIR}/adapter_config.json" ]; then
    echo "Submitting eval for final model..."
    sbatch --job-name=eval-sft-final \
        --output=logs/slurm/%j-eval-sft-recovery-final.out \
        --error=logs/slurm/%j-eval-sft-recovery-final.err \
        launch_evaluation.sh ${MODEL} ${PROJECT_NAME} ${TIMESTAMP} ""
fi

echo "Done. Check queue with: sq"
