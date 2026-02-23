#!/bin/bash
#SBATCH --gres=gpu:h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-2:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=sft-recovery
#SBATCH --output=logs/slurm/sft_recovery_%j.out
#SBATCH --error=logs/slurm/sft_recovery_%j.err

module load cuda httpproxy

export WANDB_MODE=offline
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S-%6N")

# Defaults
MODEL=${1:-Qwen/Qwen3-8B}
CHECKPOINT=${2:-cache/lpa-regular-config_IPIP-14_fewer_steps_2026-02-13_13-35-34-111503/checkpoint_50}
PROJECT_NAME=${3:-lpa-sft-recovery_alpaca_checkpoint50}
BATCH_SIZE=${4:-4}
SYSTEM_PROMPT=${5:-system_prompt/minimal.txt}
CONFIG_PATH=${6:-latent_at/sft_recovery_config.json}

echo "=== SFT Recovery Experiment ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${MODEL}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Project: ${PROJECT_NAME}"
echo "Batch size: ${BATCH_SIZE}"
echo "System prompt: ${SYSTEM_PROMPT}"
echo "Config: ${CONFIG_PATH}"
echo "==============================="

time python -m latent_at.lat_sft_recovery \
    --model_name ${MODEL} \
    --checkpoint_path ${CHECKPOINT} \
    --sft_dataset tatsu-lab/alpaca \
    --system_prompt_path ${SYSTEM_PROMPT} \
    --project_name ${PROJECT_NAME} \
    --config_path ${CONFIG_PATH} \
    --batch_size ${BATCH_SIZE} \
    --timestamp ${TIMESTAMP} \
    --wandb-offline

echo ""
echo "Training complete. Launching evaluation sweep..."

# Submit evaluation jobs for each checkpoint
# With N_checkpoints=10 over 500 steps, checkpoints are at steps 50, 100, ..., 500
for STEP in 100 200 300 400 500; do
    CKPT_DIR="cache/${PROJECT_NAME}_${TIMESTAMP}/checkpoint_${STEP}"
    if [ -d "${CKPT_DIR}" ]; then
        echo "Submitting eval for checkpoint_${STEP}"
        sbatch --job-name=eval-sft-${STEP} \
            --output=logs/slurm/%j-eval-sft-recovery-step${STEP}.out \
            --error=logs/slurm/%j-eval-sft-recovery-step${STEP}.err \
            launch_evaluation.sh ${MODEL} ${PROJECT_NAME} ${TIMESTAMP} ${STEP}
    else
        echo "Skipping checkpoint_${STEP} (not found)"
    fi
done
