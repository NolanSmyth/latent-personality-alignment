#!/bin/bash
# Launch paired LPA experiments: with-SFT vs without-SFT
# Usage: bash launch_interleaved_sft.sh
#
# Submits two SLURM jobs with identical LPA config, differing only in whether
# interleaved SFT (Alpaca) is enabled. Both auto-submit eval + lm_eval on completion.

set -e

MODEL="Qwen/Qwen3-8B"
DATASET="IPIP-14"
SYSTEM_PROMPT="system_prompt/alpha.txt"
BATCH_SIZE=4
LAT_CONFIG="latent_at/lat_config_fewer_steps.json"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S-%6N")

echo "=== Paired Interleaved SFT Experiment ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Config: ${LAT_CONFIG}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# --- Job 1: LPA WITH interleaved SFT ---
PROJECT_WITH="lpa-with-sft"
echo "Submitting: ${PROJECT_WITH}"
JOB_WITH=$(sbatch --parsable \
    --job-name=${PROJECT_WITH} \
    --gres=gpu:h100_3g.40gb \
    --cpus-per-task=2 \
    --mem=48G \
    --time=0-2:00:00 \
    --account=rrg-lplevass \
    --output=logs/slurm/${PROJECT_WITH}_%j.out \
    --error=logs/slurm/${PROJECT_WITH}_%j.err \
    --wrap="
module load cuda httpproxy
export WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTHONBREAKPOINT=0
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

time python -m latent_at.lat_training \
    --model_name ${MODEL} \
    --benign_dataset data/alpaca_sft/benign_alpaca.csv \
    --harmful_dataset data/${DATASET}/harmful_trait.csv \
    --cache_dir cache \
    --system_prompt_path ${SYSTEM_PROMPT} \
    --project_name ${PROJECT_WITH} \
    --lat_config_path ${LAT_CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --timestamp ${TIMESTAMP}

# Auto-submit eval
sbatch --job-name=eval-with-sft \
    --output=logs/slurm/%j-eval-with-sft.out \
    --error=logs/slurm/%j-eval-with-sft.err \
    launch_evaluation.sh ${MODEL} ${PROJECT_WITH} ${TIMESTAMP} \"\"

")
echo "  Submitted job ${JOB_WITH}"

# --- Job 2: LPA WITHOUT SFT (baseline) ---
PROJECT_WITHOUT="lpa-without-sft"
echo "Submitting: ${PROJECT_WITHOUT}"
JOB_WITHOUT=$(sbatch --parsable \
    --job-name=${PROJECT_WITHOUT} \
    --gres=gpu:h100_3g.40gb \
    --cpus-per-task=2 \
    --mem=48G \
    --time=0-2:00:00 \
    --account=rrg-lplevass \
    --output=logs/slurm/${PROJECT_WITHOUT}_%j.out \
    --error=logs/slurm/${PROJECT_WITHOUT}_%j.err \
    --wrap="
module load cuda httpproxy
export WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTHONBREAKPOINT=0
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

time python -m latent_at.lat_training_no_sft \
    --model_name ${MODEL} \
    --benign_dataset data/${DATASET}/benign_trait.csv \
    --harmful_dataset data/${DATASET}/harmful_trait.csv \
    --cache_dir cache \
    --system_prompt_path ${SYSTEM_PROMPT} \
    --project_name ${PROJECT_WITHOUT} \
    --lat_config_path ${LAT_CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --timestamp ${TIMESTAMP}

# Auto-submit eval
sbatch --job-name=eval-without-sft \
    --output=logs/slurm/%j-eval-without-sft.out \
    --error=logs/slurm/%j-eval-without-sft.err \
    launch_evaluation.sh ${MODEL} ${PROJECT_WITHOUT} ${TIMESTAMP} \"\"

")
echo "  Submitted job ${JOB_WITHOUT}"

echo ""
echo "=== Both jobs submitted with shared timestamp: ${TIMESTAMP} ==="
echo "With-SFT:    cache/${PROJECT_WITH}_${TIMESTAMP}"
echo "Without-SFT: cache/${PROJECT_WITHOUT}_${TIMESTAMP}"
