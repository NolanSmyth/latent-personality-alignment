#!/bin/bash
#SBATCH --gres=gpu:h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0-8:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-lm-eval
#SBATCH --output=logs/slurm/lm_eval_%j.out
#SBATCH --error=logs/slurm/lm_eval_%j.err

MODEL=${1}
PROJECT_NAME=${2}
TIMESTAMP=${3}
EPOCH=${4}

echo ${MODEL}
echo ${PROJECT_NAME}
echo ${TIMESTAMP}
echo ${EPOCH}

module load cuda httpproxy

export WANDB_MODE=offline
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0

WORKDIR=/home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
PEFT_PATH=${WORKDIR}/cache/${PROJECT_NAME}_${TIMESTAMP}/checkpoint_${EPOCH}

# If checkpoint subdir doesn't exist, fall back to the project root (final model)
if [ ! -d "${PEFT_PATH}" ] || [ ! -f "${PEFT_PATH}/adapter_config.json" ]; then
    echo "WARNING: ${PEFT_PATH} not found, using project root adapter instead"
    PEFT_PATH=${WORKDIR}/cache/${PROJECT_NAME}_${TIMESTAMP}
fi

time python -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},peft=${PEFT_PATH}  \
    --tasks mmlu,gsm8k,truthfulqa,super-glue-lm-eval-v1,bigbench_multiple_choice_b \
    --batch_size auto \
    --output_path cache/${PROJECT_NAME}_${TIMESTAMP} \
    --log_samples \
    --system_instruction '"$(cat system_prompt/minimal.txt)"' \
    --wandb_args id=${TIMESTAMP},project=${PROJECT_NAME},resume="allow"
