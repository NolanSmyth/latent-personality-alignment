#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-1:30:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-experiment
#SBATCH --output=logs/slurm/experiment_%j.out
#SBATCH --error=logs/slurm/experiment_%j.err

module load cuda httpproxy

export WANDB_MODE=offline
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S-%6N")
MODEL=${1}
DATASET=${2}
SYSTEM_PROMPT=${3}
PROJECT_NAME=${4}
BATCH_SIZE=${5}

echo ${TIMESTAMP}
echo ${MODEL}
echo ${DATASET}
echo ${SYSTEM_PROMPT}
echo ${PROJECT_NAME}
echo ${BATCH_SIZE}

time python -m latent_at.lat_training_no_sft \
    --model_name ${MODEL} \
    --benign_dataset data/${DATASET}/benign_trait.csv \
    --harmful_dataset data/${DATASET}/harmful_trait.csv \
    --cache_dir cache \
    --system_prompt_path ${SYSTEM_PROMPT} \
    --project_name ${PROJECT_NAME} \
    --lat_config_path latent_at/lat_config_fewer_steps.json \
    --batch_size ${BATCH_SIZE} \
    --timestamp ${TIMESTAMP} \
    # --eval --eval_freq 2


# Evaluate the final model (saved at project root, not in checkpoint subdir)
sbatch --job-name=eval-${MODEL}-${DATASET} --output=logs/slurm/%j-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.out --error=logs/slurm/%j-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.err launch_evaluation.sh ${MODEL} ${PROJECT_NAME} ${TIMESTAMP} ""

# Run lm_eval on the final model (use same config as training to get correct num_steps)
LAT_CONFIG=latent_at/lat_config_fewer_steps.json
NUM_STEPS=$(python -c "import json; print(json.load(open('${LAT_CONFIG}'))['num_steps'])")
sbatch --job-name=lm-eval-${MODEL}-${DATASET} --output=logs/slurm/%j-lm-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.out --error=logs/slurm/%j-lm-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.err launch_lm_eval.sh ${MODEL} ${PROJECT_NAME} ${TIMESTAMP} ${NUM_STEPS}
