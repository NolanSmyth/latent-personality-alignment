#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-0:45:00

module load python cuda cudnn gcc arrow

python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cd ~/scratch/latent-personality-alignment
pip install --no-index -r requirements.txt
pip install fastchat

#git clone https://github.com/magikarp01/tasks.git

export HF_HOME=~/scratch/hf_home
export PYTHONBREAKPOINT=0
#export WANDB_MODE=disabled


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
    --lat_config_path latent_at/lat_config.json \
    --batch_size ${BATCH_SIZE} \
    --timestamp ${TIMESTAMP} \
    # --eval --eval_freq 2


for epoch in {30..30..10}; do
  sbatch --job-name=eval-${MODEL}-${DATASET} --output=logs/slurm/%j-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.out --error=logs/slurm/%j-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.err launch_evaluation.sh ${MODEL} ${PROJECT_NAME} ${TIMESTAMP} ${epoch}
done

sbatch --job-name=lm-eval-${MODEL}-${DATASET} --output=logs/slurm/%j-lm-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.out --error=logs/slurm/%j-lm-eval-${MODEL/\/}-${DATASET}-bs${BATCH_SIZE}.err launch_lm_eval.sh ${MODEL} ${PROJECT_NAME} ${TIMESTAMP} 30
