#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-4:00:00

MODEL=${1}
PROJECT_NAME=${2}
TIMESTAMP=${3}
EPOCH=${4}

echo ${MODEL}
echo ${PROJECT_NAME}
echo ${TIMESTAMP}
echo ${EPOCH}

module load python/3.12.4 cuda cudnn gcc arrow

python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cd ~/scratch/latent-personality-alignment
pip install --no-index transformers wandb
pip install fastchat lm_eval

#git clone https://github.com/magikarp01/tasks.git

export HF_HOME=~/scratch/hf_home
export PYTHONBREAKPOINT=0
#export WANDB_MODE=disabled

time lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},peft=cache/${PROJECT_NAME}_${TIMESTAMP}/checkpoint_${EPOCH}  \
    --tasks mmlu,gsm8k,truthfulqa,super-glue-lm-eval-v1,bigbench_multiple_choice_b \
    --batch_size auto \
    --output_path cache/${PROJECT_NAME}_${TIMESTAMP} \
    --log_samples \
    --system_instruction '"$(cat system_prompt/minimal.txt)"' \
    --wandb_args id=${TIMESTAMP},project=${PROJECT_NAME},resume="allow"
