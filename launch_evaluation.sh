#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-0:45:00
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err

MODEL=${1}
PROJECT_NAME=${2}
TIMESTAMP=${3}
EPOCH=${4}

echo ${TIMESTAMP}
echo ${MODEL}
echo ${PROJECT_NAME}
echo ${EPOCH}

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

time python -m eval \
    --model_name ${MODEL} \
    --project_name ${PROJECT_NAME} \
    --run_id ${TIMESTAMP} \
    --epoch ${EPOCH}

