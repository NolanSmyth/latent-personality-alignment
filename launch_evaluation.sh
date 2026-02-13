#!/bin/bash
#SBATCH --gres=gpu:h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-1:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-evaluation
#SBATCH --output=logs/slurm/evaluation_%j.out
#SBATCH --error=logs/slurm/evaluation_%j.err

MODEL=${1}
PROJECT_NAME=${2}
TIMESTAMP=${3}
EPOCH=${4}
# Any additional args (e.g. --base_model) are forwarded to eval.py
shift 4 2>/dev/null
EXTRA_ARGS="$@"

echo ${TIMESTAMP}
echo ${MODEL}
echo ${PROJECT_NAME}
echo ${EPOCH}

module load cuda httpproxy

export WANDB_MODE=offline
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0

# NOTE: Eval datasets must be pre-cached before running offline.
# Run on a login node first:  python cache_eval_datasets.py

EPOCH_ARG=""
if [ -n "${EPOCH}" ]; then
    EPOCH_ARG="--epoch ${EPOCH}"
fi

time python -m eval \
    --model_name ${MODEL} \
    --project_name ${PROJECT_NAME} \
    --run_id ${TIMESTAMP} \
    ${EPOCH_ARG} \
    ${EXTRA_ARGS}

