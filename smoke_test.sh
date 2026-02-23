#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-0:30:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-smoke-test
#SBATCH --output=logs/slurm/smoke_test_%j.out
#SBATCH --error=logs/slurm/smoke_test_%j.err

# Smoke test for lat_training_no_sft pipeline
# Uses reduced config: 3 steps, 2 PGD iters, 1 model iter

# module load python cuda cudnn gcc arrow
module load cuda httpproxy

# python -m venv $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
# pip install --no-index -r requirements.txt
# pip install fastchat

export WANDB_MODE=offline
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

# paths.py auto-discovers HF cache; HF_HOME only needed if non-standard location
export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S-%6N")
echo "=== SMOKE TEST ==="
echo "Timestamp: ${TIMESTAMP}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

time python -m latent_at.lat_training_no_sft \
    --model_name Qwen/Qwen3-8B \
    --benign_dataset data/IPIP-08/benign_trait.csv \
    --harmful_dataset data/IPIP-08/harmful_trait.csv \
    --cache_dir cache \
    --system_prompt_path system_prompt/alpha.txt \
    --project_name lpa-smoke-test \
    --lat_config_path latent_at/lat_config_smoke.json \
    --batch_size 1 \
    --timestamp ${TIMESTAMP} \
    --wandb-offline

echo "=== SMOKE TEST COMPLETE ==="
