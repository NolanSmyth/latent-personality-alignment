#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-0:15:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-diag
#SBATCH --output=logs/slurm/diagnostics_%j.out
#SBATCH --error=logs/slurm/diagnostics_%j.err

# Diagnostic: LoRA vs full-rank analysis + activation determinism check
# Should take ~5 min (model load + a few forward passes, no training)

module load cuda httpproxy

export WANDB_MODE=offline
cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0

echo "=== DIAGNOSTIC: LoRA & Activation Determinism ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

time python diagnostics/check_activation_determinism.py

echo "=== DIAGNOSTIC COMPLETE ==="
