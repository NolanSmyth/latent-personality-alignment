#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-1:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-pgd-dirs
#SBATCH --output=logs/slurm/pgd_dirs_%j.out
#SBATCH --error=logs/slurm/pgd_dirs_%j.err

# PGD Gradient Direction Diagnostic
#
# Runs PGD (batch_size=1) on each IPIP-14 item separately and computes
# cosine similarity between positive-item δ vectors and negative-item δ
# vectors at each instrumented layer.
#
# Key question: do positive and negative items push the representation in
# OPPOSITE directions?  If cos(δ_pos, δ_neg) << 0 at any layer, gradient
# cancellation in a mixed batch explains the agree-collapse.

module load cuda httpproxy

export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0
export WANDB_MODE=offline

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

echo "============================================================"
echo "PGD gradient direction diagnostic — all 67 IPIP-14 items"
echo "16 PGD iters, ε=6.0, layers=[embedding,8,16,24,30]"
echo "============================================================"

python diagnostics/probe_pgd_gradient_directions.py \
    --model_name Qwen/Qwen3-8B \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --pgd_iterations 16 \
    --epsilon 6.0 \

echo "=== DONE ==="
