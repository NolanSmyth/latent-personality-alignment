#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-4:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=refusal-layer-sweep-v2
#SBATCH --output=logs/slurm/refusal_layer_sweep_v2_%j.out
#SBATCH --error=logs/slurm/refusal_layer_sweep_v2_%j.err

# EXP-019 — Refusal direction layer sweep (layer_sweep_refusal.py)
#
# Steers the refusal direction at negative alphas on HarmBench prompts,
# scores with harm judge, plots mean harm score vs. layer.
#
# Usage:
#   sbatch launch_refusal_layer_sweep_v2.sh
#
# Optional overrides via env vars:
#   ALPHAS="0 -10 -20 -30"   (default)
#   N_PROMPTS=40              (default)
#   MAX_NEW_TOKENS=256        (default)

module load cuda

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

ALPHAS="${ALPHAS:-0 -10 -20 -30}"
N_PROMPTS="${N_PROMPTS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DIRECTIONS="results/refusal_direction_layer*_all_completion.pt"
OUT_JSONL="results/layer_sweep_refusal.jsonl"
OUT_FIG="diagnostics/figures/layer_sweep_refusal.png"

mkdir -p logs/slurm

echo "=== REFUSAL DIRECTION LAYER SWEEP ==="
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Alphas       : ${ALPHAS}"
echo "N prompts    : ${N_PROMPTS}"
echo "Directions   : ${DIRECTIONS}"
echo "Out JSONL    : ${OUT_JSONL}"
echo "Out figure   : ${OUT_FIG}"
echo "======================================"

time python diagnostics/layer_sweep_refusal.py \
    --directions "${DIRECTIONS}" \
    --alphas     ${ALPHAS} \
    --n_prompts  "${N_PROMPTS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --out_jsonl  "${OUT_JSONL}" \
    --out_fig    "${OUT_FIG}"

echo "=== SWEEP COMPLETE ==="
