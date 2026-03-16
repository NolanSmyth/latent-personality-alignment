#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-2:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-over-refusal
#SBATCH --output=logs/slurm/over_refusal_%j.out
#SBATCH --error=logs/slurm/over_refusal_%j.err

# Over-refusal sweep: harmfulness + refusal steering vectors on benign Alpaca prompts.
# Measures refusal rate (keyword-based) at each coefficient.
#
# Usage:
#   sbatch launch_over_refusal_sweep.sh
#
# Optional overrides via env vars:
#   ALPHAS="-30 -20 -10 0 10 20 30"   (default)
#   N_PROMPTS=40                        (default)
#   MAX_NEW_TOKENS=256                  (default)

module load cuda

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

ALPHAS="${ALPHAS:--30 -20 -10 0 10 20 30}"
N_PROMPTS="${N_PROMPTS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

HARM_DIR="results/direction_scored_layer15_t50.pt"
REFUSAL_DIR="results/refusal_direction_layer15_all_completion.pt"
OUT_JSONL="results/over_refusal_sweep.jsonl"
OUT_FIG="diagnostics/figures/over_refusal_rate.png"

echo "=== OVER-REFUSAL SWEEP ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Alphas      : ${ALPHAS}"
echo "N prompts   : ${N_PROMPTS}"
echo "Harm dir    : ${HARM_DIR}"
echo "Refusal dir : ${REFUSAL_DIR}"
echo "Out JSONL   : ${OUT_JSONL}"
echo "Out figure  : ${OUT_FIG}"
echo "=========================="

time python diagnostics/over_refusal_sweep.py \
    --harm_dir    "${HARM_DIR}" \
    --refusal_dir "${REFUSAL_DIR}" \
    --alphas      ${ALPHAS} \
    --n_prompts   "${N_PROMPTS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --out_jsonl   "${OUT_JSONL}" \
    --out_fig     "${OUT_FIG}"

echo "=== SWEEP COMPLETE ==="
