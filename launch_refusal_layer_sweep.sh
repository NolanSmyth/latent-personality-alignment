#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-2:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=refusal-layer-sweep
#SBATCH --output=logs/slurm/refusal_layer_sweep_%j.out
#SBATCH --error=logs/slurm/refusal_layer_sweep_%j.err

# Sweeps the refusal direction across 8 evenly-spaced layers of Qwen3-8B
# (layers 0, 5, 10, 15, 20, 25, 30, 35 out of 36 total).
#
# Step 1 — Extract directions for all 8 layers in a SINGLE model load,
#           registering hooks for all layers simultaneously per sample.
# Step 2 — Probe each .pt file (Cohen's d, linear accuracy) — no GPU needed.
# Step 3 — Run steering tests for all layers in a SINGLE model load,
#           applying -alpha (toward compliance) only.
#
# Outputs (all in results/):
#   refusal_direction_layer{L}_first_n8.pt    — one per layer
#   steer_refusal_layer{L}_first_n8_alpha20.jsonl — one per layer

module load cuda httpproxy

export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0
export WANDB_MODE=offline

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

# ── Config ────────────────────────────────────────────────────────────────
# Qwen3-8B has 36 layers (indices 0–35).
# 8 evenly spaced: step = 35/7 = 5  →  0, 5, 10, 15, 20, 25, 30, 35
LAYERS="0 5 10 15 20 25 30 35"
ALPHA=20
N_SAMPLES=64
N_PROMPTS=10
POOL_MODE="first_n"
N_COMP_TOKENS=8

echo "========================================================"
echo "Step 1: Extract refusal directions (all layers, one model load)"
echo "  Layers:  $LAYERS"
echo "  Pool:    ${POOL_MODE}${N_COMP_TOKENS}"
echo "  Samples: $N_SAMPLES"
echo "========================================================"
python diagnostics/extract_refusal_direction.py \
    --layers $LAYERS \
    --pool_mode $POOL_MODE \
    --n_completion_tokens $N_COMP_TOKENS \
    --n_samples $N_SAMPLES

echo ""
echo "========================================================"
echo "Step 2: Probe direction quality for each layer"
echo "========================================================"
for L in $LAYERS; do
    PT="results/refusal_direction_layer${L}_${POOL_MODE}${N_COMP_TOKENS}.pt"
    echo "--- Layer $L ---"
    python diagnostics/probe_refusal_direction.py --direction "$PT" --layer $L
done

echo ""
echo "========================================================"
echo "Step 3: Steering tests (all layers, one model load, -alpha only)"
echo "  Alpha: $ALPHA  |  Prompts: $N_PROMPTS"
echo "========================================================"
# Build list of direction files in layer order
DIRECTION_FILES=""
for L in $LAYERS; do
    DIRECTION_FILES="$DIRECTION_FILES results/refusal_direction_layer${L}_${POOL_MODE}${N_COMP_TOKENS}.pt"
done

python diagnostics/steer_refusal.py \
    --directions $DIRECTION_FILES \
    --alpha $ALPHA \
    --n_prompts $N_PROMPTS \
    --out_dir results

echo ""
echo "========================================================"
echo "Layer sweep complete."
echo "Directions:  results/refusal_direction_layer*_${POOL_MODE}${N_COMP_TOKENS}.pt"
echo "Steering:    results/steer_refusal_layer*_${POOL_MODE}${N_COMP_TOKENS}_alpha${ALPHA}.jsonl"
echo "========================================================"
