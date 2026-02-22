#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-1:30:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-probe
#SBATCH --output=logs/slurm/probe_%j.out
#SBATCH --error=logs/slurm/probe_%j.err

# Runs probe_ipip_responses.py for:
#   Base model prior — base Qwen3-8B on IPIP-14, both free-generation and
#                      training-template mode (critical for adversary analysis)
#   Exp 1c — IPIP-10 checkpoint (step 50) evaluated on IPIP-10 statements
#   Exp 2c — IPIP-14 checkpoint (step 50) evaluated on IPIP-14 statements
#   Exp 1d — IPIP-10 checkpoint (step 50) cross-evaluated on IPIP-14 statements

module load cuda httpproxy

export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0
export WANDB_MODE=offline

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

IPIP10_CKPT="cache/lpa-ipip10-allpositive_2026-02-21_18-54-03-596485/checkpoint_50"
IPIP14_CKPT="cache/lpa-ipip14-mixed_2026-02-21_18-54-03-596470/checkpoint_50"

# --- Base model prior: free generation (full thinking, max tokens) ---
# Previously showed 0% because max_new_tokens=80 was too short to close </think>.
# Now fixed: default max_new_tokens=512.
echo "============================================================"
echo "Base model prior: free generation (IPIP-14 statements)"
echo "============================================================"
python diagnostics/probe_ipip_responses.py \
    --model_name Qwen/Qwen3-8B \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --base_model

# --- Base model prior: training template (pre-filled empty think block) ---
# Tests the model in the EXACT context the LAT adversary operates in.
# The adversary's loss is computed on tokens *after* </think>, so this reveals
# the true base prior that drives (or doesn't drive) adversary gradient signal.
echo "============================================================"
echo "Base model prior: training template (IPIP-14 statements)"
echo "============================================================"
python diagnostics/probe_ipip_responses.py \
    --model_name Qwen/Qwen3-8B \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --base_model --use_training_template

# --- Exp 1c: IPIP-10 model, IPIP-10 statements ---
echo "============================================================"
echo "Exp 1c: IPIP-10 checkpoint vs. IPIP-10 statements"
echo "============================================================"
python diagnostics/probe_ipip_responses.py \
    --model_name Qwen/Qwen3-8B \
    --checkpoint_dir "$IPIP10_CKPT" \
    --csv_path data/IPIP-10/harmful_trait.csv \
    --use_training_template

# --- Exp 2c: IPIP-14 model, IPIP-14 statements ---
echo "============================================================"
echo "Exp 2c: IPIP-14 checkpoint vs. IPIP-14 statements"
echo "============================================================"
python diagnostics/probe_ipip_responses.py \
    --model_name Qwen/Qwen3-8B \
    --checkpoint_dir "$IPIP14_CKPT" \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --use_training_template

# --- Exp 1d: IPIP-10 model cross-evaluated on IPIP-14 statements ---
echo "============================================================"
echo "Exp 1d (cross-dataset): IPIP-10 checkpoint vs. IPIP-14 statements"
echo "============================================================"
python diagnostics/probe_ipip_responses.py \
    --model_name Qwen/Qwen3-8B \
    --checkpoint_dir "$IPIP10_CKPT" \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --use_training_template

echo "=== ALL PROBES COMPLETE ==="
