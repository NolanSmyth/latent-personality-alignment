#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-2:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-diag5-7
#SBATCH --output=logs/slurm/diag5-7_%j.out
#SBATCH --error=logs/slurm/diag5-7_%j.err

# Experiments 5–7 diagnostics:
#   Exp 5: Token probability diagnostic (base model log-probs for agree/disagree/refuse)
#   Exp 6: Negative-only training (sbatch separately — this script just verifies the dataset)
#   Exp 7: Per-item δ norm diagnostic (PGD perturbation magnitudes by valence)

module load cuda httpproxy

export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0
export WANDB_MODE=offline

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

# ============================================================
# Exp 5: Token probability diagnostic
# ============================================================
echo "============================================================"
echo "Exp 5: Token probabilities — base model on IPIP-14"
echo "============================================================"
python diagnostics/probe_token_probabilities.py \
    --model_name Qwen/Qwen3-8B \
    --csv_path data/IPIP-14/harmful_trait.csv

# ============================================================
# Exp 7: Per-item δ norm diagnostic
# ============================================================
echo "============================================================"
echo "Exp 7: Per-item δ norms — base model + fresh LoRA on IPIP-14"
echo "============================================================"
python diagnostics/probe_delta_norms.py \
    --model_name Qwen/Qwen3-8B \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --batch_size 4 \
    --pgd_iterations 16 \
    --epsilon 6.0

echo "=== DIAGNOSTICS 5 & 7 COMPLETE ==="
echo ""
echo "To run Exp 6 (negative-only training), submit:"
echo "  sbatch launch_experiment.sh Qwen/Qwen3-8B IPIP-14-neg system_prompt/alpha.txt lpa-ipip14-negonly 4"
