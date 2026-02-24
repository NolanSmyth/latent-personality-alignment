#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0-3:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-away-only-ipip14
#SBATCH --output=logs/slurm/away_only_%j.out
#SBATCH --error=logs/slurm/away_only_%j.err

# =============================================================================
# Away-only loss ablation on IPIP-14 mixed data
#
# Hypothesis: setting toward=0 removes the gradient cancellation between
# positive ("I agree") and negative ("I disagree") items, so the defense
# can finally overfit the mixed-valence IPIP-14 set.
#
# Loss config:  adv_toward=0.0, adv_away=1.0
#               def_toward=0.0, def_away=1.0
#
# After training, checkpoints are probed to measure agree/disagree accuracy.
# =============================================================================

module load cuda httpproxy

export HF_HUB_OFFLINE=1
export PYTHONBREAKPOINT=0
export WANDB_MODE=offline

cd /home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment
source .venv/bin/activate

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S-%6N")
MODEL="Qwen/Qwen3-8B"
DATASET="IPIP-14"
SYSTEM_PROMPT="system_prompt/alpha.txt"
PROJECT_NAME="lpa-away-only-ipip14"
BATCH_SIZE=4
LAT_CONFIG="latent_at/lat_config_fewer_steps.json"

echo "============================================================"
echo "Away-only IPIP-14 mixed training"
echo "  adv: toward=0.0, away=1.0"
echo "  def: toward=0.0, away=1.0"
echo "  dataset: ${DATASET} (mixed valence)"
echo "  timestamp: ${TIMESTAMP}"
echo "============================================================"

time python -m latent_at.lat_training_no_sft \
    --model_name ${MODEL} \
    --benign_dataset data/${DATASET}/benign_trait.csv \
    --harmful_dataset data/${DATASET}/harmful_trait.csv \
    --cache_dir cache \
    --system_prompt_path ${SYSTEM_PROMPT} \
    --project_name ${PROJECT_NAME} \
    --lat_config_path ${LAT_CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --timestamp ${TIMESTAMP} \
    --adv_toward_coef 0.0 \
    --adv_away_coef 1.0 \
    --def_toward_coef 0.0 \
    --def_away_coef 1.0

TRAIN_EXIT=$?
if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "Training failed (exit ${TRAIN_EXIT}), skipping probe."
    exit ${TRAIN_EXIT}
fi

# Derive checkpoint paths (N_checkpoints=20, num_steps=100 → saves every 5 steps)
CACHE_DIR="cache/${PROJECT_NAME}_${TIMESTAMP}"
CKPT_50="${CACHE_DIR}/checkpoint_50"
CKPT_100="${CACHE_DIR}/checkpoint_100"

echo ""
echo "============================================================"
echo "Probing checkpoint 50 — does away-only training overfit?"
echo "============================================================"
python diagnostics/probe_ipip_responses.py \
    --model_name ${MODEL} \
    --checkpoint_dir "${CKPT_50}" \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --use_training_template

echo ""
echo "============================================================"
echo "Probing checkpoint 100 (final)"
echo "============================================================"
python diagnostics/probe_ipip_responses.py \
    --model_name ${MODEL} \
    --checkpoint_dir "${CKPT_100}" \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --use_training_template

echo ""
echo "=== DONE ==="
