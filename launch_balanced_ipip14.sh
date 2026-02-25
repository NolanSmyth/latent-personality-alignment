#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0-3:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=lpa-balanced-ipip14
#SBATCH --output=logs/slurm/balanced_%j.out
#SBATCH --error=logs/slurm/balanced_%j.err

# =============================================================================
# Balanced (toward=0.5, away=0.5) loss on IPIP-14 mixed data
#
# Control condition: standard default coefficients on the mixed-valence set.
# Paired with launch_toward_only_ipip14.sh and launch_away_only_ipip14.sh
# to isolate which loss term is responsible for the agree-collapse.
#
# Loss config:  adv_toward=0.5, adv_away=0.5
#               def_toward=0.5, def_away=0.5
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
PROJECT_NAME="lpa-balanced-ipip14"
BATCH_SIZE=4
LAT_CONFIG="latent_at/lat_config_fewer_steps.json"

echo "============================================================"
echo "Balanced IPIP-14 mixed training (control)"
echo "  adv: toward=0.5, away=0.5"
echo "  def: toward=0.5, away=0.5"
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
    --adv_toward_coef 0.5 \
    --adv_away_coef 0.5 \
    --def_toward_coef 0.5 \
    --def_away_coef 0.5

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
echo "Probing checkpoint 50"
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
