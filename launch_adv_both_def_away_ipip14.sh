#!/bin/bash
# =============================================================================
# Adv-both / Def-away IPIP-14 Training + Eval Dispatcher
#
# Adversary uses both toward and away loss (balanced).
# Defender uses only the away loss.
#
#
# Submits three chained SLURM jobs:
#   1. Training + probe   (h100_3g.40gb, 3 h)
#   2. Eval epoch 50      (h100, 1 h — runs after training succeeds)
#   3. Eval epoch 100     (h100, 1 h — runs after training succeeds)
#
# Usage: bash launch_adv_both_def_away_ipip14.sh
# =============================================================================

MODEL="Qwen/Qwen3-8B"
DATASET="IPIP-14"
PROJECT_NAME="lpa-adv-both-def-away-ipip14"
LAT_CONFIG="latent_at/lat_config_fewer_steps.json"
BATCH_SIZE=4
WORKDIR="/home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S-%6N")
CACHE_DIR="${WORKDIR}/cache/${PROJECT_NAME}_${TIMESTAMP}"

echo "============================================================"
echo "Adv-both / Def-away IPIP-14 dispatcher"
echo "  adv: toward=1.0, away=0.0"
echo "  def: toward=0.5, away=0.5"
echo "  project:   ${PROJECT_NAME}"
echo "  timestamp: ${TIMESTAMP}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Training job
# ---------------------------------------------------------------------------
TRAIN_JOB=$(sbatch --parsable \
    --account="rrg-lplevass" \
    --gres=gpu:h100_3g.40gb \
    --cpus-per-task=2 \
    --mem=48G \
    --time=0-3:00:00 \
    --job-name="${PROJECT_NAME}" \
    --output="${WORKDIR}/logs/slurm/adv_both_def_away_train_%j.out" \
    --error="${WORKDIR}/logs/slurm/adv_both_def_away_train_%j.err" \
    --wrap="
module load cuda httpproxy
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0
cd ${WORKDIR}
source .venv/bin/activate

echo '============================================================'
echo 'Adv-both / Def-away IPIP-14 training'
echo '  adv: toward=1.0, away=0.0'
echo '  def: toward=0.5, away=0.5'
echo '  timestamp: ${TIMESTAMP}'
echo '============================================================'

time python -m latent_at.lat_training_no_sft \
    --model_name ${MODEL} \
    --benign_dataset data/${DATASET}/benign_trait.csv \
    --harmful_dataset data/${DATASET}/harmful_trait.csv \
    --cache_dir cache \
    --system_prompt_path system_prompt/alpha.txt \
    --project_name ${PROJECT_NAME} \
    --lat_config_path ${LAT_CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --timestamp ${TIMESTAMP} \
    --adv_toward_coef 1.0 \
    --adv_away_coef 0.0 \
    --def_toward_coef 0.5 \
    --def_away_coef 0.5

TRAIN_EXIT=\$?
if [ \${TRAIN_EXIT} -ne 0 ]; then
    echo 'Training failed (exit '\${TRAIN_EXIT}'), skipping probe.'
    exit \${TRAIN_EXIT}
fi

echo ''
echo '============================================================'
echo 'Probing checkpoint 50'
echo '============================================================'
python diagnostics/probe_ipip_responses.py \
    --model_name ${MODEL} \
    --checkpoint_dir ${CACHE_DIR}/checkpoint_50 \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --use_training_template

echo ''
echo '============================================================'
echo 'Probing checkpoint 100 (final)'
echo '============================================================'
python diagnostics/probe_ipip_responses.py \
    --model_name ${MODEL} \
    --checkpoint_dir ${CACHE_DIR}/checkpoint_100 \
    --csv_path data/IPIP-14/harmful_trait.csv \
    --use_training_template

echo ''
echo '=== TRAINING + PROBE DONE ==='
")

echo "Training job -> ${TRAIN_JOB}"

# ---------------------------------------------------------------------------
# 2 & 3. Eval jobs — chained on training success
# ---------------------------------------------------------------------------
ALL_EVAL_JOBS=()

for EPOCH in 50 100; do
    EVAL_JOB=$(sbatch --parsable \
        --account="rrg-lplevass" \
        --gres=gpu:h100 \
        --cpus-per-task=4 \
        --mem=48G \
        --time=0-1:00:00 \
        --job-name="eval-adv-both-def-away-ep${EPOCH}" \
        --output="${WORKDIR}/logs/slurm/adv_both_def_away_eval_ep${EPOCH}_%j.out" \
        --error="${WORKDIR}/logs/slurm/adv_both_def_away_eval_ep${EPOCH}_%j.err" \
        --dependency="afterok:${TRAIN_JOB}" \
        --wrap="
module load cuda httpproxy
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0
cd ${WORKDIR}
source .venv/bin/activate

echo '=== Eval: adv-both def-away  epoch=${EPOCH} ==='
time python -m eval \
    --model_name ${MODEL} \
    --project_name ${PROJECT_NAME} \
    --run_id ${TIMESTAMP} \
    --epoch ${EPOCH} \
    --attacks DirectRequest \
    --evals MMLU \
    --no_wandb
")
    echo "Eval ep${EPOCH} job  -> ${EVAL_JOB}"
    ALL_EVAL_JOBS+=("${EVAL_JOB}")
done

echo ""
echo "All 3 jobs submitted."
echo "  Training:    ${TRAIN_JOB}"
echo "  Eval ep50:   ${ALL_EVAL_JOBS[0]}"
echo "  Eval ep100:  ${ALL_EVAL_JOBS[1]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs: ${WORKDIR}/logs/slurm/adv_both_def_away_*.out"
