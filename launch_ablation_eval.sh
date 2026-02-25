#!/bin/bash
# =============================================================================
# Ablation Eval Dispatcher
#
# Submits lightweight eval (DirectRequest ASR + MMLU, same as checkpoint sweep)
# for the toward-only, away-only, and balanced IPIP-14 ablation checkpoints
# at epochs 50 and 100.  One job per (run, epoch) — ~1 hour each.
#
# Usage: bash launch_ablation_eval.sh
# =============================================================================

MODEL="Qwen/Qwen3-8B"
WORKDIR="/home/nsmyth/links/projects/def-hezaveh/nsmyth/latent-personality-alignment"

# Format: "PROJECT TIMESTAMP"
declare -A RUNS
RUNS["toward-only"]="lpa-toward-only-ipip14 2026-02-23_14-32-32-594977"
RUNS["away-only"]="lpa-away-only-ipip14 2026-02-23_15-15-06-597192"
RUNS["balanced"]="lpa-balanced-ipip14 2026-02-23_15-33-51-527422"

EPOCHS=(50 100)

ALL_JOB_IDS=()

for LABEL in "${!RUNS[@]}"; do
    read -r PROJECT TIMESTAMP <<< "${RUNS[$LABEL]}"
    for EPOCH in "${EPOCHS[@]}"; do

        echo "=== ${LABEL}  epoch=${EPOCH} ==="

        JOB=$(sbatch \
            --account="rrg-lplevass" \
            --gres=gpu:h100 \
            --cpus-per-task=4 \
            --mem=48G \
            --time=0-1:00:00 \
            --job-name="eval-${LABEL}-ep${EPOCH}" \
            --output="${WORKDIR}/logs/slurm/ablation_eval_${LABEL}_ep${EPOCH}_%j.out" \
            --error="${WORKDIR}/logs/slurm/ablation_eval_${LABEL}_ep${EPOCH}_%j.err" \
            --wrap="
module load cuda httpproxy
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONBREAKPOINT=0
cd ${WORKDIR}
source .venv/bin/activate

echo '=== ${LABEL}  epoch=${EPOCH} ==='
time python -m eval \
    --model_name ${MODEL} \
    --project_name ${PROJECT} \
    --run_id ${TIMESTAMP} \
    --epoch ${EPOCH} \
    --attacks DirectRequest \
    --evals MMLU \
    --no_wandb
" \
            | awk '{print $NF}')
        echo "  -> job ${JOB}"
        ALL_JOB_IDS+=("${JOB}")

    done
done

# Submit a lightweight plot job that runs after all eval jobs succeed
DEP=$(IFS=:; echo "afterok:${ALL_JOB_IDS[*]}")
PLOT_JOB=$(sbatch \
    --account="rrg-lplevass" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=0-0:10:00 \
    --job-name="ablation-plot" \
    --output="${WORKDIR}/logs/slurm/ablation_plot_%j.out" \
    --error="${WORKDIR}/logs/slurm/ablation_plot_%j.err" \
    --dependency="${DEP}" \
    --wrap="
cd ${WORKDIR}
source .venv/bin/activate
python diagnostics/plot_ablation_comparison.py \
    --output_dir diagnostics/figures \
    --output_name ablation_comparison.png
" \
    | awk '{print $NF}')
echo ""
echo "Plot job -> ${PLOT_JOB} (runs after all evals complete)"
echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Figure will be saved to: diagnostics/figures/ablation_comparison.png"
