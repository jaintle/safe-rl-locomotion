#!/usr/bin/env bash
# reproduce_hopper_v4_500k.sh
# ============================
# Reproduce the Hopper-v4 benchmark at 500 000 environment steps.
#
# Runs PPO (seeds 0,1,2) and C-PPO (seeds 0,1,2) with the calibrated
# cost configuration, then generates per-run plots and an aggregated
# summary report.
#
# Cost configuration:
#   cost_fn                        = action_magnitude
#   cost_action_magnitude_threshold = 0.25
#   cost_limit (C-PPO)             = 80.0
#
# Save directory convention:
#   runs/ppo_hopper_v4_t025_seed{S}_500k
#   runs/cppo_hopper_v4_t025_limit80_seed{S}_500k
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/reproduce_hopper_v4_500k.sh
#
# Expected wall time (CPU): ~3–6 h total (6 runs × ~30–60 min each).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUDGET=500000
EVAL_EVERY=10000
COST_LIMIT=80.0
COST_FN="action_magnitude"
THRESHOLD=0.25
SEEDS=(0 1 2)
ENV_ID="Hopper-v4"
TAG="500k"

echo "========================================================"
echo " Robot-Safe PPO — Hopper-v4 benchmark (${BUDGET} steps)"
echo " cost_fn=${COST_FN}  threshold=${THRESHOLD}  cost_limit=${COST_LIMIT}"
echo " Seeds: ${SEEDS[*]}"
echo "========================================================"

# ------------------------------------------------------------------ #
# 1. PPO runs                                                         #
# ------------------------------------------------------------------ #
for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${REPO_ROOT}/runs/ppo_hopper_v4_t025_seed${SEED}_${TAG}"
    echo ""
    echo ">>> PPO | seed=${SEED} | save_dir=${SAVE_DIR}"
    python "${REPO_ROOT}/scripts/train_ppo.py" \
        --env_id "${ENV_ID}" \
        --seed "${SEED}" \
        --total_timesteps "${BUDGET}" \
        --save_dir "${SAVE_DIR}" \
        --eval_every "${EVAL_EVERY}"

    # Per-run plots
    PLOTS_DIR="${SAVE_DIR}/plots"
    echo "    Generating per-run plots → ${PLOTS_DIR}"
    python "${REPO_ROOT}/scripts/make_plots.py" \
        --run_dir "${SAVE_DIR}" \
        --env_id "${ENV_ID}" \
        --seed "${SEED}" \
        --total_timesteps "${BUDGET}" \
        --cost_limit "${COST_LIMIT}" \
        --out_dir "${PLOTS_DIR}"
done

# ------------------------------------------------------------------ #
# 2. C-PPO runs                                                       #
# ------------------------------------------------------------------ #
for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${REPO_ROOT}/runs/cppo_hopper_v4_t025_limit80_seed${SEED}_${TAG}"
    echo ""
    echo ">>> C-PPO | seed=${SEED} | save_dir=${SAVE_DIR}"
    python "${REPO_ROOT}/scripts/train_cppo.py" \
        --env_id "${ENV_ID}" \
        --seed "${SEED}" \
        --total_timesteps "${BUDGET}" \
        --save_dir "${SAVE_DIR}" \
        --eval_every "${EVAL_EVERY}" \
        --cost_limit "${COST_LIMIT}" \
        --cost_fn "${COST_FN}"

    # Per-run plots
    PLOTS_DIR="${SAVE_DIR}/plots"
    echo "    Generating per-run plots → ${PLOTS_DIR}"
    python "${REPO_ROOT}/scripts/make_plots.py" \
        --run_dir "${SAVE_DIR}" \
        --env_id "${ENV_ID}" \
        --seed "${SEED}" \
        --total_timesteps "${BUDGET}" \
        --cost_limit "${COST_LIMIT}" \
        --out_dir "${PLOTS_DIR}"
done

# ------------------------------------------------------------------ #
# 3. Aggregated summary plots + markdown table                        #
# ------------------------------------------------------------------ #
PPO_DIRS=""
CPPO_DIRS=""
for SEED in "${SEEDS[@]}"; do
    PPO_DIRS="${PPO_DIRS} ${REPO_ROOT}/runs/ppo_hopper_v4_t025_seed${SEED}_${TAG}"
    CPPO_DIRS="${CPPO_DIRS} ${REPO_ROOT}/runs/cppo_hopper_v4_t025_limit80_seed${SEED}_${TAG}"
done

SUMMARY_OUT="${REPO_ROOT}/reports/figures/hopper_v4"
echo ""
echo ">>> Aggregating results → ${SUMMARY_OUT}"
# shellcheck disable=SC2086
python "${REPO_ROOT}/scripts/aggregate_results.py" \
    --ppo_dirs  ${PPO_DIRS} \
    --cppo_dirs ${CPPO_DIRS} \
    --env_id "${ENV_ID}" \
    --budget "${BUDGET}" \
    --cost_limit "${COST_LIMIT}" \
    --out_dir "${SUMMARY_OUT}" \
    --tag "${TAG}"

echo ""
echo "========================================================"
echo " 500k benchmark complete."
echo " Per-run artifacts : runs/ppo_hopper_v4_t025_seed*_${TAG}/"
echo "                   : runs/cppo_hopper_v4_t025_limit80_seed*_${TAG}/"
echo " Summary figures   : ${SUMMARY_OUT}/"
echo "========================================================"
