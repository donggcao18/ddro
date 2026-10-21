#!/usr/bin/env bash
set -euo pipefail

# Defaults: 2,000 queries, one epoch, LR 2e-5, and reference EMA decay 0.98.
# Override the environment variables below to run a separate experiment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export QUERIES_PER_ROUND="${QUERIES_PER_ROUND:-2000}"
export EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-1}"
export REFERENCE_EMA_DECAY="${REFERENCE_EMA_DECAY:-0.98}"
export EVAL_EVERY_ROUNDS="${EVAL_EVERY_ROUNDS:-40}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vault-url-branch-preference-mixed-q${QUERIES_PER_ROUND}-lr${LEARNING_RATE}-ema${REFERENCE_EMA_DECAY}-epoch${EPOCHS_PER_ROUND}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"

# Inherit checkpoint, training/BM25/infer files, negative budgets, batch sizes,
# and the latest-three-round retention policy from the mixed run.
# For extra CLI overrides, set EXPERIMENT_NAME or OUTPUT_DIR to distinguish runs.
exec bash "${SCRIPT_DIR}/run_vault_branch_preference_mixed.sh" "$@"
