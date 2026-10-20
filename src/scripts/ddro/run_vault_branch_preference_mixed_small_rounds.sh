#!/usr/bin/env bash
set -euo pipefail

# Refresh model negatives every 2,000 queries; train on all usable round pairs.
# Decay 0.98 incorporates 2% of the latest policy after each completed round.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-url-branch-preference-mixed-q2000-ema098-epoch1}"
QUERIES_PER_ROUND="${QUERIES_PER_ROUND:-2000}"
export EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-1}"
REFERENCE_EMA_DECAY="${REFERENCE_EMA_DECAY:-0.98}"
EVAL_EVERY_ROUNDS="${EVAL_EVERY_ROUNDS:-40}"

# Inherit checkpoint, training/BM25/infer files, negative budgets, batch sizes,
# learning rate and the latest-three-round retention policy from the mixed run.
exec bash "${SCRIPT_DIR}/run_vault_branch_preference_mixed.sh" \
  --queries-per-round "${QUERIES_PER_ROUND}" \
  --reference-ema-decay "${REFERENCE_EMA_DECAY}" \
  --eval-every-rounds "${EVAL_EVERY_ROUNDS}" \
  "$@"
