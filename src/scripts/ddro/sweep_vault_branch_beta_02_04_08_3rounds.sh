#!/usr/bin/env bash
set -euo pipefail

# Run all three trials sequentially on the same GPUs.
# Each trial starts from CHECKPOINT_PATH, trains rounds 000-002, then infers.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TUNING_SCRIPT="${SCRIPT_DIR}/run_vault_branch_preference_tune_3rounds.sh"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-url-branch-beta-sweep-r3-q10000-lr2e-5-ema0.9-epoch1-beta0.4-bw0.1-s42}"

# Hold these settings fixed across trials; vary only BRANCH_BETA below.
export LEARNING_RATE=2e-5
export QUERIES_PER_ROUND=10000
export REFERENCE_EMA_DECAY=0.9
export EPOCHS_PER_ROUND=1
export BETA=0.4
export BRANCH_LOSS_WEIGHT=0.1
export SEED=42
export MINING_BATCH_SIZE=16
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

printf '\nTrial 1/3: branch beta 0.2\n'
BRANCH_BETA=0.2 \
OUTPUT_DIR="${SWEEP_OUTPUT_ROOT}/branch-beta-0.2" \
bash "${TUNING_SCRIPT}" "$@"

printf '\nTrial 2/3: branch beta 0.4 (baseline)\n'
BRANCH_BETA=0.4 \
OUTPUT_DIR="${SWEEP_OUTPUT_ROOT}/branch-beta-0.4" \
bash "${TUNING_SCRIPT}" "$@"

printf '\nTrial 3/3: branch beta 0.8\n'
BRANCH_BETA=0.8 \
OUTPUT_DIR="${SWEEP_OUTPUT_ROOT}/branch-beta-0.8" \
bash "${TUNING_SCRIPT}" "$@"

printf '\nSweep complete. Inference metrics:\n'
printf '%s\n' \
  "${SWEEP_OUTPUT_ROOT}/branch-beta-0.2/final_inference.json" \
  "${SWEEP_OUTPUT_ROOT}/branch-beta-0.4/final_inference.json" \
  "${SWEEP_OUTPUT_ROOT}/branch-beta-0.8/final_inference.json"
