#!/usr/bin/env bash
set -euo pipefail

# Standalone three-round tuning run: mine -> mix BM25 -> branch DPO -> EMA.
# Train every usable pair for one epoch per round, then infer after round-002.
# Copy this file with a descriptive experiment name to preserve a trial's settings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr/checkpoint-161800}"
QUERY_FILE="${QUERY_FILE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_merged_multilabel.jsonl}"
BM25_FILE="${BM25_FILE:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25/dpo_pairs_url.jsonl}"
INFER_FILE="${INFER_FILE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_test_r32.0_multilabel.json}"

# Parameters to vary between trials. EMA decay is the previous reference's weight.
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
QUERIES_PER_ROUND="${QUERIES_PER_ROUND:-10000}"
REFERENCE_EMA_DECAY="${REFERENCE_EMA_DECAY:-0.9}"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-1}"
BETA="${BETA:-0.4}"
BRANCH_BETA="${BRANCH_BETA:-0.4}"
BRANCH_LOSS_WEIGHT="${BRANCH_LOSS_WEIGHT:-0.1}"
SEED="${SEED:-42}"
MINING_BATCH_SIZE="${MINING_BATCH_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vault-url-branch-preference-tune-r3-q${QUERIES_PER_ROUND}-lr${LEARNING_RATE}-ema${REFERENCE_EMA_DECAY}-epoch${EPOCHS_PER_ROUND}-beta${BETA}-bb${BRANCH_BETA}-bw${BRANCH_LOSS_WEIGHT}-s${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

# Invoke Python directly; no other experiment shell script is inherited.
# Trailing arguments support --resume and --prepare-only. Use a distinct output
# name if overriding training settings through CLI flags or changing input files.
# The controller disables beam early stopping for final inference automatically.
exec "${PYTHON_BIN}" -u "${REPO_ROOT}/src/pretrain/launch_branch_preference.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --query-file "${QUERY_FILE}" \
  --bm25-file "${BM25_FILE}" \
  --infer-file "${INFER_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-type url_based_id \
  --arm branch \
  --rounds 3 \
  --queries-per-round "${QUERIES_PER_ROUND}" \
  --epochs-per-round "${EPOCHS_PER_ROUND}" \
  --reference-ema-decay "${REFERENCE_EMA_DECAY}" \
  --learning-rate "${LEARNING_RATE}" \
  --warmup-ratio 0.1 \
  --beta "${BETA}" \
  --branch-beta "${BRANCH_BETA}" \
  --branch-loss-weight "${BRANCH_LOSS_WEIGHT}" \
  --num-gpus 4 \
  --mining-num-gpus 4 \
  --precision fp16 \
  --num-beams 15 \
  --mining-batch-size "${MINING_BATCH_SIZE}" \
  --policy-mining-negatives-per-query 4 \
  --bm25-training-negatives-per-query 6 \
  --total-training-negatives-per-query 10 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --max-prompt-length 256 \
  --max-target-length 128 \
  --length-penalty 1.0 \
  --target-length-policy truncate \
  --target-collision-policy allow \
  --target-token-policy allow \
  --validation-fraction 0.05 \
  --eval-every-rounds 3 \
  --eval-at-end \
  --keep-last-round-checkpoints 3 \
  --save-steps 100 \
  --logging-steps 10 \
  --seed "${SEED}" \
  "$@"
