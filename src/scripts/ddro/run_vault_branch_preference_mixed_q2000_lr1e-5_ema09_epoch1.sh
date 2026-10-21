#!/usr/bin/env bash
set -euo pipefail

# Standalone URL-ID experiment: BM25 + current-policy branch DPO.
# LR 1e-5; 1,000 queries/round; one epoch/round; reference EMA decay 0.99.
# All hyperparameters are explicit here; this script does not call another run script.
# Copy and rename this file for a different experiment, including its output directory.
# Append --resume to continue this experiment with the same settings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr/checkpoint-161800}"
QUERY_FILE="${QUERY_FILE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_merged_multilabel.jsonl}"
BM25_FILE="${BM25_FILE:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25/dpo_pairs_url.jsonl}"
INFER_FILE="${INFER_FILE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_test_r32.0_multilabel.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-url-branch-preference-mixed-q1000-lr1e-5-ema0.99-epoch1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

# The launcher evaluates INFER_FILE after the final round with beam early stopping off.
exec "${PYTHON_BIN}" -u "${REPO_ROOT}/src/pretrain/launch_branch_preference.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --query-file "${QUERY_FILE}" \
  --bm25-file "${BM25_FILE}" \
  --infer-file "${INFER_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-type url_based_id \
  --arm branch \
  --rounds all \
  --queries-per-round 2000 \
  --epochs-per-round 1 \
  --reference-ema-decay 0.9 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --beta 0.4 \
  --branch-beta 0.4 \
  --branch-loss-weight 0.1 \
  --num-gpus 4 \
  --mining-num-gpus 4 \
  --precision fp16 \
  --num-beams 15 \
  --mining-batch-size 16 \
  --policy-mining-negatives-per-query 4 \
  --bm25-training-negatives-per-query 6 \
  --total-training-negatives-per-query 10 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --max-prompt-length 256 \
  --max-target-length 128 \
  --target-length-policy truncate \
  --target-collision-policy allow \
  --target-token-policy allow \
  --eval-every-rounds 80 \
  --eval-at-end \
  --keep-last-round-checkpoints 3 \
  --save-steps 100 \
  --logging-steps 10 \
  --seed 42 \
  "$@"
