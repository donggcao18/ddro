#!/usr/bin/env bash
set -euo pipefail

# Fresh full-query run. Override paths through environment variables if needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-url-bm25/checkpoint-50020}"
QUERY_FILE="${QUERY_FILE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_merged_multilabel.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-url-branch-preference-full-fresh}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

exec "${PYTHON_BIN}" -u "${REPO_ROOT}/src/pretrain/launch_branch_preference.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --query-file "${QUERY_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-type url_based_id \
  --arm branch \
  --rounds all \
  --queries-per-round 10000 \
  --steps-per-round 1000 \
  --num-gpus 4 \
  --mining-num-gpus 4 \
  --precision fp16 \
  --num-beams 15 \
  --mining-batch-size 25 \
  --total-training-negatives-per-query 4 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --learning-rate 1e-6 \
  --warmup-ratio 0.1 \
  --beta 0.4 \
  --branch-beta 0.4 \
  --branch-loss-weight 0.1 \
  --max-prompt-length 256 \
  --max-target-length 128 \
  --target-length-policy truncate \
  --target-collision-policy allow \
  --target-token-policy allow \
  --eval-every-rounds 8 \
  --eval-at-end \
  --save-steps 100 \
  --logging-steps 10 \
  --seed 42 \
  "$@"
