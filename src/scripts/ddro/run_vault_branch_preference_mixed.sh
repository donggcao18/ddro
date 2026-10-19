#!/usr/bin/env bash
set -euo pipefail

# Full-query branch DPO with BM25 + current-policy negatives.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-url-branch-preference-full-fresh/round-019/training/checkpoint-1000}"
QUERY_FILE="${QUERY_FILE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_merged_multilabel_dpo.jsonl}"
BM25_FILE="${BM25_FILE:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25/dpo_pairs_url.jsonl}"
BM25_NEGATIVES_PER_QUERY="${BM25_NEGATIVES_PER_QUERY:-2}"
INFER_FILE="${INFER_FILE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_test_r32.0_multilabel.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-url-branch-preference-full-fresh-epoch2-bm25-mixed}"
INFER_ARGS=()
if [[ -n "${INFER_FILE:-}" ]]; then
  INFER_ARGS=(--infer-file "${INFER_FILE}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

exec "${PYTHON_BIN}" -u "${REPO_ROOT}/src/pretrain/launch_branch_preference.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --query-file "${QUERY_FILE}" \
  --bm25-file "${BM25_FILE}" \
  --bm25-negatives-per-query "${BM25_NEGATIVES_PER_QUERY}" \
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
  --mining-batch-size 16 \
  --negatives-per-query 4 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --learning-rate 1e-5 \
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
  "${INFER_ARGS[@]}" "$@"
