#!/usr/bin/env bash
set -euo pipefail

# Mine structure_id_v3 hard negatives with BM25 only, then train DPO.
# Lucene uses the legacy text_id internally; chosen/rejected are structure_id_v3.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start}"
TRAIN_ORIGINAL="${TRAIN_ORIGINAL:-${DATA_ROOT}/Ruby_train_r32.0.json}"
TEST_ORIGINAL="${TEST_ORIGINAL:-${DATA_ROOT}/Ruby_test_r32.0.json}"
AUGMENTATION="${AUGMENTATION:-${DATA_ROOT}/Ruby_ready_to_feed_multilabel.jsonl}"
STRUCTURE_ID_SOURCE="${STRUCTURE_ID_SOURCE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/augmented_dsi/Ruby_merged.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_structure_v3__qg_t5/checkpoint-130000}"
WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25_structure_v3}"
PREFERENCE_OBJECTIVE="${PREFERENCE_OBJECTIVE:-dpo}"
TDPO_ALPHA="${TDPO_ALPHA:-0.5}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-structure-v3-bm25-${PREFERENCE_OBJECTIVE}-t5-base}"

BM25_PYTHON_BIN="${BM25_PYTHON_BIN:-python}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"
RUN_BM25="${RUN_BM25:-1}"
REUSE_INDEX="${REUSE_INDEX:-0}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
PRECISION="${PRECISION:-bf16}"

BM25_THREADS="${BM25_THREADS:-16}"
BM25_BATCH_SIZE="${BM25_BATCH_SIZE:-16}"
BM25_HITS="${BM25_HITS:-200}"
NEGATIVES_PER_QUERY="${NEGATIVES_PER_QUERY:-8}"
RANK_RANGES="${RANK_RANGES:-1:20,21:100,101:200}"
RANK_QUOTAS="${RANK_QUOTAS:-3,2,3}"
SEED="${SEED:-42}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
DPO_BETA="${DPO_BETA:-0.4}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"

DPO_PAIRS="${WORK_DIR}/dpo_pairs_structure_id_v3.jsonl"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 2
  fi
}

require_directory() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory does not exist: $1" >&2
    exit 2
  fi
}

add_precision_flag() {
  local -n target_array="$1"
  case "${PRECISION}" in
    bf16) target_array+=(--bf16) ;;
    fp16) target_array+=(--fp16) ;;
    fp32) ;;
    *) echo "PRECISION must be bf16, fp16, or fp32" >&2; exit 2 ;;
  esac
}

require_directory "${CHECKPOINT_PATH}"
require_file "${STRUCTURE_ID_SOURCE}"
mkdir -p "${WORK_DIR}" "${DPO_OUTPUT_DIR}"

if [[ "${RUN_BM25}" == "1" ]]; then
  require_file "${TRAIN_ORIGINAL}"
  require_file "${AUGMENTATION}"
  BM25_ARGS=(
    src/scripts/bm25/the_vault/run_pipeline.py
    --train-original "${TRAIN_ORIGINAL}"
    --augmentation "${AUGMENTATION}"
    --structure-id-source "${STRUCTURE_ID_SOURCE}"
    --structure-id-join-key url_based_id
    --target-type structure_id_v3
    --work-dir "${WORK_DIR}"
    --threads "${BM25_THREADS}"
    --batch-size "${BM25_BATCH_SIZE}"
    --hits "${BM25_HITS}"
    --negatives-per-query "${NEGATIVES_PER_QUERY}"
    --rank-ranges "${RANK_RANGES}"
    --rank-quotas "${RANK_QUOTAS}"
    --seed "${SEED}"
    --strict
  )
  if [[ -n "${TEST_ORIGINAL}" ]]; then
    require_file "${TEST_ORIGINAL}"
    BM25_ARGS+=(--test-original "${TEST_ORIGINAL}")
  fi
  if [[ "${REUSE_INDEX}" == "1" ]]; then BM25_ARGS+=(--reuse-index); fi

  echo "=== Stage 1/2: BM25 structure_id_v3 hard-negative mining ==="
  "${BM25_PYTHON_BIN}" "${BM25_ARGS[@]}"
else
  echo "=== Stage 1/2: Reusing existing BM25 structure pairs ==="
fi
require_file "${DPO_PAIRS}"

TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${DPO_PAIRS}"
  --output_dir "${DPO_OUTPUT_DIR}"
  --validation_split "${VALIDATION_SPLIT}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --beta "${DPO_BETA}"
  --preference_objective "${PREFERENCE_OBJECTIVE}"
  --tdpo_alpha "${TDPO_ALPHA}"
  --max_steps "${MAX_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --seed "${SEED}"
  --gradient_checkpointing
)
add_precision_flag TRAIN_ARGS

echo "=== Stage 2/2: ${PREFERENCE_OBJECTIVE} validation and training ==="
if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi
if [[ "${RUN_TRAINING}" == "1" ]]; then
  if (( NUM_GPUS > 1 )); then
    "${DPO_PYTHON_BIN}" -m torch.distributed.run --standalone \
      --nproc_per_node="${NUM_GPUS}" "${TRAIN_ARGS[@]}"
  else
    "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}"
  fi
fi

echo "BM25 DPO pairs: ${DPO_PAIRS}"
echo "Final DPO model: ${DPO_OUTPUT_DIR}/final"
