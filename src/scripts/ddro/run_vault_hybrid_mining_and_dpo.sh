#!/usr/bin/env bash
set -euo pipefail

# Mine an 8-pair Vault DPO dataset (up to 4 model-confusion + BM25 fallback)
# and train DPO from the same frozen SFT checkpoint.
#
# Every setting can be overridden without editing this file:
#   CHECKPOINT_PATH=/models/vault-sft/checkpoint-196000 \
#   WORK_DIR=/scratch/vault_hybrid \
#   NUM_GPUS=2 \
#   bash src/scripts/ddro/run_vault_hybrid_mining_and_dpo.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Data and output paths
# ---------------------------------------------------------------------------

DATA_ROOT="${DATA_ROOT:-/home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start}"
TRAIN_ORIGINAL="${TRAIN_ORIGINAL:-${DATA_ROOT}/Ruby_train_r32.0.json}"
TEST_ORIGINAL="${TEST_ORIGINAL:-${DATA_ROOT}/Ruby_test_r32.0.json}"
AUGMENTATION="${AUGMENTATION:-${DATA_ROOT}/Ruby_ready_to_feed_numeric.jsonl}"
WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/east/CodeGR/model/DSI_QG_Ruby_t5-base_multilabel_RQ_8_16_decoder_start/checkpoint-196000}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-hybrid-prompt256-id20}"

BM25_PAIRS="${WORK_DIR}/dpo_pairs.jsonl"
MODEL_PAIRS="${WORK_DIR}/model_confusion_pairs.jsonl"
HYBRID_PAIRS="${WORK_DIR}/dpo_pairs_hybrid.jsonl"
QUERY_METADATA="${WORK_DIR}/query_metadata.jsonl"
DOCUMENT_METADATA="${WORK_DIR}/document_metadata.jsonl"

# Use separate interpreters when Pyserini and training dependencies are in
# different environments. They may point to the same executable.
BM25_PYTHON_BIN="${BM25_PYTHON_BIN:-python}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# Mining configuration
# ---------------------------------------------------------------------------

RUN_BM25="${RUN_BM25:-1}"                 # 0 reuses an existing BM25 work dir
REUSE_INDEX="${REUSE_INDEX:-0}"
BM25_THREADS="${BM25_THREADS:-16}"
BM25_BATCH_SIZE="${BM25_BATCH_SIZE:-16}"
BM25_HITS="${BM25_HITS:-200}"

MODEL_NEGATIVES="${MODEL_NEGATIVES:-4}"
TOTAL_NEGATIVES="${TOTAL_NEGATIVES:-8}"
NUM_BEAMS="${NUM_BEAMS:-8}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-20}"
MINING_DEVICE="${MINING_DEVICE:-auto}"
MINE_LIMIT_QUERIES="${MINE_LIMIT_QUERIES:-}"  # e.g. 100 for a smoke test
REQUIRE_EXACT_MIX="${REQUIRE_EXACT_MIX:-0}"
SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
# DPO configuration
# ---------------------------------------------------------------------------

RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
PRECISION="${PRECISION:-bf16}"            # bf16, fp16, or fp32
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
DPO_BETA="${DPO_BETA:-0.4}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"


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
    *)
      echo "PRECISION must be bf16, fp16, or fp32" >&2
      exit 2
      ;;
  esac
}


require_directory "${CHECKPOINT_PATH}"
mkdir -p "${WORK_DIR}" "${DPO_OUTPUT_DIR}"

echo "=== Vault hybrid mining and DPO ==="
echo "Repository:       ${REPO_ROOT}"
echo "SFT checkpoint:   ${CHECKPOINT_PATH}"
echo "Mining work dir:  ${WORK_DIR}"
echo "DPO output dir:   ${DPO_OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Preserve/create the existing 8-negative BM25 baseline
# ---------------------------------------------------------------------------

if [[ "${RUN_BM25}" == "1" ]]; then
  require_file "${TRAIN_ORIGINAL}"
  require_file "${AUGMENTATION}"

  BM25_ARGS=(
    src/scripts/bm25/the_vault/run_pipeline.py
    --train-original "${TRAIN_ORIGINAL}"
    --augmentation "${AUGMENTATION}"
    --work-dir "${WORK_DIR}"
    --threads "${BM25_THREADS}"
    --batch-size "${BM25_BATCH_SIZE}"
    --hits "${BM25_HITS}"
    --negatives-per-query 8
    --rank-ranges "1:20,21:100,101:200"
    --rank-quotas "3,2,3"
    --seed "${SEED}"
  )
  if [[ -n "${TEST_ORIGINAL}" ]]; then
    require_file "${TEST_ORIGINAL}"
    BM25_ARGS+=(--test-original "${TEST_ORIGINAL}")
  fi
  if [[ "${REUSE_INDEX}" == "1" ]]; then
    BM25_ARGS+=(--reuse-index)
  fi

  echo "=== Stage 1/4: BM25 baseline mining ==="
  "${BM25_PYTHON_BIN}" "${BM25_ARGS[@]}"
else
  echo "=== Stage 1/4: Reusing BM25 baseline ==="
  require_file "${BM25_PAIRS}"
  require_file "${QUERY_METADATA}"
  require_file "${DOCUMENT_METADATA}"
fi

# ---------------------------------------------------------------------------
# 2. Mine the SFT model's highest-ranked valid incorrect DocIDs
# ---------------------------------------------------------------------------

MODEL_ARGS=(
  src/scripts/bm25/the_vault/mine_model_confusion_negatives.py
  --checkpoint-path "${CHECKPOINT_PATH}"
  --query-metadata "${QUERY_METADATA}"
  --document-metadata "${DOCUMENT_METADATA}"
  --output "${MODEL_PAIRS}"
  --negatives-per-query "${MODEL_NEGATIVES}"
  --num-beams "${NUM_BEAMS}"
  --batch-size "${MODEL_BATCH_SIZE}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-target-length "${MAX_TARGET_LENGTH}"
  --device "${MINING_DEVICE}"
)
if [[ -n "${MINE_LIMIT_QUERIES}" ]]; then
  MODEL_ARGS+=(--limit-queries "${MINE_LIMIT_QUERIES}")
fi
add_precision_flag MODEL_ARGS

echo "=== Stage 2/4: Model-confusion mining ==="
"${DPO_PYTHON_BIN}" "${MODEL_ARGS[@]}"

# ---------------------------------------------------------------------------
# 3. Create the separate hybrid dataset; dpo_pairs.jsonl stays BM25-only
# ---------------------------------------------------------------------------

COMBINE_ARGS=(
  src/scripts/bm25/the_vault/combine_hybrid_dpo.py
  --bm25-input "${BM25_PAIRS}"
  --model-input "${MODEL_PAIRS}"
  --document-metadata "${DOCUMENT_METADATA}"
  --output "${HYBRID_PAIRS}"
  --model-per-query "${MODEL_NEGATIVES}"
  --total-per-query "${TOTAL_NEGATIVES}"
  --seed "${SEED}"
)
if [[ "${REQUIRE_EXACT_MIX}" == "1" ]]; then
  COMBINE_ARGS+=(--require-exact-mix)
fi

echo "=== Stage 3/4: Combining model and BM25 negatives ==="
"${DPO_PYTHON_BIN}" "${COMBINE_ARGS[@]}"
require_file "${HYBRID_PAIRS}"

# ---------------------------------------------------------------------------
# 4. Validate and train DPO
# ---------------------------------------------------------------------------

TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${HYBRID_PAIRS}"
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
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --seed "${SEED}"
  --gradient_checkpointing
)
add_precision_flag TRAIN_ARGS

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  echo "=== Stage 4/4: DPO compatibility dry run ==="
  "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  echo "=== Stage 4/4: DPO training ==="
  if (( NUM_GPUS > 1 )); then
    "${DPO_PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NUM_GPUS}" \
      "${TRAIN_ARGS[@]}"
  else
    "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}"
  fi
else
  echo "RUN_TRAINING=0; mining completed without starting DPO training."
fi

echo "BM25 baseline:     ${BM25_PAIRS}"
echo "Model candidates:  ${MODEL_PAIRS}"
echo "Hybrid DPO data:   ${HYBRID_PAIRS}"
echo "DPO model output:  ${DPO_OUTPUT_DIR}/final"
