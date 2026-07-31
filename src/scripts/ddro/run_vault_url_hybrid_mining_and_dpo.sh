#!/usr/bin/env bash
set -euo pipefail

# End-to-end URL-DocID workflow:
#   1. prepare/retrieve the unchanged BM25 text-ID baseline;
#   2. convert BM25 pairs to URL decoder targets;
#   3. mine constrained URL predictions from the frozen URL SFT checkpoint;
#   4. combine up to 4 URL-model confusions with BM25 fallback;
#   5. validate and train URL-DocID DPO.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Paths
DATA_ROOT="${DATA_ROOT:-/home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start}"
TRAIN_ORIGINAL="${TRAIN_ORIGINAL:-${DATA_ROOT}/Ruby_train_r32.0.json}"
TEST_ORIGINAL="${TEST_ORIGINAL:-${DATA_ROOT}/Ruby_test_r32.0.json}"
AUGMENTATION="${AUGMENTATION:-${DATA_ROOT}/Ruby_ready_to_feed_numeric.jsonl}"
WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-url-hybrid}"

# Use separate interpreters if Pyserini and PyTorch live in different environments.
BM25_PYTHON_BIN="${BM25_PYTHON_BIN:-python}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"

# Mining
RUN_BM25="${RUN_BM25:-1}"
REUSE_INDEX="${REUSE_INDEX:-0}"
BM25_THREADS="${BM25_THREADS:-16}"
BM25_BATCH_SIZE="${BM25_BATCH_SIZE:-16}"
BM25_HITS="${BM25_HITS:-200}"
MODEL_NEGATIVES="${MODEL_NEGATIVES:-4}"
TOTAL_NEGATIVES="${TOTAL_NEGATIVES:-8}"
NUM_BEAMS="${NUM_BEAMS:-8}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-64}"
MINING_DEVICE="${MINING_DEVICE:-auto}"
MINE_LIMIT_QUERIES="${MINE_LIMIT_QUERIES:-}"
REQUIRE_EXACT_MIX="${REQUIRE_EXACT_MIX:-0}"
SEED="${SEED:-42}"

# Training
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
PRECISION="${PRECISION:-bf16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
DPO_BETA="${DPO_BETA:-0.4}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"

HYBRID_URL_PAIRS="${WORK_DIR}/dpo_pairs_hybrid_url.jsonl"


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
if [[ "${RUN_BM25}" == "1" ]]; then
  require_file "${TRAIN_ORIGINAL}"
  require_file "${AUGMENTATION}"
  if [[ -n "${TEST_ORIGINAL}" ]]; then
    require_file "${TEST_ORIGINAL}"
  fi
fi
mkdir -p "${WORK_DIR}" "${DPO_OUTPUT_DIR}"

echo "=== URL-DocID Vault hybrid mining ==="
echo "URL SFT checkpoint: ${CHECKPOINT_PATH}"
echo "Work directory:     ${WORK_DIR}"

MINING_ARGS=(
  src/scripts/bm25/the_vault/run_hybrid_pipeline.py
  --train-original "${TRAIN_ORIGINAL}"
  --augmentation "${AUGMENTATION}"
  --work-dir "${WORK_DIR}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --target-type url
  --bm25-python "${BM25_PYTHON_BIN}"
  --model-python "${DPO_PYTHON_BIN}"
  --threads "${BM25_THREADS}"
  --bm25-batch-size "${BM25_BATCH_SIZE}"
  --hits "${BM25_HITS}"
  --model-negatives-per-query "${MODEL_NEGATIVES}"
  --total-negatives-per-query "${TOTAL_NEGATIVES}"
  --num-beams "${NUM_BEAMS}"
  --model-batch-size "${MODEL_BATCH_SIZE}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-target-length "${MAX_TARGET_LENGTH}"
  --device "${MINING_DEVICE}"
  --seed "${SEED}"
)
if [[ -n "${TEST_ORIGINAL}" ]]; then
  MINING_ARGS+=(--test-original "${TEST_ORIGINAL}")
fi
if [[ "${RUN_BM25}" == "0" ]]; then
  MINING_ARGS+=(--reuse-bm25-output)
fi
if [[ "${REUSE_INDEX}" == "1" ]]; then
  MINING_ARGS+=(--reuse-index)
fi
if [[ -n "${MINE_LIMIT_QUERIES}" ]]; then
  MINING_ARGS+=(--limit-queries "${MINE_LIMIT_QUERIES}")
fi
if [[ "${REQUIRE_EXACT_MIX}" == "1" ]]; then
  MINING_ARGS+=(--require-exact-mix)
fi
add_precision_flag MINING_ARGS

"${DPO_PYTHON_BIN}" "${MINING_ARGS[@]}"
require_file "${HYBRID_URL_PAIRS}"

echo "=== URL-DocID DPO validation and training ==="
TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${HYBRID_URL_PAIRS}"
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
  "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  if (( NUM_GPUS > 1 )); then
    "${DPO_PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NUM_GPUS}" \
      "${TRAIN_ARGS[@]}"
  else
    "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}"
  fi
else
  echo "RUN_TRAINING=0; URL mining completed without DPO training."
fi

echo "BM25 text-ID pairs: ${WORK_DIR}/dpo_pairs.jsonl"
echo "BM25 URL pairs:     ${WORK_DIR}/dpo_pairs_url.jsonl"
echo "Model URL pairs:    ${WORK_DIR}/model_confusion_pairs_url.jsonl"
echo "Hybrid URL pairs:   ${HYBRID_URL_PAIRS}"
echo "Final DPO model:    ${DPO_OUTPUT_DIR}/final"
