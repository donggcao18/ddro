#!/usr/bin/env bash
set -euo pipefail

# End-to-end structure_id_v3 workflow:
#   1. join structure targets by numeric_id while preparing BM25 metadata;
#   2. retrieve with whitespace-safe legacy text IDs and emit structure targets;
#   3. mine constrained structure_id_v3 model confusions;
#   4. combine model and BM25 negatives and train DPO.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start}"
TRAIN_ORIGINAL="${TRAIN_ORIGINAL:-${DATA_ROOT}/Ruby_train_r32.0.json}"
TEST_ORIGINAL="${TEST_ORIGINAL:-${DATA_ROOT}/Ruby_test_r32.0.json}"
AUGMENTATION="${AUGMENTATION:-${DATA_ROOT}/Ruby_ready_to_feed_numeric.jsonl}"
STRUCTURE_ID_SOURCE="${STRUCTURE_ID_SOURCE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/augmented_dsi/Ruby_merged.jsonl}"
WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25_structure_v3}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-structure-v3}"

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
MODEL_NEGATIVES="${MODEL_NEGATIVES:-4}"
TOTAL_NEGATIVES="${TOTAL_NEGATIVES:-8}"
NUM_BEAMS="${NUM_BEAMS:-8}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
TARGET_COLLISION_POLICY="${TARGET_COLLISION_POLICY:-error}"
TARGET_LENGTH_POLICY="${TARGET_LENGTH_POLICY:-error}"
SEED="${SEED:-42}"

HYBRID_PAIRS="${WORK_DIR}/dpo_pairs_hybrid_structure_id_v3.jsonl"

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

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to an SFT checkpoint trained on structure_id_v3." >&2
  exit 2
fi
require_directory "${CHECKPOINT_PATH}"
require_file "${STRUCTURE_ID_SOURCE}"
if [[ "${RUN_BM25}" == "1" ]]; then
  require_file "${TRAIN_ORIGINAL}"
  require_file "${AUGMENTATION}"
  if [[ -n "${TEST_ORIGINAL}" ]]; then require_file "${TEST_ORIGINAL}"; fi
fi
mkdir -p "${WORK_DIR}" "${DPO_OUTPUT_DIR}"

MINING_ARGS=(
  src/scripts/bm25/the_vault/run_hybrid_pipeline.py
  --train-original "${TRAIN_ORIGINAL}"
  --augmentation "${AUGMENTATION}"
  --structure-id-source "${STRUCTURE_ID_SOURCE}"
  --work-dir "${WORK_DIR}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --target-type structure_id_v3
  --target-collision-policy "${TARGET_COLLISION_POLICY}"
  --target-length-policy "${TARGET_LENGTH_POLICY}"
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
  --seed "${SEED}"
)
if [[ -n "${TEST_ORIGINAL}" ]]; then MINING_ARGS+=(--test-original "${TEST_ORIGINAL}"); fi
if [[ "${RUN_BM25}" == "0" ]]; then MINING_ARGS+=(--reuse-bm25-output); fi
if [[ "${REUSE_INDEX}" == "1" ]]; then MINING_ARGS+=(--reuse-index); fi
add_precision_flag MINING_ARGS

"${DPO_PYTHON_BIN}" "${MINING_ARGS[@]}"
require_file "${HYBRID_PAIRS}"

TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${HYBRID_PAIRS}"
  --output_dir "${DPO_OUTPUT_DIR}"
  --validation_split 0.01
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --num_train_epochs 2
  --per_device_train_batch_size 32
  --per_device_eval_batch_size 32
  --gradient_accumulation_steps 1
  --learning_rate 1e-6
  --beta 0.4
  --eval_steps 2000
  --save_steps 2000
  --seed "${SEED}"
  --gradient_checkpointing
)
add_precision_flag TRAIN_ARGS

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

echo "BM25 structure pairs: ${WORK_DIR}/dpo_pairs_structure_id_v3.jsonl"
echo "Hybrid DPO pairs:     ${HYBRID_PAIRS}"
echo "Final DPO model:      ${DPO_OUTPUT_DIR}/final"
