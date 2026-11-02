#!/usr/bin/env bash
set -euo pipefail

# Shared preparation and launch flow for the URL and structure-v6 presets.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
TARGET_TYPE="${TARGET_TYPE:-structure_id_v6}"
WORK_DIR="${WORK_DIR:-data/vault_bm25_structure_v6}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vault-grpo-${TARGET_TYPE}}"
GRPO_PYTHON_BIN="${GRPO_PYTHON_BIN:-${PYTHON_BIN:-python}}"
QUERY_METADATA="${QUERY_METADATA:-${WORK_DIR}/query_metadata.jsonl}"
DOCUMENT_METADATA="${DOCUMENT_METADATA:-${WORK_DIR}/document_metadata.jsonl}"
GRPO_DATA_DIR="${GRPO_DATA_DIR:-${WORK_DIR}/grpo}"
TRAIN_FILE="${TRAIN_FILE:-${GRPO_DATA_DIR}/train.jsonl}"
CORPUS_FILE="${CORPUS_FILE:-${GRPO_DATA_DIR}/corpus.jsonl}"
# Explicitly empty EVAL_FILE disables validation. Preparation with split=0
# writes an empty validation file, which also disables validation.
EVAL_FILE="${EVAL_FILE-${GRPO_DATA_DIR}/validation.jsonl}"
NUM_GPUS="${NUM_GPUS:-4}"
PRECISION="${PRECISION:-fp16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
AUTO_TUNE_BATCH_SIZE="${AUTO_TUNE_BATCH_SIZE:-0}"
MAX_TRAIN_BATCH_SIZE="${MAX_TRAIN_BATCH_SIZE:-32}"
BATCH_SIZE_RESERVE_GIB="${BATCH_SIZE_RESERVE_GIB:-2}"
BATCH_SIZE_REPORT="${BATCH_SIZE_REPORT:-${OUTPUT_DIR}/batch_size_probe.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
GRPO_EPSILON="${GRPO_EPSILON:-0.2}"
GRPO_BETA="${GRPO_BETA:-0.0}"
EVAL_NUM_BEAMS="${EVAL_NUM_BEAMS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MAX_STEPS="${MAX_STEPS:--1}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-none}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Required file is missing or empty: $1" >&2
    exit 2
  fi
}
for name in NUM_GPUS MAX_PROMPT_LENGTH MAX_TARGET_LENGTH NUM_GENERATIONS TRAIN_BATCH_SIZE MAX_TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS EVAL_NUM_BEAMS EVAL_STEPS SAVE_STEPS LOGGING_STEPS SAVE_TOTAL_LIMIT; do
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer; received '${!name}'." >&2
    exit 2
  fi
done
if (( NUM_GENERATIONS < 2 )); then
  echo "NUM_GENERATIONS must be at least 2 for group-relative advantages." >&2
  exit 2
fi
for name in RUN_PREPARE RUN_DRY_RUN RUN_TRAINING AUTO_TUNE_BATCH_SIZE; do
  if [[ "${!name}" != 0 && "${!name}" != 1 ]]; then
    echo "${name} must be 0 or 1." >&2
    exit 2
  fi
done
if [[ "${PRECISION}" != fp16 && "${PRECISION}" != fp32 ]]; then
  echo "V100 requires PRECISION=fp16 or fp32; bf16 is unsupported." >&2
  exit 2
fi
if [[ "${RUN_DRY_RUN}" == 1 || "${RUN_TRAINING}" == 1 ]]; then
  if [[ -z "${CHECKPOINT_PATH}" || ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "Set CHECKPOINT_PATH to an existing ${TARGET_TYPE} SFT checkpoint: ${CHECKPOINT_PATH}" >&2
    exit 2
  fi
  REQUIRED_GPUS=1
  if [[ "${RUN_TRAINING}" == 1 ]]; then REQUIRED_GPUS="${NUM_GPUS}"; fi
  AVAILABLE_GPUS="$("${GRPO_PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
  if (( AVAILABLE_GPUS < REQUIRED_GPUS )); then
    echo "Need ${REQUIRED_GPUS} visible CUDA GPU(s); ${GRPO_PYTHON_BIN} sees ${AVAILABLE_GPUS}." >&2
    exit 2
  fi
fi
if [[ "${RUN_PREPARE}" == 1 ]]; then
  require_file "${QUERY_METADATA}"
  require_file "${DOCUMENT_METADATA}"
  echo "=== Preparing ${TARGET_TYPE} GRPO data from full-corpus metadata ==="
  "${GRPO_PYTHON_BIN}" src/scripts/grpo/prepare_vault_grpo_data.py \
    --query-metadata "${QUERY_METADATA}" --document-metadata "${DOCUMENT_METADATA}" \
    --target-type "${TARGET_TYPE}" --output-dir "${GRPO_DATA_DIR}" \
    --validation-split "${VALIDATION_SPLIT}" --seed "${SEED}"
fi
if [[ "${RUN_DRY_RUN}" == 1 || "${RUN_TRAINING}" == 1 ]]; then
  require_file "${TRAIN_FILE}"
  require_file "${CORPUS_FILE}"
  if [[ -n "${EVAL_FILE}" && ! -f "${EVAL_FILE}" ]]; then
    echo "Validation file is missing: ${EVAL_FILE}. Set EVAL_FILE= to disable validation explicitly." >&2
    exit 2
  fi
  if [[ -n "${EVAL_FILE}" && -s "${EVAL_FILE}" ]] && (( SAVE_STEPS % EVAL_STEPS != 0 )); then
    echo "SAVE_STEPS must be a multiple of EVAL_STEPS when validation is enabled." >&2
    exit 2
  fi
  if [[ -n "${RESUME_FROM_CHECKPOINT}" && ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "GRPO resume checkpoint directory does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${AUTO_TUNE_BATCH_SIZE}" == 1 ]]; then
    echo "For resume, set AUTO_TUNE_BATCH_SIZE=0 and TRAIN_BATCH_SIZE to the previous run's recorded batch to preserve its data-loader position." >&2
    exit 2
  fi
  mkdir -p "${OUTPUT_DIR}"
  if [[ "${RUN_TRAINING}" == 1 && "${AUTO_TUNE_BATCH_SIZE}" == 1 ]]; then
    CALIBRATION_ARGS=(
      src/pretrain/calibrate_grpo_batch_size.py
      --checkpoint_path "${CHECKPOINT_PATH}" --corpus_file "${CORPUS_FILE}"
      --output_file "${BATCH_SIZE_REPORT}" --num_gpus "${NUM_GPUS}"
      --num_generations "${NUM_GENERATIONS}"
      --max_prompt_length "${MAX_PROMPT_LENGTH}" --max_target_length "${MAX_TARGET_LENGTH}"
      --start_batch_size "${TRAIN_BATCH_SIZE}" --max_batch_size "${MAX_TRAIN_BATCH_SIZE}"
      --reserve_gib "${BATCH_SIZE_RESERVE_GIB}" --temperature "${TEMPERATURE}" --beta "${GRPO_BETA}"
    )
    if [[ "${PRECISION}" == fp16 ]]; then CALIBRATION_ARGS+=(--fp16); fi
    echo "=== Calibrating the largest GRPO query batch up to ${MAX_TRAIN_BATCH_SIZE}/GPU ==="
    "${GRPO_PYTHON_BIN}" "${CALIBRATION_ARGS[@]}"
    TRAIN_BATCH_SIZE="$("${GRPO_PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["batch_size"])' "${BATCH_SIZE_REPORT}")"
    if ! [[ "${TRAIN_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( TRAIN_BATCH_SIZE > MAX_TRAIN_BATCH_SIZE )); then
      echo "Calibration returned an invalid batch size: ${TRAIN_BATCH_SIZE}" >&2
      exit 2
    fi
  fi
  TRAIN_ARGS=(
    src/pretrain/train_grpo_vault.py
    --checkpoint_path "${CHECKPOINT_PATH}"
    --train_file "${TRAIN_FILE}" --corpus_file "${CORPUS_FILE}"
    --output_dir "${OUTPUT_DIR}" --validation_split 0
    --max_prompt_length "${MAX_PROMPT_LENGTH}" --max_target_length "${MAX_TARGET_LENGTH}"
    --num_generations "${NUM_GENERATIONS}" --temperature "${TEMPERATURE}"
    --epsilon "${GRPO_EPSILON}" --beta "${GRPO_BETA}" --eval_num_beams "${EVAL_NUM_BEAMS}"
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" --max_steps "${MAX_STEPS}"
    --learning_rate "${LEARNING_RATE}" --warmup_ratio "${WARMUP_RATIO}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --eval_steps "${EVAL_STEPS}" --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}" --logging_steps "${LOGGING_STEPS}"
    --seed "${SEED}" --report_to "${REPORT_TO}" --gradient_checkpointing
  )
  if [[ -n "${EVAL_FILE}" && -s "${EVAL_FILE}" ]]; then
    TRAIN_ARGS+=(--eval_file "${EVAL_FILE}")
  fi
  if [[ "${PRECISION}" == fp16 ]]; then TRAIN_ARGS+=(--fp16); fi
  if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi
  QUERIES_PER_STEP=$(( NUM_GPUS * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))
  ROLLOUTS_PER_STEP=$(( QUERIES_PER_STEP * NUM_GENERATIONS ))
  echo "=== Vault ${TARGET_TYPE} GRPO ==="
  echo "SFT checkpoint: ${CHECKPOINT_PATH}"
  echo "Train queries: ${TRAIN_FILE}; corpus: ${CORPUS_FILE}"
  echo "GPUs: ${NUM_GPUS}; precision: ${PRECISION}; prompt/target lengths: ${MAX_PROMPT_LENGTH}/${MAX_TARGET_LENGTH}"
  echo "Queries/GPU: ${TRAIN_BATCH_SIZE}; rollouts/query: ${NUM_GENERATIONS}; trajectories/GPU: $(( TRAIN_BATCH_SIZE * NUM_GENERATIONS ))"
  echo "Accumulation: ${GRADIENT_ACCUMULATION_STEPS}; effective batch: ${QUERIES_PER_STEP} queries / ${ROLLOUTS_PER_STEP} rollouts per optimizer step"
  echo "Learning rate: ${LEARNING_RATE}; GRPO KL beta: ${GRPO_BETA}; epochs: ${NUM_TRAIN_EPOCHS}"
  echo "Evaluation/save interval: ${EVAL_STEPS}/${SAVE_STEPS}; output: ${OUTPUT_DIR}"
  if [[ "${RUN_DRY_RUN}" == 1 ]]; then
    "${GRPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
  fi
  if [[ "${RUN_TRAINING}" == 1 ]]; then
    if (( NUM_GPUS > 1 )); then
      "${GRPO_PYTHON_BIN}" -m torch.distributed.run --standalone \
        --nproc_per_node="${NUM_GPUS}" "${TRAIN_ARGS[@]}"
    else
      "${GRPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}"
    fi
    echo "Final model: ${OUTPUT_DIR}/final"
  fi
fi
