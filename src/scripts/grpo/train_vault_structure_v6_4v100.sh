#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Existing metadata can be reused; neither BM25 rankings nor DPO pairs are read.
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to your local structure-v6 SFT checkpoint}"
: "${MAX_TARGET_LENGTH:?Set MAX_TARGET_LENGTH to the SFT decoder token limit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-4}"
PRECISION="${PRECISION:-fp16}"
WORK_DIR="${WORK_DIR:-data/vault_bm25_structure_v6}"
QUERY_METADATA="${QUERY_METADATA:-${WORK_DIR}/query_metadata.jsonl}"
DOCUMENT_METADATA="${DOCUMENT_METADATA:-${WORK_DIR}/document_metadata.jsonl}"
GRPO_DATA_DIR="${GRPO_DATA_DIR:-${WORK_DIR}/grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vault-ruby-grpo-structure-v6}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
SEED="${SEED:-42}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"

if [[ "${PRECISION}" != fp16 && "${PRECISION}" != fp32 ]]; then
  echo "V100 requires PRECISION=fp16 or fp32." >&2
  exit 2
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer." >&2
  exit 2
fi
AVAILABLE_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if (( AVAILABLE_GPUS < NUM_GPUS )); then
  echo "Need ${NUM_GPUS} visible GPUs; found ${AVAILABLE_GPUS}." >&2
  exit 2
fi

if [[ "${RUN_PREPARE}" == 1 ]]; then
  "${PYTHON_BIN}" src/scripts/grpo/prepare_vault_grpo_data.py \
    --query-metadata "${QUERY_METADATA}" --document-metadata "${DOCUMENT_METADATA}" \
    --target-type structure_id_v6 --output-dir "${GRPO_DATA_DIR}" \
    --validation-split "${VALIDATION_SPLIT}" --seed "${SEED}"
fi

TRAIN_ARGS=(
  src/pretrain/train_grpo_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${GRPO_DATA_DIR}/train.jsonl"
  --corpus_file "${GRPO_DATA_DIR}/corpus.jsonl"
  --output_dir "${OUTPUT_DIR}"
  --validation_split 0
  --max_prompt_length "${MAX_PROMPT_LENGTH:-256}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --num_generations "${NUM_GENERATIONS:-8}"
  --temperature "${TEMPERATURE:-1.0}"
  --epsilon "${GRPO_EPSILON:-0.2}"
  --beta "${GRPO_BETA:-0.0}"
  --eval_num_beams "${EVAL_NUM_BEAMS:-10}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-1}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-2}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
  --max_steps "${MAX_STEPS:--1}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --eval_steps "${EVAL_STEPS:-500}" --save_steps "${SAVE_STEPS:-500}"
  --logging_steps "${LOGGING_STEPS:-10}" --seed "${SEED}"
  --report_to "${REPORT_TO:-none}"
  --gradient_checkpointing
)
if [[ -s "${GRPO_DATA_DIR}/validation.jsonl" ]]; then
  TRAIN_ARGS+=(--eval_file "${GRPO_DATA_DIR}/validation.jsonl")
fi
if [[ "${PRECISION}" == fp16 ]]; then TRAIN_ARGS+=(--fp16); fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${RUN_DRY_RUN}" == 1 ]]; then
  "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi
if [[ "${RUN_TRAINING}" == 1 ]]; then
  if (( NUM_GPUS > 1 )); then
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
      --nproc_per_node="${NUM_GPUS}" "${TRAIN_ARGS[@]}"
  else
    "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
  fi
fi
