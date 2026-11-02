#!/usr/bin/env bash
set -euo pipefail

# Native URL data -> GRPO -> multi-positive test inference. No metadata joins.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
TARGET_TYPE=url
WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url}"
TRAIN_INPUT_FILE="${TRAIN_INPUT_FILE:-${WORK_DIR}/Ruby_merged_multilabel_dpo.jsonl}"
TEST_INPUT_FILE="${TEST_INPUT_FILE:-${WORK_DIR}/Ruby_test_r32.0_multilabel.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr_sft_3_5/checkpoint-96600}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-ruby-grpo-url}"
GRPO_DATA_DIR="${GRPO_DATA_DIR:-${OUTPUT_DIR}/grpo_data}"
GRPO_PYTHON_BIN="${GRPO_PYTHON_BIN:-${PYTHON_BIN:-python}}"
NUM_GPUS="${NUM_GPUS:-4}"
PRECISION="${PRECISION:-fp16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
# Exclude unusable full IDs rather than truncating or changing their identity.
# This policy is inherited by calibration, training, and final inference.
GRPO_INVALID_TARGET_POLICY="${GRPO_INVALID_TARGET_POLICY:-skip}"
if [[ "${GRPO_INVALID_TARGET_POLICY}" != skip && "${GRPO_INVALID_TARGET_POLICY}" != error ]]; then
  echo "GRPO_INVALID_TARGET_POLICY must be skip or error." >&2
  exit 2
fi
export GRPO_INVALID_TARGET_POLICY
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
# Deterministic beam candidates by default; DO_SAMPLE=true restores sampled GRPO.
DO_SAMPLE="${DO_SAMPLE:-false}"
# Start at the DDRO query batch, then calibrate capacity on the allocated GPUs.
# 16 queries * 8 rollouts = 128 trajectories per GPU, not 16 DPO pairs.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
AUTO_TUNE_BATCH_SIZE="${AUTO_TUNE_BATCH_SIZE:-1}"
MAX_TRAIN_BATCH_SIZE="${MAX_TRAIN_BATCH_SIZE:-32}"
# Extra headroom for runtime/DDP variation with the frozen KL reference enabled.
BATCH_SIZE_RESERVE_GIB="${BATCH_SIZE_RESERVE_GIB:-4}"
# Evaluation expands each query into EVAL_NUM_BEAMS beams.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
# With batch 16, four GPUs give 64 queries / 512 rollouts without accumulation.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
# Conservative starting point for policy updates; tune after a pilot.
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
# Regularize toward the frozen SFT checkpoint; GRPO_BETA=0 disables the reference.
GRPO_BETA="${GRPO_BETA:-0.04}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SEED="${SEED:-42}"
URL_RUN_PREPARE="${RUN_PREPARE:-1}"
# Automatically evaluate after training; explicitly enable for inference-only.
RUN_INFERENCE="${RUN_INFERENCE:-${RUN_TRAINING:-1}}"
INFERENCE_MODEL_PATH="${INFERENCE_MODEL_PATH:-${OUTPUT_DIR}/final}"
INFERENCE_OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:-${OUTPUT_DIR}/inference}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INFERENCE_NUM_BEAMS="${INFERENCE_NUM_BEAMS:-10}"
INFERENCE_NUM_GPUS="${INFERENCE_NUM_GPUS:-${NUM_GPUS}}"
for name in URL_RUN_PREPARE RUN_INFERENCE; do
  if [[ "${!name}" != 0 && "${!name}" != 1 ]]; then
    echo "${name} must be 0 or 1." >&2
    exit 2
  fi
done
if [[ "${RUN_INFERENCE}" == 1 ]]; then
  if [[ ! -s "${TEST_INPUT_FILE}" ]]; then
    echo "Test input is missing or empty: ${TEST_INPUT_FILE}" >&2
    exit 2
  fi
  for name in INFERENCE_BATCH_SIZE INFERENCE_NUM_BEAMS INFERENCE_NUM_GPUS; do
    if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
      echo "${name} must be a positive integer." >&2
      exit 2
    fi
  done
  if (( INFERENCE_NUM_BEAMS < 10 )); then
    echo "INFERENCE_NUM_BEAMS must be at least 10 for Hit/MRR@10." >&2
    exit 2
  fi
fi
if [[ "${URL_RUN_PREPARE}" == 1 ]]; then
  if [[ ! -s "${TRAIN_INPUT_FILE}" ]]; then
    echo "Training input is missing or empty: ${TRAIN_INPUT_FILE}" >&2
    exit 2
  fi
  echo "=== Preparing URL GRPO directly from ${TRAIN_INPUT_FILE} ==="
  "${GRPO_PYTHON_BIN}" src/scripts/grpo/prepare_vault_url_grpo_data.py \
    --input-file "${TRAIN_INPUT_FILE}" --output-dir "${GRPO_DATA_DIR}" \
    --validation-split "${VALIDATION_SPLIT}" --seed "${SEED}"
fi
# Reuse the training/calibration flow, bypassing its metadata preparation stage.
RUN_PREPARE=0
source "${SCRIPT_DIR}/train_vault_grpo_4v100.sh"

if [[ "${RUN_INFERENCE}" == 1 ]]; then
  require_file "${CORPUS_FILE}"
  if [[ ! -d "${INFERENCE_MODEL_PATH}" ]]; then
    echo "Inference model directory does not exist: ${INFERENCE_MODEL_PATH}" >&2
    exit 2
  fi
  AVAILABLE_INFERENCE_GPUS="$("${GRPO_PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
  if (( AVAILABLE_INFERENCE_GPUS < INFERENCE_NUM_GPUS )); then
    echo "Need ${INFERENCE_NUM_GPUS} inference GPUs; found ${AVAILABLE_INFERENCE_GPUS}." >&2
    exit 2
  fi
  INFERENCE_ARGS=(
    src/pretrain/infer_grpo_vault_url.py
    --model_path "${INFERENCE_MODEL_PATH}" --test_file "${TEST_INPUT_FILE}"
    --corpus_file "${CORPUS_FILE}" --output_dir "${INFERENCE_OUTPUT_DIR}"
    --batch_size "${INFERENCE_BATCH_SIZE}" --num_beams "${INFERENCE_NUM_BEAMS}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}" --max_target_length "${MAX_TARGET_LENGTH}"
  )
  if [[ "${PRECISION}" == fp16 ]]; then INFERENCE_ARGS+=(--fp16); fi
  echo "=== URL test inference: Hit@1/5/10 and MRR@1/5/10 ==="
  echo "Model: ${INFERENCE_MODEL_PATH}; test data: ${TEST_INPUT_FILE}"
  if (( INFERENCE_NUM_GPUS > 1 )); then
    "${GRPO_PYTHON_BIN}" -m torch.distributed.run --standalone \
      --nproc_per_node="${INFERENCE_NUM_GPUS}" "${INFERENCE_ARGS[@]}"
  else
    "${GRPO_PYTHON_BIN}" "${INFERENCE_ARGS[@]}"
  fi
  echo "Inference metrics: ${INFERENCE_OUTPUT_DIR}/metrics.json"
  echo "Per-query predictions: ${INFERENCE_OUTPUT_DIR}/predictions.jsonl"
fi
