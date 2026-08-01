#!/usr/bin/env bash
set -euo pipefail

# Hybrid URL-DocID DPO launcher derived from launch_ddro_training_vault_url.sh.
# Override any path or setting as an environment variable when launching.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/beegfs/scratch/congthanh_le/east/ddro/data/vault_bm25/dpo_pairs_hybrid_url.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-url-hybrid}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-1}"
PRECISION="${PRECISION:-bf16}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-64}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
DPO_BETA="${DPO_BETA:-0.4}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SEED="${SEED:-42}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Hybrid URL DPO file does not exist: ${TRAIN_FILE}" >&2
  echo "Create it with prepare_vault_url_hybrid_negatives.sh first." >&2
  exit 2
fi
if [[ ! -s "${TRAIN_FILE}" ]]; then
  echo "Hybrid URL DPO file is empty: ${TRAIN_FILE}" >&2
  exit 2
fi

TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${OUTPUT_DIR}"
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

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

case "${PRECISION}" in
  bf16) TRAIN_ARGS+=(--bf16) ;;
  fp16) TRAIN_ARGS+=(--fp16) ;;
  fp32) ;;
  *) echo "PRECISION must be bf16, fp16, or fp32" >&2; exit 2 ;;
esac

echo "=== Vault URL hybrid DPO training ==="
echo "Checkpoint:       ${CHECKPOINT_PATH}"
echo "Training file:    ${TRAIN_FILE}"
echo "Training rows:    $(wc -l < "${TRAIN_FILE}")"
echo "Output directory: ${OUTPUT_DIR}"
echo "GPUs:             ${NUM_GPUS}"
echo "Precision:        ${PRECISION}"
echo "Target length:    ${MAX_TARGET_LENGTH}"
echo "Batch/GPU:        ${TRAIN_BATCH_SIZE}"
echo "Grad accumulation:${GRADIENT_ACCUMULATION_STEPS}"

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  echo "=== Preflight ==="
  "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi

if [[ "${RUN_TRAINING}" != "1" ]]; then
  echo "RUN_TRAINING=0; preflight completed without starting training."
  exit 0
fi

echo "=== Training ==="
if (( NUM_GPUS > 1 )); then
  "${DPO_PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    "${TRAIN_ARGS[@]}"
else
  "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}"
fi

echo "Final model: ${OUTPUT_DIR}/final"
