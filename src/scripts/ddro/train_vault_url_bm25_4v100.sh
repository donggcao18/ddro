#!/usr/bin/env bash
set -euo pipefail

# Train URL-target DPO from BM25 preference pairs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25}"
BM25_PAIRS="${TRAIN_FILE:-${WORK_DIR}/dpo_pairs_url.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr/checkpoint-161800}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-url-bm25}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"

NUM_GPUS="${NUM_GPUS:-4}"
PRECISION="${PRECISION:-fp16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
# Use the decoder target limit from the URL SFT checkpoint.
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
DPO_BETA="${DPO_BETA:-0.4}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SEED="${SEED:-42}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${HF_DATASETS_CACHE:-}}"
TOKENIZATION_WRITER_BATCH_SIZE="${TOKENIZATION_WRITER_BATCH_SIZE:-1000}"
DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
STARTUP_DEBUG="${STARTUP_DEBUG:-0}"

if [[ -z "${CHECKPOINT_PATH}" || ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to an existing URL SFT checkpoint." >&2
  exit 2
fi
if [[ -z "${MAX_TARGET_LENGTH}" ]]; then
  echo "Set MAX_TARGET_LENGTH to the decoder limit used for URL SFT." >&2
  exit 2
fi
if [[ "${PRECISION}" != "fp16" && "${PRECISION}" != "fp32" ]]; then
  echo "V100 requires PRECISION=fp16 or fp32; bf16 is unsupported." >&2
  exit 2
fi
if [[ ! -s "${BM25_PAIRS}" ]]; then
  echo "BM25 DPO pair file is missing or empty: ${BM25_PAIRS}" >&2
  exit 2
fi
if [[ "${RUN_TRAINING}" == "1" ]]; then
  AVAILABLE_GPUS="$("${DPO_PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
  if (( AVAILABLE_GPUS < NUM_GPUS )); then
    echo "Need ${NUM_GPUS} visible CUDA GPUs; ${DPO_PYTHON_BIN} sees ${AVAILABLE_GPUS}." >&2
    exit 2
  fi
fi
mkdir -p "${OUTPUT_DIR}"

TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${BM25_PAIRS}"
  --output_dir "${OUTPUT_DIR}"
  --validation_split "${VALIDATION_SPLIT}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --tokenization_writer_batch_size "${TOKENIZATION_WRITER_BATCH_SIZE}"
  --ddp_timeout "${DDP_TIMEOUT}"
  --startup_debug "${STARTUP_DEBUG}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --learning_rate "${LEARNING_RATE}"
  --beta "${DPO_BETA}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --seed "${SEED}"
  --gradient_checkpointing
)
if [[ "${PRECISION}" == "fp16" ]]; then TRAIN_ARGS+=(--fp16); fi
if [[ -n "${DATASET_CACHE_DIR}" ]]; then
  TRAIN_ARGS+=(--dataset_cache_dir "${DATASET_CACHE_DIR}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
TRAIN_ARGS+=("$@")

echo "=== Vault URL BM25-only DPO ==="
echo "Pairs: ${BM25_PAIRS} ($(wc -l < "${BM25_PAIRS}") rows)"
echo "GPUs: ${NUM_GPUS}; precision: ${PRECISION}; target length: ${MAX_TARGET_LENGTH}"
echo "Batch/GPU: ${TRAIN_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi
if [[ "${RUN_TRAINING}" == "1" ]]; then
  if (( NUM_GPUS > 1 )); then
    DISTRIBUTED_ARGS=(-m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}")
    if [[ "${STARTUP_DEBUG}" != "0" ]]; then
      DISTRIBUTED_ARGS+=(--log-dir "${OUTPUT_DIR}/startup-logs" --tee 3)
    fi
    "${DPO_PYTHON_BIN}" -u "${DISTRIBUTED_ARGS[@]}" "${TRAIN_ARGS[@]}"
  else
    "${DPO_PYTHON_BIN}" "${TRAIN_ARGS[@]}"
  fi
fi
echo "Final model: ${OUTPUT_DIR}/final"
