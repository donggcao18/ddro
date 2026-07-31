#!/usr/bin/env bash
set -euo pipefail

# Hybrid URL-DocID DPO launcher derived from launch_ddro_training_vault_url.sh.
# Override any path or setting as an environment variable when launching.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000}"
TRAIN_FILE="${TRAIN_FILE:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25/dpo_pairs_hybrid_url.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-url-hybrid}"
NUM_GPUS="${NUM_GPUS:-1}"
PRECISION="${PRECISION:-bf16}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-96}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Hybrid URL DPO file does not exist: ${TRAIN_FILE}" >&2
  echo "Create it from dpo_pairs_hybrid.jsonl with postprocess_dpo_urls.py first." >&2
  exit 2
fi

TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --validation_split 0.01
  --max_prompt_length 256
  --max_target_length "${MAX_TARGET_LENGTH}"
  --num_train_epochs 2
  --per_device_train_batch_size 32
  --per_device_eval_batch_size 32
  --gradient_accumulation_steps 1
  --learning_rate 1e-6
  --beta 0.4
  --eval_steps 2000
  --save_steps 2000
  --gradient_checkpointing
)

case "${PRECISION}" in
  bf16) TRAIN_ARGS+=(--bf16) ;;
  fp16) TRAIN_ARGS+=(--fp16) ;;
  fp32) ;;
  *) echo "PRECISION must be bf16, fp16, or fp32" >&2; exit 2 ;;
esac

if (( NUM_GPUS > 1 )); then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${TRAIN_ARGS[@]}"
else
  python "${TRAIN_ARGS[@]}"
fi
