#!/usr/bin/env bash
set -euo pipefail

# Override any of these paths/settings as environment variables when launching.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000}"
TRAIN_FILE="${TRAIN_FILE:-/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25/dpo_pairs_url.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-url}"
NUM_GPUS="${NUM_GPUS:-1}"
PRECISION="${PRECISION:-bf16}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --validation_split 0.01
  --max_prompt_length 256
  --max_target_length 64
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
