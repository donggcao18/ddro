#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="."
if [[ "${BASH_SOURCE[0]}" == */* ]]; then
  SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
fi
SCRIPT_DIR="$(cd -- "${SCRIPT_DIR}" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Honor GPU visibility supplied by the scheduler or calling shell.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"
export PYTHONUNBUFFERED=1
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
for value in "${NUM_GPUS}" "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS, TRAIN_BATCH_SIZE, and GRADIENT_ACCUMULATION_STEPS must be positive integers." >&2
    exit 2
  fi
done
if [[ "${CUDA_VISIBLE_DEVICES}" == "-1" ]] || (( NUM_GPUS != ${#GPU_IDS[@]} )); then
  echo "Set CUDA_VISIBLE_DEVICES to exactly NUM_GPUS assigned GPUs (for example 0,1)." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PREFERENCE_OBJECTIVE="${PREFERENCE_OBJECTIVE:-tdpo2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/vault_bm25/dpo_pairs_url.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vault-url-${PREFERENCE_OBJECTIVE}-full}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
# The persistent fallback is not assumed to be local SSD. Override for faster I/O.
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${HF_DATASETS_CACHE:-${REPO_ROOT}/data/hf_cache_w1000}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/tdpo-training}"

TRAIN_ARGS=(
  "${REPO_ROOT}/src/pretrain/train_ddro_vault.py"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_cache_dir "${DATASET_CACHE_DIR}"
  --tokenization_writer_batch_size "${TOKENIZATION_WRITER_BATCH_SIZE:-1000}"
  --ddp_timeout "${DDP_TIMEOUT:-7200}"
  --startup_debug "${STARTUP_DEBUG:-0}"
  --preference_objective "${PREFERENCE_OBJECTIVE}"
  --tdpo_alpha "${TDPO_ALPHA:-0.5}"
  --beta "${BETA:-0.4}"
  --max_prompt_length "${MAX_PROMPT_LENGTH:-256}"
  --max_target_length "${MAX_TARGET_LENGTH:-128}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}"
  --validation_split "${VALIDATION_SPLIT:-0}"
  --logging_steps "${LOGGING_STEPS:-10}"
  --eval_steps "${EVAL_STEPS:-2000}"
  --save_steps "${SAVE_STEPS:-2000}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --dataset_num_proc "${DATASET_NUM_PROC:-1}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}"
)

case "${RESUME_FROM_CHECKPOINT}" in
  latest) TRAIN_ARGS+=(--resume_from_checkpoint) ;;
  none) ;;
  *) TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}") ;;
esac
case "${PRECISION:-bf16}" in
  bf16) TRAIN_ARGS+=(--bf16) ;;
  fp16) TRAIN_ARGS+=(--fp16) ;;
  fp32) ;;
  *) echo "PRECISION must be bf16, fp16, or fp32." >&2; exit 2 ;;
esac
case "${GRADIENT_CHECKPOINTING:-1}" in
  1) TRAIN_ARGS+=(--gradient_checkpointing) ;;
  0) TRAIN_ARGS+=(--no-gradient_checkpointing) ;;
  *) echo "GRADIENT_CHECKPOINTING must be 0 or 1." >&2; exit 2 ;;
esac

COMMAND=("${PYTHON_BIN}" -u)
if (( NUM_GPUS > 1 )); then
  COMMAND+=(-m torch.distributed.run --standalone "--nproc_per_node=${NUM_GPUS}"
    --log-dir "${LOG_DIR}" --tee 3)
fi
COMMAND+=("${TRAIN_ARGS[@]}" "$@")

echo "GPUs: ${CUDA_VISIBLE_DEVICES}; effective batch size: $(( NUM_GPUS * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))"
echo "Resume: ${RESUME_FROM_CHECKPOINT}; dataset cache: ${DATASET_CACHE_DIR}"
echo "For fast cache reads, set DATASET_CACHE_DIR to sufficiently large node-local disk shared by this node's ranks."
printf 'Command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ ! -d "${CHECKPOINT_PATH}" || ! -f "${TRAIN_FILE}" ]]; then
  echo "SFT checkpoint or training file not found; check CHECKPOINT_PATH and TRAIN_FILE." >&2
  exit 2
fi
if [[ "${RESUME_FROM_CHECKPOINT}" == "latest" ]]; then
  shopt -s nullglob
  CHECKPOINTS=("${OUTPUT_DIR}"/checkpoint-*/trainer_state.json)
  if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "No training checkpoint found in ${OUTPUT_DIR}. Use RESUME_FROM_CHECKPOINT=none only for a new run." >&2
    exit 2
  fi
fi
mkdir -p -- "${DATASET_CACHE_DIR}"
exec "${COMMAND[@]}"
