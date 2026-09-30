#!/usr/bin/env bash
set -euo pipefail

# Stage 2: use the BM25 files from prepare_vault_structure_v6_bm25.sh,
# mine model confusions, combine hybrid DPO pairs, and train on four V100s.
# Run this script in the conda environment that contains PyTorch/Transformers.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WORK_DIR="${WORK_DIR:-/mnt/beegfs/scratch/congthanh_le/east/ddro/data/vault_bm25_structure_v6}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${DPO_OUTPUT_DIR:-/home/users/congthanh_le/scratch/east/ddro/outputs/vault-ruby-ddro-structure-v6-hybrid}}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"

BM25_PAIRS="${WORK_DIR}/dpo_pairs_structure_id_v6.jsonl"
QUERY_METADATA="${WORK_DIR}/query_metadata.jsonl"
DOCUMENT_METADATA="${WORK_DIR}/document_metadata.jsonl"
MODEL_PAIRS="${WORK_DIR}/model_confusion_pairs_structure_id_v6.jsonl"
COLLISION_EXCLUSIONS="${WORK_DIR}/model_confusion_pairs_structure_id_v6.excluded_text_ids.json"
HYBRID_PAIRS="${WORK_DIR}/dpo_pairs_hybrid_structure_id_v6.jsonl"

RUN_MODEL_MINING="${RUN_MODEL_MINING:-1}"
RUN_COMBINE="${RUN_COMBINE:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
MODEL_NEGATIVES="${MODEL_NEGATIVES:-4}"
TOTAL_NEGATIVES="${TOTAL_NEGATIVES:-8}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-16}"
MINING_DEVICE="${MINING_DEVICE:-auto}"
MINE_LIMIT_QUERIES="${MINE_LIMIT_QUERIES:-}"
TARGET_COLLISION_POLICY="${TARGET_COLLISION_POLICY:-skip}"
TARGET_LENGTH_POLICY="${TARGET_LENGTH_POLICY:-skip}"
SEED="${SEED:-42}"

NUM_GPUS="${NUM_GPUS:-4}"
PRECISION="${PRECISION:-fp16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
# Must match the decoder target limit used by structure_id_v6 SFT.
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
DPO_BETA="${DPO_BETA:-0.4}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 2
  fi
}

if [[ -z "${CHECKPOINT_PATH}" || ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to an existing structure_id_v6 SFT checkpoint." >&2
  exit 2
fi
if [[ -z "${MAX_TARGET_LENGTH}" ]]; then
  echo "Set MAX_TARGET_LENGTH to the decoder limit used for structure_id_v6 SFT." >&2
  exit 2
fi
if [[ "${PRECISION}" != "fp16" && "${PRECISION}" != "fp32" ]]; then
  echo "V100 requires PRECISION=fp16 or fp32; bf16 is unsupported." >&2
  exit 2
fi
for input in "${BM25_PAIRS}" "${QUERY_METADATA}" "${DOCUMENT_METADATA}"; do
  require_file "${input}"
done
if [[ ! -s "${BM25_PAIRS}" ]]; then
  echo "BM25 pair file is empty: ${BM25_PAIRS}" >&2
  exit 2
fi

if [[ "${RUN_MODEL_MINING}" == "1" || "${RUN_TRAINING}" == "1" ]]; then
  AVAILABLE_GPUS="$("${DPO_PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
  REQUIRED_GPUS=1
  if [[ "${RUN_TRAINING}" == "1" ]]; then REQUIRED_GPUS="${NUM_GPUS}"; fi
  if (( AVAILABLE_GPUS < REQUIRED_GPUS )); then
    echo "Need ${REQUIRED_GPUS} visible CUDA GPU(s); ${DPO_PYTHON_BIN} sees ${AVAILABLE_GPUS}." >&2
    exit 2
  fi
fi
mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_MODEL_MINING}" == "1" ]]; then
  MODEL_ARGS=(
    src/scripts/bm25/the_vault/mine_model_confusion_negatives.py
    --target-type structure_id_v6
    --target-collision-policy "${TARGET_COLLISION_POLICY}"
    --target-length-policy "${TARGET_LENGTH_POLICY}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --query-metadata "${QUERY_METADATA}"
    --document-metadata "${DOCUMENT_METADATA}"
    --output "${MODEL_PAIRS}"
    --negatives-per-query "${MODEL_NEGATIVES}"
    --num-beams "${NUM_BEAMS}"
    --batch-size "${MODEL_BATCH_SIZE}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-target-length "${MAX_TARGET_LENGTH}"
    --device "${MINING_DEVICE}"
  )
  if [[ -n "${MINE_LIMIT_QUERIES}" ]]; then
    MODEL_ARGS+=(--limit-queries "${MINE_LIMIT_QUERIES}")
  fi
  if [[ "${PRECISION}" == "fp16" ]]; then MODEL_ARGS+=(--fp16); fi
  echo "=== Stage 2a: structure_id_v6 model-confusion mining ==="
  "${DPO_PYTHON_BIN}" "${MODEL_ARGS[@]}"
fi

if [[ "${RUN_COMBINE}" == "1" ]]; then
  require_file "${MODEL_PAIRS}"
  require_file "${COLLISION_EXCLUSIONS}"
  echo "=== Stage 2b: combining model and BM25 negatives ==="
  "${DPO_PYTHON_BIN}" src/scripts/bm25/the_vault/combine_hybrid_dpo.py \
    --bm25-input "${BM25_PAIRS}" \
    --model-input "${MODEL_PAIRS}" \
    --document-metadata "${DOCUMENT_METADATA}" \
    --excluded-text-ids "${COLLISION_EXCLUSIONS}" \
    --output "${HYBRID_PAIRS}" \
    --model-per-query "${MODEL_NEGATIVES}" \
    --total-per-query "${TOTAL_NEGATIVES}" \
    --seed "${SEED}"
fi

require_file "${HYBRID_PAIRS}"
if [[ ! -s "${HYBRID_PAIRS}" ]]; then
  echo "Hybrid DPO file is empty: ${HYBRID_PAIRS}" >&2
  exit 2
fi
TRAIN_ARGS=(
  src/pretrain/train_ddro_vault.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_file "${HYBRID_PAIRS}"
  --output_dir "${OUTPUT_DIR}"
  --validation_split "${VALIDATION_SPLIT}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --learning_rate "${LEARNING_RATE}"
  --beta "${DPO_BETA}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --seed "${SEED}"
  --gradient_checkpointing
)
if [[ "${PRECISION}" == "fp16" ]]; then TRAIN_ARGS+=(--fp16); fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo "=== Stage 2c: structure_id_v6 DPO ==="
echo "Hybrid pairs: ${HYBRID_PAIRS} ($(wc -l < "${HYBRID_PAIRS}") rows)"
echo "GPUs: ${NUM_GPUS}; precision: ${PRECISION}; target length: ${MAX_TARGET_LENGTH}"
echo "Batch/GPU: ${TRAIN_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
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
echo "Final model: ${OUTPUT_DIR}/final"
