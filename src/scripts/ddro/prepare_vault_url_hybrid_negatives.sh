#!/usr/bin/env bash
set -euo pipefail

# Preprocessing-only URL hybrid workflow.
#
# Prerequisite: BM25 preparation and URL conversion have already produced:
#   dpo_pairs_url.jsonl
#   query_metadata.jsonl
#   document_metadata.jsonl
#
# This script runs only URL model-confusion mining and hybrid combination. It
# does not invoke Pyserini and does not start DPO training.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WORK_DIR="${WORK_DIR:-/mnt/beegfs/scratch/congthanh_le/east/ddro/data/vault_bm25}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"

BM25_URL_PAIRS="${BM25_URL_PAIRS:-${WORK_DIR}/dpo_pairs_url.jsonl}"
QUERY_METADATA="${QUERY_METADATA:-${WORK_DIR}/query_metadata.jsonl}"
DOCUMENT_METADATA="${DOCUMENT_METADATA:-${WORK_DIR}/document_metadata.jsonl}"
MODEL_URL_PAIRS="${MODEL_URL_PAIRS:-${WORK_DIR}/model_confusion_pairs_url.jsonl}"
HYBRID_URL_PAIRS="${HYBRID_URL_PAIRS:-${WORK_DIR}/dpo_pairs_hybrid_url.jsonl}"
COLLISION_EXCLUSIONS="${COLLISION_EXCLUSIONS:-${WORK_DIR}/model_confusion_pairs_url.excluded_text_ids.json}"

RUN_MODEL_MINING="${RUN_MODEL_MINING:-1}"
MODEL_NEGATIVES="${MODEL_NEGATIVES:-4}"
TOTAL_NEGATIVES="${TOTAL_NEGATIVES:-8}"
NUM_BEAMS="${NUM_BEAMS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-96}"
MINING_DEVICE="${MINING_DEVICE:-auto}"
LIMIT_QUERIES="${LIMIT_QUERIES:-}"
REQUIRE_EXACT_MIX="${REQUIRE_EXACT_MIX:-0}"
TARGET_COLLISION_POLICY="${TARGET_COLLISION_POLICY:-skip}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"


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


require_directory "${CHECKPOINT_PATH}"
require_file "${BM25_URL_PAIRS}"
require_file "${QUERY_METADATA}"
require_file "${DOCUMENT_METADATA}"
mkdir -p "$(dirname -- "${MODEL_URL_PAIRS}")" "$(dirname -- "${HYBRID_URL_PAIRS}")"

if [[ "${RUN_MODEL_MINING}" == "1" ]]; then
  # Fail before loading the checkpoint if this is accidentally run with the
  # CPU-only Pyserini environment.
  "${DPO_PYTHON_BIN}" -c "import torch, transformers; print('Python:', __import__('sys').executable); print('CUDA:', torch.cuda.is_available())"
  if [[ "${PRECISION}" != "fp32" ]]; then
    if ! "${DPO_PYTHON_BIN}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
      echo "${PRECISION} model mining requires CUDA, but DPO_PYTHON_BIN has no CUDA." >&2
      echo "Activate ddro_env on a GPU node or set DPO_PYTHON_BIN to its Python executable." >&2
      exit 2
    fi
  fi

  MODEL_ARGS=(
    src/scripts/bm25/the_vault/mine_model_confusion_negatives.py
    --target-type url
    --target-collision-policy "${TARGET_COLLISION_POLICY}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --query-metadata "${QUERY_METADATA}"
    --document-metadata "${DOCUMENT_METADATA}"
    --output "${MODEL_URL_PAIRS}"
    --negatives-per-query "${MODEL_NEGATIVES}"
    --num-beams "${NUM_BEAMS}"
    --batch-size "${BATCH_SIZE}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-target-length "${MAX_TARGET_LENGTH}"
    --device "${MINING_DEVICE}"
  )
  if [[ -n "${LIMIT_QUERIES}" ]]; then
    MODEL_ARGS+=(--limit-queries "${LIMIT_QUERIES}")
  fi
  case "${PRECISION}" in
    bf16) MODEL_ARGS+=(--bf16) ;;
    fp16) MODEL_ARGS+=(--fp16) ;;
    fp32) ;;
    *) echo "PRECISION must be bf16, fp16, or fp32" >&2; exit 2 ;;
  esac

  echo "=== Stage 1/2: URL model-confusion mining ==="
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Beams: ${NUM_BEAMS}; batch size: ${BATCH_SIZE}; precision: ${PRECISION}"
  "${DPO_PYTHON_BIN}" "${MODEL_ARGS[@]}"
else
  echo "=== Stage 1/2: Reusing existing URL model-confusion pairs ==="
fi
require_file "${MODEL_URL_PAIRS}"
require_file "${COLLISION_EXCLUSIONS}"

COMBINE_ARGS=(
  src/scripts/bm25/the_vault/combine_hybrid_dpo.py
  --bm25-input "${BM25_URL_PAIRS}"
  --model-input "${MODEL_URL_PAIRS}"
  --document-metadata "${DOCUMENT_METADATA}"
  --output "${HYBRID_URL_PAIRS}"
  --excluded-text-ids "${COLLISION_EXCLUSIONS}"
  --model-per-query "${MODEL_NEGATIVES}"
  --total-per-query "${TOTAL_NEGATIVES}"
  --seed "${SEED}"
)
if [[ "${REQUIRE_EXACT_MIX}" == "1" ]]; then
  COMBINE_ARGS+=(--require-exact-mix)
fi

echo "=== Stage 2/2: Combining URL model and BM25 negatives ==="
"${DPO_PYTHON_BIN}" "${COMBINE_ARGS[@]}"

echo "BM25 URL pairs:       ${BM25_URL_PAIRS}"
echo "Model-confusion pairs: ${MODEL_URL_PAIRS}"
echo "Collision exclusions:  ${COLLISION_EXCLUSIONS}"
echo "Hybrid URL pairs:     ${HYBRID_URL_PAIRS}"
echo "Hybrid statistics:    ${HYBRID_URL_PAIRS%.jsonl}.stats.json"
