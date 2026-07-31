#!/usr/bin/env bash
set -euo pipefail

# Mine URL-DocID model-confusion negatives after the BM25/data-preparation
# stage has produced query_metadata.jsonl and document_metadata.jsonl.
#
# Expected usage:
#   conda activate ddro_env
#   bash src/scripts/ddro/mine_vault_url_model_confusions.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WORK_DIR="${WORK_DIR:-/mnt/beegfs/scratch/congthanh_le/east/ddro/data/vault_bm25}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000}"
DPO_PYTHON_BIN="${DPO_PYTHON_BIN:-python}"

QUERY_METADATA="${QUERY_METADATA:-${WORK_DIR}/query_metadata.jsonl}"
DOCUMENT_METADATA="${DOCUMENT_METADATA:-${WORK_DIR}/document_metadata.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${WORK_DIR}/model_confusion_pairs_url.jsonl}"

NEGATIVES_PER_QUERY="${NEGATIVES_PER_QUERY:-4}"
NUM_BEAMS="${NUM_BEAMS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-96}"
MINING_DEVICE="${MINING_DEVICE:-auto}"
LIMIT_QUERIES="${LIMIT_QUERIES:-}"
PRECISION="${PRECISION:-bf16}"
TARGET_COLLISION_POLICY="${TARGET_COLLISION_POLICY:-skip}"


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
require_file "${QUERY_METADATA}"
require_file "${DOCUMENT_METADATA}"
mkdir -p "$(dirname -- "${OUTPUT_FILE}")"

MINING_ARGS=(
  src/scripts/bm25/the_vault/mine_model_confusion_negatives.py
  --target-type url
  --target-collision-policy "${TARGET_COLLISION_POLICY}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --query-metadata "${QUERY_METADATA}"
  --document-metadata "${DOCUMENT_METADATA}"
  --output "${OUTPUT_FILE}"
  --negatives-per-query "${NEGATIVES_PER_QUERY}"
  --num-beams "${NUM_BEAMS}"
  --batch-size "${BATCH_SIZE}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-target-length "${MAX_TARGET_LENGTH}"
  --device "${MINING_DEVICE}"
)

if [[ -n "${LIMIT_QUERIES}" ]]; then
  MINING_ARGS+=(--limit-queries "${LIMIT_QUERIES}")
fi

case "${PRECISION}" in
  bf16) MINING_ARGS+=(--bf16) ;;
  fp16) MINING_ARGS+=(--fp16) ;;
  fp32) ;;
  *) echo "PRECISION must be bf16, fp16, or fp32" >&2; exit 2 ;;
esac

echo "=== Vault URL model-confusion mining ==="
echo "Checkpoint:        ${CHECKPOINT_PATH}"
echo "Query metadata:    ${QUERY_METADATA}"
echo "Document metadata: ${DOCUMENT_METADATA}"
echo "Output:            ${OUTPUT_FILE}"
echo "Beams:             ${NUM_BEAMS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Precision:         ${PRECISION}"

"${DPO_PYTHON_BIN}" "${MINING_ARGS[@]}"

echo "Model-confusion pairs: ${OUTPUT_FILE}"
echo "Mining statistics:     ${OUTPUT_FILE%.jsonl}.stats.json"
echo "Collision exclusions:  ${OUTPUT_FILE%.jsonl}.excluded_text_ids.json"
