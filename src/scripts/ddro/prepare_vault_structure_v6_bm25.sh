#!/usr/bin/env bash
set -euo pipefail

# Stage 1: reuse an existing Vault BM25 run, attach structure_id_v6 targets to
# its metadata, and re-mine pairs. Set RUN_BM25=1 only for a fresh retrieval.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/users/congthanh_le/scratch/veil/CodeGR/data/indexed}"
TRAIN_ORIGINAL="${TRAIN_ORIGINAL:-${DATA_ROOT}/Ruby_train_r32.0.json}"
TEST_ORIGINAL="${TEST_ORIGINAL:-${DATA_ROOT}/Ruby_test_r32.0.json}"
AUGMENTATION="${AUGMENTATION:-${/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_v6/Ruby_merged_multilabel.jsonl}"
AUGMENTATION_ID_MODE="${AUGMENTATION_ID_MODE:-auto}"
STRUCTURE_ID_SOURCE="${STRUCTURE_ID_SOURCE:-}"
WORK_DIR="${WORK_DIR:-/mnt/beegfs/scratch/congthanh_le/east/ddro/data/vault_bm25_structure_v6}"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-}"
BM25_PYTHON_BIN="${BM25_PYTHON_BIN:-python}"
RUN_BM25="${RUN_BM25:-0}"
BM25_THREADS="${BM25_THREADS:-16}"
BM25_BATCH_SIZE="${BM25_BATCH_SIZE:-16}"
BM25_HITS="${BM25_HITS:-200}"
REUSE_INDEX="${REUSE_INDEX:-0}"
SEED="${SEED:-42}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 2
  fi
}

if [[ -z "${STRUCTURE_ID_SOURCE}" ]]; then
  echo "Set STRUCTURE_ID_SOURCE to JSON/JSONL with url_based_id and structure_id_v6." >&2
  exit 2
fi
require_file "${STRUCTURE_ID_SOURCE}"
if [[ "${RUN_BM25}" == "1" ]]; then
  require_file "${TRAIN_ORIGINAL}"
  require_file "${AUGMENTATION}"
  if [[ -n "${TEST_ORIGINAL}" ]]; then require_file "${TEST_ORIGINAL}"; fi
  "${BM25_PYTHON_BIN}" -c 'import pyserini' || {
    echo "Activate the Pyserini environment or set BM25_PYTHON_BIN to its Python." >&2
    exit 2
  }
  BM25_ARGS=(
    src/scripts/bm25/the_vault/run_pipeline.py
    --train-original "${TRAIN_ORIGINAL}"
    --augmentation "${AUGMENTATION}"
    --augmentation-id-mode "${AUGMENTATION_ID_MODE}"
    --structure-id-source "${STRUCTURE_ID_SOURCE}"
    --structure-id-field structure_id_v6
    --structure-id-join-key url_based_id
    --target-type structure_id_v6
    --work-dir "${WORK_DIR}"
    --threads "${BM25_THREADS}"
    --batch-size "${BM25_BATCH_SIZE}"
    --hits "${BM25_HITS}"
    --negatives-per-query 8
    --rank-ranges 1:20,21:100,101:200
    --rank-quotas 3,2,3
    --seed "${SEED}"
  )
  if [[ -n "${TEST_ORIGINAL}" ]]; then BM25_ARGS+=(--test-original "${TEST_ORIGINAL}"); fi
  if [[ "${AUGMENTATION_ID_MODE}" == "url_based_id" ]]; then BM25_ARGS+=(--strict); fi
  if [[ "${REUSE_INDEX}" == "1" ]]; then BM25_ARGS+=(--reuse-index); fi
  echo "=== Vault structure_id_v6: fresh sparse retrieval ==="
  "${BM25_PYTHON_BIN}" "${BM25_ARGS[@]}"
elif [[ "${RUN_BM25}" == "0" ]]; then
  if [[ -z "${SOURCE_WORK_DIR}" ]]; then
    echo "Set SOURCE_WORK_DIR to the existing BM25 work directory (containing bm25_run.txt and metadata)." >&2
    exit 2
  fi
  SOURCE_RUN="${SOURCE_WORK_DIR}/bm25_run.txt"
  SOURCE_QUERIES="${SOURCE_WORK_DIR}/query_metadata.jsonl"
  SOURCE_DOCUMENTS="${SOURCE_WORK_DIR}/document_metadata.jsonl"
  require_file "${SOURCE_RUN}"
  require_file "${SOURCE_QUERIES}"
  require_file "${SOURCE_DOCUMENTS}"
  if [[ ! -s "${SOURCE_RUN}" ]]; then
    echo "Existing BM25 run is empty: ${SOURCE_RUN}" >&2
    exit 2
  fi
  echo "=== Vault structure_id_v6: reuse existing BM25 retrieval ==="
  echo "Existing run: ${SOURCE_RUN}"
  "${BM25_PYTHON_BIN}" src/scripts/bm25/the_vault/relabel_structure_v6_metadata.py \
    --source-query-metadata "${SOURCE_QUERIES}" \
    --source-document-metadata "${SOURCE_DOCUMENTS}" \
    --structure-id-source "${STRUCTURE_ID_SOURCE}" \
    --output-query-metadata "${WORK_DIR}/query_metadata.jsonl" \
    --output-document-metadata "${WORK_DIR}/document_metadata.jsonl"
  "${BM25_PYTHON_BIN}" src/scripts/bm25/the_vault/mine_dpo_negatives.py \
    --run "${SOURCE_RUN}" \
    --query-metadata "${WORK_DIR}/query_metadata.jsonl" \
    --document-metadata "${WORK_DIR}/document_metadata.jsonl" \
    --output "${WORK_DIR}/dpo_pairs_structure_id_v6.jsonl" \
    --target-type structure_id_v6 \
    --negatives-per-query 8 \
    --rank-ranges 1:20,21:100,101:200 \
    --rank-quotas 3,2,3 \
    --seed "${SEED}"
else
  echo "RUN_BM25 must be 0 (reuse) or 1 (fresh retrieval)." >&2
  exit 2
fi

for output in query_metadata.jsonl document_metadata.jsonl dpo_pairs_structure_id_v6.jsonl; do
  require_file "${WORK_DIR}/${output}"
done
if [[ ! -s "${WORK_DIR}/dpo_pairs_structure_id_v6.jsonl" ]]; then
  echo "BM25 pair file is empty: ${WORK_DIR}/dpo_pairs_structure_id_v6.jsonl" >&2
  exit 2
fi
echo "BM25 pairs: ${WORK_DIR}/dpo_pairs_structure_id_v6.jsonl"
echo "Next: activate ddro_env and run train_vault_structure_v6_hybrid_4v100.sh"
