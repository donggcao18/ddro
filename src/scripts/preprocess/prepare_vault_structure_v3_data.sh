#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start}"
INPUT_FILE="${INPUT_FILE:-${DATA_ROOT}/Ruby_ready_to_feed_multilabel.jsonl}"
TRAIN_ORIGINAL="${TRAIN_ORIGINAL:-${DATA_ROOT}/Ruby_train_r32.0.json}"
TEST_ORIGINAL="${TEST_ORIGINAL:-${DATA_ROOT}/Ruby_test_r32.0.json}"
STRUCTURE_ID_SOURCE="${STRUCTURE_ID_SOURCE:-/home/users/congthanh_le/scratch/veil/CodeGR/data/augmented_dsi/Ruby_merged.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${DATA_ROOT}/Ruby_ready_to_feed_structure_id_v3.jsonl}"

for required_file in \
  "${INPUT_FILE}" \
  "${TRAIN_ORIGINAL}" \
  "${TEST_ORIGINAL}" \
  "${STRUCTURE_ID_SOURCE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" src/scripts/preprocess/merge_structure_id_v3.py \
  --input "${INPUT_FILE}" \
  --original "${TRAIN_ORIGINAL}" \
  --original "${TEST_ORIGINAL}" \
  --structure-source "${STRUCTURE_ID_SOURCE}" \
  --join-key url_based_id \
  --output "${OUTPUT_FILE}" \
  --expand-multilabel \
  --on-missing skip

STATS_FILE="${OUTPUT_FILE%.*}.stats.json"
MISSING_FILE="${OUTPUT_FILE%.*}.missing.jsonl"

echo ""
echo "=== structure_id_v3 preprocessing complete ==="
echo "Output:      ${OUTPUT_FILE}"
echo "Statistics:  ${STATS_FILE}"
echo "Missing log: ${MISSING_FILE}"
echo ""
cat "${STATS_FILE}"
