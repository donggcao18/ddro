#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 WORK_DIR [THREADS] [HITS] [BATCH_SIZE]" >&2
  exit 2
fi

WORK_DIR=$1
THREADS=${2:-16}
HITS=${3:-200}
BATCH_SIZE=${4:-16}
INDEX_DIR="$WORK_DIR/index"
RUN_FILE="$WORK_DIR/bm25_run.txt"

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$WORK_DIR/corpus" \
  --index "$INDEX_DIR" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "$THREADS" \
  --storePositions \
  --storeDocvectors \
  --storeRaw

python -m pyserini.search.lucene \
  --index "$INDEX_DIR" \
  --topics "$WORK_DIR/queries.tsv" \
  --output "$RUN_FILE" \
  --output-format msmarco \
  --hits "$HITS" \
  --bm25 \
  --k1 0.82 \
  --b 0.68 \
  --threads "$THREADS" \
  --batch-size "$BATCH_SIZE"

echo "BM25 run written to $RUN_FILE"
