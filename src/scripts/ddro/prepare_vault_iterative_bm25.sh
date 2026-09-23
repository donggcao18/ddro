#!/usr/bin/env bash
set -euo pipefail

# 'prepare' needs only Python; 'build' must run inside the Pyserini/Java env.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
exec "${PYTHON_BIN:-python}" "${REPO_ROOT}/src/pretrain/offline_iterative_bm25.py" "$@"
