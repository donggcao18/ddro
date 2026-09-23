#!/usr/bin/env bash
set -euo pipefail

# Run in ddro_env after the offline BM25 SQLite cache has been built.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo_hybrid_v100.json}"
exec bash "${SCRIPT_DIR}/run_vault_iterative_dpo.sh" "$@"
