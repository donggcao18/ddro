#!/usr/bin/env bash
set -euo pipefail

# Run inside ddro_env. Pass --resume or --prepare-only through to the controller.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/src/pretrain/train_iterative_ddro_vault.py" \
  --config "${CONFIG}" "$@"
