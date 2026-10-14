#!/usr/bin/env bash
set -euo pipefail

# Run once from the controller, not through torchrun. It launches mining/training workers.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -u "${REPO_ROOT}/src/pretrain/launch_branch_preference.py" "$@"
