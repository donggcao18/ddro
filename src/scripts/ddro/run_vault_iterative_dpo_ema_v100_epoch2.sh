#!/usr/bin/env bash
set -euo pipefail

# Allocate the same four V100 GPUs on one node. Inherit settings from the run.
# Respect CUDA_VISIBLE_DEVICES supplied by the scheduler.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
exec "${PYTHON_BIN:-python}" "${REPO_ROOT}/src/pretrain/start_next_iterative_epoch.py" "$@"
