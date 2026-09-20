#!/usr/bin/env bash
set -euo pipefail

# Allocate four V100 GPUs on ONE node before launching. Respect the scheduler's
# CUDA_VISIBLE_DEVICES setting; the controller starts all four local workers.
# CONFIG and PYTHON_BIN overrides work as in the regular launcher.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo_frozen_v100.json}"
exec bash "${SCRIPT_DIR}/run_vault_iterative_dpo.sh" "$@"
