#!/usr/bin/env bash
set -euo pipefail

# Run the structure_v6 on-policy EMA experiment on four V100 GPUs on one node.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo_ema_v100_structure_v6_paper_lr.json}"
exec bash "${SCRIPT_DIR}/run_vault_iterative_dpo.sh" "$@"
