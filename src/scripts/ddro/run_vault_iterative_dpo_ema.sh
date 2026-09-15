#!/usr/bin/env bash
set -euo pipefail

# Run inside ddro_env. CONFIG can select a customized EMA configuration.
# Pass --resume only when continuing an existing EMA run.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo_ema.json}"
exec bash "${SCRIPT_DIR}/run_vault_iterative_dpo.sh" "$@"
