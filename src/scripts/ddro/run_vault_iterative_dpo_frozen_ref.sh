#!/usr/bin/env bash
set -euo pipefail

# Run inside ddro_env. Custom CONFIG must keep EMA decay at 1.0 to freeze the reference.
# Pass --resume only to continue this same frozen-reference experiment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo_frozen_ref.json}"
exec bash "${SCRIPT_DIR}/run_vault_iterative_dpo.sh" "$@"
