#!/usr/bin/env bash
set -euo pipefail

# Four 32-GB V100 GPUs on one node: 4 * 8 pairs/GPU * 2 accumulation = 64.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo_ema_v100_url_paper_lr.json}"
exec bash "${SCRIPT_DIR}/run_vault_iterative_dpo.sh" "$@"
