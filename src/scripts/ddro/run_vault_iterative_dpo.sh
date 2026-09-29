#!/usr/bin/env bash
set -euo pipefail

# Run inside ddro_env. Select an experiment with --config PATH (or CONFIG).
# Other options, such as --resume and --prepare-only, go to the controller.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/src/scripts/configs/iterative_vault_dpo.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONTROLLER_ARGS=()
while (( $# )); do
  case "$1" in
    --config)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "--config requires a path" >&2
        exit 2
      fi
      CONFIG="$2"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#--config=}"
      if [[ -z "$CONFIG" ]]; then
        echo "--config requires a path" >&2
        exit 2
      fi
      shift
      ;;
    *)
      CONTROLLER_ARGS+=("$1")
      shift
      ;;
  esac
done

exec "${PYTHON_BIN}" "${REPO_ROOT}/src/pretrain/train_iterative_ddro_vault.py" \
  --config "${CONFIG}" "${CONTROLLER_ARGS[@]}"
