#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/east/ddro/data/vault_JS_bm25}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-javascript-ddro-url-bm25}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to your JavaScript URL SFT checkpoint directory}"
export STARTUP_DEBUG="${STARTUP_DEBUG:-60}"

exec bash "${SCRIPT_DIR}/train_vault_url_bm25_4v100.sh" "$@"
