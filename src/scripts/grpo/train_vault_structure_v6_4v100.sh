#!/usr/bin/env bash
set -euo pipefail

# Existing structure-v6 entry point; URL checkpoints use train_vault_url_4v100.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_TYPE=structure_id_v6
WORK_DIR="${WORK_DIR:-data/vault_bm25_structure_v6}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vault-ruby-grpo-structure-v6}"
if [[ "${RUN_DRY_RUN:-1}" == 1 || "${RUN_TRAINING:-1}" == 1 ]]; then
  : "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to your local structure-v6 SFT checkpoint}"
  : "${MAX_TARGET_LENGTH:?Set MAX_TARGET_LENGTH to the SFT decoder token limit}"
fi
source "${SCRIPT_DIR}/train_vault_grpo_4v100.sh"
