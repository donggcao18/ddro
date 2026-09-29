#!/usr/bin/env bash
set -euo pipefail

# Single-node DDP on four V100 GPUs. Honor scheduler-assigned GPU visibility.
SCRIPT_DIR="."
if [[ "${BASH_SOURCE[0]}" == */* ]]; then
  SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
fi
SCRIPT_DIR="$(cd -- "${SCRIPT_DIR}" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
if [[ "${NUM_GPUS}" != "4" ]]; then
  echo "This launcher requires NUM_GPUS=4 on one node." >&2
  exit 2
fi

export PREFERENCE_OBJECTIVE="${PREFERENCE_OBJECTIVE:-tdpo2-weight}"
export TDPO_PREFIX_TOKENS="${TDPO_PREFIX_TOKENS:-3}"
export TDPO_PREFIX_WEIGHT="${TDPO_PREFIX_WEIGHT:-3.0}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
# V100 has no native BF16. FP32 avoids T5 FP16 activation overflow;
# opt into PRECISION=fp16 after checking finite loss/gradients on this checkpoint.
export PRECISION="${PRECISION:-fp32}"
case "${PRECISION}" in
  fp32|fp16) ;;
  *) echo "V100 launcher supports PRECISION=fp32 or fp16." >&2; exit 2 ;;
esac

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vault-url-${PREFERENCE_OBJECTIVE}-k${TDPO_PREFIX_TOKENS}-w${TDPO_PREFIX_WEIGHT}-4v100}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/torchrun-logs}"
# New experiments start from the SFT model. Set latest/path explicitly to resume.
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"

exec bash "${SCRIPT_DIR}/launch_tdpo_training_vault_url.sh" "$@"
