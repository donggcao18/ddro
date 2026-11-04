#!/usr/bin/env bash
set -euo pipefail

# Experiment 2: token-level trajectory rewards, multi-positive URL targets.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS=tca_grpo
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr/checkpoint-160000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/url-tca-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
DO_SAMPLE="${DO_SAMPLE:-true}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
EVAL_NUM_BEAMS="${EVAL_NUM_BEAMS:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_TRAIN_BATCH_SIZE="${MAX_TRAIN_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_STEPS="${MAX_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
HIT1_TOLERANCE="${HIT1_TOLERANCE:-0}"
MIN_VALIDATION_RECALL_GAIN="${MIN_VALIDATION_RECALL_GAIN:-0.005}"
RUN_BASELINE_INFERENCE="${RUN_BASELINE_INFERENCE:-0}"
BASELINE_DIR="${BASELINE_DIR:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/url-coverage-20261103T185157Z/sft_baseline}"
echo "=== Token-trajectory GRPO pilot: SFT checkpoint ${CHECKPOINT_PATH} ==="
echo "Rollouts: do_sample=${DO_SAMPLE}, G=${NUM_GENERATIONS}; validation/inference beams=${EVAL_NUM_BEAMS}"
echo "Set EVAL_INPUT_FILE to queries unseen by SFT for generalization-based selection; the default is a post-training query split."
source "${SCRIPT_DIR}/run_url_coverage_experiments_4v100.sh"
