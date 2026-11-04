#!/usr/bin/env bash
set -euo pipefail

# Three independent post-SFT comparisons; the existing GRPO launcher is unchanged.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${GRPO_PYTHON_BIN:-${PYTHON_BIN:-python}}"
WORK_DIR="${WORK_DIR:-/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url}"
TRAIN_INPUT_FILE="${TRAIN_INPUT_FILE:-${WORK_DIR}/Ruby_merged_multilabel_dpo.jsonl}"
TEST_INPUT_FILE="${TEST_INPUT_FILE:-${WORK_DIR}/Ruby_test_r32.0_multilabel.json}"
EVAL_INPUT_FILE="${EVAL_INPUT_FILE:-}"
FIXED_CORPUS_FILE="${FIXED_CORPUS_FILE:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr/checkpoint-160000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/url-coverage-$(date -u +%Y%m%dT%H%M%SZ)}"
NUM_GPUS="${NUM_GPUS:-4}"
PRECISION="${PRECISION:-fp16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
NUM_GENERATIONS="${NUM_GENERATIONS:-10}"
EVAL_NUM_BEAMS="${EVAL_NUM_BEAMS:-10}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
AUTO_TUNE_BATCH_SIZE="${AUTO_TUNE_BATCH_SIZE:-1}"
MAX_TRAIN_BATCH_SIZE="${MAX_TRAIN_BATCH_SIZE:-16}"
QUERIES_PER_GPU_PER_STEP="${QUERIES_PER_GPU_PER_STEP:-16}"
BATCH_SIZE_RESERVE_GIB="${BATCH_SIZE_RESERVE_GIB:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
PREFERENCE_BETA="${PREFERENCE_BETA:-0.1}"
SFT_ANCHOR_WEIGHT="${SFT_ANCHOR_WEIGHT:-0.1}"
POSITIVES_PER_QUERY="${POSITIVES_PER_QUERY:-2}"
NEGATIVES_PER_QUERY="${NEGATIVES_PER_QUERY:-4}"
MISSED_POSITIVE_WEIGHT="${MISSED_POSITIVE_WEIGHT:-2}"
HIT1_TOLERANCE="${HIT1_TOLERANCE:-0.005}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
SEED="${SEED:-42}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-${RUN_TRAINING}}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
EXPERIMENTS="${EXPERIMENTS:-multi_positive_dpo missed_positive_dpo multi_negative_dpo}"
DO_SAMPLE="${DO_SAMPLE:-false}"
DO_SAMPLE="${DO_SAMPLE,,}"
export GRPO_INVALID_TARGET_POLICY="${GRPO_INVALID_TARGET_POLICY:-skip}"
export PYTHONUNBUFFERED=1
if [[ "${PRECISION}" != fp16 && "${PRECISION}" != fp32 ]]; then
  echo "V100 precision must be fp16 or fp32." >&2; exit 2
fi
for name in NUM_GPUS MAX_PROMPT_LENGTH MAX_TARGET_LENGTH NUM_GENERATIONS EVAL_NUM_BEAMS INFERENCE_BATCH_SIZE EVAL_BATCH_SIZE TRAIN_BATCH_SIZE MAX_TRAIN_BATCH_SIZE QUERIES_PER_GPU_PER_STEP POSITIVES_PER_QUERY NEGATIVES_PER_QUERY EVAL_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT; do
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then echo "${name} must be a positive integer." >&2; exit 2; fi
done
for name in RUN_DRY_RUN RUN_TRAINING RUN_INFERENCE PREPARE_ONLY AUTO_TUNE_BATCH_SIZE; do
  if [[ "${!name}" != 0 && "${!name}" != 1 ]]; then echo "${name} must be 0 or 1." >&2; exit 2; fi
done
read -r -a EXPERIMENT_NAMES <<< "${EXPERIMENTS}"
if (( ${#EXPERIMENT_NAMES[@]} == 0 )); then echo "Select at least one experiment." >&2; exit 2; fi
for name in "${EXPERIMENT_NAMES[@]}"; do
  case "${name}" in multi_positive_dpo|missed_positive_dpo|multi_negative_dpo|tca_grpo) ;; *) echo "Unknown experiment: ${name}" >&2; exit 2 ;; esac
done
IS_TCA=0
if [[ " ${EXPERIMENTS} " == *" tca_grpo "* ]]; then
  if (( ${#EXPERIMENT_NAMES[@]} != 1 )); then echo "Run TCA separately with run_url_tca_grpo_4v100.sh." >&2; exit 2; fi
  IS_TCA=1
fi
if [[ "${DO_SAMPLE}" != true && "${DO_SAMPLE}" != false ]]; then echo "DO_SAMPLE must be true or false." >&2; exit 2; fi
if (( EVAL_NUM_BEAMS < 10 || NUM_GENERATIONS < 2 )); then echo "Need at least ten evaluation beams and two rollouts." >&2; exit 2; fi
if (( IS_TCA == 0 )) && { [[ "${DO_SAMPLE}" != false ]] || (( NUM_GENERATIONS != EVAL_NUM_BEAMS )); }; then
  echo "DPO experiments require matching mining/evaluation beam widths and DO_SAMPLE=false." >&2; exit 2
fi
if (( SAVE_STEPS % EVAL_STEPS != 0 )); then echo "SAVE_STEPS must be a multiple of EVAL_STEPS." >&2; exit 2; fi
if [[ "${RUN_TRAINING}" == 0 && "${RUN_INFERENCE}" == 1 ]]; then echo "This fresh-run suite requires training before candidate inference." >&2; exit 2; fi
PREPARE_ARGS=(src/pretrain/coverage_experiments.py prepare --train_input_file "${TRAIN_INPUT_FILE}"
  --test_input_file "${TEST_INPUT_FILE}" --output_dir "${OUTPUT_ROOT}" --validation_split "${VALIDATION_SPLIT}" --seed "${SEED}"
  --experiments "${EXPERIMENT_NAMES[@]}")
if [[ -n "${EVAL_INPUT_FILE}" ]]; then PREPARE_ARGS+=(--eval_input_file "${EVAL_INPUT_FILE}"); fi
if [[ -n "${FIXED_CORPUS_FILE}" ]]; then PREPARE_ARGS+=(--corpus_file "${FIXED_CORPUS_FILE}"); fi
echo "=== Preparing coverage suite: ${OUTPUT_ROOT} ==="
"${PYTHON_BIN}" "${PREPARE_ARGS[@]}"
if [[ "${PREPARE_ONLY}" == 1 ]]; then echo "Prepared suite: ${OUTPUT_ROOT}"; exit 0; fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then echo "Missing SFT checkpoint: ${CHECKPOINT_PATH}" >&2; exit 2; fi
AVAILABLE_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if (( AVAILABLE_GPUS < NUM_GPUS )); then echo "Need ${NUM_GPUS} visible CUDA GPUs; found ${AVAILABLE_GPUS}." >&2; exit 2; fi
DATA_DIR="${OUTPUT_ROOT}/data"
PRECISION_ARGS=()
if [[ "${PRECISION}" == fp16 ]]; then PRECISION_ARGS+=(--fp16); fi
run_distributed() {
  if (( NUM_GPUS > 1 )); then
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}
infer_model() {
  run_distributed src/pretrain/infer_grpo_vault_url.py --model_path "$1" --output_dir "$2" \
    --test_file "${TEST_INPUT_FILE}" --corpus_file "${DATA_DIR}/corpus.jsonl" --catalog_mode fixed \
    --score_normalization trie --num_beams "${EVAL_NUM_BEAMS}" --batch_size "${INFERENCE_BATCH_SIZE}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" --max_target_length "${MAX_TARGET_LENGTH}" "${PRECISION_ARGS[@]}"
}
if [[ "${RUN_INFERENCE}" == 1 ]]; then
  echo "=== SFT baseline test inference (${NUM_GPUS} GPUs, ${EVAL_NUM_BEAMS} beams); training starts after this finishes ==="
  infer_model "${CHECKPOINT_PATH}" "${OUTPUT_ROOT}/sft_baseline" 2>&1 | tee "${OUTPUT_ROOT}/sft_baseline.log"
fi
if [[ "${RUN_TRAINING}" == 1 && "${AUTO_TUNE_BATCH_SIZE}" == 1 ]]; then
  echo "=== Calibrating preference-training memory and query batch size ==="
  if (( MAX_TRAIN_BATCH_SIZE > QUERIES_PER_GPU_PER_STEP )); then MAX_TRAIN_BATCH_SIZE="${QUERIES_PER_GPU_PER_STEP}"; fi
  if (( TRAIN_BATCH_SIZE > MAX_TRAIN_BATCH_SIZE )); then TRAIN_BATCH_SIZE="${MAX_TRAIN_BATCH_SIZE}"; fi
  if (( IS_TCA )); then
    "${PYTHON_BIN}" src/pretrain/calibrate_tca_grpo_batch_size.py \
      --checkpoint_path "${CHECKPOINT_PATH}" --corpus_file "${DATA_DIR}/corpus.jsonl" \
      --output_file "${OUTPUT_ROOT}/batch_size_probe.json" --num_gpus "${NUM_GPUS}" \
      --num_generations "${NUM_GENERATIONS}" --do_sample "${DO_SAMPLE}" --temperature "${TEMPERATURE:-1.0}" \
      --max_prompt_length "${MAX_PROMPT_LENGTH}" --max_target_length "${MAX_TARGET_LENGTH}" \
      --start_batch_size "${TRAIN_BATCH_SIZE}" --max_batch_size "${MAX_TRAIN_BATCH_SIZE}" --reserve_gib "${BATCH_SIZE_RESERVE_GIB}" \
      --gradient_accumulation_steps "${QUERIES_PER_GPU_PER_STEP}" --beta "${GRPO_BETA:-0.04}" \
      --gold_candidates_per_query "${GOLD_CANDIDATES_PER_QUERY:-2}" --reference_batch_size "${REFERENCE_BATCH_SIZE:-16}" \
      --all_zero_ranking_weight "${ALL_ZERO_RANKING_WEIGHT:-0.1}" --sft_anchor_weight "${SFT_ANCHOR_WEIGHT}" \
      --negatives_per_query "${NEGATIVES_PER_QUERY}" --invalid_target_policy "${GRPO_INVALID_TARGET_POLICY}" "${PRECISION_ARGS[@]}" \
      --trajectory_alignment "${TRAJECTORY_ALIGNMENT:-pre_token}" \
      2>&1 | tee "${OUTPUT_ROOT}/calibration.log"
  else
    "${PYTHON_BIN}" src/pretrain/calibrate_grpo_batch_size.py \
    --checkpoint_path "${CHECKPOINT_PATH}" --corpus_file "${DATA_DIR}/corpus.jsonl" \
    --output_file "${OUTPUT_ROOT}/batch_size_probe.json" --num_gpus "${NUM_GPUS}" \
    --num_generations "${NUM_GENERATIONS}" --do_sample false --score_normalization trie \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" --max_target_length "${MAX_TARGET_LENGTH}" \
    --start_batch_size "${TRAIN_BATCH_SIZE}" --max_batch_size "${MAX_TRAIN_BATCH_SIZE}" --reserve_gib "${BATCH_SIZE_RESERVE_GIB}" \
    --gradient_accumulation_steps "${QUERIES_PER_GPU_PER_STEP}" --preference_objective multi_negative_dpo \
    --positives_per_query "${POSITIVES_PER_QUERY}" --negatives_per_query "${NEGATIVES_PER_QUERY}" \
    --preference_beta "${PREFERENCE_BETA}" --sft_anchor_weight "${SFT_ANCHOR_WEIGHT}" "${PRECISION_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/calibration.log"
  fi
  CAPACITY="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_size"])' "${OUTPUT_ROOT}/batch_size_probe.json")"
  if ! [[ "${CAPACITY}" =~ ^[1-9][0-9]*$ ]] || (( CAPACITY > MAX_TRAIN_BATCH_SIZE )); then
    echo "Invalid capacity returned by calibration: ${CAPACITY}" >&2; exit 2
  fi
  # Equal effective query batches across the three objectives, including when
  # the capacity probe picks a non-divisor (e.g. 13 -> use batch 8, accumulation 2).
  TRAIN_BATCH_SIZE="${CAPACITY}"
  while (( QUERIES_PER_GPU_PER_STEP % TRAIN_BATCH_SIZE != 0 )); do TRAIN_BATCH_SIZE=$(( TRAIN_BATCH_SIZE - 1 )); done
fi
if (( QUERIES_PER_GPU_PER_STEP % TRAIN_BATCH_SIZE != 0 )); then
  echo "TRAIN_BATCH_SIZE must divide QUERIES_PER_GPU_PER_STEP." >&2; exit 2
fi
ACCUMULATION=$(( QUERIES_PER_GPU_PER_STEP / TRAIN_BATCH_SIZE ))
echo "Common query batch/GPU=${TRAIN_BATCH_SIZE}; accumulation=${ACCUMULATION}; global queries/update=$(( NUM_GPUS * QUERIES_PER_GPU_PER_STEP ))"
for name in "${EXPERIMENT_NAMES[@]}"; do
  TRAIN_ENTRY=src/pretrain/train_coverage_vault.py
  if [[ "${name}" == tca_grpo ]]; then TRAIN_ENTRY=src/pretrain/train_tca_grpo_vault.py; fi
  TRAIN_ARGS=("${TRAIN_ENTRY}" --checkpoint_path "${CHECKPOINT_PATH}"
    --train_file "${DATA_DIR}/train.jsonl" --eval_file "${DATA_DIR}/validation.jsonl" --corpus_file "${DATA_DIR}/corpus.jsonl"
    --output_dir "${OUTPUT_ROOT}/${name}" --validation_split 0 --max_prompt_length "${MAX_PROMPT_LENGTH}" --max_target_length "${MAX_TARGET_LENGTH}"
    --num_generations "${NUM_GENERATIONS}" --eval_num_beams "${EVAL_NUM_BEAMS}" --do_sample "${DO_SAMPLE}" --score_normalization trie
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${ACCUMULATION}" --learning_rate "${LEARNING_RATE}" --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --max_steps "${MAX_STEPS}" --sft_anchor_weight "${SFT_ANCHOR_WEIGHT}"
    --negatives_per_query "${NEGATIVES_PER_QUERY}" --hit1_tolerance "${HIT1_TOLERANCE}"
    --eval_steps "${EVAL_STEPS}" --save_steps "${SAVE_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}" --seed "${SEED}"
    --dataloader_num_workers 2 --gradient_checkpointing "${PRECISION_ARGS[@]}")
  if [[ "${name}" == tca_grpo ]]; then
    TRAIN_ARGS+=(--beta "${GRPO_BETA:-0.04}" --temperature "${TEMPERATURE:-1.0}"
      --gold_candidates_per_query "${GOLD_CANDIDATES_PER_QUERY:-2}" --gold_cache_mb "${GOLD_CACHE_MB:-128}"
      --reference_batch_size "${REFERENCE_BATCH_SIZE:-16}" --hit_bonus "${HIT_BONUS:-2.0}"
      --overflow_penalty "${OVERFLOW_PENALTY:--0.5}" --reward_std_floor "${REWARD_STD_FLOOR:-0.001}"
      --all_zero_ranking_weight "${ALL_ZERO_RANKING_WEIGHT:-0.1}"
      --trajectory_alignment "${TRAJECTORY_ALIGNMENT:-pre_token}"
      --min_validation_recall_gain "${MIN_VALIDATION_RECALL_GAIN:-0.005}")
  else
    TRAIN_ARGS+=(--objective "${name}" --preference_beta "${PREFERENCE_BETA}"
      --positives_per_query "${POSITIVES_PER_QUERY}" --missed_positive_weight "${MISSED_POSITIVE_WEIGHT}")
  fi
  if [[ "${RUN_DRY_RUN}" == 1 ]]; then
    echo "=== ${name}: one-query dry run ==="
    "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" --per_device_train_batch_size 1 --dry_run 2>&1 | tee "${OUTPUT_ROOT}/${name}-dry-run.log"
  fi
  if [[ "${RUN_TRAINING}" == 1 ]]; then
    echo "=== ${name}: preference training (${NUM_GPUS} GPUs) ==="
    run_distributed "${TRAIN_ARGS[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${name}-train.log"
  fi
  if [[ "${RUN_INFERENCE}" == 1 ]]; then
    echo "=== ${name}: trained-candidate test inference ==="
    infer_model "${OUTPUT_ROOT}/${name}/candidate_final" "${OUTPUT_ROOT}/${name}/candidate_inference" \
      2>&1 | tee "${OUTPUT_ROOT}/${name}-inference.log"
  fi
done
if [[ "${RUN_INFERENCE}" == 1 ]]; then
  "${PYTHON_BIN}" src/pretrain/coverage_experiments.py summarize --output_dir "${OUTPUT_ROOT}" --experiments "${EXPERIMENT_NAMES[@]}"
  echo "Comparison: ${OUTPUT_ROOT}/comparison.csv and comparison.json"
fi
echo "Suite complete: ${OUTPUT_ROOT}"
