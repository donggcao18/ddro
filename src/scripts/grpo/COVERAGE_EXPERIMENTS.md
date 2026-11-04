# Three post-SFT coverage experiments

Run all three, sequentially on the same four V100s:

```bash
bash src/scripts/grpo/run_url_coverage_experiments_4v100.sh
```

Each experiment starts independently from the same SFT checkpoint, not from the
preceding experiment or an already GRPO-trained model. The existing GRPO
launchers and their defaults remain unchanged.

## Objectives

| Experiment | Preference loss | Positive selection |
| --- | --- | --- |
| `multi_positive_dpo` | Mean DPO logistic loss over selected positive/negative pairs | Uniform sampling without replacement |
| `missed_positive_dpo` | Same pairwise DPO, weighting selected missed positives by 2 | Reserve up to half the positive budget for positives absent from current top ten; fill from remaining positives |
| `multi_negative_dpo` | Per-positive multi-negative contrast on DPO log ratios | Same missed-positive selection and weights as experiment 2 |

For sequence score `s`, frozen SFT score `s_ref`, and beta `0.1`, define
`r(d) = beta * (s(d) - s_ref(d))`. The first two objectives average
`softplus(r(negative) - r(positive))` over negatives for each positive. The third
uses `log(1 + sum_negative exp(r(negative) - r(positive)))`. Positive losses are
weight-normalized within the query and query losses averaged. Other positives
never enter a positive's negative denominator. The third method is a
multi-negative preference extension, not a reproduction of a named paper.

All three add `0.1 * positive SFT NLL` under the full token vocabulary. This
anchor uses mean token NLL, first per positive, then per query. Preference and
beam scores are summed token log probabilities under the same full-catalog
trie, including EOS, without length normalization.

Ten deterministic beams mine current-policy negatives each microbatch. Any ID
in the query's entire annotated positive set is excluded from negatives. When
beams provide too few negatives, the remaining slots are sampled from the fixed
inventory after excluding all positives. This fallback is logged. Up to two
known positives and four negatives per query are teacher-forced; positives do
not have to appear in the beams. All-wrong groups can therefore learn. A query
whose entire inventory is positive has no preference term and uses the SFT
anchor only. These are supervised preference experiments, not GRPO estimators.
Negative safety depends on annotation completeness: an unannotated relevant ID
can still be treated as negative; the code does not infer additional relevance
from URL-prefix similarity.

## Data and fair comparison

Defaults use:

- Training: `/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_merged_multilabel_dpo.jsonl`
- Test: `/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_test_r32.0_multilabel.json`
- SFT: `/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr/checkpoint-160000`

Native rows need `text`, `url_based_id`, and `positive_url_based_id`. The script
does not automatically separate your five pseudoqueries. Set `TRAIN_INPUT_FILE`
to the intended post-training subset. `EVAL_INPUT_FILE` optionally supplies a
separate native validation file; otherwise a reproducible 1% query split is
used. If the starting SFT checkpoint already saw those validation queries,
this split is not an unseen-generalization evaluation. Use new validation
queries for reliable selection. The test file is never used for mining,
positive selection, loss computation, or checkpoint selection.

One fixed catalog is shared by mining, validation, SFT baseline inference, and
all candidate inference. By default it is the training inventory plus the
document URLs of validation/test records. No test-label-derived URLs are added.
Set `FIXED_CORPUS_FILE` to an explicit canonical full-corpus JSONL (`{"target":
"..."}`) to use exactly that inventory. This is useful when a post-training query
subset does not contain the full corpus. Unusable targets follow the existing
`skip` policy and are reported. Final test queries/positive labels stay in the
denominators even when an ID cannot be decoded.

## Batch size, checkpoint selection, and outputs

Defaults: 4 GPUs, FP16, 256 prompt tokens, 128 target tokens, 10 beams, LR
`2e-6`, one epoch, 10% warmup, gradient checkpointing, evaluation/saving every
500 optimizer steps, checkpoint retention limit 2. These are pilot settings,
not demonstrated optimal hyperparameters.

A representative multi-negative preference memory probe includes the SFT
reference, six full-length scored candidates, ten-beam generation, the SFT
anchor, backward, resident gradients and AdamW state, with 4 GiB runtime/DDP
headroom. The largest fitting batch up to 16 is rounded down to a divisor of
16. Gradient accumulation makes each experiment use 16 queries/GPU/update
(64 globally). For example, capacity 13 selects batch 8 and accumulation 2.
Increasing `QUERIES_PER_GPU_PER_STEP` and `MAX_TRAIN_BATCH_SIZE` permits larger
microbatches. Calibration is an estimate; evaluation/inference batches are
independent and default to two queries/GPU. Dry runs use one query.

Every run evaluates the initial SFT policy on validation. Checkpoints are
selected by maximum validation Recall@10, subject to Hit@1 being no more than
`HIT1_TOLERANCE=0.005` (0.5 percentage points) below that baseline. If the selected
trained candidate does not strictly improve validation recall within this
constraint, `final/` contains the original SFT policy instead. Set tolerance to
zero to forbid any measured validation Hit@1 regression. This guard does not
guarantee preserved test Hit@1.

Each experiment preserves both `candidate_final/` (trained candidate, even if
rejected) and `final/` (validation-selected candidate or SFT fallback), plus
`selection.json` explaining the decision. Test inference always evaluates the
trained candidate, so unsuccessful training is visible. `comparison.csv` and
`comparison.json` show candidate and selected results alongside the SFT
baseline: Hit/MRR, Recall, Precision, nDCG, and distinct relevant counts at
1/5/10, metric deltas, and per-query Hit@1 losses/recoveries and coverage changes.
SFT fallback test results reuse the baseline inference. No test-based winner is
automatically selected.

The suite runs SFT baseline test inference **before** memory calibration or
training. A `Target filtering` message at this stage is not a training update.
Inference prints a start message and per-GPU local query counts, throughput,
and ETA after the first batch, approximately every 30 seconds after subsequent
batches, and at completion. Reports are independent per rank, with no extra
distributed synchronization. Prediction shards are flushed at these reports.
The stage log is `sft_baseline.log`; temporary predictions are under
`sft_baseline/.predictions.rank*.jsonl` until merged into `predictions.jsonl`.
Standalone inference accepts `--progress_interval` to change the interval.

To reuse a completed baseline, set `RUN_BASELINE_INFERENCE=0` and
`BASELINE_DIR=/path/to/previous-run/sft_baseline`. The script checks saved
checkpoint/data/catalog/decoding provenance and copies the finished metrics and
predictions into the new suite without running baseline model inference.
The standalone TCA preset defaults to reusing your completed
`url-coverage-20261103T185157Z/sft_baseline`; set `RUN_BASELINE_INFERENCE=1` to
compute a new baseline. Validation and candidate inference remain enabled.

`OUTPUT_ROOT` defaults to a timestamped directory. The suite refuses nonempty
directories, preventing accidental overwrites. Individual training supports
`--resume_from_checkpoint` through `train_coverage_vault.py`; the suite runner
starts fresh comparisons rather than automatically resuming partial runs.

Examples:

```bash
# A short end-to-end pilot of all three experiments.
MAX_STEPS=20 EVAL_STEPS=10 SAVE_STEPS=10 \
OUTPUT_ROOT=/path/to/new/coverage-pilot \
bash src/scripts/grpo/run_url_coverage_experiments_4v100.sh

# Supply unseen post-training/validation queries and an explicit full inventory.
TRAIN_INPUT_FILE=/data/two-posttraining-queries.jsonl \
EVAL_INPUT_FILE=/data/unseen-validation.jsonl \
FIXED_CORPUS_FILE=/data/full-url-corpus.jsonl \
bash src/scripts/grpo/run_url_coverage_experiments_4v100.sh

# Prepare and inspect the data only; no model or GPU required.
PREPARE_ONLY=1 OUTPUT_ROOT=/path/to/new/prepared-suite \
bash src/scripts/grpo/run_url_coverage_experiments_4v100.sh

# Run just one objective with a fixed batch, still using the common baseline.
EXPERIMENTS=missed_positive_dpo AUTO_TUNE_BATCH_SIZE=0 TRAIN_BATCH_SIZE=4 \
bash src/scripts/grpo/run_url_coverage_experiments_4v100.sh
```

Method context: [DPO](https://arxiv.org/abs/2305.18290) supplies the reference-log
ratio preference objective. [RAD-DPO](https://arxiv.org/abs/2602.23964) motivates
explicit multi-positive coverage supervision in structured generative retrieval.
This implementation does not claim to reproduce RAD-DPO or guarantee gains.
