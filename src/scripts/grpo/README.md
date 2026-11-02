# Binary multi-positive GRPO for Vault

This path trains encoder-decoder T5/CodeT5 retrievers from an existing SFT
checkpoint. Each query produces independently sampled, trie-constrained document
IDs. A completed ID in the query's known positive set receives **1**, everything
else receives **0**. Every positive in a group is rewarded. No reward model,
critic, BM25 run, or negative-pair mining is required.

## Dependencies

Use the existing `ddro_env.yml` training environment. The implementation is
tested with PyTorch 2.5.1, Transformers 4.45.2, Accelerate 1.1.0, and Datasets
3.1.0. No additional GRPO library is required. The separate `requirements.txt`
contains older/inconsistent pins; prefer the conda environment for this path.

[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) documents causal
model support. This implementation uses Transformers `Trainer` for optimizer,
scheduler, gradient accumulation, mixed precision, DDP, evaluation, and resumable
checkpoints, with a custom encoder-decoder GRPO loss and rollout path.

## Prepare data

The **URL launcher** now reads a single native file with `text`, `url_based_id`,
and `positive_url_based_id`; see the URL workflow below. Metadata preparation in
this section remains available for structure-v6 and the other existing presets.

Reuse `query_metadata.jsonl` and `document_metadata.jsonl` from the existing
Vault preparation stage. Creating these metadata files does not require building
a BM25 index. Structure-v6 metadata must already have the correct URL-based
cross-build mappings (see the Vault BM25 README).

```bash
python src/scripts/grpo/prepare_vault_grpo_data.py \
  --query-metadata /data/vault/query_metadata.jsonl \
  --document-metadata /data/vault/document_metadata.jsonl \
  --target-type structure_id_v6 \
  --output-dir /data/vault/grpo
```

Supported target types: `text_id`, `url`, `structure_id_v3`, `structure_id_v6`.
The script unions all mapped and explicitly annotated positives, consolidates
repeated normalized queries before splitting, and writes:

- `train.jsonl` and `validation.jsonl`: `{"prompt":"find a","positive_targets":["A","B"]}`
- `corpus.jsonl`: one `{"target":"A"}` record per distinct corpus decoder target
- `prepare_stats.json`: counts and split configuration

You can also supply these formats directly without running preparation. The
corpus must represent the **full retrieval universe**, including negative IDs,
not just positives. Same-target documents share a retrieval action. The generic
Python entry points default to strict target validation. The URL launcher uses
the documented skip policy below for unusable IDs. Unmapped positive IDs and
train/validation query overlap remain data-validation errors.

Validation labels must be appropriate to that split; the trainer does not infer
new positives from held-out examples. Separate train/eval files are checked for
normalized query overlap and overlap after prompt tokenization/truncation.

## Train or inspect rollouts

```bash
python src/pretrain/train_grpo_vault.py \
  --checkpoint_path /models/ruby-structure-v6-sft/checkpoint-N \
  --train_file /data/vault/grpo/train.jsonl \
  --eval_file /data/vault/grpo/validation.jsonl \
  --corpus_file /data/vault/grpo/corpus.jsonl \
  --output_dir outputs/vault-grpo \
  --max_target_length 128 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --fp16
```

Set `max_target_length` to your actual SFT decoder limit; it includes EOS and
excludes decoder start. Append `--dry_run` to validate inputs and sample one
query batch without optimization. For a short training smoke run, add
`--max_steps 2 --save_steps 1 --eval_steps 1 --logging_steps 1`.

Batch size is **queries per device**: one query with eight generations yields
eight trajectories. Each complete group stays on its local DDP rank. For V100,
use FP16 or FP32, not BF16. Reduce query batch size or generation count for memory.

### URL checkpoint on four V100s

The URL preset reads your native multi-label training file, trains GRPO, then
runs test inference. No numeric/semantic-ID joins or metadata files are needed:

```bash
bash src/scripts/grpo/train_vault_url_4v100.sh
```

Its defaults are:

| Setting | URL GRPO preset |
| --- | --- |
| SFT checkpoint | `/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr_sft_3_5/checkpoint-96600` |
| Training input (`TRAIN_INPUT_FILE`) | `/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_merged_multilabel_dpo.jsonl` |
| Test input (`TEST_INPUT_FILE`) | `/home/users/congthanh_le/scratch/veil/CodeGR/data/updated_url/Ruby_test_r32.0_multilabel.json` |
| Prepared GRPO data | `${OUTPUT_DIR}/grpo_data` |
| Output directory | `/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/vault-ruby-grpo-url` |
| GPUs / precision | 4 / FP16 |
| Prompt / target limits | 256 / 128 tokens |
| Invalid target policy | `skip` for calibration, training, and inference |
| Starting training queries per GPU | 16, then automatically calibrated |
| Maximum training queries per GPU to probe | 32 |
| Training trajectories per query | 8 deterministic beam candidates (`DO_SAMPLE=false`) |
| Calibration memory reserve | 4 GiB, or at least two parameter-sized buffers for DDP |
| Gradient accumulation | 1 |
| Effective full optimizer batch | `4 * calibrated_batch` queries, eight trajectories each |
| Evaluation queries per GPU / beams per query | 2 / 10 |
| Epochs / validation fraction | 2 / 0.01 |
| Learning rate / KL beta | `1e-6` / `0.04` (frozen SFT reference enabled) |
| Evaluation / save interval | 2000 / 2000 optimizer steps |
| DataLoader workers per process | 2 |
| Post-training inference | enabled, on `${OUTPUT_DIR}/final` |
| Inference queries per GPU / beams per query | 8 / 10 |
| Inference GPUs | 4 (defaults to `NUM_GPUS`) |

Each source record has this relevant shape (other fields are ignored):

```json
{"text":"partition a predicate by operand type","url_based_id":"repo/path/partition(predicate)","positive_url_based_id":["repo/path/partition(predicate)","repo/other/equivalent(predicate)"]}
```

`text` becomes the query. The explicitly supplied `positive_url_based_id` is
the authoritative positive set; the row's own URL is not automatically added
to that set. Both scalar and list-valued URL fields are accepted. A missing or
empty positive set is an error. Optional `Query:` prefixes are removed;
`Code:` rows contribute document URLs to the catalog but are excluded from
query training and scoring. Duplicate normalized queries have their positive
sets unioned before splitting. JSON arrays and JSONL files are both supported.

`prepare_vault_url_grpo_data.py` creates canonical train/validation rows and a
corpus containing every row URL plus the training positive URLs from that same
file. `original.code`, numeric IDs, semantic IDs, and structure IDs are not used.
You can run preparation alone with no checkpoint or GPU:

```bash
RUN_DRY_RUN=0 RUN_TRAINING=0 RUN_INFERENCE=0 \
bash src/scripts/grpo/train_vault_url_4v100.sh
```

The URL preset starts from DDRO's batch of 16 queries per GPU, but GRPO expands
each query into eight trajectories: batch 16 means 128 trajectories per GPU.
`AUTO_TUNE_BATCH_SIZE=1` runs a GPU capacity probe before starting DDP. It can
increase or decrease that batch and finds the largest integer batch that passes
the probe up to `MAX_TRAIN_BATCH_SIZE=32`. Raise the cap to search beyond 32.
The selected batch and probe results are saved in `OUTPUT_DIR/batch_size_probe.json`.

The probe runs on the least-free of the requested training GPUs. It uses full
256-token prompts, the longest corpus target within the configured decoder
limit, cached generation, gradient checkpointing, backward, and two AdamW steps.
It also includes Accelerate's FP32 output conversion for FP16 training.
The second step accounts for resident optimizer state. It reserves at least
`BATCH_SIZE_RESERVE_GIB=4` GiB in the URL preset (and at least two parameter-sized
buffers for DDP). The shared/structure-v6 preset defaults to 2 GiB.
Probe weights are temporary; the SFT checkpoint is never modified. Probing
starts fresh on each attempted batch and happens before launching distributed
workers, so an OOM retry does not require recovering a failed DDP group.

### Skipping unusable URL IDs

The URL preset exports `GRPO_INVALID_TARGET_POLICY=skip`. The same tokenizer
validation is applied during memory calibration, training, and test inference:

- IDs exceeding `MAX_TARGET_LENGTH`, including EOS, are excluded in full.
  A 195-token URL is skipped when the limit is 128; it is never truncated.
- All members of each tokenizer-collision group are excluded; an arbitrary
  spelling is not kept as a representative.
- Unknown/PAD tokens, malformed EOS/BOS placement, tokenization errors, and
  control/surrogate/replacement characters are excluded. Ordinary Unicode and
  punctuation remain allowed when the tokenizer can represent them correctly.

Training and internal validation remove excluded positives from each query.
Queries with no usable positives left are dropped; a multi-positive query with
at least one usable positive is retained. If every validation query is dropped,
internal validation is disabled. The run fails clearly if no usable corpus or
training query remains. Missing fields or structurally malformed input data are
still errors; this policy concerns document-ID representability.

Final test inference keeps the original queries and positive label sets. An
excluded target cannot be predicted, but its query stays in the metric
denominator. This prevents target filtering from silently improving scores by
dropping difficult test cases. Metrics report missing targets and queries with
no retrievable positives.

Audit files:

- `${OUTPUT_DIR}/target_filtering.json` and `excluded_targets.jsonl`: training
  corpus counts, exclusion reasons, and exact excluded IDs.
- `${OUTPUT_DIR}/query_filtering.json`: retained/dropped train and validation
  query counts and removed positive-label counts.
- `${OUTPUT_DIR}/calibration_target_filtering/`: the calibration target report.
- `${INFERENCE_OUTPUT_DIR}/target_filtering.json` and `excluded_targets.jsonl`:
  the inference catalog's target report.

Use `GRPO_INVALID_TARGET_POLICY=error` to restore strict URL-launcher behavior.
Direct Python training, calibration, or inference accepts
`--invalid_target_policy skip`; its default is `error` unless the environment
variable is set. Prepared URL files preserve malformed strings using JSON
escaping so the tokenizer validation stage can audit them safely.

This is a memory stress estimate, not a guarantee against changes in GPU
occupancy or other runtime allocations. All training GPUs should have the same
model/hardware configuration. The calibration does not tune evaluation batch
size; it stays at two queries and ten beams. If needed, lower `EVAL_BATCH_SIZE`
separately, or increase the reserved margin. Calibration runs only when
`RUN_TRAINING=1`; a dry-run-only launch uses the requested batch unchanged.

For a fixed batch with no calibration, use:

```bash
AUTO_TUNE_BATCH_SIZE=0 TRAIN_BATCH_SIZE=16 \
bash src/scripts/grpo/train_vault_url_4v100.sh
```

With batch 16 and four GPUs, accumulation 1 yields 64 queries / 512 trajectories
per optimizer step. The startup summary reports the actual values after tuning.
The starting learning rate is lower than DDRO's `2e-5` for a policy-gradient
pilot. DPO's `beta=0.4` is not transferred: GRPO's beta controls the optional KL
penalty. Every setting remains overridable through environment variables.

For the three-SFT/two-GRPO pseudo-query experiment, set `TRAIN_INPUT_FILE` to a
native file containing only the two post-training queries, with their correct
positive URL sets. The launcher does not select those variants automatically.

To run a short pilot before the full job:

```bash
MAX_STEPS=20 EVAL_STEPS=10 SAVE_STEPS=10 \
OUTPUT_DIR=/path/to/url-grpo-pilot \
bash src/scripts/grpo/train_vault_url_4v100.sh
```

### URL inference and metrics

After training succeeds, `RUN_INFERENCE=1` loads `final/` and evaluates
`TEST_INPUT_FILE` with deterministic trie-constrained beam search. Test queries
and positive sets are never used in GRPO training or checkpoint selection.
Inference uses the prepared training catalog plus the test rows' own
`url_based_id` entries (including document-only `Code:` records). Test positive
label sets do not add candidates. Labels missing from this candidate catalog
remain in the evaluation and are reported in `metrics.json`; such missing
targets cannot be retrieved. The files must collectively identify the intended
retrieval universe.

The output directory defaults to `${OUTPUT_DIR}/inference`:

- `metrics.json`: `hit1`, `hit5`, `hit10`, `mrr1`, `mrr5`, `mrr10`, query counts,
  corpus coverage, model path, and decoding settings.
- `predictions.jsonl`: each unique query's positives, top-1/top-10 predictions,
  and its individual metric values, in input query order.

Hit@K is 1 if **any** positive is in the first K unique predictions. MRR@K
uses the reciprocal rank of the first positive if its rank is at most K,
otherwise 0. Dataset metrics are means over unique normalized queries, reported
as fractions in [0, 1]. Consequently MRR@1 equals Hit@1.

Inference defaults to the same number of GPUs as training and shards queries
across workers without duplicates. `INFERENCE_NUM_GPUS=1` selects a single GPU.
`INFERENCE_BATCH_SIZE`, `INFERENCE_NUM_BEAMS` (at least 10), `INFERENCE_MODEL_PATH`,
and `INFERENCE_OUTPUT_DIR` override inference settings. `RUN_INFERENCE=0` skips
post-training inference. When `RUN_TRAINING=0`, inference is off by default;
explicitly enable it for an inference-only run:

```bash
RUN_PREPARE=0 RUN_DRY_RUN=0 RUN_TRAINING=0 RUN_INFERENCE=1 \
bash src/scripts/grpo/train_vault_url_4v100.sh
```

Or call inference directly (CPU supported when `--fp16` is omitted):

```bash
python src/pretrain/infer_grpo_vault_url.py \
  --model_path /models/url-grpo/final \
  --test_file /data/Ruby_test_r32.0_multilabel.json \
  --corpus_file /models/url-grpo/grpo_data/corpus.jsonl \
  --output_dir /models/url-grpo/inference \
  --batch_size 8 --num_beams 10 --max_target_length 128 --fp16
```

### Structure-v6 checkpoint

The existing structure-v6 entry point remains available and shares the same
preparation/launch flow. It retains its earlier batch and interval defaults.
Set `NUM_GPUS=1` for one GPU:

```bash
CHECKPOINT_PATH=/models/ruby-structure-v6-sft/checkpoint-N \
MAX_TARGET_LENGTH=128 \
WORK_DIR=/data/vault_bm25_structure_v6 \
OUTPUT_DIR=/models/ruby-grpo-v6 \
bash src/scripts/grpo/train_vault_structure_v6_4v100.sh
```

Both presets prepare data, run a rollout dry run, then train. Set
`RUN_PREPARE=0`, `RUN_DRY_RUN=0`, or `RUN_TRAINING=0` to disable stages.
Preparation-only (`RUN_DRY_RUN=0 RUN_TRAINING=0`) requires neither CUDA nor a
checkpoint. A dry-run-only job requires one GPU; training requires `NUM_GPUS`.
For structure-v6, `QUERY_METADATA` and `DOCUMENT_METADATA` override source paths.
For URL data, use `TRAIN_INPUT_FILE` and `TEST_INPUT_FILE`. `GRPO_DATA_DIR`
overrides the prepared-data directory in both workflows.
`TRAIN_FILE`, `CORPUS_FILE`, and `EVAL_FILE` can override prepared inputs when
`RUN_PREPARE=0`; explicitly set `EVAL_FILE=` to disable validation. An empty
validation file from `VALIDATION_SPLIT=0` also disables validation. A missing
configured validation file fails instead of silently disabling evaluation.
Use `GRPO_PYTHON_BIN` (or the existing `PYTHON_BIN` alias) to choose Python.
`NUM_GENERATIONS`, `TRAIN_BATCH_SIZE`, `TEMPERATURE`, `GRPO_BETA`, `EVAL_NUM_BEAMS`,
`MAX_STEPS`, and other variables in the script expose common training settings.
The startup summary reports query and trajectory counts per GPU and optimizer
step. `DATASET_NUM_PROC` from the DDRO script is not forwarded: GRPO preparation
uses its own query consolidation and the trainer has no such CLI argument.

Resume with `--resume_from_checkpoint outputs/vault-grpo/checkpoint-N` or the
launcher's `RESUME_FROM_CHECKPOINT` variable. HF checkpoints contain optimizer,
scheduler, trainer and RNG state. The final portable model/tokenizer is saved in
`OUTPUT_DIR/final/`. `grpo_run.json` records the launch configuration.

For example, resume URL GRPO with the same prepared data and output directory:

```bash
RUN_PREPARE=0 \
AUTO_TUNE_BATCH_SIZE=0 TRAIN_BATCH_SIZE=16 \
RESUME_FROM_CHECKPOINT=/path/to/url-grpo/checkpoint-2000 \
OUTPUT_DIR=/path/to/url-grpo \
bash src/scripts/grpo/train_vault_url_4v100.sh
```

`CHECKPOINT_PATH` still points to the original SFT model; the resume path points
to a GRPO checkpoint containing optimizer/trainer state. Replace batch 16 in the
resume example with the batch recorded in that run's `grpo_run.json` or
`batch_size_probe.json`. Reusing the same batch preserves the data-loader resume
position; use calibration for a new run rather than changing the batch mid-run.

## Deterministic training rollouts

The URL launcher now defaults to `DO_SAMPLE=false`. It uses deterministic
constrained beam search with `num_beams=NUM_GENERATIONS` and returns that many
candidates per query, rather than repeating greedy decoding. For example:

```bash
DO_SAMPLE=false NUM_GENERATIONS=8 \
bash src/scripts/grpo/train_vault_url_4v100.sh
```

`DO_SAMPLE=true` restores independent sampled GRPO. The shared and structure-v6
launchers still default to sampling. Direct Python training and calibration
accept `--do_sample false` (default: `true`). Dry runs and memory calibration
use the selected training mode, and run manifests record it. Calibration should
be rerun for a new beam experiment because generation memory differs. Use a new
`OUTPUT_DIR` to compare it with an existing sampled run.

Beam rollouts retain binary multi-positive rewards and **signed** group
advantages. They are an experimental selection-biased variant, not standard
on-policy sampled GRPO; the token-level ratio does not correct beam selection.
Temperature is ignored for beam decoding and scoring uses temperature 1.
Beam scores use `length_penalty=0`. `NUM_GENERATIONS` must not exceed the usable
corpus target count. The separate evaluation/inference beam settings are unchanged.
Beam candidates can still all receive the same reward, producing zero relevance
advantages, and the centered loss can still log zero with nonzero gradients.

Generation explicitly enters Accelerate's autocast context, including standalone
dry runs before the training model is prepared. With `PRECISION=fp16` this keeps
dry-run beam generation in mixed precision, consistent with calibration. Older
versions could run this stage in FP32 and exhaust memory at a calibrated batch.
If memory remains insufficient, reduce the per-device query batch (for example,
`AUTO_TUNE_BATCH_SIZE=0 TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=4` preserves
the effective query batch of batch 16 with accumulation 1). For a new run you can
instead rerun calibration with a larger `BATCH_SIZE_RESERVE_GIB` for more headroom
(the URL preset defaults to 4 GiB).

## Objective and diagnostics

The objective uses group-centered rewards divided by group population std plus
`1e-4`, the clipped token-level policy surrogate, and trajectory-then-batch
averaging. In sampling mode it uses fresh on-policy rollouts per training
microbatch, equivalent to the **single-iteration GRPO regime**. In beam mode the
same surrogate is applied to deterministically selected candidates. Old-policy log probabilities are the
detached current-policy scores because no update occurs between sampling and
scoring; there is no stale rollout cache or multi-iteration replay. Consequently,
ratios start at one and clipping does not activate in this initial version.
The scalar policy loss can be zero even while its gradient is nonzero, because
group-centered advantages sum to zero. This remains true with deterministic
beam rollouts: changing candidate generation does not change that cancellation.
`loss=0` alone does not diagnose a broken gradient or successful learning.
Track reward, mixed groups, gradient norm, and held-out retrieval metrics.

Two additional diagnostics are logged without changing the optimized loss:

- `grpo/pg_loss`: `-mean(advantage * mean_valid_token_logp)` on the current
  candidates. Its derivative matches the policy term in this single-iteration
  regime, but it is logged detached. It can have either sign and need not
  decrease across changing rollout groups; it is not a retrieval-quality metric.
- `grpo/advantage_abs_mean`: average absolute normalized advantage. Zero means
  no group-relative relevance signal in that logging window (KL may still act).

Regression tests verify that a zero surrogate has the expected nonzero gradient
for mixed reward groups with unequal sequence lengths, and that a real tiny-T5
update increases an annotated positive target's probability while decreasing the
combined probability of its negative candidates. This checks the implementation,
not the retrieval performance of a GPU training run.

Sampling has no top-k/top-p truncation, repetition penalty, or forced-token
processors. Scoring uses exactly the same trie-normalized distribution and
temperature. Dropout is disabled, including T5 attention's functional dropout.
The first EOS is included in the loss; decoder start and padding are excluded.
The trie validates that every corpus ID can finish within the decoder limit.

`beta=0` (Python/shared-launcher default) loads only the policy. The URL preset
defaults to `GRPO_BETA=0.04`, adding a frozen copy of the initial SFT checkpoint
for the KL penalty, using the same
constrained distribution. This reference is **not a reward model**. DeepSpeed,
FSDP, vLLM, multiple optimization iterations per rollout, auxiliary SFT losses,
partial-prefix rewards, and diversity rewards are outside this implementation.

All-zero/all-one groups have zero relevance advantages and keep a connected
zero loss so DDP can run backward. With nonzero beta, KL can still contribute.
There is no positive injection, reward-based resampling, or duplicate removal
during sampling. Monitor:

- `grpo/reward`, `grpo/mixed_reward_groups`, `grpo/all_zero_groups`, `grpo/all_one_groups`
- `grpo/duplicate_fraction`, `grpo/distinct_positives`
- `grpo/completion_length`, `grpo/truncated_fraction`, `grpo/kl`

Evaluation uses deterministic constrained beam search with raw sequence
log-probability scores (`length_penalty=0`), reports `eval_mrr@10` and
`eval_recall@k`, and selects the best checkpoint by MRR@10. K is
`min(eval_num_beams, corpus_target_count)` and is recorded in `grpo_run.json`.
Recall counts all annotated positive target IDs, not document multiplicity.
Changing decoding settings can affect comparisons to older evaluation scripts;
use identical decoding for SFT/DPO/GRPO comparisons.

Tests (CPU, no network or downloaded checkpoints):

```bash
python -m unittest discover -s src/pretrain/tests -p 'test_grpo*.py' -v
```

Validation includes offline tiny-T5 training/evaluation, variable-length rollout
probability checks, and resume equivalence to uninterrupted training. A separate
two-process CPU DDP smoke run also passed. CUDA/FP16 and four-V100 execution must
be validated on GPU hardware.
