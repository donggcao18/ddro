# Iterative Vault DPO

This experiment runs entirely in `ddro_env`. It uses model-confusion negatives
only and never invokes BM25, Pyserini, Java, or the hybrid combiner. The existing
single-run and hybrid launchers remain available.

## Run

Copy `src/scripts/configs/iterative_vault_dpo.json`, edit its input/output paths,
and activate the training environment. Paths in the JSON are relative to that
JSON file unless absolute.

```bash
conda activate ddro_env
python src/pretrain/train_iterative_ddro_vault.py --config /path/to/experiment.json
# Or:
CONFIG=/path/to/experiment.json bash src/scripts/ddro/run_vault_iterative_dpo.sh
```

For the dedicated EMA experiment, edit the paths in
`src/scripts/configs/iterative_vault_dpo_ema.json`, then run:

```bash
conda activate ddro_env
CUDA_VISIBLE_DEVICES=0,1 bash src/scripts/ddro/run_vault_iterative_dpo_ema.sh
# Continue the same EMA experiment:
CUDA_VISIBLE_DEVICES=0,1 bash src/scripts/ddro/run_vault_iterative_dpo_ema.sh --resume
```

This config uses two GPUs for training and mining, one epoch per round, and
`reference_update: "ema"` with decay `0.9`: the next reference is 90% of the
previous reference plus 10% of the just-trained policy. Results go to
`vault-iterative-dpo-ema-d09`. Start a separate EMA run when switching from an
existing replacement experiment; changing the reference rule is not a
resume-compatible setting. `CONFIG` and `PYTHON_BIN` overrides work as in the
regular launcher; a custom `CONFIG` must itself specify EMA.

To keep the SFT reference weights frozen while refreshing negatives from the
latest policy each round, use the dedicated frozen-reference config/launcher:

```bash
conda activate ddro_env
CUDA_VISIBLE_DEVICES=0,1 bash src/scripts/ddro/run_vault_iterative_dpo_frozen_ref.sh
# Resume this same frozen-reference run:
CUDA_VISIBLE_DEVICES=0,1 bash src/scripts/ddro/run_vault_iterative_dpo_frozen_ref.sh --resume
```

Edit paths in `src/scripts/configs/iterative_vault_dpo_frozen_ref.json` before
running. It uses `reference_update: "ema"`, `reference_ema_decay: 1.0`, two GPUs
for mining/training, and training batch size 16 per GPU. For this T5 experiment,
decay 1.0 keeps the reference weights at the initial SFT values; policy training
and policy-based negative mining continue normally. The existing EMA machinery
still loads and exports reference snapshots each round. Results go to the
separate `vault-iterative-dpo-frozen-ref` directory. Start fresh when switching
from replacement or decay-0.9 EMA; do not resume those runs with this config.
`CONFIG` overrides must themselves retain EMA decay 1.0.

To inspect metadata and splits before GPU work:

```bash
python src/pretrain/train_iterative_ddro_vault.py --config /path/to/experiment.json --prepare-only
python src/pretrain/train_iterative_ddro_vault.py --config /path/to/experiment.json --resume
```

Always launch the controller with `python`, not `torchrun`. Set `num_gpus` for
single-node distributed training; the controller launches and joins each
training process group. Set `mining_num_gpus` for mining and evaluation; when
omitted or null, it defaults to `num_gpus`. Each GPU loads the same policy and
processes a disjoint query shard. These processes exit before training starts,
so their model memory is released before training. `fp32` with
`device: cpu` supports small smoke tests; mixed precision requires CUDA.

For two GPUs, use the following settings in your existing config:

```json
"num_gpus": 2,
"mining_num_gpus": 2,
"mining_batch_size": 64
```

`mining_batch_size` is **per GPU**. Beam width and negative quotas are unchanged.
For N GPUs, set `mining_num_gpus: N` and expose at least N GPUs through
`CUDA_VISIBLE_DEVICES`. Multi-GPU mining requires `device: auto` or `cuda` and
uses logical devices `cuda:0` through `cuda:N-1` within that visible list.
For example, `CUDA_VISIBLE_DEVICES=2,3` selects physical GPUs 2 and 3. Set
`mining_num_gpus: 1` to retain the original single-device mining path.

Shards preserve the original contiguous batches and merge in original query
order. A small dataset may use fewer workers than requested if it has fewer
batches. Evaluation metrics are weighted by processed query count; shared
corpus statistics are not multiplied by the number of workers.

Per-worker progress is in `round-000/preferences.shards/gpu-000/worker.log`
(and `gpu-001`, etc.). Evaluation uses `validation_predictions.shards/`.
The merged preference file remains `round-000/preferences.jsonl`.
Completed shards have hash-checked `complete.json` markers and are reused
after interruption. Failed or incomplete shards rerun; training cannot start
with a partially merged preference dataset. Changing the GPU count during an
unfinished mining stage regenerates that stage's shards.

To upgrade an existing run, stop the running job, copy both
`src/pretrain/train_iterative_ddro_vault.py` and `src/pretrain/mine_on_gpus.py`,
add `mining_num_gpus` to the existing config, and restart with `--resume`.
Keep training settings (including `num_gpus`), data, and output paths unchanged.
The known preceding controllers are supported; completed stages and model
checkpoints are preserved. Arbitrary code changes still fail resume checks.

## Evaluation frequency and resuming an existing run

### Four 32 GB V100 GPUs on one node

Use `src/scripts/configs/iterative_vault_dpo_ema_v100.json` and the dedicated
launcher after allocating four GPUs on the same node:

```bash
conda activate ddro_env
CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/scripts/ddro/run_vault_iterative_dpo_ema_v100.sh
# Resume this V100 experiment:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/scripts/ddro/run_vault_iterative_dpo_ema_v100.sh --resume
```

If your scheduler already sets `CUDA_VISIBLE_DEVICES`, preserve its allocation
and simply run the bash command without overriding that variable. Launch one
controller, not four copies of the script. This is a single-node configuration;
four GPUs spread across multiple nodes require a different distributed launcher.

V100 uses FP16 rather than the previous BF16 setting. Training uses mixed
precision with gradient scaling; mining loads FP16 model weights. EMA reference
averaging still uses FP32 on CPU. The config retains EMA decay 0.9, beam width
10, one epoch per partition, and the evaluation schedule. The initial batch
sizes target 32 GB V100s: 8 training pairs per GPU with accumulation 1 gives an
effective batch of 32 (matching 2 GPUs x 16 pairs x accumulation 1); mining uses
16 queries per GPU. These are starting settings, not a guarantee of memory fit
for every checkpoint. The existing controller already enables gradient
checkpointing. For more training headroom, 4 pairs/GPU with accumulation 2 also
gives effective batch 32. If FP16 produces non-finite losses or scores for your
checkpoint, use FP32 with smaller batches in a separate run.

Edit checkpoint/query paths for the new cluster. Results default to
`vault-iterative-dpo-ema-v100-fp16-d09`. Start a fresh experiment: an old
two-GPU BF16 run cannot use this config with `--resume`, since GPU count,
precision, and training batch settings are part of its saved identity.
Once this V100 run exists, `--resume` and `--start-round` work normally.

The example config evaluates after every eight completed training epochs:

```json
"eval_every_epochs": 8,
"eval_at_end": true
```

With `epochs_per_round: 1`, scheduled evaluation runs after zero-based rounds
7, 15, 23, and so on. The final round is also evaluated; set `eval_at_end: false`
to omit an extra final evaluation before the next eight-epoch boundary. These
are epochs over each round's preference partition, not eight passes over the
whole original query dataset. Empty rounds add no training epochs. With longer
rounds, evaluation occurs at the first round boundary that crosses each
eight-epoch threshold. `eval_every_epochs: null` (the default when omitted)
preserves evaluation after every round. A numeric interval requires epoch-based
rounds. The SFT baseline still runs once and is reused on resume.

To change the schedule while an old run is in `evaluate-0`:

1. Stop the current evaluation/job before starting another controller.
2. Update `src/pretrain/train_iterative_ddro_vault.py`, copy the new
   `src/pretrain/mine_on_gpus.py`, and add the two settings
   above to your existing config. Keep all existing training settings and paths,
   including the same `output_dir`.
3. Restart the launcher with `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0,1 CONFIG=/path/to/your/existing-config.json \
  bash src/scripts/ddro/run_vault_iterative_dpo.sh --resume
```

This release accepts the immediately preceding controller's fingerprints and
evaluation-schedule/mining-concurrency changes during resume. It records the prior identity in
`run_manifest.json` under `identity_updates`; it does not rewrite round input
manifests, preferences, checkpoint completion markers, or model weights.
Training settings, source files, other runtime code, package versions, and
checkpoint/artifact integrity are still checked. Unrelated older or modified
controller versions are not automatically accepted.

Completed evaluations are reused even when they precede the new interval.
An interrupted evaluation that is no longer due is bypassed; its round's
completed policy remains the starting policy for the next round. Deferred
evaluations have `metrics: null` and `evaluation_status: "deferred"` in the
run manifest. Best-checkpoint selection uses only the baseline and completed
evaluations; training always proceeds from the latest trained policy.

## Round semantics

1. Fix train/validation query-family membership once.
2. Evaluate the SFT checkpoint for a retrieval baseline.
3. Shuffle the training queries once using the configured seed, partition the
   entire shuffled dataset into rounds, and persist each round's queries before
   generation. The final partition can be smaller; queries do not repeat.
4. Generate candidates from the round-start checkpoint and write immutable
   preferences. Every verified positive is excluded from rejection, even when
   the correct document is not generated. Incorrect lower-ranked candidates
   are used even if top-1 is correct. Use one chosen target per query row.
5. Initialize the policy from its latest checkpoint and load the round's
   reference snapshot separately. Freeze the reference, reset optimizer/scheduler,
   and train on the fixed preferences. Both start from SFT in the first round.
6. Export the latest policy, update the reference for the next round, evaluate
   fixed validation queries when the evaluation schedule is due, and repeat.

The example config uses `reference_update: "ema"` and `reference_ema_decay: 0.9`:

```text
reference_next = decay * reference_previous + (1 - decay) * policy_after_training
```

With decay 0.9, each update retains 90% of the previous reference and adds 10%
of the newly trained policy's weights. This is a parameter EMA applied once
after each trained round, never per minibatch. It is a tunable experiment choice,
not a guarantee of better retrieval; a larger decay makes the reference lag
further behind the policy. Set `reference_update: "replace"` for the previous
direct replacement behavior (also the default when the option is omitted).
Decay 0 is direct weight replacement; decay 1 retains the floating-point
reference weights. Empty rounds keep both policy and reference unchanged.

EMA runs in a separate CPU process after distributed training finishes. It
loads both checkpoints in FP32, blends each unique parameter once (including
tied T5 embeddings), and atomically exports `round-NNN/reference/`. Floating
buffers are averaged and nonfloating buffers copied from the policy. This
requires CPU RAM for two FP32 models and extra disk space per round, but no
additional GPU model during training. Policy and reference checkpoint paths
and fingerprints are recorded separately in each round manifest. Generation
always uses the latest policy, not the EMA reference.

Parameter mixing for DPO references is also supported by
[TRL's reference synchronization options](https://huggingface.co/docs/trl/dpo_trainer).
Here the update stays at round boundaries to preserve fixed references within
each round.

Reference log-probabilities are recomputed during training; reference caching
and automatic in-round reference synchronization are disabled. In iterative
T5 training, functional attention dropout is also disabled alongside TRL's
module dropout handling, so identical policy/reference copies start with
matching likelihoods even while the policy is in train mode. Under EMA, later
rounds have different policy/reference weights, so their starting DPO loss
need not be log(2).

Omit `rounds` (or set it to `null`) to derive the round count from the complete
prepared training dataset:

```text
number_of_rounds = ceil(training_query_count / queries_per_round)
```

For 25,000 training queries and `queries_per_round: 10000`, the partitions are
10,000, 10,000, and 5,000 queries. Every prepared training query is selected
exactly once, including queries that ultimately produce no negatives. The
last partition never wraps back to the beginning. `queries_per_round: null`
puts all training queries into one round. Validation remains held out; counts
and partition sizes are saved in `round_plan.json`, also by `--prepare-only`.

An explicit positive `rounds` value retains the old fixed-round scheduling,
which uses rotating windows and can stop before full coverage or repeat queries
after wrapping. The example config omits this limit and uses one epoch per
partition. The shuffle is deterministic and does not change on resume.

Each query supplies up to `negatives_per_query` unique negatives. Shortfalls are
retained; queries with zero negatives are skipped and counted. An entirely
empty preference set skips training for that round, retains the current policy,
and continues to the next round's queries. The skip is recorded in
`round_skipped.json` and the run manifest with zero optimizer updates; it can
be resumed without remaking the completed mining data. There is no replay accumulation.

`steps_per_round` counts **optimizer updates**, including gradient accumulation.
The trainer reshuffles and repeats the generated dataset as needed. Nominal
pair presentations are steps × per-device batch size × GPU count × accumulation;
partial batches and distributed sampler padding can change the exact count.
`pair_audit.json` reports the resulting nominal number of dataset passes so an
undersized query subset is visible. For epoch-based rounds, replace
`steps_per_round` with `epochs_per_round`; specify exactly one budget.

The example config uses `epochs_per_round: 1`, `num_beams: 10`, and up to four
negatives per query. Each round trains for one pass over its actual mined pairs;
distributed sampler padding may repeat a few pairs. One, two, or three usable
negatives are all accepted; queries with zero contribute no pairs. The negative
quota is an upper bound and may exceed the beam count without causing an error.
The partition size remains `queries_per_round: 10000`; the round count follows
the training dataset size automatically.

Ordinary uniform pair sampling is used. Queries with more surviving negatives
have proportionally greater training weight; quotas bound this imbalance and
the audit records the distribution. Round budgets do not promise each pair is
used exactly once or that each selected query appears in a short step budget.

## Inputs and labels

The example config uses the merged multilabel input directly:

```json
{
  "input_format": "multilabel",
  "target_type": "url_based_id",
  "query_file": "/path/to/Ruby_merged_multilabel.jsonl"
}
```

For `target_type: "A"`, each row must contain the source ID in column `A`
and a nonempty list of positive ID strings in `positive_A`. For this experiment:

```json
{"text":"find this function", "url_based_id":"repo/lib/file.rb/function(arg)", "positive_url_based_id":["repo/lib/file.rb/function(arg)","repo/lib/other.rb/equivalent(arg)"]}
```

IDs are read verbatim from those columns, with no numeric/semantic/structure
mapping. Every row keeps its own positive list, even if another row has the
same prompt or source document. Duplicate IDs inside a list are removed. The
chosen target is the source ID when it is explicitly positive, otherwise the
lexicographically first listed positive. All listed positives are excluded
from negatives. Missing/empty labels fail preparation; the source ID is never
silently added as a positive. Rows with the same normalized prompt, source ID,
and positive set are deduplicated; different positive lists remain separate. Conflicting lists for the
same prompt remain authoritative; check their consistency in the source data.

`corpus_files` can be omitted. Candidate IDs are the union of source IDs and
all listed positives, including documents without their own query row.
The original training file, code bodies, and BM25 are unnecessary. To search
a larger collection, optional `corpus_files` may supply additional rows with
column `A`; their query labels are ignored. Without them, evaluation/mining
searches only the collection represented in the merged file. Candidate IDs
are shared with validation, but validation queries and positive lists are
never used to construct training preferences.

Prepared metadata retains internal names such as `text_id`, `positive_text_ids`
and `chosen_text_id`; in this mode their values are the selected IDs themselves
(URLs here), not mapped semantic IDs. `doc_id_type` records their namespace.
Use a checkpoint SFT-trained to generate that same ID type. The example config
uses `max_target_length: 128` and `target_length_policy: "truncate"`. Mining and
training both use the tokenizer's truncation with special tokens, retaining EOS
within the limit. Original full URL strings remain in metadata and preference
JSONL. The constrained decoder's shortened token sequence is mapped back to
the full ID for mining and retrieval evaluation. Queries are not dropped just
because their target is long. Truncation means learning a shortened representation of
the URL, not generating its complete text. Log files named
`*.excluded_text_ids.json` include `truncated_targets` and the original token
lengths in `overlength_targets`; statistics also report the truncation count.

`target_length_policy: "error"` keeps the original fail-on-overlength behavior
(the default when omitted). `"skip"` excludes long IDs from candidate generation
and skips training queries with a long chosen target. Evaluation in skip mode
retains all query labels and counts unreachable long positives as misses.
The example config also sets `target_collision_policy: "allow"` and
`target_token_policy: "allow"`. These make token collisions and existing
unknown/padding tokens non-fatal in both mining and DPO training. The tokenizer's
output is retained, not rewritten by substituting other tokens. Normal batch
padding is separate from padding tokens occurring inside the encoded target.
The latter are retained through EOS when looking up generated sequences.
Counts of unknown/padding target sequences are saved in generation statistics.

Under collision policy `allow`, different full IDs sharing one token sequence
are retained as aliases. Generation uses the lexicographically first full ID
as that sequence's representative. All aliases of every positive are excluded
from negatives, so indistinguishable sequences are not trained against each
other. Per-query positive lists and chosen full IDs remain unchanged. Evaluation
scores the deterministic representative against the original full-ID labels;
`predicted_text_id_groups` records all aliases for each generated sequence.
The model cannot distinguish aliases using the shortened/unknown-token encoding.
Query variations for the same document were already supported and are unaffected.

For stricter runs, `target_collision_policy: "error"` rejects collisions;
`"skip"` excludes them. `target_token_policy: "error"` restores rejection of
unknown/padding tokens. Both default to `error` when omitted.

Use a new output directory when changing data, ID type, or length policy;
existing rounds cannot be resumed with different inputs.

Legacy inputs remain supported with `input_format: "legacy"` (the default):

`corpus_files` supply the allowed document identities and mappings, not query
relevance labels. They may contain original Vault code/query records; only ID
metadata is read. The `query_file` must contain training queries/augmentations
with trusted targets, never held-out test-query labels.

Supported query records:

```json
{"prompt":"find this function", "target_text_id":"doc-a", "positive_text_ids":["doc-a","doc-b"], "family_id":"original-query-7"}
```

Raw q10 (`text_id` is numeric) and mapped Vault augmentation rows (`numeric_id`,
canonical `text_id`, `text`) are also supported. Ambiguous numeric/canonical IDs
require explicit `id_mode`. Queries must resolve to corpus documents. For an
augmented paraphrase, its declared target is assumed relevant; additional
positives must be explicit or come from the same normalized query text.

For both input formats, rows sharing normalized query text, a known positive
document, or explicit family ID are grouped into the same split. Multilabel
rows also group by source ID even when it is not positive. This is conservative:
augmentations of the same source document cannot leak across validation.
Grouping is used only for splitting; it does **not** propagate relevance
transitively between different queries. At least two independent families are
required. A large family may make the realized validation fraction differ from
the requested value; counts and input hashes are saved in `split_manifest.json`.

In legacy mode, `target_type` supports `text_id`, `url`, and `structure_id_v3`. For structure
targets, provide `structure_id_sources` with URL-to-structure mappings, or
include the structure fields in corpus records. Match the SFT decoder namespace.
Every corpus target must have an unambiguous mapping. The selected
`target_length_policy` also applies to legacy inputs; `error` requires complete
targets to fit within `max_target_length`, including EOS.

## Checkpoints and recovery

If only a round's `queries.jsonl` was deleted, restore the exact partition
without rerunning mining or training:

```bash
python src/pretrain/restore_iterative_queries.py --output-dir /path/to/existing/run --round 16
```

Stop the running controller first. This command uses the config recorded in
`run_manifest.json` and the hash-verified `metadata/train_queries.jsonl`. It
publishes the reconstructed file only if its bytes match the original
`select-16` hash, and leaves manifests, preferences, and checkpoints unchanged.
Existing mismatched files are not overwritten. Then resume normally (or use
`--start-round 16` if earlier round directories were deleted too). Copying this
standalone recovery script does not change the controller's resume identity.

If you deleted early round directories, resume from a specific **zero-based**
round using the same configuration and output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/scripts/ddro/run_vault_iterative_dpo_ema.sh --resume --start-round 3
# The same option is supported by the frozen-reference and regular launchers.
```

This starts at `round-003` and skips artifact checks/work for rounds 0–2.
The run manifest must record all earlier rounds as completed. Keep the policy
and reference snapshots used to start round 3 (normally round 2's
`training/latest` and, for EMA/frozen-reference, `reference`). Those snapshots
are still fingerprint-checked. If they are missing, restore them from backup or
choose a later boundary whose required snapshots survive; the launcher cannot
recreate deleted learned weights. Shared metadata, baseline, and artifacts at
or after the requested round remain checked normally. Completed stages within
the requested round are reused, and interrupted training resumes as usual.

Continue to pass `--start-round 3` on subsequent resumes while the historical
directories remain deleted. Existing epoch counts and round history are
preserved. Deleted historical best checkpoints are excluded from best-model
selection. Previous-pair overlap diagnostics use the preceding preferences
only when that file survives with its recorded hash. To upgrade an existing
run for this option, copy the updated `train_iterative_ddro_vault.py`; the known
preceding controller version is accepted without changing training settings.

```text
output_dir/
  run_manifest.json
  round_plan.json
  best_checkpoint.json
  metadata/{document_metadata,train_queries,validation_queries}.jsonl
  metadata/split_manifest.json
  baseline/validation_predictions{.jsonl,.stats.json}
  round-000/
    queries.jsonl
    preferences.jsonl
    preferences.stats.json
    pair_audit.json
    round_inputs.json
    reference/                   # EMA weights/tokenizer + reference_complete.json
    training/checkpoint-N/        # optimizer, scheduler, RNG, round identity
    training/latest/              # portable latest policy + completion marker
    training/training_metrics.json
    validation_predictions{.jsonl,.stats.json}
```

Use `--resume` with the same configuration. Committed preparation/mining phases
are reused after fingerprint checks; incomplete single-device mining is
regenerated, while multi-GPU mining reuses matching completed shards.
An interrupted training round resumes from its last fully written checkpoint,
with the **original round-start reference** and the same preferences. If no
complete training checkpoint exists, that round restarts from its starting
policy. Work since the last checkpoint is lost. A completed snapshot is reused
if interruption occurred during subsequent evaluation.

New rounds reset optimizer/scheduler state. Resume within a round restores it.
An interrupted EMA export reuses a matching completed reference snapshot or
regenerates it from its recorded input snapshots; the blend is never applied
twice to an already updated reference. `latest_reference` in the run manifest
points to the reference available for the next round.

Model, tokenizer, source-data, training-configuration, relevant-code and package-version
changes are rejected on resume, except the compatible controller/launcher upgrades,
evaluation scheduling, and `mining_num_gpus` changes described above. Use a new output directory for a changed
experiment. Keep all round-start/latest and EMA reference snapshots; checkpoint rotation affects
only periodic checkpoints inside an individual round. Completion markers and
atomic manifests prevent an unfinished snapshot from becoming the next reference.

`latest/` always contains the final policy of the round; best-model restoration
is disabled. `best_checkpoint.json` selects by fixed-validation retrieval
metrics and may point to SFT if no round improves. It is a pointer, not another
copy of the weights. Training always continues from latest. There is no
automatic early stopping or rollback in this initial experiment.

Recall@K and MRR@K use the original beam ranks, retaining invalid beams in the
denominator/rank positions. Candidate validity and mining coverage are logged.
Beam mining is policy-driven hard-negative mining rather than stochastic
on-policy sampling. Changing references changes the loss anchor; compare
retrieval metrics across rounds, not raw DPO reward/loss values.

## Second full dataset pass after EMA V100 training

After all partitions finish, start another pass from both the final policy and
the final EMA reference. Allocate the same four V100 GPUs on one node, then run:

```bash
bash src/scripts/ddro/run_vault_iterative_dpo_ema_v100_epoch2.sh \
  --previous-output outputs/vault-iterative-dpo-ema-v100-fp16-d09 \
  --output-dir outputs/vault-iterative-dpo-ema-v100-fp16-d09-epoch2
```

The launcher reads the completed run's `run_manifest.json` and inherits its
configuration (GPU counts, precision, batch sizes, beam count, EMA decay and
learning rate). It verifies the final policy and EMA reference fingerprints.
It uses `latest` rather than the best checkpoint, and preserves EMA history
through the new optional `initial_reference_checkpoint` configuration field.
The source output is read only. No mining implementation changes are required.

Each partition is mined again using the current policy and trained for one
epoch. The seed, validation split and partition order stay the same as the
first pass. Optimizer/scheduler state starts fresh at round boundaries, as in
the existing pipeline. A baseline evaluation runs once in the new directory;
the inherited evaluation interval counts from the new run's round zero.

The generated configuration is saved beside the output as
`<output-dir>.config.json`. Omitting `--output-dir` appends `-epoch2` to the
source directory name. Repeat the command with `--resume` to resume epoch 2.
Keep the source run's final policy/reference snapshots and prepared metadata.
Use `--prepare-only` to prepare the new split first, then `--resume` to train.

## Hybrid iterative DPO: offline BM25 + on-policy negatives

Use `src/scripts/configs/iterative_vault_dpo_hybrid_v100.json` for the separate
4-V100 FP16 EMA experiment. It keeps beam 10 and requests four on-policy
negatives (`negatives_per_query`) plus four BM25 negatives
(`bm25_negatives_per_query`). Set `bm25_cache` to the immutable SQLite cache.
Existing configurations with no cache continue to use model-only mining.
The previous trie/inference optimization is not required or reintroduced.

1. Check the query/checkpoint/output paths in the hybrid configuration, then
   prepare the offline inputs (no Pyserini or model loading needed):

   ```bash
   bash src/scripts/ddro/prepare_vault_iterative_bm25.sh prepare \
     --config src/scripts/configs/iterative_vault_dpo_hybrid_v100.json \
     --work-dir outputs/vault-bm25-offline
   ```

2. Switch to your existing BM25 environment with Pyserini and Java available.
   Index and retrieve a deeper pool once, then package the completed retrieval:

   ```bash
   bash src/scripts/ddro/prepare_vault_iterative_bm25.sh build \
     --work-dir outputs/vault-bm25-offline \
     --output-cache outputs/vault-bm25-offline/candidates.sqlite \
     --hits 200 --threads 16 --batch-size 32
   ```

3. Switch back to `ddro_env`, allocate four V100 GPUs on one node, and train:

   ```bash
   bash src/scripts/ddro/run_vault_iterative_dpo_hybrid_v100.sh
   # To resume this SAME hybrid run:
   bash src/scripts/ddro/run_vault_iterative_dpo_hybrid_v100.sh --resume
   ```

The offline workflow uses the same metadata preparation, seed, query keys and
family split as training. Only training queries are searched; validation
labels are never added to preferences. The full candidate corpus remains
searchable, as in model-only retrieval. Corpus text comes exclusively from
`code` or `original.code`, grouped by the exact selected document ID column;
query text and positive lists are never inserted into document contents.
Pseudo-query variants therefore produce one indexed code document per ID.
The exporter rejects conflicting code bodies for the same ID. Add repeatable
`--code-file PATH` arguments during preparation if some bodies are stored in
another file with the same ID column. IDs without code are omitted from Lucene
and counted in `export_manifest.json`; they remain in the model corpus.

Numeric retrieval IDs avoid URL/parameter whitespace parsing problems. The
cache maps them back to the original IDs and binds candidates to the prepared
train/corpus hashes and per-query prompt/positive signatures. SQLite indexed
lookups fetch just the current partition's candidates without rescanning the
entire retrieval file or loading all hits into memory. Training needs only
Python's standard SQLite support, with no Java/Pyserini dependency. The
index/search commands follow the [Pyserini Lucene workflow](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md),
using the repository's existing BM25 settings `k1=0.82`, `b=0.68` and TREC
output to retain both ranks and scores. These settings are not newly tuned
for Ruby code.

Each round first commits `model_preferences.jsonl` and its mining sidecars.
It then selects up to four model negatives and the first four eligible BM25
hits in rank order, scanning deeper past all positives and previously selected
IDs. Tokenizer-identical IDs (including collisions caused by target truncation)
are deduplicated using that round's miner collision report; any alias of a
positive is ineligible. Excluded model targets are also excluded from BM25
pairs. The fused `preferences.jsonl` is the only file passed to DPO. Source
tags, BM25 rank/score, round identity and cache fingerprint are retained.
No policy/reference log-probabilities are cached with offline BM25 candidates.

Quotas are separate: a shortfall is not filled from the other source and a
duplicate is never repeated to reach eight. Training continues with fewer
pairs when necessary, including BM25-only pairs when model mining finds none.
`preferences.stats.json` and `pair_audit.json` report per-source counts,
shortfalls and the actual mix (for example `4+4`, `2+4`, `0+4`). Missing cache
queries, changed labels/corpus, or corrupt artifacts fail explicitly rather
than masquerading as a genuine candidate shortfall. More pairs normally mean
more optimizer steps per partition when training for one epoch.

Cache contents, quotas and fusion code are part of the run identity. Resuming
reuses completed model mining and fused pairs; a failed fusion does not require
repeating committed inference. The known previous model-only controller can
still resume its old runs. Enabling hybrid data or changing the cache/quota in
an existing run is intentionally rejected: use a new output directory. To
start from previously trained weights, set `checkpoint_path` to that policy
and `initial_reference_checkpoint` to its EMA reference before starting the
new hybrid run. A second dataset pass of a hybrid run can reuse the same cache
as long as the split, IDs and labels are unchanged.

Offline preparation/retrieval never overwrites a cache. If retrieval finished
but packaging was interrupted, rerun only `pack` with `--work-dir` and
`--output-cache`. It requires a hash-verified `retrieval_complete.json` from
successful indexing/search. For an interrupted retrieval without that marker,
prepare a fresh work directory and rebuild. Keep the final SQLite file for
training/resume; the Lucene index and raw retrieval output are not needed by
the training process.

## Verification

```bash
python -m unittest discover -s src/pretrain/tests -p 'test_iterative*.py' -v
python -m unittest discover -s src/scripts/bm25/the_vault -p 'test_*.py' -v
```

The controller tests use small deterministic worker fixtures. The optional
real-model tests use a locally constructed tiny T5 and tokenizer, require the
training dependencies, and download no pretrained models. Set
`RUN_DPO_INTEGRATION=1` to enable them. Exact bitwise reproducibility on GPU is
not guaranteed; resume preserves the training state and input identities.
