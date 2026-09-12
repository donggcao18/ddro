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

To inspect metadata and splits before GPU work:

```bash
python src/pretrain/train_iterative_ddro_vault.py --config /path/to/experiment.json --prepare-only
python src/pretrain/train_iterative_ddro_vault.py --config /path/to/experiment.json --resume
```

Always launch the controller with `python`, not `torchrun`. Set `num_gpus` for
single-node distributed training; the controller launches and joins each
training process group. Mining/evaluation use one device and run in separate
processes, so their model memory is released before training. `fp32` with
`device: cpu` supports small smoke tests; mixed precision requires CUDA.

## Round semantics

1. Fix train/validation query-family membership once.
2. Evaluate the SFT checkpoint for a retrieval baseline.
3. Select the round's queries from a seeded, rotating permutation of training
   queries and persist them before generation. No query repeats within a round.
4. Generate candidates from the round-start checkpoint and write immutable
   preferences. Every verified positive is excluded from rejection, even when
   the correct document is not generated. Incorrect lower-ranked candidates
   are used even if top-1 is correct. Use one chosen target per query row.
5. Initialize separate policy/reference copies from that checkpoint. Freeze the
   reference, reset optimizer/scheduler, and train on the fixed preferences.
6. Export the latest policy, evaluate fixed validation queries, and repeat.

Reference log-probabilities are recomputed during training; reference caching
and automatic in-round reference synchronization are disabled. In iterative
T5 training, functional attention dropout is also disabled alongside TRL's
module dropout handling, so identical policy/reference copies start with
matching likelihoods even while the policy is in train mode.

`queries_per_round: null` uses all training queries. A positive value caps the
selected subset; the rotating window covers queries before wrapping. Each
query supplies up to `negatives_per_query` unique negatives. Shortfalls are
retained; queries with zero negatives are skipped and counted. An entirely
empty preference set fails before training. There is no replay accumulation.

`steps_per_round` counts **optimizer updates**, including gradient accumulation.
The trainer reshuffles and repeats the generated dataset as needed. Nominal
pair presentations are steps × per-device batch size × GPU count × accumulation;
partial batches and distributed sampler padding can change the exact count.
`pair_audit.json` reports the resulting nominal number of dataset passes so an
undersized query subset is visible. For epoch-based rounds, replace
`steps_per_round` with `epochs_per_round`; specify exactly one budget.

Ordinary uniform pair sampling is used. Queries with more surviving negatives
have proportionally greater training weight; quotas bound this imbalance and
the audit records the distribution. Round budgets do not promise each pair is
used exactly once or that each selected query appears in a short step budget.

## Inputs and labels

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

All rows sharing normalized query text, a known positive document, or explicit
family ID are grouped into the same split. This is deliberately conservative:
augmentations of the same source document cannot leak across validation.
Grouping is used only for splitting; it does **not** propagate relevance
transitively between different queries. At least two independent families are
required. A large family may make the realized validation fraction differ from
the requested value; counts and input hashes are saved in `split_manifest.json`.

`target_type` supports `text_id`, `url`, and `structure_id_v3`. For structure
targets, provide `structure_id_sources` with URL-to-structure mappings, or
include the structure fields in corpus records. Match the SFT decoder namespace.
Every corpus target must have an unambiguous mapping and fit entirely within
`max_target_length`, including EOS. Increase the limit or correct the mapping
when preflight fails; this path never truncates document identities.

## Checkpoints and recovery

```text
output_dir/
  run_manifest.json
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
    training/checkpoint-N/        # optimizer, scheduler, RNG, round identity
    training/latest/              # portable latest policy + completion marker
    training/training_metrics.json
    validation_predictions{.jsonl,.stats.json}
```

Use `--resume` with the same configuration. Committed preparation/mining phases
are reused after fingerprint checks; an incomplete mining phase is regenerated.
An interrupted training round resumes from its last fully written checkpoint,
with the **original round-start reference** and the same preferences. If no
complete training checkpoint exists, that round restarts from its starting
policy. Work since the last checkpoint is lost. A completed snapshot is reused
if interruption occurred during subsequent evaluation.

New rounds reset optimizer/scheduler state. Resume within a round restores it.
Model, tokenizer, source-data, configuration, relevant-code and package-version
changes are rejected on resume. Use a new output directory for a changed
experiment. Keep all round-start/latest snapshots; checkpoint rotation affects
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
