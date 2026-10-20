# Branch-preference continuation after DPO

The `branch-preference` branch implements whole-ID sigmoid DPO plus a local
preference at the first divergent target token. Both actions are scored under
the same chosen-prefix history. Existing constrained on-policy mining,
multi-GPU generation, query-family splits, and retrieval evaluation are reused.

For `chosen=[A,B,C,D,EOS]` and `rejected=[A,B,X,Y,EOS]`, the local preference is
`C > X` under `[A,B]`. If X also leads to another known positive, the original
sequence pair remains in training but its branch loss is zero. Duplicate leaf
pairs representing the same decision share its auxiliary weight.

The objective is `L_sequence_DPO + branch_loss_weight * L_branch_DPO`.
The reference is initialized from the supplied completed DPO checkpoint and
kept frozen through the existing EMA-decay-1 mechanism. This implementation
uses uniform decision preferences with query/decision normalization; adaptive
weights and prefix SFT are separate follow-up experiments.

## Start a fresh full-query run

Run in the repository root with the existing training environment (TRL 0.11.4,
Transformers 4.45.2). Supply the path to your **completed DPO checkpoint**:

```bash
conda activate ddro_env
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/scripts/ddro/run_vault_branch_preference.sh \
  --checkpoint /path/to/completed-dpo-checkpoint \
  --query-file /path/to/Ruby_merged_multilabel.jsonl \
  --target-type url_based_id \
  --output-dir outputs/vault-url-branch-C \
  --arm branch \
  --rounds all \
  --num-beams 15 \
  --mining-batch-size 25 \
  --eval-every-rounds 8
```

For the configured Ruby URL experiment, the complete launch script is
`src/scripts/ddro/run_vault_branch_preference_full.sh`. It initializes from
the original normal-DPO checkpoint, uses a new `vault-url-branch-preference-full-fresh`
output directory, and contains all training/mining settings explicitly. Run
it with `bash`; override `CHECKPOINT_PATH`, `QUERY_FILE`, or `OUTPUT_DIR` through
environment variables when needed.

For structure IDs, use the matching checkpoint/data and
`--target-type structure_id_v6`. The query file requires `positive_url_based_id`
or `positive_structure_id_v6`, respectively, as a nonempty list on every query
row. The ordinary ID column is also required. Missing positives do not fall
back to the provenance label.

If the query file does not cover the full intended retrieval collection, add
`--corpus-file /path/to/full-corpus.jsonl`; each corpus row must include the
selected ID column. Additional corpus files provide identities, not query
relevance labels.

Defaults: 4 GPUs, FP16, enough rounds to cover the prepared training queries
once, up to 10,000 selected queries per round, 1,000 optimizer updates per round,
retrieval evaluation every 8 completed rounds plus the final round, 15 beams,
4 negatives/query, mining batch 25/GPU,
training batch 16/GPU with accumulation 1, LR 1e-6, beta 0.4, branch coefficient
0.1, prompt limit 256, target limit 128. These are pilot settings, not verified
V100 memory/throughput measurements. Lower `--mining-batch-size` or `--batch-size`
and adjust accumulation if your checkpoint needs more memory.

The launcher writes a resolved config beside the output directory, e.g.
`outputs/vault-url-branch-C.config.json`. It launches the controller once;
the controller starts the mining and training workers. Do not wrap the launcher
in `torchrun`.

## Mix BM25 and current-policy preferences in every round

Use the original multilabel `QUERY_FILE` for query selection and relevance
labels, and pass a separate `--bm25-file /path/to/bm25_preferences.jsonl`.
BM25 rows require `prompt`, `chosen_url_based_id`, and
`rejected_url_based_id` for URL runs. `bm25_rank` is an optional positive
integer; `bm25_score` can be null. For other selected ID types the fields are
`chosen_<target_type>` and `rejected_<target_type>`.

The explicit selected-ID fields are authoritative. The source file's numeric
`chosen_text_id`, `rejected_text_id`, `positive_text_ids`, and `query_key` are
not used to join the datasets. Rows are matched by normalized query text AND
the designated chosen target, then rewritten using this run's canonical IDs,
query key and full positive list. A BM25 row for a different chosen target
does not join just because its prompt matches. Validation queries are not
imported into the training pool. No relevance labels are added from BM25.

The launcher default reserves two BM25 and two policy negatives per selected
query. The dedicated mixed run script currently requests six BM25 plus four
policy negatives, with ten total training negatives per query:

```bash
conda activate ddro_env
BM25_FILE=/path/to/bm25_preferences.jsonl \
bash src/scripts/ddro/run_vault_branch_preference_mixed.sh
```

The mixed script uses training batch 16/GPU, accumulation 1, 15 beams, mining
batch 16/GPU, evaluation every eight rounds, and the configured final inference
file. Its default BM25 file is
`/home/users/congthanh_le/scratch/east/ddro/data/vault_bm25/dpo_pairs_url.jsonl`,
matching the existing URL-DPO launch script. Override `BM25_FILE` for another
file. It uses a separate `...-full-bm25-mixed` output directory.
`OUTPUT_DIR`, `CHECKPOINT_PATH`, `QUERY_FILE`, and `INFER_FILE` remain
environment overrides. Add `--arm dpo` for the ordinary DPO control, or
`--resume` after an interruption. The epoch2 and SFT scripts retain their
policy-only configuration.

The same options work directly with `launch_branch_preference.py`:
`--bm25-file ... --bm25-training-negatives-per-query 2
--total-training-negatives-per-query 4 --policy-mining-negatives-per-query 4`.

Negative budgets are named separately:

| Launcher option | Mixed-script environment variable | Current mixed default |
| --- | --- | --- |
| `--policy-mining-negatives-per-query` | `POLICY_MINING_NEGATIVES_PER_QUERY` | 4 candidates retained from the model's 15 beams |
| `--bm25-training-negatives-per-query` | `BM25_TRAINING_NEGATIVES_PER_QUERY` | 6 BM25 negatives targeted in the final mix |
| `--total-training-negatives-per-query` | `TOTAL_TRAINING_NEGATIVES_PER_QUERY` | 10 training negatives maximum across sources |

The policy training target is total minus BM25 (currently 10 minus 6 = 4).
Source shares are targets; existing backfilling may shift the actual mixture.
Mining can retain a larger candidate pool than the policy training share.
If unspecified, the mining limit defaults to the total training quota, preserving
the old behavior. Without BM25, an explicit mining limit must equal the total
training quota.

`--negatives-per-query` and `--bm25-negatives-per-query` remain CLI aliases.
`BM25_NEGATIVES_PER_QUERY` remains an environment alias in the mixed script.
Saved JSON configs retain the legacy `negatives_per_query` (training total) and
`bm25_negatives_per_query` (BM25 training share) keys; the optional
`policy_mining_negatives_per_query` key separates mining when needed. Merely
renaming the CLI flags does not change a run's identity. The source quota must be between
one and total-minus-one. BM25 candidates are ordered by ascending BM25 rank
(file order for ties/missing ranks); policy candidates retain beam order.
The two sources' scores are never compared to each other.

The BM25 file is indexed once during `prepare-bm25`, using the same corpus,
tokenizer, length and collision policies as mining. Known positives and their
token aliases are filtered. Unknown corpus IDs are skipped and reported with
examples; add `--corpus-file` for a more complete retrieval collection. A file
with no usable training matches fails before GPU work. Each round then:

1. Mines up to the policy-mining quota into `preferences.jsonl`.
2. Runs `mix-N` to select the reserved source shares, deduplicate effective
   token sequences across sources, then fill empty slots from unused policy
   candidates followed by BM25 candidates. Fewer than the requested total of valid distinct
   negatives is allowed; no pair is duplicated to satisfy the quota.
3. Writes `mixed_preferences.jsonl`, `mixed_preferences.audit.json` (actual
   source counts and per-query mixture distribution), and `mixed_pair_audit.json`.
4. Extracts branch annotations and query/decision weights from the combined
   dataset, then trains on those annotations. Ordinary DPO trains directly on
   `mixed_preferences.jsonl`.

The original mining outputs remain available for inspection. Training exposure
and coverage use the mixed pair counts. BM25 file content, source quota, pool,
and mixed artifacts are fingerprinted for resume. Start a new output directory
to introduce mixing into a previously policy-only experiment; initialize it
from the desired existing checkpoint. Once a mixed run starts, resume it with
the same BM25 file and quota. Final inference behavior is unchanged.

## Evaluate a separate inference file on the final policy

Set `INFER_FILE` alongside the training `QUERY_FILE`:

```bash
INFER_FILE=/path/to/Ruby_infer.jsonl \
bash src/scripts/ddro/run_vault_branch_preference_full_sft_3_5.sh
```

The generic full-run script supports the same variable. With the Python
launcher, use `--infer-file /path/to/Ruby_infer.jsonl`. A JSON controller config
can contain `"infer_file": "/path/to/Ruby_infer.jsonl"`.

Only the latest policy from the final round is evaluated on this file. Its
queries/labels do not enter preference mining, training, the validation split,
or best-checkpoint selection. Periodic validation keeps the existing schedule.
Final inference is run even with `--no-eval-at-end` (that switch controls the
training-validation pass). Its beam search explicitly uses
`early_stopping=False`; ordinary mining and validation retain their existing
early-stopping behavior. Generation still uses the configured length limit.

For URL IDs, inference records can use either format:

```json
{"text": "find a function", "url_based_id": "repo/file.rb/function(arg)"}
{"text": "find related functions", "positive_url_based_id": ["repo/a.rb/f()", "repo/b.rb/g()"]}
```

The single ID is the ground truth only when `positive_url_based_id` is absent.
An explicit positive list is used verbatim and must be nonempty. For structure
IDs, use `structure_id_v6` / `positive_structure_id_v6`. Every inference record
is evaluated, including duplicates. Legacy inputs can use canonical
`target_text_id` / `positive_text_ids` metadata.

Inference uses the same corpus as mining. Supply all retrieval identities
through `--corpus-file` when the training query file does not cover the full
collection. Missing inference positives are reported during preparation before
GPU work; the inference file does not silently expand the training corpus.

Outputs are `round-NNN/infer_predictions.jsonl`, its `.stats.json`,
`round-NNN/infer_summary.json`, and the top-level `final_inference.json` summary.
They record retrieval metrics, final checkpoint identity, and disabled beam
early stopping. Metrics include `hit@1`, `hit@5`, and `hit@10` with the default
15 beams, alongside MRR and Recall at each reported cutoff (including 15).
Hit@K is the fraction of evaluated queries with at least one relevant document
in the top K; Recall@K measures the fraction of all relevant documents retrieved
per query. GPU shard metrics are averaged by their evaluated query counts.
Previously committed evaluation results are preserved on resume; newly run
evaluations include Hit@K. Inference inputs and prepared artifacts are fingerprinted;
interrupted final inference can resume without retraining completed rounds.

To add inference to an existing run, use the updated controller with its saved
config, `--resume`, and `--infer-file /path/to/Ruby_infer.jsonl`. The known
preceding code version is supported without changing trained outputs. Repeat
the file override on later controller resumes, or save `infer_file` in the
existing config. Branch-launcher resumes retain a previously saved inference
file automatically. Once inference metadata is committed, its file/corpus and
preparation code must remain unchanged.

The launcher uses the on-policy compatibility defaults for every arm:
`--target-length-policy truncate --target-collision-policy allow
--target-token-policy allow`. Mining, branch extraction, and training use the
same effective token sequences, including tokenizer truncation and terminal EOS.
The target limit can remain 128 when some original IDs are longer. Alias groups
use the existing miner's representative IDs and positive filtering. Pairs whose
effective sequences are identical receive zero auxiliary loss. The branch audit
records overlength IDs, alias groups, excluded targets, and inactive pairs.

To require complete, distinct encodings, select `--target-length-policy error
--target-collision-policy error --target-token-policy error` and set an adequate
`--max-target-length`. The `skip` length/collision policies exclude affected IDs
from mining and branch-positive support; original relevance labels are preserved.

## Controls and refreshing negatives

Repeat the same command with identical settings and separate output directories:

```text
--arm dpo    --output-dir outputs/vault-url-branch-A
--arm zero   --output-dir outputs/vault-url-branch-C0
--arm branch --output-dir outputs/vault-url-branch-C
```

`dpo` continues ordinary DPO. `zero` builds branch annotations but sets the
auxiliary coefficient to zero. `branch` adds the local preference loss.
All arms use the same initial snapshot, selection seed, mining settings, and
frozen reference. Original sequence row order/content is preserved during
branch extraction.

The launcher defaults to full-query coverage (`--rounds all`). Each round mines
from the latest policy and re-extracts its branch preferences. The reference
remains frozen at the initial DPO checkpoint. The existing controller restarts
optimizer/scheduler state between rounds; use the same lifecycle in comparison
arms. Select `--rounds 1` explicitly for a single-round pilot or another integer
for a fixed-round budget. Full-query mode keeps a smaller final round without
wrapping; a fixed-round budget does not guarantee full query coverage.

To resume an interruption, repeat the same command with `--resume`. You can
also run the saved config directly:

```bash
python src/pretrain/train_iterative_ddro_vault.py \
  --config outputs/vault-url-branch-C.config.json --resume
```

### Continue a completed one-round pilot over the full dataset

Use the updated controller with the **original saved config and output directory**:

```bash
conda activate ddro_env
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u src/pretrain/train_iterative_ddro_vault.py \
  --config outputs/vault-url-branch-C.config.json \
  --resume --start-round 1 --extend-rounds all
```

`--extend-rounds all` derives the total from the number of prepared training
queries and the saved queries-per-round value. The existing round 0 remains
unchanged. Round 1 starts from its latest policy and recorded reference;
branch runs keep their original frozen reference. The final round uses the
remaining queries without wrapping, so each prepared training query is selected
once across the pilot and continuation. Queries with no negatives still produce
no preference pairs. Each added round retains the saved training budget.

Repeat this exact command after any interruption. Keep the original config
file unchanged; the extension is recorded in `run_manifest.json`, with an
additional `round_plan_extended_NNN.json`. The initial `round_plan.json` stays
unchanged. For an explicit total instead, use e.g. `--extend-rounds 5` (five
rounds total, including the pilot), which retains the fixed rotating schedule.
Only completed fixed-round runs can be extended; data, package versions, model
code, training settings, and checkpoint identities must still match. The update
accepts the known preceding controller version without altering trained outputs.

### Resume at round 3 and evaluate every 8 completed rounds

```bash
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u src/pretrain/train_iterative_ddro_vault.py \
  --config outputs/vault-url-branch-C.config.json \
  --resume --start-round 3 --extend-rounds all --eval-every-rounds 8
```

Round IDs are zero-based: this preserves rounds 0, 1, and 2 and resumes
`round-003` from the retained round-2 policy and reference. Retrieval evaluation
runs after 8, 16, 24, ... completed rounds (directories `round-007`, `round-015`,
`round-023`, ...), plus the final round. Add `--no-eval-at-end` to evaluate only
at exact multiples of eight. Previously committed metrics are retained.

The round interval works with the saved 1,000-step training budget; it does not
change the number of optimizer updates or treat each round as an epoch. It
overrides an existing epoch-based evaluation interval. Repeat the full command,
including `--eval-every-rounds 8`, after any interruption. The saved config file
stays unchanged; the runtime schedule update is recorded in the run manifest.

### Resume mining after GPU out-of-memory

Use the controller to lower the generation batch while retaining the saved
config and output directory. For a failure during `round-004`:

```bash
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u src/pretrain/train_iterative_ddro_vault.py \
  --config outputs/vault-url-branch-preference-full-fresh.config.json \
  --resume --start-round 4 --mining-batch-size 8 --eval-every-rounds 8
```

The override changes mining/evaluation generation batch size per GPU. Beam
count, training batch, optimizer settings, and existing trained checkpoints
remain unchanged. Completed rounds and committed mining datasets are retained.
The failed round's shard identities include batch size, so its mining shards
are regenerated at the new batch size; outputs from different batch settings
are never mixed. Shards completed with the new setting can be reused on a
subsequent retry. Repeat the same override on later resumes. The controller
records this runtime change without rewriting the original config file.

Use a new output directory when changing objective, coefficient, checkpoint,
data, target policies, or training settings. Extending the round plan through
the command above keeps the existing output directory. After this compatibility update,
restart a previously failed strict-policy run with a new output directory.
Existing pre-update manifests do not silently
migrate into branch experiments. `--prepare-only` prepares the split; continue
that run with `--resume` after removing `--prepare-only`.

## Smaller rounds with an updating EMA reference

For the 2,000-query experiment, reuse the existing mixed script through:

```bash
conda activate ddro_env
bash src/scripts/ddro/run_vault_branch_preference_mixed_small_rounds.sh
```

This starts a new full-query run in `.../vault-url-branch-preference-mixed-q2000-ema098`:

- 2,000 queries and 200 optimizer steps per round.
- Reference EMA decay 0.98: 98% old reference plus 2% latest policy per round.
- Validation every 40 completed rounds (approximately 80,000 queries), plus the
  final round. The separate inference file is evaluated only after training,
  with beam early stopping disabled.
- The existing mixed checkpoint and query/BM25/infer paths, LR 2e-5, training
  batch 16/GPU, accumulation 1, 15 beams, mining batch 16/GPU, six BM25 plus four
  policy training targets, and latest-three-round checkpoint retention.

The smaller final query partition keeps the same 200-step round budget.
The 200 versus 1,000 steps scales the nominal full-query training budget by the
same factor as 2,000 versus 10,000 queries. Five EMA updates give
`0.98^5 ~= 0.904` retention of the old reference, close to one decay-0.9 update
per 10,000 queries. These are experimental starting values, not measured optima.
The ordinary launcher's default remains decay 1.0 (frozen reference).

Override `QUERIES_PER_ROUND`, `STEPS_PER_ROUND`, `REFERENCE_EMA_DECAY`, and
`EVAL_EVERY_ROUNDS` through environment variables. For example, the 1,000-query
alternative uses 100 steps, decay 0.99, and validation every 80 rounds; give it
a separate `OUTPUT_DIR`. The launcher also accepts `--reference-ema-decay`
directly. Start a new output directory when changing the round size, step
budget, or EMA decay. To resume the same small-round experiment, repeat its
script with `--resume` and the same environment overrides.

## Keep only the latest three checkpoint rounds

Branch-preference launchers default to `--keep-last-round-checkpoints 3`.
After a round's training, reference update, and scheduled validation have
completed and its result is committed, older checkpoint payloads are pruned.
For example, after `round-005`, checkpoint weights for `round-003`,
`round-004`, and `round-005` remain. Skipped rounds do not count toward this
window because they have no new weights.

Only recognized model-weight, optimizer, scheduler, scaler, RNG and training-args
binary files in the old rounds' `training/latest`, `training/final`,
`training/checkpoint-N`, training root, and EMA `reference` directories are
removed. JSON/JSONL data, mined and mixed pairs, branch annotations, audits,
metrics, predictions, logs, completion markers, and tokenizer/config metadata
remain. No round or data directory is recursively removed. Initial policy and
reference inputs and the currently required policy/reference are protected.

Cleanup is journaled in `run_manifest.json` before deletion. An interrupted
cleanup resumes from that journal. Ordinary `--resume` automatically skips the
completed historical rounds after pruning; it does not try to reload their
deleted weights. Retained artifacts are still checked for changes. An explicit
`--start-round` requires the preceding policy/reference weights to still exist.

This is a strict recent-round policy: an older best checkpoint can be pruned
too. Its historical metrics and path remain in `best_checkpoint.json`, with
`checkpoint_available: false`. No extra best-model copy is kept. Final inference
continues to use the latest retained policy.

The mixed script exposes `KEEP_LAST_ROUND_CHECKPOINTS=3`. Set the CLI option to
`0` to disable future pruning; this does not restore previously removed weights.
Existing JSON configurations may add `keep_last_round_checkpoints: 3`, or use
the controller's resume-compatible override:

```bash
python -u src/pretrain/train_iterative_ddro_vault.py \
  --config /path/to/existing.config.json \
  --resume --keep-last-round-checkpoints 3
```

Updated branch launch scripts also enable retention when resumed with an older
saved config, provided the training settings remain the same. A controller that
is already running must be restarted with `--resume` to use the new retention
code. Generic controller configurations without the retention option retain
their old keep-all behavior.

## Outputs

Each round preserves `preferences.jsonl` and adds:

- `branch_preferences.jsonl`: all original sequence rows plus branch annotations.
- `branch_audit.json`: eligible rows/queries, skip reasons, distinct decisions,
  divergence positions, normalization weights, and source fingerprints.
- `training/latest/`: exported policy, tokenizer, and saved training arguments.

Training logs separate sequence/branch losses, eligible fraction, action margin,
and action accuracy. `best_checkpoint.json` selects by held-out MRR@10;
`run_manifest.json` records checkpoint identities and round retrieval metrics.
True multi-positive Recall is also reported by the existing evaluator.
If no pairs are branch-eligible, the auxiliary is zero and the run reports this
coverage explicitly. Action accuracy alone does not establish retrieval gains.

## Verification

The tests use locally created tiny T5 weights, with no model download:

```bash
python -m unittest discover -s src/pretrain/tests -p 'test_branch_preference*.py' -v
RUN_DPO_INTEGRATION=1 python -m unittest discover \
  -s src/pretrain/tests -p 'test_branch_preference*.py' -v
```

The integration flag enables real mining, annotation, training, frozen-reference
checks, interrupted checkpoint resume, and stale-artifact rejection under both
strict policies and truncated aliases. Tests also
verify zero-coefficient loss/gradient parity and same-state local gradients.
