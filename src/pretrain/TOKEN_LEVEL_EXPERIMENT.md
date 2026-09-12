# Token-Level DPO for Ruby Generative Code Retrieval

Experiment summary and improvement proposal. Snapshot: 2026-09-11.

This document combines the current repository implementation with observations
reported during the experiment. **Implemented**, **observed**, and **proposed**
are distinguished throughout. Numerical retrieval improvements have not yet
been provided; none are inferred from training loss or qualitative feedback.

## 1. Executive Summary

The experiment adapts DDRO's preference-training pipeline to **TDPO1 and
TDPO2**, retaining the existing query preparation, multi-label relevance
handling, BM25 mining, and `prompt/chosen/rejected` training format. The model
generates a document identifier for a natural-language code-search query.

The current training experiment uses **URL document IDs**. A separate
`structure_id_v3` dataset and mining run also exist. Their audit statistics
must not be presented as URL retrieval results.

Two observations motivate further work:

- TDPO has shown promise according to the experimenter, but a quantitative
  SFT/DPO/TDPO comparison is still needed.
- Supplying the correct first three ID tokens during an oracle experiment
  substantially improved retrieval. This suggests investigating early decoding
  decisions, but does not establish that prefix weighting alone will recover
  the oracle gain.

**Recommended order:** measure prefix survival, improve the mixture of
negatives, test an auxiliary positive-prefix loss, and only then redesign IDs.
Keep standard TDPO1 and TDPO2 intact as reproducible baselines.

## 2. Task and Data Pipeline

### Retrieval Formulation

- Input: a natural-language query, including augmented/pseudo-query examples.
- Output: an autoregressively generated document ID.
- Chosen response: an ID associated with a relevant code document.
- Rejected response: a mined non-positive document ID.
- Model: the existing encoder-decoder retriever, initialized from its SFT
  checkpoint; preference training uses a separate frozen SFT reference.

TDPO can be applied to these preference pairs because document IDs are token
sequences. However, its objective is a surrogate for retrieval quality, not a
direct optimization of Recall or MRR. The original TDPO paper introduces
tokenwise forward-KL control for preference optimization; its text-generation
results are not evidence of gains on this Ruby retrieval task.
[TDPO, ICML 2024](https://proceedings.mlr.press/v235/zeng24c.html)

### Identity Handling

| Identifier | Role in this experiment |
| --- | --- |
| `numeric_id` | Dataset-build-specific identifier; unsafe as an assumed cross-build join key. |
| `url_based_id` | Primary exact-match key when merging the new structure-ID source with legacy data. Also the decoder target in the current URL experiment. |
| Legacy `text_id` | Internal document identity used by the BM25 pipeline and positive-label metadata. |
| `structure_id_v3` | Alternative decoder target, without the old hash suffix; can be shared by multiple documents. |

The original numeric-ID merge was invalid because numeric assignments differed
between source files. For example, legacy `text_id=1002061209121201` was linked
to `numeric_id=41249`, which did not appear in the new structure-ID source.
The corrected [merge script](../scripts/preprocess/merge_structure_id_v3.py)
joins through `url_based_id`, using the original records to connect URLs to
legacy text IDs. Earlier missing counts from the numeric join are obsolete as
an assessment of the corrected merge.

Structure-ID source:

```text
/home/users/congthanh_le/scratch/veil/CodeGR/data/augmented_dsi/Ruby_merged.jsonl
```

Example alternative ID:

```text
compare_stats|item1,item2|ShopStatusCompare.rb/Misc Objects/src
```

### Preprocessing to Training

```text
Original Ruby Query:/Code: records + augmented multi-label query records
  -> resolve document identities and all known positives
  -> optional URL-keyed structure_id_v3 merge and multi-label expansion
  -> prepare code corpus, query metadata, document metadata, and qrels
  -> BM25: query text searches code-document contents
  -> exclude known positives and equivalent positive decoder targets
  -> deduplicate candidate targets and sample negatives by rank band
  -> preference JSONL: prompt, chosen, rejected, audit metadata
  -> URL target conversion OR direct structure_id_v3 targets
  -> normalize and tokenize preferences
  -> DPO / TDPO1 / TDPO2 training
  -> document-ID decoding and retrieval evaluation
```

The structure-ready file adds the new target namespace; it does not replace the
original code corpus needed for BM25. URL pairs are produced through
[postprocess_dpo_urls.py](../scripts/bm25/the_vault/postprocess_dpo_urls.py).
The current URL training file is `data/vault_bm25/dpo_pairs_url.jsonl`.

### Current BM25 Negative Selection

The default [miner](../scripts/bm25/the_vault/mine_dpo_negatives.py) uses up to
200 hits, with eight negatives per query: three from ranks 1-20, two from
21-100, and three from 101-200. The pipeline defaults are `k1=0.82`, `b=0.68`,
and sampling seed 42. These are rank-stratified negatives, not a guarantee that
each candidate is difficult for the generative model.

All known positive text IDs are excluded, not only the scalar chosen ID.
Equivalent positive decoder targets are also excluded, and repeated candidate
targets are deduplicated. Default sampling skips a query if a rank-band quota
cannot be filled, even when some negatives remain. `--fill-shortfall` enables
fallback selection. `--pair-all-positives` is separate from positive filtering:
it expands chosen responses instead of merely excluding all positives.

The repository already has model-confusion mining and a BM25/model hybrid
combiner. Their existence does not establish that the current TDPO run used
hybrid data. Prefix-based stratification is a proposed extension.

### Reported Structure-V3 Mining Audit

These counts came from the **structure-v3** run, not a URL evaluation:

| Statistic | Reported value |
| --- | ---: |
| Input queries | 202,300 |
| Queries in BM25 run | 202,299 |
| Queries without a run | 1 |
| Queries without a usable sampled negative set | 201 |
| Filtered positive/multi-label/equivalent-target hits | 129,965 |
| Unknown-document hits / invalid target mappings | 0 / 0 |
| Decoder targets shared by multiple text IDs | 225 |
| Text IDs involved in those collisions | 477 |
| Extra text IDs sharing a target | 252 |
| Output preference pairs | 1,616,784 |

At eight pairs per retained query, the output corresponds to 202,098 queries.
Skipping 202 queries is a small coverage loss, but allowing duplicate decoder
targets does not make them uniquely retrievable. Evaluation must explicitly
handle an ID resolving to multiple documents. Also audit collisions after
tokenization and truncation, not just equality of ID strings.

## 3. Implemented Token-Level Objective

Implementation: [tdpo_trainer.py](tdpo_trainer.py). Its provenance comment cites
`Token-level-Direct-Preference-Optimization/trainers.py`; that reference folder
is not present in this checkout.

For query `q` and decoder target `y`, the code computes:

```text
r(y, q) = sum_t log[p_policy(y_t | q, y_<t) / p_reference(y_t | q, y_<t)]
K(y, q) = sum_t KL(p_reference(. | q, y_<t) || p_policy(. | q, y_<t))

TDPO1 loss = -logsigmoid(beta * [r_chosen - r_rejected
                               - (K_rejected - K_chosen)])

TDPO2 loss = -logsigmoid(beta * [r_chosen - r_rejected
                               - alpha * (K_rejected - stopgrad(K_chosen))])
```

Implementation details that must be preserved:

- Decoder labels already align with encoder-decoder logits; there is no extra
  causal-LM shift in the statistics function.
- Padding labels `-100` are masked; actual EOS positions contribute.
- Log probabilities and KL calculations use FP32 log-softmax.
- The reference is frozen and evaluated without gradients.
- TDPO1 ignores `alpha`; TDPO2 with `alpha=0` reduces algebraically to DPO.
- Statistics are summed over target tokens, not length averaged.
- Full-vocabulary reference probabilities are needed for KL. Ordinary DPO's
  cached scalar reference log probabilities are insufficient.

This remains pairwise preference training using token-level statistics. It is
not an independent chosen-versus-rejected classification loss at each position,
and it does not currently give the first three tokens extra weight. Logged
reward accuracy is a training diagnostic, not retrieval accuracy; for TDPO2 it
is not necessarily even the sign of the objective's preference logit.

## 4. Current Run Configuration

The latest [URL launcher](../scripts/ddro/launch_tdpo_training_vault_url.sh)
encodes the following setup. The two-GPU configuration is requested/configured;
a completed run with it has not yet been reported.

| Setting | Current value |
| --- | --- |
| Objective | TDPO2 |
| `beta` / TDPO2 `alpha` | 0.4 / 0.5 |
| Learning rate | 1e-6 |
| Epochs | 2 |
| Prompt / target token limits | 256 / 128 |
| Requested GPU configuration | Two GPUs, batch 16 per GPU, accumulation 1 |
| Effective batch size | 32 preference pairs per optimizer update |
| Recent single-GPU alternative | Batch 8, accumulation 4, also effective batch 32 |
| Precision / activation memory | bf16 / gradient checkpointing enabled |
| Validation split | 0; no internal validation unless an explicit eval file is supplied |
| Logging / checkpoint interval | 10 / 2,000 optimizer updates |
| Checkpoints retained | 3 |
| Dataset map processes / loader workers | 1 / 0 |
| Distributed timeout | 7,200 seconds |
| Tokenization writer batch size | 1,000 |

Paths on the training machine:

```text
SFT policy initialization and frozen reference:
/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000

Training data, relative to the DDRO repository:
data/vault_bm25/dpo_pairs_url.jsonl

TDPO output and trainer-resume directory:
outputs/vault-url-tdpo2-full
```

The URL tokenization log also reported 1,616,784 pairs. This is separate evidence
from the structure-v3 mining audit above. A pair contains both chosen and
rejected sequences, so a per-device batch of 16 processes 32 decoder responses
in each concatenated policy/reference forward.

Two epochs are the current run budget, not an experimentally established
optimum. The discussion mentioned a three-epoch DPO baseline; verify that run's
saved configuration before comparing. Compare matched update budgets and
development-set retrieval quality rather than choosing epochs by convention.

### Reproduce or Resume

Run from the Linux repository in the existing training environment:

```bash
# Resume the latest TDPO checkpoint; GPUs must be assigned to your job.
CUDA_VISIBLE_DEVICES=0,1 \
bash src/scripts/ddro/launch_tdpo_training_vault_url.sh

# One GPU, same effective batch size of 32.
CUDA_VISIBLE_DEVICES=0 TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
bash src/scripts/ddro/launch_tdpo_training_vault_url.sh

# Fresh, separate baseline experiment. Use tdpo1 to retain the other variant.
CUDA_VISIBLE_DEVICES=0,1 RESUME_FROM_CHECKPOINT=none \
PREFERENCE_OBJECTIVE=tdpo1 OUTPUT_DIR="$PWD/outputs/vault-url-tdpo1-baseline" \
bash src/scripts/ddro/launch_tdpo_training_vault_url.sh
```

The launcher defaults to resuming and fails if no trainer checkpoint exists.
Use `RESUME_FROM_CHECKPOINT=none` and a new output directory for a fresh ablation.
Set `PRINT_ONLY=1` to inspect the generated command without launching training.
Preserve scheduler-provided GPU visibility when it differs from these examples.

The SFT checkpoint is not the TDPO resume checkpoint. Resume restores the
trained policy and trainer state; the reference must remain the original SFT
model. Changing GPU count is not an exact replay of minibatches/RNG state,
even if effective batch size is unchanged. Do not resume a URL run into a new
ID namespace or silently change its objective.

**Resume batch-size caveat:** Transformers 4.45.2 restores internal
`_train_batch_size` from `trainer_state.json` before building the training
dataloader. Thus, the launcher's printed batch size is not sufficient to verify
the batch actually used after resume. Check the saved `train_batch_size` and
Trainer's startup summary. The behavior is visible in
[Trainer.train](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py).

## 5. Observations and Runtime Issue

| Observation | What it establishes |
| --- | --- |
| Single-A100 smoke test completed | The exercised configuration can execute training. |
| TDPO described as promising | Qualitative feedback; no numerical effect size supplied. |
| Oracle first-three-token decoding gave a large gain | Motivation for prefix diagnostics; exact protocol and metrics still needed. |
| Single-GPU progress: `17742/101048 [7:12:24<34:17:06]` | That run was slow; not a controlled throughput benchmark. |
| Roughly two hours between tokenization completion and training | Startup overhead is distinct from optimization cost. |
| Two-GPU rank 1 timed out at a barrier while rank 0 was reopening Arrow data | The reported timeout followed a data-loading stall; it does not by itself prove broken GPU communication. |
| Latest two-GPU progress: `24726/404196 21:07<153:07:37` | An advancing training run is now reported, but the total update count needs configuration verification. |

For 1,616,784 pairs and two epochs, effective batch 32 requires approximately
101,050 optimizer updates, allowing for batch rounding. The reported 404,196
updates instead exactly matches effective batch 8. One plausible explanation
is a saved batch size of 4 restored on two GPUs with accumulation 1. This is
a hypothesis until the actual resume checkpoint and runtime configuration are
checked; different data size, epochs, or explicit `max_steps` can also change
the total. The ETA implies approximately 1.45 seconds per remaining update,
so the inflated update count is an important part of the apparent runtime.

### Cache Diagnosis and Current Mitigation

Rank 0's repeated stack traces were inside
`datasets.table._memory_mapped_arrow_table_from_file`, reached from TRL's
tokenization `Dataset.map`. Rank 1 was waiting in
`local_main_process_first()`. Tokenization progress reaching 100% does not mean
Arrow finalization and reopening have completed. One GPU has no second rank
waiting at this barrier, so the same delay can eventually finish without that
distributed timeout.

The exact storage/RAM cause has not been profiled. The stack establishes the
blocked stage, not the cache's physical filesystem. Small Arrow record batches,
shared-storage latency, and memory pressure are candidates. Memory mapping
does not mean the entire dataset is eagerly copied into RAM.

[preference_cache.py](preference_cache.py) now intercepts TRL 0.11.4's
tokenization map and changes its writer batch size from 10 to 1,000. For about
1.62 million rows, this targets roughly 1,617 instead of 161,679 record batches,
subject to sharding boundaries. These are record batches, not that many files,
and the reduction does not imply a 100-fold speedup. The original setting is
visible in the [TRL 0.11.4 source](https://github.com/huggingface/trl/blob/v0.11.4/trl/trainer/dpo_trainer.py).

The adapter gives the new cache layout a distinct fingerprint, preserves row
selection/formatting, and logs map-plus-reopen time and cache paths. Set
`DATASET_CACHE_DIR` to sufficiently large node-local storage accessible to both
ranks on the node. The launcher's repository fallback is not guaranteed to be
local SSD. Keep checkpoints on persistent storage. Cache reuse depends on
matching dataset/tokenizer/configuration fingerprints.

Increasing the timeout only permits a longer wait; it does not speed up cache
loading. A two-hour timeout can still be too short when waiting starts before
tokenization. `--startup_debug 60` helps separate trainer initialization from
later checkpoint/optimizer restoration. Neither post-fix startup speed nor a
successful full two-GPU run has yet been reported. Local GPU verification was
not available; ML-dependent tests require the training environment.

## 6. Interpreting the Oracle Prefix Result

The useful factorization is:

```text
p(doc_id | query) = p(prefix | query) * p(suffix | query, prefix)
```

Oracle prefixes bypass part of the first factor and restrict the search space.
A large gain is compatible with poor early-token prediction, beam pruning,
or an ID organization that is difficult to infer from a query. It is not proof
that the first three tokens are semantically meaningful: even an arbitrary
correct prefix can reveal substantial document identity.

Before changing training, record which namespace the oracle experiment used,
whether "three tokens" means actual tokenizer IDs, and exactly how gold
prefixes were selected for multi-label queries. Exclude decoder-start and
padding tokens when counting ID tokens. Three subword tokens are not three
words or three pipe-separated fields.

**Important gradient distinction:** when chosen and rejected IDs share a
prefix, their log-ratio terms at identical histories cancel in the pairwise
margin. Shared-history KL terms also cancel in TDPO1. In TDPO2, detaching
chosen KL allows remaining regularization gradients, but does not create a
direct preference between the shared prefix and an alternative branch.
Same-prefix negatives are useful for suffix discrimination; using only them
does not directly teach which early branch should win.

Early-prefix pruning is also studied in generative retrieval. The recent PRO
preprint proposes prefix retention methods for multimodal retrieval, which
supports investigating the mechanism, not assuming its results transfer to
Ruby URL IDs. [Prefix Retention Optimization, 2026 preprint](https://arxiv.org/abs/2606.09241)

## 7. Proposed Improvements, in Priority Order

### P0: Measure Prefix Survival Before Modifying the Loss

Add diagnostics to the actual constrained-decoding evaluation path:

- At depths `k = 1, 2, 3, 5`, measure whether the live beam retains at least
  one prefix belonging to any known relevant document.
- Record the first depth at which all relevant paths are pruned. Also inspect
  gold-prefix scores to distinguish scoring errors from search limitations.
- Report ordinary and oracle-prefix retrieval at the same beam width, target
  limit, length penalty, candidate trie, and number of returned documents.
- Record documents per prefix, prefix frequency, target lengths, and
  truncation/tokenization collisions. Compare cardinality-matched controls
  when interpreting oracle gains across different ID designs.

Teacher-forced token accuracy alone is insufficient: it conditions on correct
histories that may never survive inference. Keep oracle results explicitly
labeled as privileged-information diagnostics, not deployable retrieval scores.

### P1: Mix Negatives According to the Model's Errors

Keep BM25 as the semantic candidate source and extend the existing
[model-confusion miner](../scripts/bm25/the_vault/mine_model_confusion_negatives.py).
Generate valid candidate IDs with a frozen checkpoint and the same decoding
configuration used for evaluation. Store token-level longest-common-prefix
lengths and stratify against **all positive prefixes for the query**.

A starting eight-negative mixture, to be tested rather than assumed optimal:

| Pool | Initial quota | Intended signal |
| --- | ---: | --- |
| High-scoring model errors outside all positive `k`-prefixes | 4 | Choose a relevant early branch. |
| Non-positive documents inside a positive `k`-prefix | 2 | Disambiguate documents within a relevant branch. |
| Additional BM25 negatives | 2 | Retain lexical/semantic coverage beyond current model errors. |

Use `k=3` initially only after matching the oracle's tokenizer definition.
Deduplicate across pools, diversify wrong-prefix branches, and use logged
fallbacks when a pool cannot fill its quota. Prefer preserving the query with
an explicit actual mixture over silently discarding it for a missing pool.
Match total pair counts in controlled comparisons.

Required safeguards and audit fields:

- Filter all known positives, equivalent decoder targets, and invalid token
  sequences before assigning pools. A non-positive document sharing a
  relevant prefix is a valid leaf negative, not a negative prefix.
- Audit likely false negatives such as code clones and semantically equivalent
  functions; missing relevance labels do not prove irrelevance.
- Retain query key, all positives, chosen/rejected IDs, mining source,
  model/BM25 rank and score, prefix bucket, longest common prefix, tokenizer,
  and mining-checkpoint identity. Do not compare BM25 and model scores as if
  they had a common scale.
- Mine training queries only. A later refresh from a stronger policy is a
  separate experiment; keep the TDPO reference frozen and include mining cost.

This is the smallest retrieval-specific intervention because the trainer can
still consume unchanged `prompt/chosen/rejected` pairs.

### P2: Add an Explicit Positive-Prefix Auxiliary Loss

Test this independently of P1 before combining them:

```text
L_total(q) = L_TDPOj(q) + lambda_prefix * L_prefix(q),  j in {1, 2}

P_k(q) = deduplicated valid prefixes of all known positive document IDs

L_prefix(q) = - mean_{p in P_k(q)} [
                 (1 / length(p)) * sum_t log p_policy(p_t | q, p_<t)
              ]
```

This proposes supervised support for every known positive prefix, with equal
weight across distinct prefixes. It is not the different objective of
maximizing only the total probability mass of the positive-prefix set, which
can concentrate on one easy prefix. Choose deliberately according to the
desired multi-label retrieval behavior.

Start with `lambda_prefix=0` as the parity control and small development-set
trials such as `0.01` and `0.1`; their usefulness depends on loss scale. Log
both components. Weight the auxiliary term once per query, or equivalently
correct for pair multiplicity, so eight negatives do not create eight times
the intended positive supervision. Deduplicate repeated prefixes from
multi-label expansion. Use actual EOS for short complete IDs, but never append
an artificial EOS to a prefix that is only a truncated part of a longer ID.

Keep the original TDPO1/TDPO2 equations, masks, and stop-gradient behavior
unchanged. Call the result **TDPO plus prefix supervision**, not unmodified
TDPO. Include a full-target SFT auxiliary control to test whether gains come
from prefix emphasis specifically or simply from additional positive training.

This is more than a loss-function edit: current
[dataset cleaning](train_ddro_vault.py) retains only `prompt/chosen/rejected`.
Query IDs and all-positive metadata must survive through a sidecar or an
extended dataset/collator path. Simply using each pair's chosen prefix is not
the multi-positive, query-balanced objective above.

### P3: Test Query-Aligned, Uniquely Decodable IDs

URL IDs often begin with repository-owner information, which may be weakly
related to a functional query. The action-first structure-v3 example is a
plausible alternative, but its observed collisions require attention.

An illustrative future design is:

```text
action_or_operation|object_or_context|arguments_or_path|stable_unique_suffix
```

Derive the semantic fields from document/code information, not held-out test
queries. Audit semantic quality, prefix balance, token lengths, and uniqueness
both as strings and tokenizer sequences. A uniqueness suffix fixes ambiguity
only if the model can retain it within its target-length budget.

Changing the namespace requires matching SFT training, a new frozen reference,
updated mappings/trie, and a separately controlled experiment. It is not an
ordinary resume of the URL TDPO run. Learned semantic IDs are a larger follow-up;
GenRet provides relevant prior work, not a minimal drop-in change.
[Learning to Tokenize for Generative Retrieval](https://arxiv.org/abs/2304.04171)

Do not initially alter ID design, negative mixture, token weights, and decoding
together. Otherwise an improvement will be difficult to explain.

## 8. Controlled Experiment Plan

All rows below are planned comparisons, not claimed completed results. The
current TDPO2 run is the anchor for E3; its final metrics remain to be recorded.

| Run | Objective | Negatives / extra supervision | Question |
| --- | --- | --- | --- |
| E0 | Original URL SFT | None | What is the starting retrieval quality? |
| E1 | DPO | Current BM25 | Does preference training help? |
| E2 | TDPO1 | Current BM25 | Does the first KL correction help? |
| E3 | TDPO2 | Current BM25 | Does detached chosen KL help? |
| E4 | TDPO2 | P1 mixed prefix-aware negatives | Does mining improve branch selection? |
| E5 | TDPO2 + prefix supervision | Current BM25 | Does explicit early-token training help? |
| E6 | TDPO2 + prefix supervision | P1 mixed negatives | Are the interventions complementary? |
| E7 | TDPO2 + full-target SFT auxiliary | Current BM25 | Is prefix emphasis better than generic positive supervision? |
| E8 | TDPO1 + best validated additions | Same selected data | Do additions also work with TDPO1? |

Run a smaller, fixed training-query subset with full negative sets first; then
confirm promising changes at the full budget. Keep the original SFT reference,
query set, pair count, effective batch size, update budget, and decoding settings
matched wherever possible. Use separate output directories. Report target-token
counts and wall time when an intervention changes compute.

Create a fixed development set grouped by query and augmentation family.
Splitting individual preference rows can put the same query in both training
and validation. The current `validation_split=0` needs an explicit external
development protocol; pairwise validation loss alone is not enough. Corpus
documents may include evaluation documents if the retrieval protocol permits
it, but held-out queries/relevance judgments must not inform mining or ID design.

Primary metrics: Recall@1/5/10 and MRR@10, using all known relevant documents.
State whether "Recall@k" means fraction of relevant documents retrieved or
at-least-one-positive Hit@k; report them separately if needed. Add prefix
survival by depth, oracle-prefix curves, invalid/ambiguous-ID rates, and target
truncation rates. Define how a shared decoder target maps back to documents.
Keep final test data sealed until configuration selection; report multiple seeds
or query-level uncertainty intervals for the selected comparisons.

Track engineering metrics separately: tokenization time, cache-reopen time,
resume time, actual post-resume batch size, steady-state pairs/second, optimizer updates/second, peak GPU RAM,
and mining time. Larger batches alone do not establish a faster end-to-end run.

## 9. Minimal Implementation Plan and Checks

| Change | Existing location | Verification |
| --- | --- | --- |
| Prefix-survival instrumentation | [eval_ddro_docid_ranking.py](eval_ddro_docid_ranking.py) or the HF evaluation path actually used | Known toy trie, multiple relevant prefixes, pruned branches, EOS handling. |
| Prefix-aware negative pools | Model-confusion miner and [hybrid combiner](../scripts/bm25/the_vault/combine_hybrid_dpo.py) | All-positive exclusion, target/token collisions, deterministic quotas/fallbacks, pool statistics. |
| Positive-prefix supervision | `train_ddro_vault.py` dataset/collator plus `tdpo_trainer.py` | `lambda=0` baseline parity; no reference gradients; original TDPO2 detach preserved. |
| Query-balanced multi-label support | Metadata retention or sidecar lookup | No duplicated-prefix overweighting; invariance to repeated pair rows under corrected weighting. |
| Prefix gradient behavior | [test_tdpo_trainer.py](test_tdpo_trainer.py) | Shared-prefix pairwise cancellation versus nonzero positive-prefix auxiliary gradient; padding/short IDs. |
| Cache and training integration | [test_preference_cache.py](test_preference_cache.py) | Identical tokenized rows, preserved selections, reduced record batches, tiny train/eval/save/resume. |

No new mining strategy or auxiliary loss is implemented by this document.
Before a full GPU run, execute the relevant tests in the working training
environment and record installed package versions. The TDPO adaptation targets
TRL 0.11.4 and Transformers 4.45.2; do not blindly reinstall the repository's
older general `requirements.txt` into an already working environment.

## 10. Results Still Needed

| Item to record | Current status |
| --- | --- |
| Exact completed TDPO checkpoint and optimizer step | Not supplied |
| SFT, DPO, TDPO1, TDPO2 Recall/MRR on the same split | Not supplied |
| Oracle ID namespace, actual token IDs, beam settings, and numerical gain | Not supplied |
| URL pair-file audit, hash, unique query count, and relevance coverage | Pair count observed; full audit not supplied |
| Corrected URL-keyed structure merge audit | Not supplied here |
| A100 memory capacity and steady-state peak allocation | Not supplied |
| Post-cache-fix timings and full two-GPU completion | An advancing two-GPU run is reported; timings/configuration and completion remain unverified |
| Git commit, package versions, seed, full command, and resume history | Capture with each measured run |

**Immediate next experiment:** evaluate the existing URL SFT and TDPO2
checkpoints under identical decoding settings, instrument positive-prefix
survival, then compare the current BM25 pairs against prefix-aware mixed
negatives with the TDPO2 loss unchanged. This tests the most actionable
hypothesis before introducing a new objective or document-ID namespace.
