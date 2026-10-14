# Branch-preference experiment: code-update plan

Prepared 2026-10-15 for the current `branch-preference` checkout. This document specifies planned implementation and experiments; it does not implement a new trainer or run training.

Run C has now been implemented. See [BRANCH_PREFERENCE.md](BRANCH_PREFERENCE.md) for the current launcher and behavior. The adaptive/curriculum items below remain follow-up experiments.

## 1. First implementation and experimental question

Implement Run C: continue from a completed normal-DPO checkpoint with the existing whole-ID sigmoid DPO loss plus a local DPO loss at the first token divergence between an existing chosen/rejected pair.

The experiment asks whether explicitly supervising a branch decision improves retrieval beyond continuing ordinary DPO with the same pairs, reference, initial checkpoint, and training budget.

The first implementation uses uniform branch preference weights. Adaptive confidence weights, fixed prefix weighting, positive-prefix SFT, all-positive pair expansion, direct next-token negative mining, curriculum schedules, and live-beam instrumentation are separate later changes. Introduce them through independent ablations after Run C works.

Start with a single mining/training round from a frozen post-DPO snapshot. This isolates the loss. Subsequently reuse the current controller to refresh negatives from the latest policy between rounds. Match that round lifecycle, including optimizer resets, in the ordinary-DPO control.

## 2. Inspected code and reuse decisions

| Existing component | Reuse / required change |
|---|---|
| `src/pretrain/train_iterative_ddro_vault.py` | Reuse query selection, generation commands, baseline/round retrieval evaluation, artifact hashing, and training lifecycle. Add an opt-in branch-extraction stage and branch configuration. |
| `src/scripts/bm25/the_vault/mine_model_confusion_negatives.py` | Reuse constrained beam generation, target encoding, known-positive filtering, and complete preference rows. No mining behavior change in v1. |
| `src/pretrain/mine_on_gpus.py` | Reuse existing sharding, deterministic merging, and completed-shard reuse. Derive branches after the merged pair file is complete; no branch-specific shard format is needed. |
| `src/data/data_prep/prepare_vault_dpo_metadata.py` | Reuse prepared query keys, exact per-row positive sets, corpus identities, and family splits. Preserve current label requirements. |
| `src/pretrain/train_ddro_vault.py` | Add opt-in trainer selection, arguments, branch-aware dataset validation, and metadata retention. Current cleaning removes every column except `prompt/chosen/rejected`. |
| `src/pretrain/iterative_dpo_utils.py` | Reuse hashing and round verification; include new files/artifacts in the controller's immutable-input contract. |
| `src/utils/trie.py` | Reuse legal-child lookup on `[decoder_start] + prefix`. No global trie rewrite. |

The repository and inspected local environment use TRL 0.11.4. Its `concatenated_forward` returns chosen/rejected sequence log probabilities and their full logits, so the branch term can reuse the existing policy/reference forwards. Its encoder-decoder labels and logits are aligned; do not add a causal-LM shift. Its collator treats additional scalar fields as Python lists, so a small wrapper must tensorize branch metadata explicitly. `max_target_length` maps to `max_completion_length` in this version; verify the effective limit during preflight.

## 3. Exact branch semantics

Tokenize both complete target IDs using the mining checkpoint's tokenizer, `add_special_tokens=True`, and the same target limit/policy used by training. Never split ID strings on slashes, separators, or characters to locate the branch.

For complete target token arrays c and r, find their longest common prefix of length k:

```text
c[:k] == r[:k]
c[k] != r[k]
h = c[:k]
a_positive = c[k]
a_negative = r[k]
```

Example: chosen `[A,B,C,D,EOS]`, rejected `[A,B,X,Y,EOS]` produces one decision `C > X` under `h=[A,B]`. Do not make an additional local comparison `D > Y`: those tokens have different histories.

At the corpus trie node for h, derive:

```text
legal_children = trie.get([decoder_start] + h)
positive_children = {
    p[k] for every known positive target p
    if p[:k] == h and len(p) > k
}
```

Both compared actions must be legal; the chosen action must be a positive child; the rejected action must not be any positive child. A non-positive leaf can live inside a relevant branch. Such a pair is valid for sequence DPO but is ineligible for this first-divergence branch auxiliary. Keep its sequence row and explicitly record the branch skip reason.

Decisions at k=0 are valid and important: the shared history is empty apart from decoder start. Genuine EOS-versus-continuation choices are valid. Decoder start is never a target action. Do not append artificial EOS to an unfinished prefix. Identical tokenized targets cannot supply a branch preference.

Use the positive set attached to the prepared `query_key`, exactly as current metadata defines it. Do not silently aggregate labels across distinct prepared query records or infer that a provenance/source ID is relevant. Unknown relevance remains a potential false-negative source and must be audited rather than described as verified irrelevance.

## 4. New modules and artifacts

### `src/pretrain/branch_preference_utils.py`

Provide reusable, testable helpers:

- Complete target tokenization/index construction using existing mining helpers.
- `first_divergence(chosen_tokens, rejected_tokens)`.
- Legal-child and per-query positive-child lookup.
- Branch eligibility and explicit skip-reason classification.
- Stable decision keys and multiplicity weights.
- Validation of branch metadata against actual training labels.

Keep pure token/list logic importable without torch where possible. A small torch loss helper can live in the trainer module.

### `src/pretrain/prepare_branch_preferences.py`

A tokenizer-only preparation CLI, invoked after existing single- or multi-GPU mining. Inputs: original pairs, selected query metadata, corpus document metadata, policy checkpoint/tokenizer, target type, target policies/limits, and round identity.

Outputs:

- `branch_preferences.jsonl`: the same original sequence pairs, in the same order, with additional branch fields. No new leaf negatives, no new chosen targets, no dropped sequence rows.
- `branch_audit.json`: eligible/skipped rows and queries, reasons, divergence-depth histogram, repeated decision counts, weighting totals, target diagnostics, and source/tokenizer fingerprints.

Retain original `preferences.jsonl` unchanged. No extra model generation or GPU inference is required to extract branches.

Minimum additional row fields:

```json
{
  "branch_schema_version": 1,
  "branch_active": true,
  "branch_skip_reason": "",
  "branch_position": 2,
  "branch_prefix_tokens": [101, 102],
  "branch_preferred_token": 103,
  "branch_rejected_token": 104,
  "branch_positive_children": [103, 105],
  "branch_key": "stable decision fingerprint",
  "branch_loss_weight": 1.0
}
```

The position is a zero-based ordinal in the unpadded target labels, excluding decoder start. Use schema-consistent sentinel values for inactive rows; their branch loss weight is zero. All rows keep `query_key`, the original provenance, and the full known-positive set for auditing. The illustrative weight above is replaced by the dataset-level normalization described below.

The audit must prove that original sequence preferences are unchanged and that their existing ground-truth/provenance checks still pass. Recompute annotations when pairs, tokenizer, corpus, selected queries, or target policies change.

### `src/pretrain/branch_preference_trainer.py`

Implement `BranchDPOTrainer` as a narrow TRL subclass, plus a branch configuration/collator wrapper. Do not modify installed TRL files and do not copy the TDPO implementation wholesale.

- Delegate ordinary sequence tokenization and padding to the existing TRL path.
- Validate that retokenized labels reproduce the recorded prefix and actions exactly; reject stale or mismatched annotations.
- Remove branch-only fields before the base collator, then add typed branch tensors. Preserve diagnostic strings separately; never forward them as model inputs.
- Extend `get_batch_loss_metrics` to reuse one policy and one frozen-reference concatenated forward, retaining chosen logits from each.
- Gather both branch action scores from the **chosen-prefix logits at the same position**. Do not gather the negative action from a later rejected-history position.
- Freeze/evaluate the reference and retain existing dropout handling.
- Store objective, coefficient, effective branch beta, and schema version in saved training configuration.
- With branch coefficient zero, delegate to the ordinary DPO loss path so a parity check can be exact.

## 5. Loss and normalization

At a valid node, use:

```text
z_branch = beta_branch * [
    log p_policy(a_positive | q,h) - log p_policy(a_negative | q,h)
  - log p_reference(a_positive | q,h) + log p_reference(a_negative | q,h)
]
loss_branch = softplus(-z_branch)
loss_total = loss_sequence_DPO + lambda_branch * weighted_branch_mean
```

Because both actions share the same state, each model's softmax normalizer cancels. Compute z directly from the two policy logits and two reference logits, casting to FP32 before differences. This avoids an additional full-vocabulary KL calculation and is algebraically the local log-ratio DPO loss.

Keep standard sequence DPO unchanged, including its EOS contribution. In v1, `beta_branch` defaults to the existing beta and `lambda_branch` starts at 0.1; tune the coefficient independently after the parity control. Positive-prefix SFT and adaptive weights start disabled.

Repeated generated leaves may encode the same local decision. Define a decision by `(query_key, prefix tokens, preferred token, rejected token)`. Preserve sequence rows but avoid repeated auxiliary weight.

For uniform row sampling, let N be the number of sequence rows, Q the number of represented query keys, B_q the number of distinct eligible branch decisions for query q, and M_b the number of rows representing decision b. Each eligible row gets:

```text
branch_loss_weight = N / (Q * B_q * M_b)
```

Inactive rows get zero. The expected auxiliary objective is then an average over queries and distinct decisions, with zero contribution from queries lacking valid branches. Do not renormalize weights independently within each minibatch. Compute normalization after final dataset validation; branch-enabled training must not silently drop rows after weights are assigned. Log maximum weights and coverage before training to identify low-coverage/high-variance datasets.

`query_key` is the balancing unit in v1. Different prepared records with the same normalized prompt remain separate according to the existing metadata contract; report that duplication rather than silently changing the experiment.

## 6. Changes to existing entry points

### Trainer

Add opt-in options to `train_ddro_vault.py`:

```text
--preference_objective {dpo,branch-dpo}     default dpo
--branch_loss_weight FLOAT               default 0 for ordinary dpo
--branch_beta FLOAT                     default existing beta
```

Branch mode requires validated branch annotations in its training file. Ordinary mode retains the current three-column cleaning and existing behavior. Branch mode retains a strict allowlist of required metadata and rejects malformed rows rather than repairing labels silently. Use a saved configuration subclass or equivalent explicit serialization for branch settings, not unsaved global variables.

`--dry_run` must show the shared token prefix, both actions, label position, eligibility/skip reason, and weighted coverage in addition to current target examples. It should catch alignment and schema failures before loading a training reference or starting distributed work.

### Controller

Add validated branch options to its configuration and training-option plumbing. Ordinary DPO remains the default; branch options become part of experiment identity.

Add this stage after existing merged mining/auditing and before `round-inputs`:

```text
prepare -> select -> mine -> audit original pairs
        -> annotate/audit branches -> snapshot round inputs
        -> train -> existing reference handling -> existing retrieval evaluation
```

For branch runs, `pairs_file` in the training manifest points to `branch_preferences.jsonl`. Retain an additional original-pairs path/hash for provenance. Hash augmented pairs, branch audit, tokenizer identity, and all new implementation modules. Verify that a completed annotation stage can resume without regenerating or changing mining outputs.

The raw-pair audit remains responsible for leaf ground truth and quotas. A branch audit validates auxiliary semantics separately. A nonempty sequence dataset with zero eligible branches should still train ordinary sequence DPO behavior and report zero auxiliary coverage; an entirely empty mined dataset retains the existing skipped-round behavior.

Add the new stage to start-round skipping/identity checks. Do not resume a pre-change experiment as a branch experiment or bypass code hashes. Newly created ordinary and branch runs must support normal interrupted resume. Any migration of older completed manifests is a separate, explicit compatibility task.

Reuse the current reference mechanism for the initial controls: initialize policy and reference from the completed DPO checkpoint, then use EMA decay 1.0 to keep the reference weights fixed. The default replacement/EMA behavior of other experiments is unchanged. No new reference-update algorithm is needed for v1.

The existing controller starts a new optimizer/scheduler per round. V1 reuses that lifecycle and matches it across controls; do not claim continuous optimizer state across refreshed rounds. A continuous-state controller is a later independent experiment.

### Launch/configuration

Reuse `src/scripts/ddro/run_vault_iterative_dpo.sh`; add separate branch-preference URL and structure-v6 example configs. Both need explicit paths to the user's completed normal-DPO checkpoint and real query/corpus assets. Current example configs point to SFT checkpoints and must not be presented as post-DPO continuation configs.

Pilot: one fixed mining round, about 10,000 selected training queries, matched optimizer updates, frozen reference, identical generation settings, branch beta equal to sequence beta, branch coefficient 0.1, and evaluation at the end. Keep learning rate/precision/batch matched to the ordinary continuation control; do not simultaneously introduce the previous configs' higher learning rate or moving reference.

For branch pilots, require complete target sequences and an unambiguous token-to-document mapping. Audit the tokenizer maximum before selecting a length limit. Existing configs allow truncation, collisions, and special tokens; if actual data require these policies, report the conflict during preflight and design an explicit alias/special-token experiment rather than silently changing IDs or excluding evaluation positives. Missing positive fields in multilabel query files continue to fail as they currently do.

## 7. Validation plan

### Pure extraction tests: `tests/test_branch_preferences.py`

- First divergence at token zero, a long shared prefix, and unequal lengths.
- Real EOS-versus-continuation branches; no synthetic prefix EOS.
- Identical tokenized targets and malformed/overlength encodings.
- A negative leaf whose divergent child leads to another positive: auxiliary inactive, sequence row retained.
- Multiple positive children, legal-child validation, and singleton nodes.
- Deterministic decisions, unchanged row order/content, and complete skip statistics.
- Duplicate leaf pairs mapping to one decision; exact query/decision normalization.
- No implicit positive-label fallback or transitive label expansion.
- Provenance mismatch and corrupted branch metadata are rejected.

### Loss/collator tests: `tests/test_branch_preference_trainer.py`

- Independent local DPO formula versus implementation and direct-logit equivalence.
- Both actions are read under the same chosen prefix.
- Correct encoder-decoder ordinal/label alignment; variable-length batching, padding, and EOS.
- Policy gradients on the selected actions, no reference gradients, and no direct local-loss gradient on other logit positions. Earlier hidden states may still receive gradients through the network.
- Coefficient zero recovers ordinary DPO; all-inactive batch has finite zero auxiliary contribution.
- Duplicate-decision weighting and batching preserve the intended full-dataset objective.
- FP32 branch arithmetic remains finite with mixed-precision input logits.
- Tokenization limits and stored branch configuration round-trip correctly.

### Integration/controller tests

- Tiny locally created T5 model: real mining -> branch extraction -> one training step -> checkpoint/export -> interrupted resume.
- Single- and multi-GPU merged-mining compatibility: annotations are derived once from the committed merged pairs.
- Fixed reference stays fixed; reference/tokenizer mismatch fails.
- Changed annotations/config/code fail resume; interrupted annotation stages regenerate deterministically.
- Ordinary-DPO default commands and datasets remain behaviorally identical.
- Existing controller, metadata, sharding, round-resume, and real-T5 integration checks still pass.

No extra model download is required for the CPU fixture. The existing `.venv-tdpo-test` provides the relevant local TRL version; record actual package versions when running the later checks. A real V100 smoke run is still needed to establish GPU memory and throughput.

## 8. Experiment and diagnostics

First compare:

| Arm | Objective / inputs |
|---|---|
| A | Ordinary DPO continuation on the original fixed mined pairs |
| C0 | Branch trainer with coefficient zero, same sequence rows |
| C | Ordinary sequence DPO + valid local branch DPO |

C0 establishes implementation parity. A versus C tests the branch auxiliary without a negative-pool or curriculum change. A fixed-prefix-weight baseline can be evaluated later using the existing token-level implementation without importing that entire branch into this implementation.

Use the same initial checkpoint, reference, prepared split, mined pair content, sequence exposure, optimizer/scheduler lifecycle, decoding limits, beam width, and length penalty. Preserve and report branch-ineligible rows. Confirm the selected coefficient with multiple seeds and query-level uncertainty; compare URL and structure IDs independently.

Log sequence/branch loss separately, eligible-row and query coverage, branch depth, action margin/accuracy, distinct decision count, weight distribution, and preprocessing/training cost. Existing retrieval evaluation supplies MRR@10 and true multi-positive Recall@10.

For a fixed development branch benchmark, mine preferences once from the initial DPO checkpoint on held-out development queries, annotate them, and evaluate action scores using the loss helpers without optimizer updates. Keep this benchmark separate from training and fixed across policy snapshots. It can be provided through a small later `eval_branch_preferences.py` entry point; do not alter the current training-only round contract to pass held-out examples into training.

Training branch accuracy is not proof of retrieval improvement. Full live-beam prefix-survival instrumentation is a later diagnostic extension; existing completed beam outputs do not establish when a path was pruned.

## 9. Follow-up Run D and curriculum

Only after C is validated:

1. Add detached, bounded branch weights based on policy scores at the shared prefix. The current miner's complete `model_score` is not a local branch margin and must not substitute for one. Gather scores from a frozen policy snapshot or add a separate prefix-scoring pass.
2. Compare static and refreshed negatives using the same ordinary/branch losses and reference, matching round resets.
3. Add a measured curriculum over early-routing and later-discrimination decisions, with replay and a shuffled-order control. Define difficulty from observed branch mistakes, not raw ID length alone.
4. Compare positive-prefix SFT separately; require explicit coefficient-zero parity and query/positive-prefix normalization.
5. If generated negatives yield too few eligible branches or too little early-branch coverage, add direct legal-child mining from teacher-forced positive prefixes as a separately named mining ablation. Do not silently backfill v1 data.

## 10. Implementation order and completion criteria

1. Pure branch helpers, extraction CLI, schema, and extraction tests.
2. Trainer/collator integration, metadata retention, loss/gradient/parity tests.
3. Controller stage/config/manifests, resume checks, and separate example configs.
4. Tiny real-T5 end-to-end validation and ordinary-DPO regression checks.
5. One matched A/C0/C pilot per ID design, then adaptive/curriculum work if the evidence warrants it.

The first implementation is ready for a full pilot when the same original pairs survive preparation, all valid branch actions share a verified state, positive branches cannot become negative actions, coefficient-zero parity holds, frozen-reference and resume checks pass, and extraction reports enough valid decisions to make the proposed comparison meaningful.
