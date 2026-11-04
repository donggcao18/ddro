# Experiment 2: token-level trajectory GRPO for URL retrieval

Run the pilot from the repository root on four allocated V100s:

```bash
bash src/scripts/grpo/run_url_tca_grpo_4v100.sh
```

The starting policy and frozen reference both use your requested full-data SFT
checkpoint:
`/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url__qg_t5_url_max_lr/checkpoint-160000`.
Override `CHECKPOINT_PATH`, `TRAIN_INPUT_FILE`, `TEST_INPUT_FILE`, and optional
`EVAL_INPUT_FILE` / `FIXED_CORPUS_FILE` as in the coverage suite. The default
training input is `Ruby_merged_multilabel_dpo.jsonl`, with native `text` and
`positive_url_based_id` fields; no metadata joins or new relevance labels are
required. The output root is a fresh timestamped `url-tca-grpo-*` directory.

## Method and correspondence to the authors

This is an independent implementation inspired by
[Zhao et al., Token-Level Credit Assignment Optimization for Generative Document Retrieval](https://arxiv.org/abs/2608.12049)
and checked against the authors' [reward functions](https://github.com/colinzhaoxp/TCA-RL4GR/blob/main/tca_grpo/rewarder.py)
and [T5 trainer](https://github.com/colinzhaoxp/TCA-RL4GR/blob/main/tca_grpo/t5_grpo_trainer.py).
It is a multi-positive adaptation, not a claim of exact paper reproduction.

For each query, the policy samples eight independent trie-constrained DocIDs.
The reference teacher-forces up to two randomly selected annotated positives.
Any additional sampled ID that is itself an annotated positive also gets its
own gold reference trajectory. Thus a correct alternative is never compared
only against an unrelated designated positive.

For a wrong completion, choose **one whole gold trajectory** maximizing its
mean content-token reward, including overflow penalties. Never take a separate
maximum over different positive documents at each token. Compare final decoder
states by cosine similarity at corresponding token positions:

- Overlapping content positions: cosine similarity, clipped to `[-1, 1]`.
- Content beyond the selected gold's content length: `-0.5` per extra token.
- EOS: no cosine or overflow reward.
- Any complete, EOS-terminated ID in the query's full annotated positive set:
  add `2.0` to **every valid position, including EOS**.
- Padding and decoder-start positions do not receive rewards or gradients.

This matches the author's absolute-cosine/global-hit reward convention for a
single gold path. At each decoding depth, center rewards and compute the
population standard deviation across **valid** candidates in that query's
group. Padding never votes. Depths with fewer than two valid candidates or
standard deviation at most `REWARD_STD_FLOOR=0.001` get zero advantage; otherwise
use the centered reward divided by that standard deviation. The floor prevents
tiny cosine rounding differences being amplified into large advantages.

The frozen gold states are computed on demand, without the reference LM head,
in chunks of at most 16 paths. An LRU stores up to 128 MiB of FP32 gold states
on CPU per process, keyed by query tokens, target ID, and alignment convention.
It is not an unbounded GPU or full-dataset trajectory cache. On resume it is
recomputed from the same immutable SFT reference.

## Alignment ablation for long shared prefixes

`TRAJECTORY_ALIGNMENT=pre_token` is the default author-style convention: state
`h_t` sees the prefix before token `y_t`. With identical prefixes, this state is
identical even if the candidates choose different next tokens. In particular,
wrong IDs differing only in their final content token can receive identical
trajectory rewards. Exact-hit bonuses can still separate correct completions;
the all-wrong auxiliary loss below supplies known-positive supervision.

To test a deliberate adaptation that evaluates the state **after** each chosen
content token, run a separate pilot:

```bash
TRAJECTORY_ALIGNMENT=post_token \
bash src/scripts/grpo/run_url_tca_grpo_4v100.sh
```

Both the policy and gold reference shift by one position. The state before EOS
evaluates the final content-token decision. This is not the author's default
alignment, and it must be evaluated rather than assumed to improve retrieval.

## Loss, rollout semantics, and all-wrong groups

The default hybrid objective is:

`L = token_GRPO + 0.04 * reference_KL + 0.1 * all_wrong_ranking + 0.1 * positive_SFT`.

Only queries with **no exact-positive sampled completion** receive the ranking
term. It compares the selected annotated positives with up to four distinct
sampled incorrect IDs using raw, summed, temperature-one trie scores:
`mean_positive log(1 + sum_negative exp(s_negative - s_positive))`.
The positive SFT anchor uses full-vocabulary mean-token NLL for the selected
positives, averaged per query. Both auxiliary terms are separate supervision;
gold IDs are not injected into the GRPO sample group. A single policy scoring
pass handles at most `G + gold_candidates_per_query` rows/query, reusing sampled
scores as negatives. Reference trajectories/reward observations are detached.

Set `ALL_ZERO_RANKING_WEIGHT=0 SFT_ANCHOR_WEIGHT=0` for an ablation isolating
token-reward GRPO plus KL. A changed parameter or nonzero gradient does not
establish a retrieval improvement; report the held-out metrics.

`DO_SAMPLE=true` is the default, temperature 1, no top-k/top-p truncation beyond
the trie. The author implementation also offers beam-selected training. To
compare that variant:

```bash
DO_SAMPLE=false NUM_GENERATIONS=10 \
bash src/scripts/grpo/run_url_tca_grpo_4v100.sh
```

Beam-selected training is selection-biased. The ratio does not correct that
bias. All modes use one fresh scoring/update pass per microbatch: the old-policy
log probabilities are detached current scores, so the ratio is initially one
and PPO clipping is inactive in this single-pass regime. This limitation is
explicit rather than implying a multi-epoch PPO replay implementation.

Preference scoring, validation, and final ten-beam inference share the same
fixed-catalog trie and score convention. Sampling temperature is applied to
sample probabilities and their KL; auxiliary ranking and inference use
temperature one. Changing sampling versus beam decoding is an intentional
rollout difference, not a different probability normalization convention.

## Pilot, memory, validation, and outputs

Defaults: FP16, LR `1e-6`, 100 optimizer steps, save/evaluate every 50 steps,
10% warmup, gradient checkpointing, checkpoint retention limit 2. To run one
full epoch set `MAX_STEPS=-1`. `HIT_BONUS`, `OVERFLOW_PENALTY`,
`REWARD_STD_FLOOR`, `GRPO_BETA`, `GOLD_CANDIDATES_PER_QUERY`,
`REFERENCE_BATCH_SIZE`, `GOLD_CACHE_MB`, and the auxiliary weights are configurable.

The dedicated GPU capacity probe runs the **actual TCA objective**, including
all returned hidden states, reference gold paths, KL, all-wrong fallback,
positive anchor, gradients, and two AdamW steps. It stresses full-length prompts
and the longest generated target. It reserves at least 4 GiB/runtime-DDP
headroom and accounts for resident accumulation gradients. The selected query
batch divides 16; accumulation maintains 16 queries/GPU/update, 64 globally.
The initial probe batch is 4, the cap 16. Memory calibration remains an estimate;
evaluation and inference batches default independently to two queries/GPU.

The runner prepares one inventory. By default the TCA preset reuses your
completed SFT baseline from
`/mnt/beegfs/scratch/congthanh_le/scratch/east/ddro/outputs/url-coverage-20261103T185157Z/sft_baseline`,
skipping its repeated test decoding. Set `BASELINE_DIR` to another completed
baseline directory if needed. Reuse verifies the checkpoint path, test-data
hash, full inventory, prompt/target limits, precision, beams, filtering policy,
and score convention. It copies the completed metrics/predictions into the new
run so comparisons and SFT fallback remain available. It accepts older coverage
suites using their saved data and training manifests; incomplete rank shards
are not accepted. Set `RUN_BASELINE_INFERENCE=1` to regenerate the baseline.
Training validation and final trained-candidate test inference still run.
Existing processes do not pick up script changes; restart into a fresh output
root if you want to skip an already-running baseline stage.
`suite_data.json` now includes positive-label-count histograms for train,
validation and test, so the prevalence of multi-positive queries is visible.

For this pilot, checkpoint selection requires **no measured validation Hit@1
regression** and at least **0.5 percentage points of validation Recall@10 gain**
over SFT (`HIT1_TOLERANCE=0`, `MIN_VALIDATION_RECALL_GAIN=0.005`). This replaces
accepting a tiny validation gain by default for the new experiment only.
`candidate_final/` always preserves the trained best candidate; `final/` contains
that candidate or an explicit SFT fallback, explained in `selection.json`.
`comparison.csv` / `comparison.json` compare candidate and selected models with
SFT using all 18 metrics and per-query ranking/coverage changes.

With the requested full-data SFT initialization, the default random 1%
post-training validation split may have already been seen by SFT. This remains
a retention experiment unless you supply independent `EVAL_INPUT_FILE` queries.
Neither the minimum gain nor the Hit@1 constraint guarantees test preservation.
Test labels are never used for training or selection.

Examples:

```bash
# Short end-to-end check, including baseline/candidate inference.
MAX_STEPS=20 EVAL_STEPS=10 SAVE_STEPS=10 \
bash src/scripts/grpo/run_url_tca_grpo_4v100.sh

# Reuse another finished baseline, without baseline GPU decoding.
BASELINE_DIR=/path/to/previous-run/sft_baseline \
bash src/scripts/grpo/run_url_tca_grpo_4v100.sh

# Use independent validation queries and an explicit corpus inventory.
EVAL_INPUT_FILE=/data/unseen-validation.jsonl FIXED_CORPUS_FILE=/data/corpus.jsonl \
bash src/scripts/grpo/run_url_tca_grpo_4v100.sh

# Data preparation only, without model/GPU requirements.
PREPARE_ONLY=1 OUTPUT_ROOT=/path/to/new/tca-data \
bash src/scripts/grpo/run_url_tca_grpo_4v100.sh
```

Monitor `grpo/token_reward`, `grpo/informative_depth_fraction`,
`grpo/advantage_abs_mean`, exact all-zero/all-one group fractions,
`grpo/all_zero_ranking_loss`, `grpo/sft_nll`, `grpo/kl`, and actual sampled
parameter updates. Group-centered policy-loss values alone are not a learning
or retrieval-quality diagnostic.
