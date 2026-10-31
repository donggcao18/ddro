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
not just positives. Same-target documents share a retrieval action; distinct
target strings that tokenize identically are rejected. Unmapped or missing
positive targets and overlong targets are rejected, rather than silently dropped.

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

Four-V100 launcher (or set `NUM_GPUS=1` for one GPU):

```bash
CHECKPOINT_PATH=/models/ruby-structure-v6-sft/checkpoint-N \
MAX_TARGET_LENGTH=128 \
WORK_DIR=/data/vault_bm25_structure_v6 \
OUTPUT_DIR=/models/ruby-grpo-v6 \
bash src/scripts/grpo/train_vault_structure_v6_4v100.sh
```

The launcher prepares data, runs a rollout dry run, then trains. Set
`RUN_PREPARE=0`, `RUN_DRY_RUN=0`, or `RUN_TRAINING=0` to disable stages.
`QUERY_METADATA`, `DOCUMENT_METADATA`, and `GRPO_DATA_DIR` override input paths.
`NUM_GENERATIONS`, `TRAIN_BATCH_SIZE`, `TEMPERATURE`, `GRPO_BETA`, `EVAL_NUM_BEAMS`,
`MAX_STEPS`, and other variables in the script expose common training settings.

Resume with `--resume_from_checkpoint outputs/vault-grpo/checkpoint-N` or the
launcher's `RESUME_FROM_CHECKPOINT` variable. HF checkpoints contain optimizer,
scheduler, trainer and RNG state. The final portable model/tokenizer is saved in
`OUTPUT_DIR/final/`. `grpo_run.json` records the launch configuration.

## Objective and diagnostics

The objective uses group-centered rewards divided by group population std plus
`1e-4`, the clipped token-level policy surrogate, and trajectory-then-batch
averaging. It uses one fresh on-policy rollout per training microbatch, equivalent
to the **single-iteration GRPO regime**. Old-policy log probabilities are the
detached current-policy scores because no update occurs between sampling and
scoring; there is no stale rollout cache or multi-iteration replay. Consequently,
ratios start at one and clipping does not activate in this initial version.
The scalar policy loss can be zero even while its gradient is nonzero, because
group-centered advantages sum to zero. Track reward, mixed groups, gradient
norm, and held-out retrieval metrics when assessing learning.

Sampling has no top-k/top-p truncation, repetition penalty, or forced-token
processors. Scoring uses exactly the same trie-normalized distribution and
temperature. Dropout is disabled, including T5 attention's functional dropout.
The first EOS is included in the loss; decoder start and padding are excluded.
The trie validates that every corpus ID can finish within the decoder limit.

`beta=0` (default) loads only the policy. Setting `--beta` above zero adds a frozen
copy of the initial SFT checkpoint for the sampled KL penalty, using the same
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
python -m unittest discover -s src/pretrain/tests -p 'test_grpo.py' -v
```

Validation includes offline tiny-T5 training/evaluation, variable-length rollout
probability checks, and resume equivalence to uninterrupted training. A separate
two-process CPU DDP smoke run also passed. CUDA/FP16 and four-V100 execution must
be validated on GPU hardware.
