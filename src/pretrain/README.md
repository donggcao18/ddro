# Vault DPO and Token-Level DPO

See [the token-level experiment summary and improvement plan](TOKEN_LEVEL_EXPERIMENT.md)
for the current Ruby URL setup, reported observations, and proposed retrieval-specific ablations.

`train_ddro_vault.py` supports `--preference_objective dpo|tdpo1|tdpo2|tdpo2-weight`.
The default remains `dpo`. All TDPO variants reuse the same SFT checkpoint,
tokenizer, JSONL `prompt/chosen/rejected` data, validation split, and distributed
training launcher as DPO. No preprocessing or BM25 rerun is required.

The implementation in `tdpo_trainer.py` adapts the loss from
`Token-level-Direct-Preference-Optimization/trainers.py` to encoder-decoder models
using the project's pinned TRL 0.11.4 and Transformers 4.45.2. It does not import
or require the reference repository's Hydra, FSDP, or dataset training loop.

For each response, `r` is the sum of token log-probability ratios between policy
and frozen reference; `K` sums `KL(reference || policy)` over decoder positions.
Padding labels (`-100`) are excluded, EOS is included, and the decoder labels
are used without the extra causal-model shift.

```text
TDPO1: -logsigmoid(beta * (r_chosen - r_rejected - (K_rejected - K_chosen)))
TDPO2: -logsigmoid(beta * (r_chosen - r_rejected
                         - alpha * (K_rejected - stop_gradient(K_chosen)))))
```

`--tdpo_alpha` defaults to `0.5` and affects only TDPO2; `--beta` is shared with
DPO. TDPO2 with alpha zero reduces to the DPO objective. Loss statistics use
FP32 log-softmax; reference logits have no gradient. The logged rewards follow
the reference implementation (`beta * (r + K)`), so their accuracy is a
diagnostic, not necessarily the sign of the TDPO2 preference logit. KL metrics
include chosen, rejected, and rejected-minus-chosen sums. The objective and
alpha are serialized in `training_args.bin` alongside ordinary training options.

## Run With Existing BM25 Pairs

Activate your existing `ddro_env` and run from the repository root:

```bash
PREFERENCE_OBJECTIVE=tdpo2 \
TDPO_ALPHA=0.5 \
RUN_BM25=0 \
NUM_GPUS=2 \
TRAIN_BATCH_SIZE=4 \
EVAL_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=4 \
bash src/scripts/ddro/run_vault_structure_v3_bm25_and_dpo.sh
```

Use `PREFERENCE_OBJECTIVE=tdpo1` to run TDPO1 (alpha is ignored), or `dpo` for
the existing baseline. The launcher includes the objective in its default output
directory. Override `CHECKPOINT_PATH`, `WORK_DIR`, and `DPO_OUTPUT_DIR` as needed;
the checkpoint must be your original SFT checkpoint. `DPO_BETA` controls beta for
all three objectives. This command's effective batch size is `2 * 4 * 4 = 32`.
The batch sizes are a starting point: TDPO retains full vocabulary distributions
and may require more memory than DPO.

For an initial training smoke run, add `MAX_STEPS=2` and a separate
`DPO_OUTPUT_DIR`. `RUN_DRY_RUN=1` only validates data/checkpoint loading and
prints tokenization examples; it does not execute the loss or backward pass.
When resuming through `--resume_from_checkpoint`, pass the same objective,
alpha, beta, and original SFT checkpoint used for that run. The CLI flags select
the objective; they are not automatically restored from a checkpoint.

## Prefix-Weighted TDPO2 on Four V100 GPUs

`tdpo2-weight` applies `--tdpo_prefix_weight` (default `3.0`) to the first
`--tdpo_prefix_tokens` valid decoder targets (default `3`), and weight `1` to
later targets. Both the token log-ratio and forward-KL sums use these weights,
for chosen and rejected responses, before the existing TDPO2 loss. The chosen
KL remains detached. There is no length/weight normalization or auxiliary SFT
loss. This is an experimental modified objective, not the original TDPO2.

For example, `N=3, weight=3` gives `[3, 3, 3, 1, 1, ...]`. Padding has no
contribution and does not consume prefix positions. EOS is included and counts
as a target if the response is shorter than N; the decoder start token and
encoder prompt are not targets. Weights are assigned independently to each
response. `--tdpo_prefix_weight 1` or `--tdpo_prefix_tokens 0` reproduces TDPO2.
The prefix settings have no effect on `dpo`, `tdpo1`, or `tdpo2`.

The four-GPU launcher defaults to a **new run**, four visible GPUs, batch size
1 preference pair per GPU, accumulation 8 (effective batch 32), gradient
checkpointing, FP32, and two epochs. It uses the existing URL JSONL and SFT
checkpoint defaults, without rerunning mining. DDP replicates policy and frozen
reference on every GPU; it does not combine four GPUs' memory into one pool.
Batch sizes are conservative starting points, not a guarantee that every model
size fits a V100. Each preference pair forwards both chosen and rejected.

V100 has no native BF16 support. FP32 is the default because T5 checkpoints can
overflow in FP16; `PRECISION=fp16` is available after a smoke run confirms finite
loss and gradients. See [Hugging Face mixed precision guidance](https://huggingface.co/docs/transformers/v5.10.1/mixed_precision_training)
and [T5 overflow debugging](https://huggingface.co/docs/transformers/main/debugging).
The launcher rejects `PRECISION=bf16`.

On a Linux training node with four assigned V100 GPUs, activate your existing
training environment and run from the repository root:

```bash
CHECKPOINT_PATH=/path/to/sft/checkpoint \
TRAIN_FILE=/path/to/dpo_pairs_url.jsonl \
TDPO_PREFIX_TOKENS=3 TDPO_PREFIX_WEIGHT=3 \
bash src/scripts/ddro/launch_tdpo2_weight_vault_url_4v100.sh
```

`CUDA_VISIBLE_DEVICES` supplied by the scheduler is honored; otherwise it
defaults to `0,1,2,3`. This is single-node `torchrun` DDP. Obtain four GPUs from
your scheduler before launching; the script does not submit a scheduler job.
`PYTHON_BIN` selects the environment's Python executable.

Compare two and three prefix tokens in separate output directories (the default
directory includes objective, N, and weight):

```bash
for n in 2 3; do
  TDPO_PREFIX_TOKENS="$n" TDPO_PREFIX_WEIGHT=3 \
    bash src/scripts/ddro/launch_tdpo2_weight_vault_url_4v100.sh
done

# Same batch/precision defaults, ordinary TDPO2 baseline.
PREFERENCE_OBJECTIVE=tdpo2 \
  bash src/scripts/ddro/launch_tdpo2_weight_vault_url_4v100.sh

# Two-step smoke test in a separate directory; performs real forward/backward.
OUTPUT_DIR=outputs/tdpo2-weight-smoke \
  bash src/scripts/ddro/launch_tdpo2_weight_vault_url_4v100.sh \
    --max_steps 2 --save_steps 2 --eval_steps 2 --logging_steps 1

# Print the command without loading data or starting GPU processes.
PRINT_ONLY=1 bash src/scripts/ddro/launch_tdpo2_weight_vault_url_4v100.sh

# Resume a matching experiment with the same SFT reference and prefix settings.
RESUME_FROM_CHECKPOINT=latest TDPO_PREFIX_TOKENS=3 TDPO_PREFIX_WEIGHT=3 \
  bash src/scripts/ddro/launch_tdpo2_weight_vault_url_4v100.sh
```

Override `OUTPUT_DIR`, `CHECKPOINT_PATH`, `TRAIN_FILE`, `DATASET_CACHE_DIR`,
`TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `BETA`,
`TDPO_ALPHA`, or `PRECISION` as needed. Use the same weight spelling and output
directory when resuming (`3` and `3.0` produce different default directory names).
To enable validation, use `VALIDATION_SPLIT=0.01` or pass `--eval_file PATH`.
The existing default `VALIDATION_SPLIT=0` trains on all pairs. Final portable
weights are saved under `OUTPUT_DIR/final`; checkpoints are resumable.

The objective, alpha, prefix count, and prefix weight are saved in
`training_args.bin`, but must still be passed consistently when resuming.
`logps/*` metrics remain ordinary unweighted log probabilities. With
`tdpo2-weight`, `kl/*` and reward diagnostics use weighted sums, so their scales
are not directly comparable to baseline TDPO2. The larger preference score may
also saturate the sigmoid sooner; compare retrieval performance, not just loss.
Identical chosen/rejected prefix log ratios cancel, so inspect tokenized URL
prefixes if the weighted positions mostly contain shared syntax.

## Faster Tokenization Cache Loading

For URL TDPO training, the convenience launcher defaults to visible GPUs 0,1
(or respects an existing scheduler-provided CUDA_VISIBLE_DEVICES), batch size
16 per GPU, accumulation 1, TDPO2, bf16, gradient checkpointing, two epochs,
and a 7200-second timeout. It resumes the latest checkpoint in
`outputs/vault-url-tdpo2-full` and does not rerun BM25:

```bash
bash src/scripts/ddro/launch_tdpo_training_vault_url.sh
```

Override `DATASET_CACHE_DIR` with a sufficiently large node-local disk path for
faster cache I/O. The persistent fallback is `data/hf_cache_w1000` under the
repository (or `HF_DATASETS_CACHE` if set); it is not assumed to be local SSD.
Other environment overrides include `CHECKPOINT_PATH`, `TRAIN_FILE`,
`OUTPUT_DIR`, `CUDA_VISIBLE_DEVICES`, `TRAIN_BATCH_SIZE`, `DDP_TIMEOUT`,
`TOKENIZATION_WRITER_BATCH_SIZE`, and `STARTUP_DEBUG`.

```bash
# Inspect the command without touching data or launching training.
PRINT_ONLY=1 bash src/scripts/ddro/launch_tdpo_training_vault_url.sh

# One GPU with the same effective batch size of 32.
CUDA_VISIBLE_DEVICES=0 TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
  bash src/scripts/ddro/launch_tdpo_training_vault_url.sh
```

`RESUME_FROM_CHECKPOINT` accepts `latest` (default), an explicit training
checkpoint path, or `none` for a new run. Use a separate `OUTPUT_DIR` for new
experiments. The original SFT checkpoint remains `CHECKPOINT_PATH`, even when
resuming. Extra arguments are passed to the Python training script; use the
environment overrides above for settings shown in the launch summary. Stop an
existing run before launching another job against the same output directory.

The Vault Python entry point now defaults to `--tokenization_writer_batch_size
1000`. TRL 0.11.4 hard-codes 10 rows per Arrow write batch; the local
`TokenizationCacheDataset` adapter overrides only TRL's `_tokenize` map call.
It keeps the original tokenizer, truncation, labels, row order, and split
selection indices. DPO, TDPO1, and TDPO2 use the same adapter. GPU batch size
and gradient accumulation are independent of this cache setting.

A versioned fingerprint includes the input dataset fingerprint and writer size,
so old ten-row caches are not silently reused. The first run rebuilds tokenized
caches without deleting the originals. Changing the writer size also rebuilds
the layout. A successful map prints `[token-cache ...] mapping/cache reopen DONE`
with elapsed time and the resulting Arrow cache paths.

Use `--dataset_cache_dir` to place both the JSON loader cache and downstream
dataset caches on sufficiently large node-local disk. For example, add these
arguments to the existing single-node training command, only after confirming
that `/tmp` is suitable disk-backed storage with enough free space:

```bash
--dataset_cache_dir "/tmp/${USER}/ddro-cache-w1000" \
--tokenization_writer_batch_size 1000
```

Every rank on the node must use the same path. Do not use a per-rank cache
directory. Multi-node jobs need an appropriate cache strategy on each node;
this example is for the current single-node, two-GPU run. Node-local caches
may disappear when a job ends, so use persistent local scratch when available
and keep model checkpoints in the existing persistent output directory.

Larger Arrow write batches reduce the number of record batches and can reduce
cache metadata overhead; they do not guarantee a specific speedup or fix all
filesystem/RAM problems. Start at 1000. Very large writer sizes can increase
CPU-memory use. A longer `--ddp_timeout` only permits a longer wait; it does not
make cache loading faster. The adapter does not change resume state, but the
installed TRL/Datasets runtime and actual startup speed should be checked with
a smoke run on the training machine.

```bash
python -m unittest discover -s src/pretrain -p test_preference_cache.py
python -m unittest discover -s src/pretrain -p test_tdpo_trainer.py
```

The cache tests compare token values and row order with the original TRL map,
check the Arrow batch count on disk-backed data with one/two CPU processes,
and exercise selection indices and cache fingerprints. The trainer tests use
the adapter during tiny CPU DPO/TDPO train/eval/save/resume runs.

## Diagnosing Distributed Startup Hangs

The Python entry point accepts `--ddp_timeout SECONDS` (default: `1800`). Use
`--ddp_timeout 7200` for a two-hour distributed process-group timeout. This is
not a training duration limit and does not accelerate preprocessing. A startup
operation taking approximately two hours can still exceed this timeout because
the waiting rank may enter the barrier before tokenization begins.

Add `--startup_debug 60` to the Python training arguments on all ranks, and add
`--log-dir ./logs/tdpo-startup --tee 3` to `torchrun` before the script path.
This enables flushed rank/PID/time markers and Python stack dumps every 60
seconds until the trainer constructor returns. Dumps during healthy, slow
startup are expected too: a dump is a snapshot, not itself an exception.
They go to each rank's stderr, captured in torchrun's log directory.

After TrainingArguments initializes distributed state, a one-element all-reduce
tests the process group on the assigned device before loading models and
entering TRL's rank-zero-first tokenization block. It logs `Communication probe
BEGIN` and `Communication probe PASSED` with the expected world-size sum. The
probe initializes communication earlier than normal and may change the symptom;
it is a diagnostic, not proof that the original problem has been fixed.

If the probe stalls, inspect both ranks' stacks and device/backend markers. If
both ranks pass but trainer initialization stalls, inspect rank zero's stack
for dataset/cache work versus a distributed/CUDA wait. A 100% tokenization bar
does not prove the surrounding dataset operation and synchronization completed.
The rank-one NCCL store timeout alone cannot distinguish these cases.

Keep the same data, SFT reference, TDPO settings, and resume checkpoint. These
diagnostics do not change the objective or checkpoint format and are disabled
by default. They do not skip training: a successful startup resumes the normal
run, and the periodic dumps stop before `trainer.train`. No dependency upgrade
is needed. A Python stack may identify a native call without showing the native
code's internal wait, so further CUDA/NCCL diagnostics may still be necessary.

The diagnostics' control flow can be tested without installing ML libraries:

```bash
python -m unittest discover -s src/pretrain -p test_startup_diagnostics.py
```

## Verification

```bash
python -m unittest discover -s src/pretrain -p test_tdpo_trainer.py
```

Tests cover loss/gradient parity with the supplied reference when its checkout
is present, T5 alignment and padding, TDPO2 gradient detachment, alpha-zero DPO
equivalence, numerical stability, and tiny CPU T5 train/eval/save/resume runs
for all four objectives. Weighted tests also cover manual loss/gradient parity,
unit-weight equivalence to TDPO2, masked prefix positions, short responses,
parameter validation, and prefix-setting persistence. No downloaded model or
dataset is needed.
