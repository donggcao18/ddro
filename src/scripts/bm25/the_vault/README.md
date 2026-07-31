# Vault BM25 hard-negative pipeline

This pipeline mines DPO negatives for the Vault generative code retriever.

Identity rules:

- `numeric_id` links raw augmented pseudo-queries back to an original Vault row.
- `text_id` is the quantized document ID and the CodeT5 decoder target.
- Original `Query:` and `Code:` records are paired by their shared `text_id`.
- Synthetic `vault-...` query keys exist only because Pyserini requires query IDs.
- BM25 searches pseudo-query text against original `Code:` documents, never query against query.

## Outputs

`prepare_bm25.py` creates:

```text
work/
├── corpus/documents.jsonl      # {id: quantized text_id, contents: searchable code}
├── queries.tsv                 # synthetic query key + pseudo-query
├── qrels.tsv                   # all multi-label positives for each pseudo-query
├── query_metadata.jsonl        # maps query keys back to the target text_id
├── document_metadata.jsonl     # audit metadata for code documents
├── unmapped_queries.jsonl
└── prepare_stats.json
```

`mine_dpo_negatives.py` creates standard explicit-prompt preference records:

```json
{"prompt":"partition the predicate into its components.","chosen":"1012060615140114","rejected":"0706080109141201005","positive_text_ids":["1012060615140114"],"bm25_rank":7}
```

`chosen` and `rejected` are quantized Vault `text_id` values. Extra fields are audit
metadata and can be removed by a trainer that requires exactly three columns.

## 1. Prepare the corpus and queries

### One-command execution

With the repository's `pyserini` environment active, the complete pipeline can
be run as:

```bash
python src/scripts/bm25/the_vault/run_pipeline.py \
  --train-original /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_train_r32.0.json \
  --test-original /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_test_r32.0.json \
  --augmentation /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_ready_to_feed_numeric.jsonl \
  --work-dir /home/users/congthanh_le/scratch/east/ddro/data/vault_bm25 \
  --negatives-per-query 8
```

The final output is `/data/vault_bm25/dpo_pairs.jsonl`. The following sections
show the same process as separate, inspectable stages.

### Separate stages

Raw augmentation (`Ruby_q10.jsonl`) and `map.py`-ready augmentation are both
accepted. Auto mode resolves, in order, `numeric_id`, raw augmentation `text_id`
as a numeric ID, and an already-quantized `text_id`.

```bash
python src/scripts/bm25/the_vault/prepare_bm25.py \
  --train-original /data/Ruby_train_r32.0.json \
  --test-original /data/Ruby_test_r32.0.json \
  --augmentation /data/Ruby_ready_to_feed_numeric.jsonl \
  --output-dir /data/vault_bm25
```

`--test-original` is optional and repeatable. Include it when, as in
`build_multilable.py`, multi-label groups must be detected across train and test.
Use `--strict` after checking the data once if every augmentation row must map.

By default searchable documents contain repository, path, identifier, parameters,
URL-based ID, and source code. Add `--code-only` to index only the `Code:` body.

## 2. Build the index and retrieve top 200

Activate the repository's `pyserini` environment, then run:

```bash
bash src/scripts/bm25/the_vault/run_bm25.sh /data/vault_bm25 16 200
```

This creates `/data/vault_bm25/bm25_run.txt`.

## 3. Filter all positives and create DPO pairs

```bash
python src/scripts/bm25/the_vault/mine_dpo_negatives.py \
  --run /data/vault_bm25/bm25_run.txt \
  --query-metadata /data/vault_bm25/query_metadata.jsonl \
  --document-metadata /data/vault_bm25/document_metadata.jsonl \
  --output /data/vault_bm25/dpo_pairs.jsonl \
  --negatives-per-query 8
```

For each pseudo-query the miner removes the current target and every `text_id` in
its normalized-query multi-label group. The Vault defaults are:

- BM25 retrieves the top 200 with `k1=0.82` and `b=0.68`.
- 8 negatives are requested per query with random seed 42.
- 3 are sampled from ranks 1-20, 2 from ranks 21-100, and 3 from ranks 101-200.
- A query is skipped if filtering leaves too few candidates to satisfy a rank
  bucket.

`--fill-shortfall` is available as an explicit non-original fallback, but is
disabled by default.

The default creates pairs only for the augmentation row's mapped target. Add
`--pair-all-positives` if every valid multi-label target should also be emitted as
a `chosen` response.

## Performance on large augmentation files

With 200,000 pseudo-queries and `--hits 200`, Pyserini may write 40 million
result rows. This output volume, not CodeT5, is normally the dominant cost. The
postprocessor streams one query group at a time and writes DPO pairs immediately,
so its memory usage is bounded by roughly one query's 200 hits rather than the
complete run.

The search defaults retain the original settings: 16 threads and batch size 16.
Set `--threads` to the CPUs actually allocated by the scheduler. A larger
`--batch-size` such as 64 or 128 can improve throughput on machines with enough
memory without changing BM25 scores; benchmark it on a small shard first.

When rerunning the same corpus, avoid rebuilding Lucene:

```bash
python src/scripts/bm25/the_vault/run_pipeline.py \
  ... \
  --reuse-index \
  --threads 32 \
  --batch-size 64
```

Keep the work directory on node-local SSD/scratch rather than network storage,
because both the index and the 40-million-line run are I/O intensive.

Alongside `dpo_pairs.jsonl`, the miner writes:

- `dpo_pairs.tsv`: `query_key<TAB>chosen_text_id<TAB>rejected_text_id`.
- `dpo_pairs.stats.json`: filtering and output counts.

## Hybrid BM25 + model-confusion negatives

The model-confusion path is additive: the existing BM25 pipeline and its
`dpo_pairs.jsonl` output are not modified. Constrained beam search is run against
the frozen SFT checkpoint, then a separate combiner creates
`dpo_pairs_hybrid.jsonl`.

The default hybrid allocation is at most 4 model-confusion negatives and enough
unique BM25 negatives to retain 8 pairs per query. When fewer than 4 valid model
predictions remain after multi-label filtering, BM25 fills the shortfall.

Run all stages with:

```bash
python src/scripts/bm25/the_vault/run_hybrid_pipeline.py \
  --train-original /data/Ruby_train_r32.0.json \
  --test-original /data/Ruby_test_r32.0.json \
  --augmentation /data/Ruby_ready_to_feed_numeric.jsonl \
  --checkpoint-path /models/vault-sft/checkpoint-196000 \
  --work-dir /data/vault_bm25 \
  --bf16
```

For a cheap checkpoint smoke test, reuse existing BM25 output and mine only 100
queries:

```bash
python src/scripts/bm25/the_vault/run_hybrid_pipeline.py \
  --train-original /data/Ruby_train_r32.0.json \
  --augmentation /data/Ruby_ready_to_feed_numeric.jsonl \
  --checkpoint-path /models/vault-sft/checkpoint-196000 \
  --work-dir /data/vault_bm25 \
  --reuse-bm25-output \
  --limit-queries 100 \
  --bf16
```

This smoke-test output uses BM25 fallback for queries beyond the first 100. Use
`--require-exact-mix` if such queries should instead be omitted.

The same stages can be run separately:

```bash
python src/scripts/bm25/the_vault/mine_model_confusion_negatives.py \
  --checkpoint-path /models/vault-sft/checkpoint-196000 \
  --query-metadata /data/vault_bm25/query_metadata.jsonl \
  --document-metadata /data/vault_bm25/document_metadata.jsonl \
  --output /data/vault_bm25/model_confusion_pairs.jsonl \
  --negatives-per-query 4 \
  --num-beams 8 \
  --batch-size 64 \
  --max-prompt-length 256 \
  --max-target-length 20 \
  --bf16

python src/scripts/bm25/the_vault/combine_hybrid_dpo.py \
  --bm25-input /data/vault_bm25/dpo_pairs.jsonl \
  --model-input /data/vault_bm25/model_confusion_pairs.jsonl \
  --document-metadata /data/vault_bm25/document_metadata.jsonl \
  --output /data/vault_bm25/dpo_pairs_hybrid.jsonl \
  --model-per-query 4 \
  --total-per-query 8 \
  --seed 42
```

The model miner uses only valid `text_id` target sequences from
`document_metadata.jsonl`. It fails rather than truncating a DocID or accepting
two DocIDs with the same tokenizer sequence. Audit counts are written to
`model_confusion_pairs.stats.json` and `dpo_pairs_hybrid.stats.json`.

Train the hybrid file with
`src/scripts/ddro/launch_ddro_training_vault_hybrid.sh`, or pass
`dpo_pairs_hybrid.jsonl` to `train_ddro_vault.py` directly.

For a URL-DocID SFT checkpoint, first convert the final hybrid pairs and then
use the URL-specific hybrid launcher:

```bash
python src/scripts/bm25/the_vault/postprocess_dpo_urls.py \
  --input /data/vault_bm25/dpo_pairs_hybrid.jsonl \
  --document-metadata /data/vault_bm25/document_metadata.jsonl \
  --output /data/vault_bm25/dpo_pairs_hybrid_url.jsonl

bash src/scripts/ddro/launch_ddro_training_vault_url_hybrid.sh
```

The launcher defaults to `max_target_length=64` and the same URL checkpoint as
`launch_ddro_training_vault_url.sh`. Override `CHECKPOINT_PATH`, `TRAIN_FILE`,
`OUTPUT_DIR`, `NUM_GPUS`, or `PRECISION` through environment variables.

For the complete URL workflow—including data preparation, BM25 retrieval, BM25
URL conversion, URL-checkpoint model-confusion generation, hybrid combination,
the trainer dry run, and DPO training—use:

```bash
CHECKPOINT_PATH=/models/DSI_Ruby_url/checkpoint-630000 \
TRAIN_ORIGINAL=/data/Ruby_train_r32.0.json \
TEST_ORIGINAL=/data/Ruby_test_r32.0.json \
AUGMENTATION=/data/Ruby_ready_to_feed_numeric.jsonl \
WORK_DIR=/scratch/vault_url_hybrid \
DPO_OUTPUT_DIR=/scratch/models/vault-url-ddro-hybrid \
BM25_PYTHON_BIN=/envs/pyserini/bin/python \
DPO_PYTHON_BIN=/envs/ddro/bin/python \
NUM_GPUS=2 \
bash src/scripts/ddro/run_vault_url_hybrid_mining_and_dpo.sh
```

This produces:

```text
dpo_pairs.jsonl                    # unchanged BM25 text-ID baseline
dpo_pairs_url.jsonl                # BM25 pairs converted to URL targets
model_confusion_pairs_url.jsonl    # wrong URL DocIDs predicted by URL SFT
dpo_pairs_hybrid_url.jsonl         # final URL-target 4+4/fallback DPO data
```

Set `RUN_BM25=0` to reuse prepared BM25 files, `MINE_LIMIT_QUERIES=100`
for a smoke test, or `RUN_TRAINING=0` to stop after URL dataset validation.

After running BM25 preparation separately, URL model-confusion generation alone
can be launched from `ddro_env` with:

```bash
conda activate ddro_env

WORK_DIR=/mnt/beegfs/scratch/congthanh_le/east/ddro/data/vault_bm25 \
CHECKPOINT_PATH=/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000 \
NUM_BEAMS=8 \
BATCH_SIZE=64 \
bash src/scripts/ddro/mine_vault_url_model_confusions.sh
```

This reads `query_metadata.jsonl` and `document_metadata.jsonl` from
`WORK_DIR`, then writes `model_confusion_pairs_url.jsonl` and
`model_confusion_pairs_url.stats.json`. Set `LIMIT_QUERIES=100` only for a
smoke test; omit it for the complete dataset.

To generate both the URL model-confusion file and the final hybrid URL dataset
from an existing `dpo_pairs_url.jsonl`, use the preprocessing-only script:

```bash
conda activate ddro_env

WORK_DIR=/mnt/beegfs/scratch/congthanh_le/east/ddro/data/vault_bm25 \
CHECKPOINT_PATH=/home/users/congthanh_le/scratch/veil/CodeGR/outputs/DSI_Ruby_url/checkpoint-630000 \
NUM_BEAMS=8 \
BATCH_SIZE=64 \
bash src/scripts/ddro/prepare_vault_url_hybrid_negatives.sh
```

This script never invokes Pyserini and never starts training. It requires
`dpo_pairs_url.jsonl`, `query_metadata.jsonl`, and `document_metadata.jsonl`,
then creates `model_confusion_pairs_url.jsonl` and
`dpo_pairs_hybrid_url.jsonl`. Set `RUN_MODEL_MINING=0` to recombine an existing
model-confusion file without rerunning the model.

URL targets must not be truncated. If the miner reports targets longer than 64
tokens, set the same larger value for preprocessing and DPO training. For
example:

```bash
MAX_TARGET_LENGTH=96 \
bash src/scripts/ddro/prepare_vault_url_hybrid_negatives.sh

MAX_TARGET_LENGTH=96 \
bash src/scripts/ddro/launch_ddro_training_vault_url_hybrid.sh
```

The miner reports the corpus-wide maximum target length and several longest
examples, so use at least that reported maximum rather than repeatedly guessing.

To mine and train in one command, use:

```bash
bash src/scripts/ddro/run_vault_hybrid_mining_and_dpo.sh
```

All paths and training settings are environment-variable overrides. For example:

```bash
CHECKPOINT_PATH=/models/vault-sft/checkpoint-196000 \
WORK_DIR=/scratch/vault_hybrid \
DPO_OUTPUT_DIR=/scratch/models/vault-ddro-hybrid \
BM25_PYTHON_BIN=/envs/pyserini/bin/python \
DPO_PYTHON_BIN=/envs/ddro/bin/python \
NUM_GPUS=2 \
bash src/scripts/ddro/run_vault_hybrid_mining_and_dpo.sh
```

Set `RUN_BM25=0` to reuse existing BM25 files, `MINE_LIMIT_QUERIES=100`
for a mining smoke test, or `RUN_TRAINING=0` to create and validate the hybrid
dataset without starting DPO training.

## Convert an existing DPO file to URL targets

BM25 does not need to be rerun when `dpo_pairs.jsonl` has already been mined.
Convert its final decoder targets with:

```bash
python src/scripts/bm25/the_vault/postprocess_dpo_urls.py \
  --input /data/vault_bm25/dpo_pairs.jsonl \
  --document-metadata /data/vault_bm25/document_metadata.jsonl \
  --output /data/vault_bm25/dpo_pairs_url.jsonl
```

The converter changes only `chosen` and `rejected` to the corresponding scalar
`url_based_id`. It preserves `chosen_text_id` and `rejected_text_id` for audit,
and adds `chosen_url_based_id` and `rejected_url_based_id`. By default it fails
if either target has no URL or has multiple URLs. Use
`--on-mapping-error skip` only when dropping such rows is intentional. Pairs
whose chosen and rejected IDs resolve to the same URL are always dropped because
they are not valid preference pairs. Conversion counts are written beside the
output as `dpo_pairs_url.stats.json`.

## Important validation checks

- `unmapped_queries.jsonl` should be empty for the full dataset.
- `queries_whose_current_positive_is_missing_from_corpus` should normally be zero.
- Every corpus `id`, qrels document ID, `chosen`, and `rejected` uses the same
  quantized `text_id` namespace.
- The dummy files in this repository are incomplete slices, so their mapped
  positives need not have corresponding `Code:` rows. The full dataset should.
