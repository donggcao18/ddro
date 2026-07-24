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
