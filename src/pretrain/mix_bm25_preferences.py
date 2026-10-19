"""Index external BM25 pairs once, then mix only the selected round's queries."""

from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import iter_json_records, normalize_query
from mine_model_confusion_negatives import build_document_targets, build_target_index
from pretrain.branch_preference_utils import encode_branch_target
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash, read_json


def prepare_bm25_pool(bm25_file, training_queries, documents_file, checkpoint, output,
                      target_type, max_target_length, length_policy, collision_policy,
                      token_policy, tokenizer=None):
    """URLs/selected target fields are authoritative; external numeric keys are not."""
    sources = {str(p): file_hash(p) for p in (bm25_file, training_queries, documents_file)}
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    documents = {r["text_id"]: r for r in iter_json_records(documents_file)}
    targets, target_to_id, invalid = build_document_targets(documents, target_type)
    if invalid:
        raise ValueError("BM25 corpus contains missing or ambiguous document targets")
    _, _, excluded, _, _ = build_target_index(
        tokenizer, targets.values(), max_target_length,
        collision_policy=collision_policy, length_policy=length_policy)

    @lru_cache(maxsize=None)
    def encode(target):
        return encode_branch_target(tokenizer, target, max_target_length, length_policy, token_policy)

    # The external query_key is from another preparation run. Match by text AND
    # the designated positive target; a shared prompt alone cannot join labels.
    matches = defaultdict(list)
    for query in iter_json_records(training_queries):
        chosen = targets[query["target_text_id"]]
        if chosen in excluded:
            continue
        positives = {encode(targets[p]) for p in query["positive_text_ids"] if targets[p] not in excluded}
        matches[(normalize_query(query["prompt"]), chosen)].append((query, positives))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    counters, matched_queries, missing_examples = Counter(), set(), set()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("""CREATE TABLE candidates (
            query_key TEXT NOT NULL, rejected_id TEXT NOT NULL, rejected TEXT NOT NULL,
            token_key TEXT NOT NULL, rank INTEGER, score REAL, source_query_key TEXT,
            source_record INTEGER NOT NULL, PRIMARY KEY(query_key, rejected_id))""")
        connection.execute("CREATE TABLE targets (text_id TEXT PRIMARY KEY, target TEXT, token_key TEXT)")
        connection.executemany("INSERT INTO targets VALUES (?, ?, ?)",
            ((key, target, json.dumps(encode(target))) for key, target in targets.items() if target not in excluded))
        for record, raw in enumerate(iter_json_records(bm25_file), 1):
            counters["input_rows"] += 1
            prompt, chosen, rejected = (raw.get("prompt"), raw.get(f"chosen_{target_type}"),
                                         raw.get(f"rejected_{target_type}"))
            if any(not isinstance(v, str) or not v.strip() for v in (prompt, chosen, rejected)):
                raise ValueError(f"BM25 record {record} requires prompt, chosen_{target_type}, rejected_{target_type}")
            if chosen != chosen.strip() or rejected != rejected.strip():
                raise ValueError(f"BM25 target IDs contain surrounding whitespace at record {record}")
            eligible = matches.get((normalize_query(prompt), chosen), [])
            if not eligible:
                counters["unmatched_rows"] += 1
                continue
            counters["matched_rows"] += 1
            rank, score = raw.get("bm25_rank"), raw.get("bm25_score")
            if rank is not None and (type(rank) is not int or rank < 1):
                raise ValueError(f"Invalid bm25_rank at record {record}")
            if score is not None and (type(score) not in {int, float} or not math.isfinite(score)):
                raise ValueError(f"Invalid bm25_score at record {record}")
            if rejected not in target_to_id:
                counters["unknown_corpus_target_rows"] += 1
                if len(missing_examples) < 3:
                    missing_examples.add(rejected)
                continue
            if rejected in excluded:
                counters["excluded_target_rows"] += 1
                continue
            tokens = encode(rejected)
            for query, positives in eligible:
                if tokens in positives:
                    counters["positive_or_alias_pairs_filtered"] += 1
                    continue
                matched_queries.add(query["query_key"])
                connection.execute("""INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_key, rejected_id) DO UPDATE SET
                        rank=excluded.rank, score=excluded.score,
                        source_query_key=excluded.source_query_key, source_record=excluded.source_record
                    WHERE COALESCE(excluded.rank, 9223372036854775807)
                        < COALESCE(candidates.rank, 9223372036854775807)""",
                    (query["query_key"], target_to_id[rejected], rejected, json.dumps(tokens),
                     rank, score, str(raw.get("query_key", "")), record))
        count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        if not count:
            raise ValueError("No usable BM25 pairs match training queries. Match prompt and "
                             f"chosen_{target_type}; supply missing retrieval identities with --corpus-file.")
        connection.commit()
        if any(file_hash(p) != digest for p, digest in sources.items()):
            raise ValueError("BM25 preparation inputs changed while reading")
        connection.close()
        temporary.replace(output)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)
    audit = {**counters, "indexed_pairs": count, "queries_with_bm25": len(matched_queries),
             "missing_target_examples": sorted(missing_examples), "sources": sources,
             "target_type": target_type, "max_target_length": max_target_length,
             "bm25_file_fingerprint": sources[str(bm25_file)]}
    atomic_json(output.with_suffix(".audit.json"), audit)
    print("BM25 pool:", json.dumps(audit), flush=True)
    if missing_examples:
        print("BM25 candidates outside the retrieval corpus were skipped; use --corpus-file to include them.", flush=True)
    return audit


def fuse_round_preferences(pool, model_pairs, queries_file, output, negatives_per_query,
                           bm25_negatives_per_query, round_id, fingerprint):
    """Preserve the total quota, reserve source shares, then backfill shortfalls."""
    if not 0 < bm25_negatives_per_query < negatives_per_query:
        raise ValueError("BM25 quota must be positive and smaller than the total negative quota")
    sources = {str(p): file_hash(p) for p in (pool, model_pairs, queries_file)}
    queries = list(iter_json_records(queries_file))
    query_map = {q["query_key"]: q for q in queries}
    policy = defaultdict(list)
    for row in iter_json_records(model_pairs):
        query = query_map.get(row.get("query_key"))
        if (query is None or row.get("prompt") != query["prompt"]
                or row.get("chosen_text_id") != query["target_text_id"]
                or row.get("round_id") != round_id or row.get("policy_fingerprint") != fingerprint
                or row.get("negative_source") != "model_confusion"):
            raise ValueError("Model pairs do not belong to the current selected round")
        policy[row["query_key"]].append(row)
    audit_path = Path(pool).with_suffix(".audit.json")
    pool_audit = read_json(audit_path)
    bm25_hash = pool_audit["bm25_file_fingerprint"]
    target_type = pool_audit["target_type"]
    counters, distribution, rows = Counter(), Counter(), []
    connection = sqlite3.connect(Path(pool).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        @lru_cache(maxsize=None)
        def target_info(text_id):
            result = connection.execute("SELECT target, token_key FROM targets WHERE text_id=?", (text_id,)).fetchone()
            if result is None:
                raise ValueError(f"Mixed preference target is absent from the usable corpus: {text_id}")
            return result

        for query in queries:
            key = query["query_key"]
            lexical = []
            for rejected_id, rejected, token_key, rank, score, source_key, record in connection.execute(
                """SELECT rejected_id, rejected, token_key, rank, score, source_query_key, source_record
                   FROM candidates WHERE query_key=?
                   ORDER BY COALESCE(rank, 9223372036854775807), source_record, rejected_id""", (key,)):
                # Both source text_id columns and positive_text_ids in the raw
                # BM25 file can be numeric. Replace them with this run's IDs.
                lexical.append({"prompt": query["prompt"], "query_key": key,
                    "numeric_id": query.get("numeric_id", ""), "chosen": target_info(query["target_text_id"])[0],
                    "chosen_text_id": query["target_text_id"], "rejected": rejected,
                    "rejected_text_id": rejected_id, "positive_text_ids": query["positive_text_ids"],
                    "negative_source": "bm25", "target_type": target_type,
                    f"chosen_{target_type}": target_info(query["target_text_id"])[0],
                    f"rejected_{target_type}": rejected, "bm25_rank": rank, "bm25_score": score,
                    "bm25_source_query_key": source_key, "bm25_source_record": record,
                    "bm25_source_fingerprint": bm25_hash, "round_id": round_id,
                    "policy_fingerprint": fingerprint, "family_id": query.get("family_id", ""),
                    "_token_key": token_key})
            selected, seen = [], set()

            def take(candidates, limit, reserve_policy=0):
                if limit <= 0:
                    return
                added = 0
                for candidate in candidates:
                    rejected = candidate["rejected_text_id"]
                    token_key = candidate.get("_token_key") or target_info(rejected)[1]
                    if rejected in query["positive_text_ids"] or token_key in seen:
                        continue
                    if reserve_policy and len(policy_keys - seen - {token_key}) < reserve_policy:
                        continue
                    selected.append(candidate)
                    seen.add(token_key)
                    added += 1
                    if added == limit:
                        break

            # Keep policy ranks and BM25 ranks separate; their scores have
            # different scales. Cross-source duplicates count only once.
            # Reserve room for policy candidates when a BM25 hit appears in
            # both pools. Otherwise overlap could erase an achievable 2+2 mix.
            policy_keys = {target_info(row["rejected_text_id"])[1] for row in policy[key]
                           if row["rejected_text_id"] not in query["positive_text_ids"]}
            policy_quota = negatives_per_query - bm25_negatives_per_query
            take(lexical, bm25_negatives_per_query, reserve_policy=min(policy_quota, len(policy_keys)))
            take(policy[key], policy_quota)
            lexical_count = sum(row["negative_source"] == "bm25" for row in selected)
            take(lexical, bm25_negatives_per_query - lexical_count)
            for candidates in (policy[key], lexical):
                remaining = negatives_per_query - len(selected)
                if remaining:
                    take(candidates, remaining)
            source_counts = Counter(row["negative_source"] for row in selected)
            counters.update(source_counts)
            counters["queries_with_pairs"] += bool(selected)
            counters["queries_without_bm25"] += source_counts["bm25"] == 0
            counters["queries_below_bm25_quota"] += source_counts["bm25"] < bm25_negatives_per_query
            counters["queries_below_total_quota"] += len(selected) < negatives_per_query
            distribution[f"{source_counts['bm25']} bm25 + {source_counts['model_confusion']} policy"] += 1
            rows.extend({k: v for k, v in row.items() if k != "_token_key"} for row in selected)
    finally:
        connection.close()
    if any(file_hash(p) != digest for p, digest in sources.items()):
        raise ValueError("Round fusion inputs changed while reading")
    atomic_jsonl(output, rows)
    audit = {"round_id": round_id, "policy_fingerprint": fingerprint, "pairs": len(rows),
             "selected_queries": len(queries), "bm25_quota": bm25_negatives_per_query,
             "total_quota": negatives_per_query, "counts": dict(counters),
             "source_mix_per_query": dict(distribution),
             "sources": sources}
    atomic_json(Path(output).with_suffix(".audit.json"), audit)
    print("Mixed preferences:", json.dumps(audit), flush=True)
    return audit
