"""Read immutable offline BM25 candidates and fuse them with round-local mining."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

from common import iter_json_records
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash, read_json


def query_signature(row):
    fields = {k: row[k] for k in ("query_key", "prompt", "target_text_id", "positive_text_ids")}
    fields["positive_text_ids"] = sorted(set(fields["positive_text_ids"]))
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class BM25Cache:
    """SQLite lookups avoid rescanning or loading millions of hits each round."""

    def __init__(self, path, training, documents, target_type):
        self.path = Path(path).resolve()
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        try:
            self.metadata = json.loads(self.connection.execute("SELECT value FROM metadata").fetchone()[0])
            expected = {"schema": 1, "train_hash": file_hash(training),
                        "documents_hash": file_hash(documents), "target_type": target_type}
            if any(self.metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("BM25 cache does not match the training split, corpus, or target type")
            self.documents = dict(self.connection.execute("SELECT id, text_id FROM documents"))
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def candidates(self, query):
        record = self.connection.execute("SELECT id, signature FROM queries WHERE query_key=?",
                                         (query["query_key"],)).fetchone()
        if record is None or record[1] != query_signature(query):
            raise ValueError(f"Missing or changed BM25 query: {query['query_key']}")
        return [(self.documents[doc], rank, score) for doc, rank, score in self.connection.execute(
            "SELECT doc, rank, score FROM hits WHERE query=? ORDER BY rank", (record[0],))]


def fuse_preferences(selected, model_pairs, exclusions, cache, output, round_id, fingerprint,
                     cache_fingerprint, model_quota=4, bm25_quota=4):
    """Separate source quotas; never duplicate IDs/token aliases or fill across quotas.

    This path uses direct multilabel IDs. The miner's exclusion/collision report
    describes the SAME tokenizer and truncation used in that round, so fusion
    needs no torch, tokenizer loading, Pyserini, or cached model log-probabilities.
    """
    query_map = {q["query_key"]: q for q in selected}
    model_by_query = defaultdict(list)
    for row in iter_json_records(model_pairs):
        if row["query_key"] not in query_map:
            raise ValueError("On-policy pair belongs to an unselected query")
        model_by_query[row["query_key"]].append(row)
    report = read_json(exclusions)
    excluded = set(report.get("model_mining_excluded_text_ids", []))
    representatives = {}
    for group in report.get("collision_groups", []):
        ids = group["text_ids"]
        for doc in ids:
            representatives[doc] = min(ids)
    canonical = lambda doc: representatives.get(doc, doc)
    known = set(cache.documents.values())
    counts, distribution = Counter(), Counter()

    def rows():
        for query in selected:
            # Check coverage even when the chosen target is excluded.
            candidates = cache.candidates(query)
            counts["queries_seen"] += 1
            chosen = query["target_text_id"]
            if chosen in excluded:
                counts["queries_skipped_excluded_chosen"] += 1
                continue
            positives = set(query["positive_text_ids"]) | {chosen}
            blocked = {canonical(doc) for doc in positives}
            picked = {"model_confusion": 0, "bm25": 0}
            model_rows = sorted(model_by_query[query["query_key"]], key=lambda r: r.get("model_rank", 10**9))
            for pair in model_rows:
                if (pair["prompt"] != query["prompt"] or pair["chosen_text_id"] != chosen
                        or pair["chosen"] != chosen or pair["rejected"] != pair["rejected_text_id"]
                        or set(pair["positive_text_ids"]) != positives
                        or pair.get("negative_source") != "model_confusion"
                        or pair.get("round_id") != round_id or pair.get("policy_fingerprint") != fingerprint):
                    raise ValueError("On-policy preference identity/provenance changed")
                doc = pair["rejected_text_id"]
                if doc not in known or doc in excluded or canonical(doc) in blocked:
                    raise ValueError("Invalid, positive, or duplicate on-policy negative")
                if picked["model_confusion"] >= model_quota:
                    raise ValueError("On-policy quota exceeded")
                blocked.add(canonical(doc))
                picked["model_confusion"] += 1
                yield pair
            for doc, rank, score in candidates:
                if picked["bm25"] >= bm25_quota:
                    break
                if doc in excluded or canonical(doc) in blocked:
                    counts["bm25_filtered_positive_duplicate_or_excluded"] += 1
                    continue
                blocked.add(canonical(doc))
                picked["bm25"] += 1
                yield {"query_key": query["query_key"], "prompt": query["prompt"],
                       "chosen": chosen, "rejected": doc, "chosen_text_id": chosen,
                       "rejected_text_id": doc, "positive_text_ids": sorted(positives),
                       "negative_source": "bm25", "target_type": cache.metadata["target_type"],
                       "round_id": round_id, "policy_fingerprint": fingerprint,
                       "bm25_cache_fingerprint": cache_fingerprint, "bm25_rank": rank,
                       "bm25_score": score, "family_id": query.get("family_id", "")}
            counts["queries_processed"] += 1
            for source, quota in (("model_confusion", model_quota), ("bm25", bm25_quota)):
                counts[source + "_pairs"] += picked[source]
                counts[source + "_shortfall_queries"] += picked[source] < quota
            distribution[f"{picked['model_confusion']}+{picked['bm25']}"] += 1
    atomic_jsonl(output, rows())
    stats = {**counts, "source_mix_distribution": dict(sorted(distribution.items())),
             "model_quota": model_quota, "bm25_quota": bm25_quota,
             "bm25_cache_fingerprint": cache_fingerprint}
    atomic_json(Path(output).with_suffix(".stats.json"), stats)
    atomic_json(Path(output).with_suffix(".excluded_text_ids.json"), report)
    return stats
