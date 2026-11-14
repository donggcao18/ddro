#!/usr/bin/env python3
"""Mine URL-target DPO pairs from run_url_bm25.py results."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from common import as_text_id_list, iter_json_records, write_jsonl
from mine_dpo_negatives import stratified_sample


def iter_pairs(args: argparse.Namespace, stats: dict[str, int]) -> Iterator[dict[str, Any]]:
    rng = random.Random(args.seed)
    for query in iter_json_records(args.input):
        stats["queries"] += 1
        positives = as_text_id_list(query["positive_url_based_id"])
        source_id = query.get("url_based_id")
        if not positives:
            raise ValueError(f"Query {query['query_id']!r} has no positive URL IDs")
        chosen = source_id if source_id in positives else positives[0]
        excluded = set(positives)
        excluded.add(source_id)
        candidates = {}
        for hit in sorted(query["hits"], key=lambda hit: hit["rank"]):
            url_id = hit["url_based_id"]
            if url_id and url_id not in excluded and not hit.get("is_positive"):
                candidates.setdefault(url_id, hit)

        negatives = stratified_sample(
            list(candidates.values()),
            8,
            [(1, 20), (21, 100), (101, 200)],
            rng,
            fill_shortfall=args.fill_shortfall,
            rank_quotas=[3, 2, 3],
            candidate_id_field="url_based_id",
        )
        if not negatives:
            stats["skipped"] += 1
            continue
        for hit in negatives:
            yield {
                "prompt": query["text"],
                "chosen": chosen,
                "rejected": hit["url_based_id"],
                "query_key": query["query_id"],
                "positive_url_based_id": positives,
                "bm25_rank": hit["rank"],
                "bm25_score": hit["score"],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="BM25 results JSONL")
    parser.add_argument("--output", required=True, type=Path, help="DPO pairs JSONL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fill-shortfall",
        action="store_true",
        help="Fill missing rank quotas from other hits and keep partial samples",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")

    stats = {"queries": 0, "skipped": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="vault-url-dpo-", dir=args.output.parent) as temporary:
        temporary_output = Path(temporary) / "pairs.jsonl"
        count = write_jsonl(temporary_output, iter_pairs(args, stats))
        if not count:
            parser.error(
                f"No DPO pairs from {stats['queries']:,} queries. "
                "Use --fill-shortfall if queries have negatives but cannot meet the rank quotas."
            )
        temporary_output.replace(args.output)

    print(
        f"Wrote {count:,} DPO pairs to {args.output} "
        f"({stats['skipped']:,} of {stats['queries']:,} queries skipped)"
    )


if __name__ == "__main__":
    main()
