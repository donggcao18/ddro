#!/usr/bin/env python3
"""Filter Vault BM25 results and export quantized-text_id DPO pairs."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import iter_json_records, write_jsonl


def load_run(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load Pyserini MS MARCO (3-column) or TREC (6-column) output."""
    rankings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    implicit_rank: dict[str, int] = defaultdict(int)

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue

            if len(fields) >= 6 and fields[1] == "Q0":
                query_key = fields[0]
                document_id = fields[2]
                rank = int(fields[3])
                score = float(fields[4])
            elif len(fields) >= 3:
                query_key = fields[0]
                document_id = fields[1]
                implicit_rank[query_key] += 1
                try:
                    rank = int(fields[2])
                    score = None
                except ValueError:
                    rank = implicit_rank[query_key]
                    score = float(fields[2])
            else:
                raise ValueError(f"Unsupported run format at {path}:{line_number}")

            rankings[query_key].append(
                {"text_id": document_id, "rank": rank, "score": score}
            )

    for query_key, candidates in rankings.items():
        candidates.sort(key=lambda candidate: candidate["rank"])
        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate["text_id"] in seen:
                continue
            seen.add(candidate["text_id"])
            deduplicated.append(candidate)
        rankings[query_key] = deduplicated

    return rankings


def parse_rank_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for raw_range in value.split(","):
        start_text, separator, end_text = raw_range.strip().partition(":")
        if not separator:
            raise argparse.ArgumentTypeError(f"Invalid rank range: {raw_range!r}")
        start, end = int(start_text), int(end_text)
        if start < 1 or end < start:
            raise argparse.ArgumentTypeError(f"Invalid rank range: {raw_range!r}")
        ranges.append((start, end))
    if not ranges:
        raise argparse.ArgumentTypeError("At least one rank range is required")
    return ranges


def stratified_sample(
    candidates: list[dict[str, Any]],
    count: int,
    rank_ranges: list[tuple[int, int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample evenly across BM25 rank bands, then fill from remaining hits."""
    if count <= 0:
        return []

    base = count // len(rank_ranges)
    remainder = count % len(rank_ranges)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for index, (start, end) in enumerate(rank_ranges):
        quota = base + (1 if index < remainder else 0)
        bucket = [candidate for candidate in candidates if start <= candidate["rank"] <= end]
        for candidate in rng.sample(bucket, min(quota, len(bucket))):
            if candidate["text_id"] not in selected_ids:
                selected_ids.add(candidate["text_id"])
                selected.append(candidate)

    missing = count - len(selected)
    if missing > 0:
        remaining = [
            candidate for candidate in candidates if candidate["text_id"] not in selected_ids
        ]
        for candidate in rng.sample(remaining, min(missing, len(remaining))):
            selected_ids.add(candidate["text_id"])
            selected.append(candidate)

    selected.sort(key=lambda candidate: candidate["rank"])
    return selected


def mine(args: argparse.Namespace) -> dict[str, Any]:
    query_rows = {
        row["query_key"]: row for row in iter_json_records(args.query_metadata)
    }
    document_rows = {
        row["text_id"]: row for row in iter_json_records(args.document_metadata)
    }
    rankings = load_run(args.run)
    rng = random.Random(args.seed)

    dpo_rows: list[dict[str, Any]] = []
    triple_rows: list[tuple[str, str, str]] = []
    queries_without_run = 0
    queries_without_negatives = 0
    filtered_positive_hits = 0
    filtered_unknown_hits = 0

    for query_key, query in query_rows.items():
        candidates = rankings.get(query_key)
        if not candidates:
            queries_without_run += 1
            continue

        positive_ids = set(query.get("positive_text_ids", []))
        positive_ids.add(query["target_text_id"])
        usable: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["text_id"] in positive_ids:
                filtered_positive_hits += 1
                continue
            if candidate["text_id"] not in document_rows:
                filtered_unknown_hits += 1
                continue
            usable.append(candidate)

        negatives = stratified_sample(
            usable,
            args.negatives_per_query,
            args.rank_ranges,
            rng,
        )
        if not negatives:
            queries_without_negatives += 1
            continue

        chosen_ids = (
            sorted(positive_ids) if args.pair_all_positives else [query["target_text_id"]]
        )
        for chosen_id in chosen_ids:
            for negative in negatives:
                rejected_id = negative["text_id"]
                rejected_metadata = document_rows[rejected_id]
                dpo_rows.append(
                    {
                        "prompt": query["prompt"],
                        "chosen": chosen_id,
                        "rejected": rejected_id,
                        "query_key": query_key,
                        "numeric_id": query.get("numeric_id", ""),
                        "chosen_text_id": chosen_id,
                        "rejected_text_id": rejected_id,
                        "positive_text_ids": sorted(positive_ids),
                        "bm25_rank": negative["rank"],
                        "bm25_score": negative["score"],
                        "rejected_url_based_ids": rejected_metadata.get("url_based_ids", []),
                    }
                )
                triple_rows.append((query_key, chosen_id, rejected_id))

    write_jsonl(args.output, dpo_rows)
    triples_output = Path(args.triples_output) if args.triples_output else Path(args.output).with_suffix(".tsv")
    triples_output.parent.mkdir(parents=True, exist_ok=True)
    with triples_output.open("w", encoding="utf-8") as handle:
        for query_key, chosen_id, rejected_id in triple_rows:
            handle.write(f"{query_key}\t{chosen_id}\t{rejected_id}\n")

    stats = {
        "queries": len(query_rows),
        "queries_in_bm25_run": len(rankings),
        "queries_without_run": queries_without_run,
        "queries_without_usable_negatives": queries_without_negatives,
        "filtered_multi_label_or_target_hits": filtered_positive_hits,
        "filtered_unknown_document_hits": filtered_unknown_hits,
        "dpo_pairs": len(dpo_rows),
        "output": str(args.output),
        "triples_output": str(triples_output),
    }
    stats_path = Path(args.output).with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Vault DPO pairs from a Pyserini BM25 run."
    )
    parser.add_argument("--run", required=True, help="Pyserini MS MARCO or TREC run file")
    parser.add_argument("--query-metadata", required=True)
    parser.add_argument("--document-metadata", required=True)
    parser.add_argument("--output", required=True, help="DPO-ready JSONL output")
    parser.add_argument("--triples-output", help="Optional qid/chosen/rejected TSV output")
    parser.add_argument("--negatives-per-query", type=int, default=12)
    parser.add_argument(
        "--rank-ranges",
        type=parse_rank_ranges,
        default=parse_rank_ranges("1:100,101:500,501:1000"),
        help="Comma-separated inclusive BM25 rank bands",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair-all-positives",
        action="store_true",
        help=(
            "Emit each known multi-label positive as chosen. By default only the "
            "augmentation row's mapped target is chosen, while all positives are filtered."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = mine(args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
