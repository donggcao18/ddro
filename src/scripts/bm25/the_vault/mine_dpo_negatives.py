#!/usr/bin/env python3
"""Filter Vault BM25 results and export target-namespace DPO pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

from common import as_text_id, document_targets, iter_json_records


def build_document_target_map(
    document_rows: dict[str, dict[str, Any]], target_type: str
) -> tuple[dict[str, str], int]:
    """Map canonical text IDs to one usable decoder target."""
    mapping: dict[str, str] = {}
    target_to_text_id: dict[str, str] = {}
    invalid = 0
    for text_id, row in document_rows.items():
        targets = document_targets(row, target_type)
        if len(targets) != 1:
            invalid += 1
            continue
        target = targets[0]
        previous_text_id = target_to_text_id.get(target)
        if previous_text_id is not None and previous_text_id != text_id:
            raise ValueError(
                f"Decoder target {target!r} maps to multiple text IDs: "
                f"{previous_text_id!r} and {text_id!r}"
            )
        mapping[text_id] = target
        target_to_text_id[target] = text_id
    return mapping, invalid


def iter_run_groups(
    path: str | Path,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Stream grouped Pyserini output while retaining at most one query's hits."""
    current_query: str | None = None
    current_candidates: list[dict[str, Any]] = []
    current_ids: set[str] = set()
    implicit_rank = 0
    completed_queries: set[str] = set()

    def finish_group() -> tuple[str, list[dict[str, Any]]] | None:
        if current_query is None:
            return None
        current_candidates.sort(key=lambda candidate: candidate["rank"])
        return current_query, current_candidates

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
                if query_key != current_query:
                    line_rank = 1
                else:
                    line_rank = implicit_rank + 1
                try:
                    rank = int(fields[2])
                    score = None
                except ValueError:
                    rank = line_rank
                    score = float(fields[2])
            else:
                raise ValueError(f"Unsupported run format at {path}:{line_number}")

            if current_query is None:
                current_query = query_key
            elif query_key != current_query:
                group = finish_group()
                if group is not None:
                    completed_queries.add(group[0])
                    yield group
                if query_key in completed_queries:
                    raise ValueError(
                        f"Run file is not grouped by query; {query_key!r} reappears at line {line_number}"
                    )
                current_query = query_key
                current_candidates = []
                current_ids = set()
                implicit_rank = 0

            implicit_rank += 1
            if document_id in current_ids:
                continue
            current_ids.add(document_id)
            current_candidates.append(
                {"text_id": document_id, "rank": rank, "score": score}
            )

    group = finish_group()
    if group is not None:
        yield group


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


def parse_rank_quotas(value: str) -> list[int]:
    quotas = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not quotas or any(quota < 0 for quota in quotas):
        raise argparse.ArgumentTypeError("Rank quotas must be non-negative integers")
    return quotas


def stratified_sample(
    candidates: list[dict[str, Any]],
    count: int,
    rank_ranges: list[tuple[int, int]],
    rng: random.Random,
    fill_shortfall: bool = False,
    rank_quotas: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Sample a configured quota from each BM25 rank band."""
    if count <= 0:
        return []

    if rank_quotas is None:
        base = count // len(rank_ranges)
        remainder = count % len(rank_ranges)
        quotas = [base] * len(rank_ranges)
        quotas[-1] += remainder
    else:
        quotas = rank_quotas
        if len(quotas) != len(rank_ranges):
            raise ValueError("rank_quotas must contain one value per rank range")
        if sum(quotas) != count:
            raise ValueError("rank_quotas must sum to negatives_per_query")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for index, (start, end) in enumerate(rank_ranges):
        quota = quotas[index]
        bucket = [candidate for candidate in candidates if start <= candidate["rank"] <= end]
        if len(bucket) < quota and not fill_shortfall:
            # Match bm25_negative_sampling_msmarco.py: skip a query if the
            # configured bands cannot provide the full requested sample.
            return []
        for candidate in rng.sample(bucket, min(quota, len(bucket))):
            if candidate["text_id"] not in selected_ids:
                selected_ids.add(candidate["text_id"])
                selected.append(candidate)

    missing = count - len(selected)
    if missing > 0 and fill_shortfall:
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
    target_type = getattr(args, "target_type", "text_id")
    text_id_to_target, invalid_target_mappings = build_document_target_map(
        document_rows, target_type
    )
    rng = random.Random(args.seed)

    queries_without_negatives = 0
    filtered_positive_hits = 0
    filtered_unknown_hits = 0
    dpo_pair_count = 0
    seen_run_queries: set[str] = set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    triples_output = Path(args.triples_output) if args.triples_output else output_path.with_suffix(".tsv")
    triples_output.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as dpo_handle, triples_output.open(
        "w", encoding="utf-8"
    ) as triples_handle:
        for query_key, candidates in iter_run_groups(args.run):
            seen_run_queries.add(query_key)
            query = query_rows.get(query_key)
            if query is None:
                continue

            positive_ids = set(query.get("positive_text_ids", []))
            positive_ids.add(query["target_text_id"])
            positive_targets = {
                text_id_to_target[positive_id]
                for positive_id in positive_ids
                if positive_id in text_id_to_target
            }
            positive_targets.update(
                as_text_id(value)
                for value in query.get("positive_structure_id_v3s", [])
                if target_type == "structure_id_v3" and as_text_id(value)
            )
            usable: list[dict[str, Any]] = []
            seen_candidate_targets: set[str] = set()
            for candidate in candidates:
                if candidate["text_id"] in positive_ids:
                    filtered_positive_hits += 1
                    continue
                candidate_target = text_id_to_target.get(candidate["text_id"])
                if candidate["text_id"] not in document_rows or not candidate_target:
                    filtered_unknown_hits += 1
                    continue
                if candidate_target in positive_targets:
                    filtered_positive_hits += 1
                    continue
                if candidate_target in seen_candidate_targets:
                    continue
                seen_candidate_targets.add(candidate_target)
                candidate["target"] = candidate_target
                usable.append(candidate)

            negatives = stratified_sample(
                usable,
                args.negatives_per_query,
                args.rank_ranges,
                rng,
                args.fill_shortfall,
                args.rank_quotas,
            )
            if not negatives:
                queries_without_negatives += 1
                continue

            chosen_ids = (
                sorted(positive_ids)
                if args.pair_all_positives
                else [query["target_text_id"]]
            )
            seen_chosen_targets: set[str] = set()
            for chosen_id in chosen_ids:
                chosen_target = text_id_to_target.get(chosen_id)
                if not chosen_target:
                    raise ValueError(
                        f"Chosen text_id={chosen_id!r} has no unique {target_type} mapping"
                    )
                if chosen_target in seen_chosen_targets:
                    continue
                seen_chosen_targets.add(chosen_target)
                for negative in negatives:
                    rejected_id = negative["text_id"]
                    rejected_target = negative["target"]
                    rejected_metadata = document_rows[rejected_id]
                    dpo_row = {
                        "prompt": query["prompt"],
                        "chosen": chosen_target,
                        "rejected": rejected_target,
                        "query_key": query_key,
                        "numeric_id": query.get("numeric_id", ""),
                        "chosen_text_id": chosen_id,
                        "rejected_text_id": rejected_id,
                        "positive_text_ids": sorted(positive_ids),
                        "target_type": target_type,
                        "bm25_rank": negative["rank"],
                        "bm25_score": negative["score"],
                        "rejected_url_based_ids": rejected_metadata.get("url_based_ids", []),
                    }
                    if target_type == "structure_id_v3":
                        dpo_row["chosen_structure_id_v3"] = chosen_target
                        dpo_row["rejected_structure_id_v3"] = rejected_target
                        dpo_row["positive_structure_id_v3s"] = sorted(positive_targets)
                    dpo_handle.write(json.dumps(dpo_row, ensure_ascii=False) + "\n")
                    triples_handle.write(
                        f"{query_key}\t{chosen_target}\t{rejected_target}\n"
                    )
                    dpo_pair_count += 1

    queries_without_run = len(set(query_rows) - seen_run_queries)

    stats = {
        "queries": len(query_rows),
        "queries_in_bm25_run": len(seen_run_queries),
        "queries_without_run": queries_without_run,
        "queries_without_usable_negatives": queries_without_negatives,
        "filtered_multi_label_or_target_hits": filtered_positive_hits,
        "filtered_unknown_document_hits": filtered_unknown_hits,
        "invalid_document_target_mappings": invalid_target_mappings,
        "target_type": target_type,
        "dpo_pairs": dpo_pair_count,
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
    parser.add_argument(
        "--target-type",
        choices=["text_id", "structure_id_v3"],
        default="text_id",
        help="Write chosen/rejected in this decoder target namespace.",
    )
    parser.add_argument("--triples-output", help="Optional qid/chosen/rejected TSV output")
    parser.add_argument(
        "--negatives-per-query",
        type=int,
        default=8,
        help="Vault default: 8 negatives per pseudo-query.",
    )
    parser.add_argument(
        "--rank-ranges",
        type=parse_rank_ranges,
        default=parse_rank_ranges("1:20,21:100,101:200"),
        help="Comma-separated inclusive BM25 rank bands",
    )
    parser.add_argument(
        "--rank-quotas",
        type=parse_rank_quotas,
        default=parse_rank_quotas("3,2,3"),
        help="Negatives sampled per rank band; must sum to --negatives-per-query.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fill-shortfall",
        action="store_true",
        help=(
            "Fill missing bucket quotas from other ranks. Disabled by default to "
            "keep strict stratification; undersupplied queries are skipped."
        ),
    )
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
