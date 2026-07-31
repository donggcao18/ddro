#!/usr/bin/env python3
"""Combine existing Vault BM25 pairs with model-confusion DPO pairs.

The BM25 input remains the stable eight-negative baseline.  For each
``(query_key, chosen_text_id)`` group this script selects at most the configured
model quota and fills the rest of the fixed total from unique BM25 negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from common import as_text_id, iter_json_records


PairKey = tuple[str, str]


def pair_key(row: dict[str, Any]) -> PairKey:
    query_key = str(row.get("query_key", "")).strip()
    chosen_id = as_text_id(row.get("chosen_text_id", row.get("chosen")))
    if not query_key or not chosen_id:
        raise ValueError("Every input row must contain query_key and chosen/chosen_text_id")
    return query_key, chosen_id


def iter_pair_groups(
    path: str | Path,
) -> Iterator[tuple[PairKey, list[dict[str, Any]]]]:
    """Stream consecutive pair groups and require deterministic sorted order."""
    current_key: PairKey | None = None
    current_rows: list[dict[str, Any]] = []
    previous_key: PairKey | None = None
    for row in iter_json_records(path):
        key = pair_key(row)
        if previous_key is not None and key < previous_key:
            raise ValueError(
                f"{path} is not sorted/grouped by (query_key, chosen_text_id): "
                f"{key!r} appears after {previous_key!r}"
            )
        previous_key = key
        if current_key is None:
            current_key = key
        elif key != current_key:
            yield current_key, current_rows
            current_key = key
            current_rows = []
        current_rows.append(row)
    if current_key is not None:
        yield current_key, current_rows


def stable_rng(seed: int, key: PairKey) -> random.Random:
    material = f"{seed}\0{key[0]}\0{key[1]}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def rejected_id(row: dict[str, Any]) -> str:
    return as_text_id(row.get("rejected_text_id", row.get("rejected")))


def validate_group(
    key: PairKey,
    rows: list[dict[str, Any]],
    known_documents: set[str],
) -> tuple[str, set[str]]:
    """Validate shared group fields and return prompt and positive set."""
    if not rows:
        raise ValueError(f"Empty pair group for {key!r}")
    prompts = {str(row.get("prompt", "")).strip() for row in rows}
    if len(prompts) != 1 or not next(iter(prompts)):
        raise ValueError(f"Inconsistent or empty prompts for group {key!r}")

    positive_ids: set[str] = {key[1]}
    for row in rows:
        positive_ids.update(
            as_text_id(value)
            for value in row.get("positive_text_ids", [])
            if as_text_id(value)
        )
    for row in rows:
        candidate_id = rejected_id(row)
        if not candidate_id:
            raise ValueError(f"Missing rejected/rejected_text_id in group {key!r}")
        if candidate_id not in known_documents:
            raise ValueError(
                f"Unknown rejected_text_id={candidate_id!r} in group {key!r}"
            )
        if candidate_id in positive_ids:
            raise ValueError(
                f"Positive text_id={candidate_id!r} appears as a rejection in group {key!r}"
            )
    return next(iter(prompts)), positive_ids


def unique_by_rejected(
    rows: list[dict[str, Any]],
    rank_field: str,
) -> list[dict[str, Any]]:
    """Keep the best-ranked row for each rejected DocID."""
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get(rank_field, 10**9)),
            rejected_id(row),
        ),
    )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        candidate_id = rejected_id(row)
        if candidate_id not in seen:
            seen.add(candidate_id)
            output.append(row)
    return output


def choose_bm25_rows(
    rows: list[dict[str, Any]],
    count: int,
    excluded_ids: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select BM25 rows with a 2/1/1 top/mid/lower base allocation."""
    if count <= 0:
        return []
    candidates = [
        row
        for row in unique_by_rejected(rows, "bm25_rank")
        if rejected_id(row) not in excluded_ids
    ]
    buckets = [
        [row for row in candidates if int(row.get("bm25_rank", 10**9)) <= 20],
        [
            row
            for row in candidates
            if 21 <= int(row.get("bm25_rank", 10**9)) <= 100
        ],
        [row for row in candidates if int(row.get("bm25_rank", -1)) >= 101],
    ]
    base_quotas = [2, 1, 1]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket, quota in zip(buckets, base_quotas):
        available = [row for row in bucket if rejected_id(row) not in selected_ids]
        for row in rng.sample(available, min(quota, len(available))):
            selected.append(row)
            selected_ids.add(rejected_id(row))

    remaining = [
        row
        for row in candidates
        if rejected_id(row) not in selected_ids
    ]
    missing = count - len(selected)
    if missing > 0:
        for row in rng.sample(remaining, min(missing, len(remaining))):
            selected.append(row)
            selected_ids.add(rejected_id(row))
    return selected[:count]


def combine_group(
    key: PairKey,
    bm25_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    known_documents: set[str],
    model_per_query: int,
    total_per_query: int,
    seed: int,
    require_exact_mix: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Combine one group and return rows plus small group-level counters."""
    _, bm25_positives = validate_group(key, bm25_rows, known_documents)
    if model_rows:
        _, model_positives = validate_group(key, model_rows, known_documents)
        if model_positives != bm25_positives:
            raise ValueError(
                f"BM25/model positive_text_ids disagree for group {key!r}: "
                f"{sorted(bm25_positives)!r} != {sorted(model_positives)!r}"
            )

    counters: Counter[str] = Counter()
    model_candidates = unique_by_rejected(model_rows, "model_rank")
    model_selected = model_candidates[:model_per_query]
    if require_exact_mix and len(model_selected) < model_per_query:
        counters["skipped_model_shortfall"] += 1
        return [], dict(counters)

    bm25_by_id = {rejected_id(row): row for row in bm25_rows}
    prepared_model_rows: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    for row in model_selected:
        candidate_id = rejected_id(row)
        copied = dict(row)
        copied["negative_source"] = "model_confusion"
        matching_bm25 = bm25_by_id.get(candidate_id)
        if matching_bm25 is not None:
            copied["negative_source"] = "model_confusion+bm25"
            copied["also_bm25_rank"] = matching_bm25.get("bm25_rank")
            copied["also_bm25_score"] = matching_bm25.get("bm25_score")
            counters["model_bm25_overlap"] += 1
        prepared_model_rows.append(copied)
        model_ids.add(candidate_id)

    bm25_needed = total_per_query - len(prepared_model_rows)
    selected_bm25 = choose_bm25_rows(
        bm25_rows,
        bm25_needed,
        model_ids,
        stable_rng(seed, key),
    )
    if len(selected_bm25) < bm25_needed:
        counters["skipped_insufficient_unique_bm25"] += 1
        return [], dict(counters)

    prepared_bm25_rows: list[dict[str, Any]] = []
    for row in selected_bm25:
        copied = dict(row)
        copied["negative_source"] = "bm25"
        prepared_bm25_rows.append(copied)

    output = prepared_model_rows + prepared_bm25_rows
    output_ids = [rejected_id(row) for row in output]
    if len(output) != total_per_query or len(set(output_ids)) != len(output_ids):
        raise AssertionError(f"Invalid hybrid selection for group {key!r}")
    if any(candidate_id in bm25_positives for candidate_id in output_ids):
        raise AssertionError(f"A positive survived hybrid selection for group {key!r}")

    counters["model_pairs_selected"] += len(prepared_model_rows)
    counters["bm25_pairs_selected"] += len(prepared_bm25_rows)
    if len(prepared_model_rows) == model_per_query:
        counters["queries_with_exact_mix"] += 1
    else:
        counters["queries_using_bm25_fallback"] += 1
    return output, dict(counters)


def atomic_output_path(path: Path) -> Path:
    return path.with_name(path.name + f".tmp.{os.getpid()}")


def combine(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_per_query < 0:
        raise ValueError("--model-per-query cannot be negative")
    if args.total_per_query <= 0:
        raise ValueError("--total-per-query must be greater than zero")
    if args.model_per_query > args.total_per_query:
        raise ValueError("--model-per-query cannot exceed --total-per-query")

    known_documents = {
        as_text_id(row.get("text_id"))
        for row in iter_json_records(args.document_metadata)
        if as_text_id(row.get("text_id"))
    }
    if not known_documents:
        raise ValueError("document_metadata contains no valid text_id values")

    bm25_groups = iter(iter_pair_groups(args.bm25_input))
    model_groups = iter(iter_pair_groups(args.model_input))
    current_model = next(model_groups, None)
    stats: Counter[str] = Counter()
    model_count_distribution: Counter[int] = Counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = atomic_output_path(output_path)

    try:
        with temporary_output.open("w", encoding="utf-8") as output_handle:
            for bm25_key, bm25_rows in bm25_groups:
                stats["bm25_groups"] += 1
                while current_model is not None and current_model[0] < bm25_key:
                    stats["model_groups_without_bm25"] += 1
                    current_model = next(model_groups, None)

                if current_model is not None and current_model[0] == bm25_key:
                    model_rows = current_model[1]
                    current_model = next(model_groups, None)
                else:
                    model_rows = []
                    stats["bm25_groups_without_model"] += 1

                selected, group_stats = combine_group(
                    bm25_key,
                    bm25_rows,
                    model_rows,
                    known_documents,
                    args.model_per_query,
                    args.total_per_query,
                    args.seed,
                    args.require_exact_mix,
                )
                stats.update(group_stats)
                if not selected:
                    stats["groups_skipped"] += 1
                    continue

                selected_model_count = sum(
                    str(row.get("negative_source", "")).startswith("model_confusion")
                    for row in selected
                )
                model_count_distribution[selected_model_count] += 1
                for row in selected:
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["hybrid_pairs"] += 1
                stats["hybrid_groups"] += 1

            while current_model is not None:
                stats["model_groups_without_bm25"] += 1
                current_model = next(model_groups, None)
        temporary_output.replace(output_path)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    result: dict[str, Any] = dict(stats)
    result["model_count_distribution"] = {
        str(count): groups
        for count, groups in sorted(model_count_distribution.items())
    }
    result.update(
        {
            "documents": len(known_documents),
            "model_per_query": args.model_per_query,
            "total_per_query": args.total_per_query,
            "require_exact_mix": args.require_exact_mix,
            "bm25_input": str(args.bm25_input),
            "model_input": str(args.model_input),
            "output": str(output_path),
        }
    )
    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine Vault BM25 and model-confusion DPO pairs."
    )
    parser.add_argument("--bm25-input", required=True)
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--document-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-per-query", type=int, default=4)
    parser.add_argument("--total-per-query", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-exact-mix",
        action="store_true",
        help="Skip groups with fewer than --model-per-query model negatives.",
    )
    return parser


def main() -> None:
    stats = combine(build_parser().parse_args())
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
