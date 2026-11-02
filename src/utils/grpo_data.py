"""Query-level, multi-positive data preparation for generative retrieval GRPO."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from scripts.bm25.the_vault.common import (
    CODE_PREFIX_RE,
    clean_prompt,
    document_targets,
    iter_json_records,
    normalize_query,
    unique_strings,
)


TARGET_TYPES = ("text_id", "url", "structure_id_v3", "structure_id_v6")


def _url_values(value: Any, field: str, row_number: int) -> list[str]:
    """Keep URL labels as strings; never convert nested metadata into an ID."""
    values = value if isinstance(value, list) else [value]
    if any(item is not None and not isinstance(item, str) for item in values):
        raise ValueError(f"Row {row_number}: {field} must contain a string or list of strings")
    return sorted({item.strip() for item in values if isinstance(item, str) and item.strip()})


def prepare_url_file(
    path: str | Path, include_positive_targets_in_corpus: bool = True
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read native Vault rows directly, without numeric/semantic-ID joins.

    Training catalog: union of row URLs and annotated positive URLs. Evaluation
    can disable label-derived catalog entries and use test row URLs only, keeping
    test positive labels separate from candidate construction.
    """
    queries: list[dict[str, Any]] = []
    corpus: set[str] = set()
    for number, row in enumerate(iter_json_records(path), 1):
        corpus.update(_url_values(row.get("url_based_id"), "url_based_id", number))
        raw_text = row.get("text")
        if not isinstance(raw_text, str):
            raise ValueError(f"Row {number}: text must be a query string")
        # Original indexed files can include Code: rows; they add corpus IDs,
        # but must not become code-to-ID training/evaluation queries.
        if CODE_PREFIX_RE.match(raw_text):
            continue
        positives = _url_values(row.get("positive_url_based_id"), "positive_url_based_id", number)
        if not positives:
            raise ValueError(f"Row {number}: positive_url_based_id must contain at least one URL")
        prompt = clean_prompt(raw_text)
        if not prompt:
            raise ValueError(f"Row {number}: empty query text")
        queries.append({"prompt": prompt, "positive_targets": positives})
        if include_positive_targets_in_corpus:
            corpus.update(positives)
    queries = consolidate_queries(queries)
    return queries, sorted(corpus)


def consolidate_queries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union positives before splitting, so repeated queries cannot leak."""
    groups: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        prompt = clean_prompt(row.get("prompt", ""))
        positives = unique_strings(row.get("positive_targets", []))
        if not prompt or not positives:
            raise ValueError(f"Query row {number} needs prompt and nonempty positive_targets")
        key = normalize_query(prompt)
        group = groups.setdefault(key, {"prompt": prompt, "positive_targets": set()})
        group["positive_targets"].update(positives)
    if not groups:
        raise ValueError("No queries were provided")
    return [
        {"prompt": group["prompt"], "positive_targets": sorted(group["positive_targets"])}
        for group in groups.values()
    ]


def split_queries(
    rows: list[dict[str, Any]], validation_split: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in [0, 1)")
    if not validation_split:
        return rows, []
    if len(rows) < 2:
        raise ValueError("At least two distinct queries are needed for a validation split")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    count = min(len(rows) - 1, max(1, round(len(rows) * validation_split)))
    validation_indices = set(indices[:count])
    return (
        [row for i, row in enumerate(rows) if i not in validation_indices],
        [row for i, row in enumerate(rows) if i in validation_indices],
    )


def validate_query_sets(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], targets: set[str]
) -> None:
    overlap = {normalize_query(row["prompt"]) for row in train} & {
        normalize_query(row["prompt"]) for row in validation
    }
    if overlap:
        raise ValueError(f"Train/validation query overlap: {next(iter(overlap))!r}")
    for row in train + validation:
        missing = set(row["positive_targets"]) - targets
        if missing:
            raise ValueError(
                f"Positive targets missing from the retrieval corpus for {row['prompt']!r}: "
                f"{sorted(missing)[:3]}"
            )


def filter_query_targets(
    rows: list[dict[str, Any]], allowed_targets: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove unusable positive labels and training queries with no labels left."""
    filtered = []
    removed_positives = 0
    affected_queries = 0
    for row in rows:
        positives = [target for target in row["positive_targets"] if target in allowed_targets]
        removed = len(row["positive_targets"]) - len(positives)
        removed_positives += removed
        affected_queries += int(removed > 0)
        if positives:
            filtered.append({**row, "positive_targets": positives})
    return filtered, {
        "input_queries": len(rows), "kept_queries": len(filtered),
        "dropped_queries": len(rows) - len(filtered),
        "affected_queries": affected_queries, "removed_positive_labels": removed_positives,
    }


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    return consolidate_queries(iter_json_records(path))


def load_corpus(path: str | Path) -> list[str]:
    targets: set[str] = set()
    for number, row in enumerate(iter_json_records(path), 1):
        target = row.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Corpus row {number} needs a nonempty string target")
        targets.add(target.strip())
    if not targets:
        raise ValueError("The retrieval corpus is empty")
    return sorted(targets)


def prepare_metadata(
    query_path: str | Path, document_path: str | Path, target_type: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reuse Vault identity mappings without requiring a BM25 index or run."""
    if target_type not in TARGET_TYPES:
        raise ValueError(f"Unsupported target type: {target_type}")
    by_text_id: dict[str, set[str]] = {}
    corpus: set[str] = set()
    for number, row in enumerate(iter_json_records(document_path), 1):
        text_id = str(row.get("text_id", "")).strip()
        targets = document_targets(row, target_type)
        if not text_id or not targets:
            raise ValueError(f"Document metadata row {number} has no text_id or {target_type} target")
        by_text_id.setdefault(text_id, set()).update(targets)
        corpus.update(targets)

    def converted() -> Iterable[dict[str, Any]]:
        for number, row in enumerate(iter_json_records(query_path), 1):
            positives = set(unique_strings(row.get("positive_targets", [])))
            if target_type in {"structure_id_v3", "structure_id_v6"}:
                for field in (f"positive_{target_type}", f"positive_{target_type}s", f"target_{target_type}"):
                    positives.update(unique_strings(row.get(field, [])))
            ids = set(unique_strings(row.get("positive_text_ids", [])))
            ids.update(unique_strings(row.get("target_text_id", [])))
            for text_id in ids:
                if text_id not in by_text_id:
                    raise ValueError(f"Query metadata row {number}: unmapped positive text_id={text_id!r}")
                positives.update(by_text_id[text_id])
            yield {"prompt": row.get("prompt", ""), "positive_targets": sorted(positives)}

    queries = consolidate_queries(converted())
    validate_query_sets(queries, [], corpus)
    return queries, sorted(corpus)
