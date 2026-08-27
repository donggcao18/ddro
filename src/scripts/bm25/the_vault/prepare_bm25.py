#!/usr/bin/env python3
"""Prepare a Vault corpus, pseudo-queries, qrels, and mappings for BM25."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    CODE_PREFIX_RE,
    QUERY_PREFIX_RE,
    as_text_id,
    as_text_id_list,
    clean_code,
    clean_prompt,
    document_targets,
    iter_json_records,
    normalize_query,
    write_jsonl,
)


def structure_join_value(row: dict[str, Any], join_key: str) -> str:
    if join_key == "url_based_id":
        return as_text_id(row.get("url_based_id", row.get("url_id", ""))).strip()
    return as_text_id(row.get(join_key)).strip()


def load_structure_id_map(
    paths: list[Path], field: str, join_key: str
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load a validated join-key -> structure target mapping."""
    key_to_structure: dict[str, str] = {}
    rows = 0
    missing_join_key = 0
    missing_structure_id = 0

    for path in paths:
        for row_number, row in enumerate(iter_json_records(path), start=1):
            rows += 1
            key = structure_join_value(row, join_key)
            structure_id = as_text_id(row.get(field)).strip()
            if not key:
                missing_join_key += 1
                continue
            if not structure_id:
                missing_structure_id += 1
                continue
            previous = key_to_structure.get(key)
            if previous is not None and previous != structure_id:
                raise ValueError(
                    f"{join_key} {key!r} maps to multiple {field} values: "
                    f"{previous!r} and {structure_id!r} ({path}:{row_number})"
                )
            key_to_structure[key] = structure_id

    return key_to_structure, {
        "structure_join_key": join_key,
        "structure_source_rows": rows,
        "structure_source_join_keys": len(key_to_structure),
        "structure_source_rows_missing_join_key": missing_join_key,
        "structure_source_rows_missing_target": missing_structure_id,
    }


def build_original_indexes(
    paths: list[Path],
    key_to_structure: dict[str, str] | None = None,
    structure_join_key: str = "url_based_id",
) -> dict[str, Any]:
    """Index Vault metadata and reproduce its normalized-query multi-label map."""
    numeric_to_target: dict[str, str] = {}
    target_metadata: dict[str, dict[str, Any]] = {}
    code_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_to_targets: dict[str, set[str]] = defaultdict(set)
    original_row_count = 0
    query_row_count = 0
    code_row_count = 0

    for path in paths:
        for row in iter_json_records(path):
            original_row_count += 1
            text_id = as_text_id(row.get("text_id"))
            numeric_id = as_text_id(row.get("numeric_id"))
            raw_text = str(row.get("text", ""))

            if not text_id:
                continue

            if numeric_id:
                previous = numeric_to_target.get(numeric_id)
                if previous is not None and previous != text_id:
                    raise ValueError(
                        f"numeric_id {numeric_id!r} maps to both {previous!r} and {text_id!r}"
                    )
                numeric_to_target[numeric_id] = text_id

            metadata = target_metadata.setdefault(
                text_id,
                {
                    "text_id": text_id,
                    "numeric_ids": [],
                    "url_based_ids": [],
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "identifier": row.get("identifier", ""),
                    "language": row.get("language", ""),
                },
            )
            if numeric_id and numeric_id not in metadata["numeric_ids"]:
                metadata["numeric_ids"].append(numeric_id)
            url_based_id = str(row.get("url_based_id", row.get("url_id", "")) or "")
            if url_based_id and url_based_id not in metadata["url_based_ids"]:
                metadata["url_based_ids"].append(url_based_id)

            if QUERY_PREFIX_RE.match(raw_text):
                query_row_count += 1
                normalized = normalize_query(raw_text)
                if normalized:
                    query_to_targets[normalized].add(text_id)
            elif CODE_PREFIX_RE.match(raw_text):
                code_row_count += 1
                code_rows[text_id].append(row)

    # A target can occur in more than one normalized query group. Union all
    # overlapping groups so no known positive can be sampled as a negative.
    target_to_positives: dict[str, set[str]] = defaultdict(set)
    for target_ids in query_to_targets.values():
        for text_id in target_ids:
            target_to_positives[text_id].update(target_ids)

    changed = True
    while changed:
        changed = False
        for text_id, positives in list(target_to_positives.items()):
            expanded = set(positives)
            for positive_id in positives:
                expanded.update(target_to_positives.get(positive_id, {positive_id}))
            if expanded != positives:
                target_to_positives[text_id] = expanded
                changed = True

    for text_id in target_metadata:
        target_to_positives[text_id].add(text_id)

    if key_to_structure:
        for metadata in target_metadata.values():
            metadata_keys = (
                metadata["url_based_ids"]
                if structure_join_key == "url_based_id"
                else metadata["numeric_ids"]
            )
            metadata["structure_id_v3s"] = sorted(
                {
                    key_to_structure[key]
                    for key in metadata_keys
                    if key in key_to_structure
                }
            )

    return {
        "numeric_to_target": numeric_to_target,
        "target_metadata": target_metadata,
        "code_rows": code_rows,
        "query_to_targets": query_to_targets,
        "target_to_positives": target_to_positives,
        "original_row_count": original_row_count,
        "query_row_count": query_row_count,
        "code_row_count": code_row_count,
    }


def make_code_contents(rows: list[dict[str, Any]], include_metadata: bool) -> str:
    """Build one searchable BM25 document for a quantized text_id."""
    sections: list[str] = []
    seen: set[str] = set()

    for row in rows:
        if include_metadata:
            parameters = row.get("parameters", [])
            parameter_names = []
            if isinstance(parameters, list):
                for parameter in parameters:
                    if isinstance(parameter, dict) and parameter.get("param"):
                        parameter_names.append(str(parameter["param"]))

            metadata_parts = [
                str(row.get("repo", "") or ""),
                str(row.get("path", "") or "").replace("/", " "),
                str(row.get("identifier", "") or ""),
                " ".join(parameter_names),
                str(row.get("url_based_id", row.get("url_id", "")) or "").replace("/", " "),
            ]
            metadata_text = "\n".join(part for part in metadata_parts if part).strip()
            if metadata_text and metadata_text not in seen:
                seen.add(metadata_text)
                sections.append(metadata_text)

        code = clean_code(row.get("text", ""))
        if code and code not in seen:
            seen.add(code)
            sections.append(code)

    return "\n\n".join(sections).strip()


def resolve_augmentation_target(
    row: dict[str, Any],
    numeric_to_target: dict[str, str],
    known_targets: set[str],
    mode: str,
) -> tuple[str, list[str], str]:
    """Resolve raw q10 rows and map.py-ready rows to a quantized text_id."""
    numeric_id = as_text_id(row.get("numeric_id"))
    raw_ids = as_text_id_list(row.get("text_id"))

    if mode in {"auto", "numeric"}:
        if numeric_id and numeric_id in numeric_to_target:
            return numeric_to_target[numeric_id], raw_ids, numeric_id

        if len(raw_ids) == 1 and raw_ids[0] in numeric_to_target:
            return numeric_to_target[raw_ids[0]], [], raw_ids[0]

        if mode == "numeric":
            return "", raw_ids, numeric_id or (raw_ids[0] if raw_ids else "")

    if raw_ids:
        if numeric_id and numeric_id in numeric_to_target:
            return numeric_to_target[numeric_id], raw_ids, numeric_id
        if raw_ids[0] in known_targets or mode == "text_id":
            return raw_ids[0], raw_ids, numeric_id

    return "", raw_ids, numeric_id


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    original_paths = [Path(args.train_original)]
    original_paths.extend(Path(path) for path in args.test_original)
    output_dir = Path(args.output_dir)
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    structure_sources = [
        Path(path) for path in getattr(args, "structure_id_source", [])
    ]
    structure_field = getattr(args, "structure_id_field", "structure_id_v3")
    structure_join_key = getattr(args, "structure_id_join_key", "url_based_id")
    key_to_structure, structure_stats = load_structure_id_map(
        structure_sources, structure_field, structure_join_key
    )
    indexes = build_original_indexes(
        original_paths, key_to_structure, structure_join_key
    )
    code_rows = indexes["code_rows"]
    target_metadata = indexes["target_metadata"]
    target_to_positives = indexes["target_to_positives"]

    corpus_records: list[dict[str, str]] = []
    document_metadata_records: list[dict[str, Any]] = []
    for text_id in sorted(code_rows):
        contents = make_code_contents(code_rows[text_id], not args.code_only)
        if not contents:
            continue
        corpus_records.append({"id": text_id, "contents": contents})
        document_metadata_records.append(target_metadata[text_id])

    write_jsonl(corpus_dir / "documents.jsonl", corpus_records)
    write_jsonl(output_dir / "document_metadata.jsonl", document_metadata_records)

    query_metadata: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    qrels: dict[str, list[str]] = {}
    known_targets = set(target_metadata)
    corpus_targets = {record["id"] for record in corpus_records}
    missing_positive_corpus = 0

    for row_number, row in enumerate(iter_json_records(args.augmentation)):
        prompt = clean_prompt(row.get("text", ""))
        if not prompt:
            continue

        target, explicit_ids, numeric_id = resolve_augmentation_target(
            row,
            indexes["numeric_to_target"],
            known_targets,
            args.augmentation_id_mode,
        )
        if not target:
            unmapped = dict(row)
            unmapped["_row_number"] = row_number
            unmapped["_reason"] = "Could not map numeric_id/text_id to an original quantized text_id"
            unmapped_rows.append(unmapped)
            if args.strict:
                raise ValueError(unmapped["_reason"] + f" at augmentation row {row_number}")
            continue

        positives = set(target_to_positives.get(target, {target}))
        for explicit_id in explicit_ids:
            if explicit_id in known_targets:
                positives.update(target_to_positives.get(explicit_id, {explicit_id}))
        positives.add(target)

        query_key = f"vault-{len(query_metadata):09d}"
        positive_ids = sorted(positives)
        qrels[query_key] = positive_ids
        if target not in corpus_targets:
            missing_positive_corpus += 1

        metadata_row = {
            "query_key": query_key,
            "prompt": prompt,
            "target_text_id": target,
            "positive_text_ids": positive_ids,
            "numeric_id": numeric_id,
            "url_based_id": row.get("url_based_id", row.get("url_id", "")),
            "is_original": bool(row.get("is_original", False)),
            "augmentation_row": row_number,
        }
        if key_to_structure:
            positive_structure_ids = sorted(
                {
                    structure_id
                    for positive_id in positive_ids
                    for structure_id in document_targets(
                        target_metadata.get(positive_id, {}), "structure_id_v3"
                    )
                }
            )
            target_join_key = structure_join_value(row, structure_join_key)
            target_structure_id = key_to_structure.get(target_join_key, "")
            if not target_structure_id:
                target_values = document_targets(
                    target_metadata.get(target, {}), "structure_id_v3"
                )
                if len(target_values) == 1:
                    target_structure_id = target_values[0]
            if target_structure_id and target_structure_id not in positive_structure_ids:
                positive_structure_ids.append(target_structure_id)
                positive_structure_ids.sort()
            metadata_row["target_structure_id_v3"] = target_structure_id
            metadata_row["positive_structure_id_v3s"] = positive_structure_ids
        query_metadata.append(metadata_row)

    write_jsonl(output_dir / "query_metadata.jsonl", query_metadata)
    write_jsonl(output_dir / "unmapped_queries.jsonl", unmapped_rows)

    with (output_dir / "queries.tsv").open("w", encoding="utf-8") as handle:
        for row in query_metadata:
            prompt = row["prompt"].replace("\t", " ").replace("\n", " ")
            handle.write(f"{row['query_key']}\t{prompt}\n")

    with (output_dir / "qrels.tsv").open("w", encoding="utf-8") as handle:
        for query_key, positive_ids in qrels.items():
            for text_id in positive_ids:
                handle.write(f"{query_key}\t0\t{text_id}\t1\n")

    multi_groups = [
        targets for targets in indexes["query_to_targets"].values() if len(targets) > 1
    ]
    structure_target_owners: dict[str, set[str]] = defaultdict(set)
    for row in document_metadata_records:
        for structure_id in document_targets(row, "structure_id_v3"):
            structure_target_owners[structure_id].add(row["text_id"])
    stats = {
        "original_files": [str(path) for path in original_paths],
        "augmentation_file": str(args.augmentation),
        "original_rows": indexes["original_row_count"],
        "original_query_rows": indexes["query_row_count"],
        "original_code_rows": indexes["code_row_count"],
        "bm25_documents": len(corpus_records),
        "augmentation_queries": len(query_metadata),
        "unmapped_augmentation_rows": len(unmapped_rows),
        "multi_label_query_groups": len(multi_groups),
        "max_multi_label_group_size": max((len(group) for group in multi_groups), default=1),
        "queries_whose_current_positive_is_missing_from_corpus": missing_positive_corpus,
        **structure_stats,
        "documents_with_structure_id_v3": sum(
            bool(document_targets(row, "structure_id_v3"))
            for row in document_metadata_records
        ),
        "documents_with_ambiguous_structure_id_v3": sum(
            len(document_targets(row, "structure_id_v3")) > 1
            for row in document_metadata_records
        ),
        "structure_id_v3_collisions": sum(
            len(text_ids) > 1 for text_ids in structure_target_owners.values()
        ),
        "queries_missing_target_structure_id_v3": sum(
            not row.get("target_structure_id_v3")
            for row in query_metadata
            if key_to_structure
        ),
    }
    with (output_dir / "prepare_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Vault Code: corpus and augmented pseudo-queries for BM25."
    )
    parser.add_argument("--train-original", required=True)
    parser.add_argument(
        "--test-original",
        action="append",
        default=[],
        help="Optional test/original files used for corpus coverage and multi-label detection. Repeatable.",
    )
    parser.add_argument("--augmentation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--structure-id-source",
        action="append",
        default=[],
        help=(
            "JSON/JSONL containing url_based_id and structure_id_v3. Repeatable; "
            "used to enrich BM25 metadata without changing Lucene document IDs."
        ),
    )
    parser.add_argument(
        "--structure-id-field",
        default="structure_id_v3",
        help="Structure target field in --structure-id-source.",
    )
    parser.add_argument(
        "--structure-id-join-key",
        choices=["url_based_id", "numeric_id"],
        default="url_based_id",
        help="Cross-file identity key for structure metadata (default: url_based_id).",
    )
    parser.add_argument(
        "--augmentation-id-mode",
        choices=["auto", "numeric", "text_id"],
        default="auto",
        help=(
            "auto accepts raw q10 rows (text_id is numeric_id), map.py-ready rows "
            "(numeric_id plus quantized text_id), and multi-label text_id lists."
        ),
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Index only Code: text; by default repository/path/function metadata is also searchable.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on the first unmapped augmentation row.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = prepare(args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
