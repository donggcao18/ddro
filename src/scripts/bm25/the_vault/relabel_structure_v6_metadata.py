#!/usr/bin/env python3
"""Attach structure_id_v6 targets to an existing Vault BM25 run's metadata.

The existing query keys and canonical text IDs are preserved so bm25_run.txt
can be re-mined without rebuilding the Lucene index or rerunning retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import as_text_id, as_text_id_list, iter_json_records, unique_strings, write_jsonl
from prepare_bm25 import load_structure_id_map, structure_join_value


def load_explicit_v6_positives(paths: list[Path]) -> dict[str, set[str]]:
    """Conservatively union known positives for each URL across merged rows."""
    positives_by_url: dict[str, set[str]] = {}
    for path in paths:
        for row in iter_json_records(path):
            url = structure_join_value(row, "url_based_id")
            if not url:
                continue
            positives = set(unique_strings(row.get("positive_structure_id_v6")))
            positives.update(unique_strings(row.get("positive_structure_id_v6s")))
            if positives:
                positives_by_url.setdefault(url, set()).update(positives)
    return positives_by_url


def relabel(args: argparse.Namespace) -> dict[str, Any]:
    source_documents = Path(args.source_document_metadata)
    source_queries = Path(args.source_query_metadata)
    output_documents = Path(args.output_document_metadata)
    output_queries = Path(args.output_query_metadata)
    if source_documents.resolve() == output_documents.resolve():
        raise ValueError("Output document metadata must differ from the source")
    if source_queries.resolve() == output_queries.resolve():
        raise ValueError("Output query metadata must differ from the source")

    structure_sources = [Path(path) for path in args.structure_id_source]
    key_to_v6, source_stats = load_structure_id_map(
        structure_sources,
        "structure_id_v6",
        "url_based_id",
    )
    if not key_to_v6:
        raise ValueError("No usable url_based_id -> structure_id_v6 mappings found")
    positives_by_url = load_explicit_v6_positives(structure_sources)

    documents: list[dict[str, Any]] = []
    text_id_to_v6: dict[str, list[str]] = {}
    text_id_to_urls: dict[str, list[str]] = {}
    for row in iter_json_records(source_documents):
        text_id = as_text_id(row.get("text_id"))
        if not text_id:
            raise ValueError("Document metadata contains a row without text_id")
        if text_id in text_id_to_v6:
            raise ValueError(f"Duplicate document text_id: {text_id!r}")
        keys = unique_strings(row.get("url_based_ids", row.get("url_based_id", "")))
        targets = sorted({key_to_v6[key] for key in keys if key in key_to_v6})
        updated = dict(row)
        updated["structure_id_v6s"] = targets
        documents.append(updated)
        text_id_to_v6[text_id] = targets
        text_id_to_urls[text_id] = keys

    queries: list[dict[str, Any]] = []
    seen_query_keys: set[str] = set()
    queries_with_explicit_positives = 0
    for row in iter_json_records(source_queries):
        query_key = as_text_id(row.get("query_key"))
        chosen_id = as_text_id(row.get("target_text_id"))
        if not query_key or query_key in seen_query_keys:
            raise ValueError(f"Missing or duplicate query_key: {query_key!r}")
        seen_query_keys.add(query_key)
        chosen_targets = text_id_to_v6.get(chosen_id, [])
        if len(chosen_targets) != 1:
            raise ValueError(
                f"{query_key}: chosen text_id={chosen_id!r} needs exactly one "
                f"structure_id_v6 mapping; found {chosen_targets!r}"
            )
        chosen_target = chosen_targets[0]
        query_url = structure_join_value(row, "url_based_id")
        if not query_url and len(text_id_to_urls.get(chosen_id, [])) == 1:
            query_url = text_id_to_urls[chosen_id][0]
        query_target = key_to_v6.get(query_url)
        if not query_target:
            raise ValueError(
                f"{query_key}: query url_based_id={query_url!r} has no "
                "structure_id_v6 mapping in the new merged file"
            )
        if query_target != chosen_target:
            raise ValueError(
                f"{query_key}: query URL maps to {query_target!r}, but its "
                f"chosen document maps to {chosen_target!r}"
            )

        positive_ids = set(as_text_id_list(row.get("positive_text_ids", [])))
        positive_ids.add(chosen_id)
        positive_targets = {
            target
            for positive_id in positive_ids
            for target in text_id_to_v6.get(positive_id, [])
        }
        positive_targets.add(chosen_target)
        explicit_positives = positives_by_url.get(query_url, set())
        if explicit_positives:
            queries_with_explicit_positives += 1
            positive_targets.update(explicit_positives)
        updated = dict(row)
        updated["target_structure_id_v6"] = chosen_target
        updated["positive_structure_id_v6s"] = sorted(positive_targets)
        queries.append(updated)

    output_documents.parent.mkdir(parents=True, exist_ok=True)
    output_queries.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_documents, documents)
    write_jsonl(output_queries, queries)
    stats = {
        **source_stats,
        "documents": len(documents),
        "documents_with_unique_structure_id_v6": sum(
            len(targets) == 1 for targets in text_id_to_v6.values()
        ),
        "documents_without_structure_id_v6": sum(
            not targets for targets in text_id_to_v6.values()
        ),
        "documents_with_ambiguous_structure_id_v6": sum(
            len(targets) > 1 for targets in text_id_to_v6.values()
        ),
        "queries": len(queries),
        "queries_with_explicit_v6_positives": queries_with_explicit_positives,
        "urls_with_explicit_v6_positives": len(positives_by_url),
        "source_document_metadata": str(source_documents),
        "source_query_metadata": str(source_queries),
        "output_document_metadata": str(output_documents),
        "output_query_metadata": str(output_queries),
    }
    stats_path = output_queries.parent / "relabel_structure_v6_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-query-metadata", required=True)
    parser.add_argument("--source-document-metadata", required=True)
    parser.add_argument("--structure-id-source", action="append", required=True)
    parser.add_argument("--output-query-metadata", required=True)
    parser.add_argument("--output-document-metadata", required=True)
    return parser


def main() -> None:
    stats = relabel(build_parser().parse_args())
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
