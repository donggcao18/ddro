"""Prepare BM25 documents and queries directly from one labeled JSONL file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from common import clean_prompt, iter_json_records, write_jsonl


def read_id(value: Any, field: str, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"{location}: {field} must be a non-empty string or integer")
    result = str(value)
    if any(character.isspace() for character in result):
        raise ValueError(f"{location}: {field} contains whitespace; use a whitespace-free ID for BM25")
    return result


def read_query(
    row: dict[str, Any], id_field: str, positive_field: str, location: str
) -> tuple[str, str, list[str]]:
    document_id = read_id(row.get(id_field), id_field, location)
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError(f"{location}: text must be a non-empty query string")
    prompt = clean_prompt(row["text"])
    if not prompt:
        raise ValueError(f"{location}: text contains no query after removing Query: prefix")
    values = row.get(positive_field)
    if not isinstance(values, list):
        raise ValueError(f"{location}: {positive_field} must be a list of positive IDs")
    # The row's own document is always positive, even if omitted from the list.
    positives = list(dict.fromkeys(
        [document_id] + [read_id(value, positive_field, location) for value in values]
    ))
    return document_id, prompt, positives


def code_contents(row: dict[str, Any], document_id: str, code_only: bool, location: str) -> str:
    original = row.get("original")
    if not isinstance(original, dict):
        raise ValueError(f"{location}: original must contain the document's source code")
    code = original.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"{location}: original.code must be a non-empty string")
    if code_only:
        return code.strip()
    parameters = original.get("parameters", [])
    names = [
        str(value.get("param", value.get("name", ""))) if isinstance(value, dict) else str(value)
        for value in parameters
    ] if isinstance(parameters, list) else []
    return "\n".join(part for part in [
        str(original.get("repo", "") or ""),
        str(original.get("path", "") or "").replace("/", " "),
        str(original.get("identifier", "") or ""),
        " ".join(names),
        document_id.replace("/", " "),
        code.strip(),
    ] if part)


def prepare_single_file(args: Any) -> dict[str, Any]:
    source = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    id_field = args.id_field
    positive_field = args.positive_id_field or f"positive_{id_field}"
    corpus_path = output_dir / "corpus" / "documents.jsonl"
    artifact_paths = [corpus_path] + [output_dir / name for name in (
        "document_metadata.jsonl", "query_metadata.jsonl", "queries.tsv", "qrels.tsv",
        "prepare_stats.json",
    )]
    if source in artifact_paths or source.is_relative_to(output_dir / "index"):
        raise ValueError("Input must not be a generated BM25 artifact or inside its index directory")
    # JsonCollection reads the entire corpus directory, so never index stale shards.
    if corpus_path.parent.exists():
        unexpected = [path for path in corpus_path.parent.iterdir() if path != corpus_path]
        if unexpected:
            raise ValueError("Corpus directory contains other files; use a new output directory")

    # Pass 1 validates every row and deduplicates code by its actual primary key.
    # Keep one code copy per document, regardless of the number of pseudo-queries.
    documents: dict[str, str] = {}
    declared_positives: set[str] = set()
    query_count = multi_label_count = 0
    for row_number, row in enumerate(iter_json_records(source), 1):
        location = f"{source}:record {row_number}"
        document_id, _, positives = read_query(row, id_field, positive_field, location)
        contents = code_contents(row, document_id, args.code_only, location)
        previous = documents.get(document_id)
        if previous is not None and previous != contents:
            raise ValueError(f"{location}: conflicting document contents for {id_field}={document_id!r}")
        documents[document_id] = contents
        declared_positives.update(positives)
        query_count += 1
        multi_label_count += len(positives) > 1
    if not documents:
        raise ValueError(f"{source}: no records found")

    corpus_hash = hashlib.sha256()

    def corpus_records() -> Iterator[dict[str, str]]:
        for document_id in sorted(documents):
            record = {"id": document_id, "contents": documents[document_id]}
            corpus_hash.update((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            yield record

    write_jsonl(corpus_path, corpus_records())
    write_jsonl(output_dir / "document_metadata.jsonl", (
        {"id": document_id, "id_field": id_field} for document_id in sorted(documents)
    ))

    # Pass 2 streams queries; repeated IDs and identical query strings remain separate rows.
    with (output_dir / "query_metadata.jsonl").open("w", encoding="utf-8") as metadata, \
         (output_dir / "queries.tsv").open("w", encoding="utf-8") as topics, \
         (output_dir / "qrels.tsv").open("w", encoding="utf-8") as qrels:
        for row_number, row in enumerate(iter_json_records(source), 1):
            document_id, prompt, positives = read_query(
                row, id_field, positive_field, f"{source}:record {row_number}"
            )
            query_key = f"vault-{row_number - 1:09d}"
            record = {
                "query_key": query_key, "prompt": prompt, "target_id": document_id,
                "positive_ids": positives, "id_field": id_field,
                "positive_id_field": positive_field, "source_row": row_number,
            }
            metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
            topics.write(f"{query_key}\t{prompt}\n")
            for positive_id in positives:
                qrels.write(f"{query_key}\t0\t{positive_id}\t1\n")

    stats = {
        "schema": "single_file", "input": str(source), "id_field": id_field,
        "positive_id_field": positive_field, "bm25_documents": len(documents),
        "queries": query_count, "multi_label_queries": multi_label_count,
        "positive_ids_missing_from_corpus": len(declared_positives - documents.keys()),
        "corpus_sha256": corpus_hash.hexdigest(),
    }
    (output_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats
