#!/usr/bin/env python3
"""Retrieve Vault code by URL ID from one labeled JSONL or JSON file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import chain, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from common import as_text_id_list, clean_prompt, iter_json_records, write_jsonl


def iter_documents(input_path: Path, document_ids: set[str]) -> Iterator[dict[str, str]]:
    for number, row in enumerate(iter_json_records(input_path), start=1):
        original = row.get("original")
        code = original.get("code") if isinstance(original, dict) else None
        if not isinstance(code, str) or not code.strip():
            continue
        url_id = row.get("url_based_id")
        if not isinstance(url_id, str) or not url_id.strip():
            raise ValueError(f"Row {number} has code but no url_based_id")
        if url_id not in document_ids:
            document_ids.add(url_id)
            yield {"id": url_id, "contents": code}


def iter_queries(input_path: Path, include_original: bool) -> Iterator[dict[str, Any]]:
    for number, row in enumerate(iter_json_records(input_path), start=1):
        is_original = row.get("is_original", False)
        if is_original and not include_original:
            continue
        text = row.get("text")
        prompt = clean_prompt(text) if isinstance(text, str) else ""
        if not prompt:
            continue
        positives = as_text_id_list(row.get("positive_url_based_id"))
        if not positives:
            raise ValueError(f"Row {number} has a query but no positive_url_based_id")
        yield {
            "query_id": f"vault-{number:09d}",
            "text": prompt,
            "url_based_id": row.get("url_based_id"),
            "is_original": is_original,
            "positive_url_based_id": positives,
        }


def iter_results(
    searcher: Any,
    queries: Iterator[dict[str, Any]],
    document_ids: set[str],
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    while batch := list(islice(queries, args.batch_size)):
        results = searcher.batch_search(
            [query["text"] for query in batch],
            [query["query_id"] for query in batch],
            k=args.hits,
            threads=args.threads,
        )
        for query in batch:
            positives = query["positive_url_based_id"]
            positive_ids = set(positives)
            yield {
                **query,
                "missing_positive_url_based_ids": [
                    url_id for url_id in positives if url_id not in document_ids
                ],
                "hits": [
                    {
                        "url_based_id": hit.docid,
                        "rank": rank,
                        "score": hit.score,
                        "is_positive": hit.docid in positive_ids,
                    }
                    for rank, hit in enumerate(results.get(query["query_id"], []), start=1)
                ],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Vault JSONL or JSON array")
    parser.add_argument("--output", required=True, type=Path, help="Ranked results in JSONL")
    parser.add_argument("--hits", type=int, default=200)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--include-original", action="store_true", help="Also search rows with is_original=true"
    )
    args = parser.parse_args()
    if min(args.hits, args.threads, args.batch_size) < 1:
        parser.error("--hits, --threads, and --batch-size must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="vault-bm25-", dir=args.output.parent) as temporary:
        work_dir = Path(temporary)
        corpus_dir = work_dir / "corpus"
        index_dir = work_dir / "index"
        document_ids: set[str] = set()
        documents = write_jsonl(
            corpus_dir / "documents.jsonl", iter_documents(args.input, document_ids)
        )
        if not documents:
            parser.error("no documents with original.code found in the input")

        queries = iter_queries(args.input, args.include_original)
        first_query = next(queries, None)
        if first_query is None:
            parser.error("no eligible query text found; use --include-original for original queries")

        from pyserini.search.lucene import LuceneSearcher

        subprocess.run(
            [
                sys.executable, "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", str(corpus_dir),
                "--index", str(index_dir),
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", str(args.threads),
            ],
            check=True,
        )
        searcher = LuceneSearcher(str(index_dir))
        output_path = work_dir / "results.jsonl"
        try:
            searcher.set_bm25(k1=0.82, b=0.68)
            count = write_jsonl(
                output_path,
                iter_results(searcher, chain([first_query], queries), document_ids, args),
            )
        finally:
            searcher.close()
        output_path.replace(args.output)

    print(f"Indexed {documents:,} documents and wrote {count:,} query results to {args.output}")


if __name__ == "__main__":
    main()
