#!/usr/bin/env python3
"""Run the complete Vault preparation, Pyserini BM25, and DPO export flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete Vault BM25 pipeline.")
    parser.add_argument("--train-original", required=True)
    parser.add_argument("--test-original", action="append", default=[])
    parser.add_argument("--augmentation", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--hits", type=int, default=1000)
    parser.add_argument("--negatives-per-query", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--pair-all-positives", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    prepare_command = [
        sys.executable,
        str(script_dir / "prepare_bm25.py"),
        "--train-original",
        args.train_original,
        "--augmentation",
        args.augmentation,
        "--output-dir",
        str(work_dir),
    ]
    for test_path in args.test_original:
        prepare_command.extend(["--test-original", test_path])
    if args.code_only:
        prepare_command.append("--code-only")
    if args.strict:
        prepare_command.append("--strict")
    run(prepare_command)

    index_dir = work_dir / "index"
    run_file = work_dir / "bm25_run.txt"
    run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(work_dir / "corpus"),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(args.threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pyserini.search.lucene",
            "--index",
            str(index_dir),
            "--topics",
            str(work_dir / "queries.tsv"),
            "--output",
            str(run_file),
            "--output-format",
            "msmarco",
            "--hits",
            str(args.hits),
            "--bm25",
            "--k1",
            "0.82",
            "--b",
            "0.68",
            "--threads",
            str(args.threads),
        ]
    )

    mine_command = [
        sys.executable,
        str(script_dir / "mine_dpo_negatives.py"),
        "--run",
        str(run_file),
        "--query-metadata",
        str(work_dir / "query_metadata.jsonl"),
        "--document-metadata",
        str(work_dir / "document_metadata.jsonl"),
        "--output",
        str(work_dir / "dpo_pairs.jsonl"),
        "--negatives-per-query",
        str(args.negatives_per_query),
        "--seed",
        str(args.seed),
    ]
    if args.pair_all_positives:
        mine_command.append("--pair-all-positives")
    run(mine_command)

    print(f"DPO-ready data: {work_dir / 'dpo_pairs.jsonl'}")


if __name__ == "__main__":
    main()
