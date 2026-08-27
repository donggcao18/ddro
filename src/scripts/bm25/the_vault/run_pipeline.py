#!/usr/bin/env python3
"""Run the complete Vault preparation, Pyserini BM25, and DPO export flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    print(f"Completed in {(time.perf_counter() - started) / 60:.1f} minutes", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete Vault BM25 pipeline.")
    parser.add_argument("--train-original", required=True)
    parser.add_argument("--test-original", action="append", default=[])
    parser.add_argument("--augmentation", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--structure-id-source",
        action="append",
        default=[],
        help="JSON/JSONL containing url_based_id -> structure_id_v3 mappings.",
    )
    parser.add_argument(
        "--target-type",
        choices=["text_id", "structure_id_v3"],
        default="text_id",
        help="Decoder target namespace for the mined DPO pairs.",
    )
    parser.add_argument(
        "--structure-id-join-key",
        choices=["url_based_id", "numeric_id"],
        default="url_based_id",
        help="Cross-file key used to attach structure_id_v3 metadata.",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hits", type=int, default=200)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--rank-ranges", default="1:20,21:100,101:200")
    parser.add_argument("--rank-quotas", default="3,2,3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Skip Lucene indexing when WORK_DIR/index already contains an index.",
    )
    parser.add_argument("--pair-all-positives", action="store_true")
    parser.add_argument(
        "--fill-shortfall",
        action="store_true",
        help="Use non-original fallback sampling when a rank band is undersupplied.",
    )
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
    for structure_path in args.structure_id_source:
        prepare_command.extend(["--structure-id-source", structure_path])
    prepare_command.extend(
        ["--structure-id-join-key", args.structure_id_join_key]
    )
    if args.target_type == "structure_id_v3" and not args.structure_id_source:
        raise ValueError(
            "--target-type structure_id_v3 requires --structure-id-source"
        )
    if args.code_only:
        prepare_command.append("--code-only")
    if args.strict:
        prepare_command.append("--strict")
    run(prepare_command)

    index_dir = work_dir / "index"
    run_file = work_dir / "bm25_run.txt"
    if not (args.reuse_index and index_dir.is_dir() and any(index_dir.iterdir())):
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
    else:
        print(f"Reusing existing Lucene index: {index_dir}", flush=True)
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
            "--batch-size",
            str(args.batch_size),
        ]
    )

    output_name = (
        "dpo_pairs_structure_id_v3.jsonl"
        if args.target_type == "structure_id_v3"
        else "dpo_pairs.jsonl"
    )
    output_path = work_dir / output_name
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
        str(output_path),
        "--target-type",
        args.target_type,
        "--negatives-per-query",
        str(args.negatives_per_query),
        "--seed",
        str(args.seed),
        "--rank-ranges",
        args.rank_ranges,
        "--rank-quotas",
        args.rank_quotas,
    ]
    if args.pair_all_positives:
        mine_command.append("--pair-all-positives")
    if args.fill_shortfall:
        mine_command.append("--fill-shortfall")
    run(mine_command)

    print(f"DPO-ready data: {output_path}")


if __name__ == "__main__":
    main()
