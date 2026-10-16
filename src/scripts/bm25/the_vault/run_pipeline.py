#!/usr/bin/env python3
"""Run the complete Vault preparation, Pyserini BM25, and DPO export flow."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--input", help="Single labeled JSONL containing queries and original.code.")
    parser.add_argument("--id-field", default="url_based_id", help="Primary key in --input.")
    parser.add_argument("--positive-id-field", help="Defaults to positive_<id-field> in --input.")
    parser.add_argument("--train-original", help="Legacy original corpus.")
    parser.add_argument("--test-original", action="append", default=[])
    parser.add_argument("--augmentation", help="Legacy augmentation file.")
    parser.add_argument(
        "--augmentation-id-mode",
        choices=["auto", "numeric", "text_id", "url_based_id"],
        default="auto",
        help="How augmentation rows map to original text IDs.",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--structure-id-source",
        action="append",
        default=[],
        help="JSON/JSONL containing url_based_id -> structure_id_v3 mappings.",
    )
    parser.add_argument(
        "--target-type",
        choices=["text_id", "structure_id_v3", "structure_id_v6"],
        default=None,
        help="Decoder target namespace for the mined DPO pairs.",
    )
    parser.add_argument(
        "--structure-id-field",
        help="Field in the structure source; defaults to the structure target type.",
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
    parser = build_parser()
    args = parser.parse_args()
    if args.input:
        if args.train_original or args.augmentation or args.test_original or args.structure_id_source or args.target_type:
            parser.error("--input uses its own IDs; omit legacy source and --target-type arguments")
    elif not (args.train_original and args.augmentation):
        parser.error("provide --input (recommended), or both --train-original and --augmentation")
    else:
        args.target_type = args.target_type or "text_id"
    structure_target = args.target_type in {"structure_id_v3", "structure_id_v6"}
    structure_field = args.structure_id_field or (
        args.target_type if structure_target else "structure_id_v3"
    )
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        source = Path(args.input).resolve()
        if source.parent == work_dir or source.is_relative_to(work_dir / "index"):
            parser.error("keep --input outside the work directory to protect the source file")
    stats_path = work_dir / "prepare_stats.json"
    previous_hash = None
    if args.input and args.reuse_index and stats_path.is_file():
        previous_hash = json.loads(stats_path.read_text(encoding="utf-8")).get("corpus_sha256")

    prepare_command = [sys.executable, str(script_dir / "prepare_bm25.py"), "--output-dir", str(work_dir)]
    if args.input:
        prepare_command.extend(["--input", args.input, "--id-field", args.id_field])
        if args.positive_id_field:
            prepare_command.extend(["--positive-id-field", args.positive_id_field])
    else:
        prepare_command.extend([
            "--train-original", args.train_original,
            "--augmentation", args.augmentation,
            "--augmentation-id-mode", args.augmentation_id_mode,
        ])
    for test_path in args.test_original:
        prepare_command.extend(["--test-original", test_path])
    for structure_path in args.structure_id_source:
        prepare_command.extend(["--structure-id-source", structure_path])
    if not args.input:
        prepare_command.extend(
            ["--structure-id-field", structure_field,
             "--structure-id-join-key", args.structure_id_join_key]
        )
    if structure_target and not args.structure_id_source:
        raise ValueError(
            f"--target-type {args.target_type} requires --structure-id-source"
        )
    if args.code_only:
        prepare_command.append("--code-only")
    if args.strict:
        prepare_command.append("--strict")
    run(prepare_command)

    index_dir = work_dir / "index"
    run_file = work_dir / "bm25_run.txt"
    reuse_index = args.reuse_index and index_dir.is_dir() and any(index_dir.iterdir())
    if args.input and reuse_index:
        current_hash = json.loads(stats_path.read_text(encoding="utf-8"))["corpus_sha256"]
        index_manifest = index_dir / "corpus_sha256.txt"
        indexed_hash = index_manifest.read_text(encoding="utf-8").strip() if index_manifest.is_file() else None
        if previous_hash != current_hash or indexed_hash != current_hash:
            raise ValueError("Cannot reuse an index with different or unverified documents; use a new --work-dir")
    if not reuse_index:
        if args.input and index_dir.is_dir() and any(index_dir.iterdir()):
            raise ValueError("Index already exists; use --reuse-index or a new --work-dir")
        index_command = [
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
        ]
        if not args.input:
            index_command.extend(["--storePositions", "--storeDocvectors", "--storeRaw"])
        run(index_command)
        if args.input:
            current_hash = json.loads(stats_path.read_text(encoding="utf-8"))["corpus_sha256"]
            (index_dir / "corpus_sha256.txt").write_text(current_hash + "\n", encoding="utf-8")
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
            "trec" if args.input else "msmarco",
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
        f"dpo_pairs_{args.target_type}.jsonl"
        if structure_target else (
            "dpo_pairs_url.jsonl" if args.input and args.id_field == "url_based_id" else "dpo_pairs.jsonl"
        )
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
        "--negatives-per-query",
        str(args.negatives_per_query),
        "--seed",
        str(args.seed),
        "--rank-ranges",
        args.rank_ranges,
        "--rank-quotas",
        args.rank_quotas,
    ]
    if not args.input:
        mine_command.extend(["--target-type", args.target_type])
    if args.pair_all_positives:
        mine_command.append("--pair-all-positives")
    if args.fill_shortfall:
        mine_command.append("--fill-shortfall")
    run(mine_command)

    print(f"DPO-ready data: {output_path}")


if __name__ == "__main__":
    main()
