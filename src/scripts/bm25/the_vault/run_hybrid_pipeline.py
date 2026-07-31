#!/usr/bin/env python3
"""Run the additive Vault BM25 + model-confusion negative pipeline."""

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
    parser = argparse.ArgumentParser(
        description="Run Vault BM25 baseline mining, model-confusion mining, and hybrid combination."
    )
    parser.add_argument("--train-original", required=True)
    parser.add_argument("--test-original", action="append", default=[])
    parser.add_argument("--augmentation", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--target-type",
        choices=["text_id", "url"],
        default="text_id",
        help="Decoder target namespace for model mining and the final hybrid file.",
    )
    parser.add_argument(
        "--bm25-python",
        default=sys.executable,
        help="Python executable for the Pyserini BM25 pipeline.",
    )
    parser.add_argument(
        "--model-python",
        default=sys.executable,
        help="Python executable for model mining, conversion, and combination.",
    )

    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--bm25-batch-size", type=int, default=16)
    parser.add_argument("--hits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument(
        "--reuse-bm25-output",
        action="store_true",
        help="Skip the existing BM25 pipeline when its prepared files and dpo_pairs.jsonl exist.",
    )
    parser.add_argument("--pair-all-positives", action="store_true")

    parser.add_argument("--model-negatives-per-query", type=int, default=4)
    parser.add_argument("--total-negatives-per-query", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--require-exact-mix", action="store_true")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    return parser


def require_reusable_bm25_files(work_dir: Path) -> None:
    required = (
        work_dir / "query_metadata.jsonl",
        work_dir / "document_metadata.jsonl",
        work_dir / "dpo_pairs.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "--reuse-bm25-output was requested, but required files are missing: "
            + ", ".join(missing)
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.model_negatives_per_query <= 0:
        raise ValueError("--model-negatives-per-query must be greater than zero")
    if not (
        args.model_negatives_per_query
        <= args.total_negatives_per_query
        <= 8
    ):
        raise ValueError(
            "The additive pipeline requires model negatives <= total negatives <= "
            "the 8-pair BM25 baseline"
        )
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_bm25_output:
        require_reusable_bm25_files(work_dir)
        print(f"Reusing existing BM25 output in {work_dir}", flush=True)
    else:
        bm25_command = [
            args.bm25_python,
            str(script_dir / "run_pipeline.py"),
            "--train-original",
            args.train_original,
            "--augmentation",
            args.augmentation,
            "--work-dir",
            str(work_dir),
            "--threads",
            str(args.threads),
            "--batch-size",
            str(args.bm25_batch_size),
            "--hits",
            str(args.hits),
            "--negatives-per-query",
            "8",
            "--rank-ranges",
            "1:20,21:100,101:200",
            "--rank-quotas",
            "3,2,3",
            "--seed",
            str(args.seed),
        ]
        for test_path in args.test_original:
            bm25_command.extend(["--test-original", test_path])
        if args.code_only:
            bm25_command.append("--code-only")
        if args.strict:
            bm25_command.append("--strict")
        if args.reuse_index:
            bm25_command.append("--reuse-index")
        if args.pair_all_positives:
            bm25_command.append("--pair-all-positives")
        run(bm25_command)

    bm25_pairs = work_dir / "dpo_pairs.jsonl"
    if args.target_type == "url":
        bm25_pairs = work_dir / "dpo_pairs_url.jsonl"
        run(
            [
                args.model_python,
                str(script_dir / "postprocess_dpo_urls.py"),
                "--input",
                str(work_dir / "dpo_pairs.jsonl"),
                "--document-metadata",
                str(work_dir / "document_metadata.jsonl"),
                "--output",
                str(bm25_pairs),
            ]
        )

    model_output_name = (
        "model_confusion_pairs_url.jsonl"
        if args.target_type == "url"
        else "model_confusion_pairs.jsonl"
    )
    model_output = work_dir / model_output_name
    model_command = [
        args.model_python,
        str(script_dir / "mine_model_confusion_negatives.py"),
        "--checkpoint-path",
        args.checkpoint_path,
        "--query-metadata",
        str(work_dir / "query_metadata.jsonl"),
        "--document-metadata",
        str(work_dir / "document_metadata.jsonl"),
        "--output",
        str(model_output),
        "--target-type",
        args.target_type,
        "--negatives-per-query",
        str(args.model_negatives_per_query),
        "--num-beams",
        str(args.num_beams),
        "--batch-size",
        str(args.model_batch_size),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-target-length",
        str(args.max_target_length),
        "--device",
        args.device,
    ]
    if args.limit_queries is not None:
        model_command.extend(["--limit-queries", str(args.limit_queries)])
    if args.pair_all_positives:
        model_command.append("--pair-all-positives")
    if args.bf16:
        model_command.append("--bf16")
    if args.fp16:
        model_command.append("--fp16")
    run(model_command)

    hybrid_output_name = (
        "dpo_pairs_hybrid_url.jsonl"
        if args.target_type == "url"
        else "dpo_pairs_hybrid.jsonl"
    )
    hybrid_output = work_dir / hybrid_output_name
    combine_command = [
        args.model_python,
        str(script_dir / "combine_hybrid_dpo.py"),
        "--bm25-input",
        str(bm25_pairs),
        "--model-input",
        str(model_output),
        "--document-metadata",
        str(work_dir / "document_metadata.jsonl"),
        "--output",
        str(hybrid_output),
        "--model-per-query",
        str(args.model_negatives_per_query),
        "--total-per-query",
        str(args.total_negatives_per_query),
        "--seed",
        str(args.seed),
    ]
    if args.require_exact_mix:
        combine_command.append("--require-exact-mix")
    run(combine_command)
    print(f"BM25 baseline data: {work_dir / 'dpo_pairs.jsonl'}")
    if args.target_type == "url":
        print(f"BM25 URL data: {bm25_pairs}")
    print(f"Model-confusion data: {model_output}")
    print(f"Hybrid DPO data: {hybrid_output}")


if __name__ == "__main__":
    main()
