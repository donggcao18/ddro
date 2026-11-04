#!/usr/bin/env python3
"""Evaluate a URL retriever on native Vault multi-positive JSON/JSONL rows."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.bm25.the_vault.common import iter_json_records, write_jsonl
from utils.grpo_data import load_corpus, prepare_url_file
from utils.grpo_trainer import TargetIndex, beam_generation_config, resolve_score_normalization
from utils.retrieval_metrics import CUTOFFS, METRIC_NAMES, score_ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Local final/ or save_pretrained checkpoint.")
    parser.add_argument("--test_file", required=True, help="Native text/url_based_id/positive_url_based_id rows.")
    parser.add_argument("--corpus_file", required=True, help="Prepared training URL catalog.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="Queries per inference worker.")
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--score_normalization", choices=("auto", "trie", "full_vocab"), default="auto",
                        help="auto reads checkpoint scoring metadata; use the same explicit rule for SFT comparisons.")
    parser.add_argument("--catalog_mode", choices=("extend_test", "fixed"), default="extend_test",
                        help="fixed searches exactly corpus_file, without extending it from test records.")
    parser.add_argument("--invalid_target_policy", choices=("error", "skip"),
                        default=os.environ.get("GRPO_INVALID_TARGET_POLICY", "error"))
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N; auto binds torchrun workers to LOCAL_RANK.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--progress_interval", type=float, default=30.,
                        help="Seconds between per-rank progress reports; 0 keeps only first/final batch reports.")
    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.batch_size, args.max_prompt_length, args.max_target_length) < 1 or args.num_beams < 10:
        raise ValueError("Batch/token limits must be positive and num_beams must be >= 10 for Hit/MRR@10")
    progress_interval = getattr(args, "progress_interval", 30.)
    if not 0 <= progress_interval < float("inf"):
        raise ValueError("progress_interval must be finite and nonnegative")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "auto":
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
        if world > 1 and device.type == "cuda":
            raise ValueError("Use --device auto with torchrun so each worker gets a separate GPU")
    if args.fp16 and device.type != "cuda":
        raise ValueError("FP16 inference requires CUDA; omit --fp16 for CPU inference")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    initialized_here = False
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        initialized_here = True
    try:
        # Test labels are read for scoring only. Candidate extension uses the
        # test records' own document URLs, never their positive label sets.
        rows, test_catalog = prepare_url_file(args.test_file, include_positive_targets_in_corpus=False)
        training_catalog = set(load_corpus(args.corpus_file))
        catalog = training_catalog if getattr(args, "catalog_mode", "extend_test") == "fixed" else training_catalog | set(test_catalog)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=True)
        if not model.config.is_encoder_decoder:
            raise ValueError("URL inference requires an encoder-decoder checkpoint")
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
        index = TargetIndex(tokenizer, model.config, sorted(catalog), args.max_target_length,
                            invalid_target_policy=getattr(args, "invalid_target_policy", "error"))
        original_catalog = catalog
        catalog = set(index.target_to_sequence)
        model.to(device)
        model.eval()
        beams = min(args.num_beams, len(catalog))
        returned = min(10, len(catalog))
        normalization = resolve_score_normalization(args.model_path, getattr(args, "score_normalization", "auto"))
        generation = beam_generation_config(index, args.max_target_length, beams, returned, normalization)
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            index.write_filter_report(destination)
            print("Target filtering:", json.dumps(index.filtering_summary), flush=True)
        shard_path = destination / f".predictions.rank{rank}.jsonl"
        sums = {name: 0.0 for name in METRIC_NAMES}
        local_count = 0
        indices = list(range(rank, len(rows), world))
        started = last_report = time.monotonic()
        print(f"[inference rank {rank}/{world}] Starting: device={device}, local queries={len(indices)}, "
              f"total queries={len(rows)}, batch={args.batch_size}, beams={beams}, catalog={len(catalog)}, "
              f"model={args.model_path}. Decoding first batch...", flush=True)
        with shard_path.open("w", encoding="utf-8") as handle:
            for offset in range(0, len(indices), args.batch_size):
                positions = indices[offset:offset + args.batch_size]
                batch = [rows[i] for i in positions]
                tokenized = tokenizer([row["prompt"] for row in batch], padding=True, truncation=True,
                                      max_length=args.max_prompt_length, return_tensors="pt")
                inputs = {name: tokenized[name].to(device) for name in ("input_ids", "attention_mask")}
                precision = torch.autocast("cuda", dtype=torch.float16) if args.fp16 else nullcontext()
                with torch.inference_mode(), precision:
                    sequences = model.generate(
                        **inputs, generation_config=generation,
                        prefix_allowed_tokens_fn=index.allowed, synced_gpus=False,
                    )
                decoded = [index.sequence_to_target.get(index.canonical(row)) for row in sequences.cpu().tolist()]
                for i, row in enumerate(batch):
                    ranked = list(dict.fromkeys(target for target in decoded[i * returned:(i + 1) * returned] if target is not None))
                    scores = score_ranking(ranked, row["positive_targets"])
                    for name, value in scores.items():
                        sums[name] += value
                    local_count += 1
                    prediction = {
                        "query_index": positions[i], "query": row["prompt"],
                        "positive_url_based_id": row["positive_targets"],
                        "top_1_prediction": ranked[0] if ranked else None,
                        "top_10_predictions": ranked,
                        **scores,
                    }
                    handle.write(json.dumps(prediction, ensure_ascii=True) + "\n")
                now = time.monotonic()
                if offset == 0 or local_count == len(indices) or (progress_interval and now - last_report >= progress_interval):
                    # Make the existing prediction shard useful for monitoring
                    # without flushing every query on a shared filesystem.
                    handle.flush()
                    elapsed = max(now - started, 1e-9)
                    rate = local_count / elapsed
                    remaining = (len(indices) - local_count) / rate
                    print(f"[inference rank {rank}/{world}] {local_count}/{len(indices)} local queries "
                          f"({100 * local_count / len(indices):.1f}%); elapsed={elapsed:.1f}s; "
                          f"rate={rate:.2f} queries/s; ETA={remaining:.1f}s", flush=True)
                    last_report = now
        print(f"[inference rank {rank}/{world}] Local decoding complete; aggregating metrics.", flush=True)
        totals = torch.tensor([local_count, *[sums[name] for name in METRIC_NAMES]], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        values = totals.cpu().tolist()
        count = int(values[0])
        if count != len(rows):
            raise RuntimeError(f"Expected {len(rows)} evaluated queries, got {count}")
        metrics = {name: values[i + 1] / count for i, name in enumerate(METRIC_NAMES)}
        result = {
            **metrics, "num_queries": count, "corpus_targets": len(catalog),
            "corpus_targets_before_filtering": len(original_catalog), "target_filtering": index.filtering_summary,
            "test_catalog_targets_added": len(catalog - training_catalog),
            "catalog_mode": getattr(args, "catalog_mode", "extend_test"),
            "queries_with_no_retrievable_positive": sum(not (set(row["positive_targets"]) & catalog) for row in rows),
            "queries_with_missing_positive_targets": sum(bool(set(row["positive_targets"]) - catalog) for row in rows),
            "missing_positive_targets": len({target for row in rows for target in row["positive_targets"] if target not in catalog}),
            "model_path": str(args.model_path), "test_file": str(args.test_file),
            "num_beams": beams, "returned_predictions": returned, "world_size": world,
            "score_normalization": normalization,
        }
        if rank == 0:
            print(f"[inference] Merging {world} prediction shard(s) into {destination / 'predictions.jsonl'}", flush=True)
            shards = [destination / f".predictions.rank{i}.jsonl" for i in range(world)]
            merged = heapq.merge(*(iter_json_records(path) for path in shards), key=lambda row: row["query_index"])
            written = write_jsonl(destination / "predictions.jsonl", merged, ensure_ascii=True)
            if written != count:
                raise RuntimeError(f"Prediction output count mismatch: {written} != {count}")
            (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            for path in shards:
                path.unlink()
            print(json.dumps(result, indent=2), flush=True)
        if world > 1:
            dist.barrier()
        return result
    finally:
        if initialized_here:
            dist.destroy_process_group()


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
