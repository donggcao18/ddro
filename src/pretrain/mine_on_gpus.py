#!/usr/bin/env python3
"""Run independent, resumable mining shards on the first N visible CUDA devices."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import subprocess
import sys
import time

SRC = Path(__file__).resolve().parents[1]
MINER = SRC / "scripts/bm25/the_vault/mine_model_confusion_negatives.py"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(MINER.parent))
from common import iter_json_records
from mine_model_confusion_negatives import build_parser
from pretrain.iterative_dpo_utils import (
    assert_file_hashes, atomic_json, atomic_jsonl, checkpoint_hash, file_hash, read_json,
)

# Corpus/configuration values describe one shared corpus, not query counters.
SHARED_STATS = set("""
documents usable_decoder_targets invalid_document_target_mappings
invalid_document_url_mappings tokenized_docid_sequences target_type
target_collision_policy target_length_policy target_token_policy
target_sequences_with_unknown_tokens target_sequences_with_padding_tokens
max_target_length tokenizer_collision_groups allowed_collision_groups
excluded_collision_targets excluded_collision_text_ids excluded_overlength_targets
truncated_targets excluded_overlength_text_ids model_mining_excluded_targets
model_mining_excluded_text_ids checkpoint num_beams negatives_per_query seed
round_id policy_fingerprint length_penalty
""".split())
OUTPUT_STATS = {"device", "output", "exclusions_output", "collision_exclusions_output"}


def split_batches(rows: list[dict], batch_size: int, workers: int) -> list[list[dict]]:
    """Partition contiguous whole batches, retaining original beam-search inputs."""
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    batches = math.ceil(len(rows) / batch_size)
    active = min(workers, batches)
    shards, offset = [], 0
    for rank in range(active):
        count = batches // active + (rank < batches % active)
        end = min(len(rows), offset + count * batch_size)
        shards.append(rows[offset:end])
        offset = end
    return shards


def validate_cuda(workers: int, device: str) -> None:
    import torch

    if device not in {"auto", "cuda"}:
        raise ValueError("Multi-GPU mining requires device=auto or cuda; use CUDA_VISIBLE_DEVICES")
    available = torch.cuda.device_count()
    if available < workers:
        raise ValueError(f"Mining requested {workers} GPUs, but only {available} CUDA devices are visible. "
                         "Set mining_num_gpus or CUDA_VISIBLE_DEVICES accordingly.")


def run_workers(jobs: list[dict], on_complete) -> None:
    """Launch all workers; commit successes immediately and stop peers on failure."""
    processes = []
    try:
        for job in jobs:
            log = Path(job["log"]).open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            processes.append((process, log, job))
            print(f"Mining GPU {job['rank']}: log={job['log']}", flush=True)
        pending = list(processes)
        last_report = time.monotonic()
        while pending:
            failed = []
            for item in list(pending):
                process, log, job = item
                status = process.poll()
                if status is None:
                    continue
                log.close()
                pending.remove(item)
                if status:
                    failed.append((status, job))
                else:
                    on_complete(job)
                    print(f"Mining GPU {job['rank']}: completed", flush=True)
            if failed:
                status, job = failed[0]
                raise RuntimeError(f"Mining GPU {job['rank']} failed (exit {status}); see {job['log']}")
            if pending:
                if time.monotonic() - last_report >= 30:
                    print(f"Mining: {len(pending)} GPU worker(s) still running; progress is in shard logs", flush=True)
                    last_report = time.monotonic()
                time.sleep(0.2)
    finally:
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, log, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def merge_stats(stats: list[dict], output: Path, workers: int) -> dict:
    result, counts, distribution, totals = {}, Counter(), Counter(), Counter()
    special = SHARED_STATS | OUTPUT_STATS | {"candidate_count_distribution", "metrics"}
    for key in SHARED_STATS:
        values = [part.get(key) for part in stats]
        if any(value != values[0] for value in values):
            raise ValueError(f"Mining shard metadata differs: {key}")
        if key in stats[0]:
            result[key] = values[0]
    metric_keys = set(stats[0].get("metrics", {}))
    for part in stats:
        for key, value in part.items():
            if key not in special:
                if type(value) is not int:
                    raise ValueError(f"Unexpected mining counter: {key}")
                counts[key] += value
        distribution.update(part.get("candidate_count_distribution", {}))
        if set(part.get("metrics", {})) != metric_keys:
            raise ValueError("Mining shards have different evaluation metric keys")
        for key, value in part.get("metrics", {}).items():
            totals[key] += value * part.get("queries_processed", 0)
    result.update(counts)
    if metric_keys:
        if not counts["queries_processed"]:
            raise ValueError("No validation queries were evaluated")
        result["metrics"] = {key: totals[key] / counts["queries_processed"] for key in sorted(metric_keys)}
    result.update(candidate_count_distribution=dict(sorted(distribution.items())),
                  output=str(output), device=[f"cuda:{i}" for i in range(len(stats))],
                  mining_num_gpus=workers, active_mining_workers=len(stats),
                  exclusions_output=str(output.with_suffix(".excluded_text_ids.json")),
                  collision_exclusions_output=str(output.with_suffix(".excluded_text_ids.json")))
    return result


def run_sharded(argv: list[str], workers: int, launcher=run_workers, device_check=validate_cuda) -> dict:
    """Use the unchanged single-GPU miner and merge only fully committed shards."""
    if type(workers) is not int or workers <= 0:
        raise ValueError("num-gpus must be a positive integer")
    args = build_parser().parse_args(argv)
    device_check(workers, args.device)
    rows = list(iter_json_records(args.query_metadata))
    if args.limit_queries is not None:
        if args.limit_queries <= 0:
            raise ValueError("limit-queries must be positive")
        rows = rows[:args.limit_queries]
    if not rows:
        raise ValueError("No queries to mine")
    keys = [row["query_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Mining query keys must be unique")
    shards = split_batches(rows, args.batch_size, workers)
    output = Path(args.output)
    directory = output.with_suffix(".shards")
    directory.mkdir(parents=True, exist_ok=True)
    policy_hash = checkpoint_hash(args.checkpoint_path)
    if args.checkpoint_fingerprint and args.checkpoint_fingerprint != policy_hash:
        raise ValueError("Mining policy checkpoint fingerprint changed")
    identity = {"args": vars(args), "workers": workers, "checkpoint": policy_hash,
                "query_hash": file_hash(args.query_metadata), "document_hash": file_hash(args.document_metadata),
                "code": {str(p): file_hash(p) for p in (
                    Path(__file__), MINER, MINER.parent / "common.py", SRC / "utils/trie.py",
                    SRC / "pretrain/iterative_dpo_utils.py")}}
    jobs, pending = [], []
    for rank, shard in enumerate(shards):
        shard_dir = directory / f"gpu-{rank:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        queries = shard_dir / "queries.jsonl"
        target = shard_dir / output.name
        atomic_jsonl(queries, shard)
        # Reconstruct parsed arguments so both '--arg=value' and '--arg value'
        # work, and the global limit is not mistakenly applied to every shard.
        options = {**vars(args), "query_metadata": str(queries), "output": str(target),
                   "device": f"cuda:{rank}", "limit_queries": None}
        command = [sys.executable, str(MINER)]
        for key, value in options.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                command.append(flag)
            elif value is not None and value is not False:
                command.extend([flag, str(value)])
        job = {"rank": rank, "command": command, "log": str(shard_dir / "worker.log"),
               "marker": shard_dir / "complete.json", "output": target,
               "keys": {row["query_key"] for row in shard},
               "identity": {**identity, "rank": rank, "shard_hash": file_hash(queries)}}
        jobs.append(job)
        if job["marker"].exists() and read_json(job["marker"])["identity"] == job["identity"]:
            assert_file_hashes(read_json(job["marker"])["artifacts"])
            print(f"Mining GPU {rank}: reusing completed shard", flush=True)
        else:
            pending.append(job)

    def commit(job):
        target = job["output"]
        stats = read_json(target.with_suffix(".stats.json"))
        if stats.get("queries_seen", 0) != len(job["keys"]):
            raise ValueError("Mining shard did not visit every assigned query")
        count = 0
        for row in iter_json_records(target):
            if row.get("query_key") not in job["keys"]:
                raise ValueError("Mining shard produced a query from another shard")
            count += 1
        expected = stats.get("queries_processed", 0) if args.evaluation_only else stats.get("model_pairs_written", 0)
        if count != expected:
            raise ValueError("Mining shard output count does not match its statistics")
        paths = [target, target.with_suffix(".stats.json"), target.with_suffix(".excluded_text_ids.json")]
        atomic_json(job["marker"], {"identity": job["identity"],
                                    "artifacts": {str(p): file_hash(p) for p in paths}})

    if pending:
        launcher(pending, commit)
    # Never publish a partial global dataset, even if a launcher exits early.
    for job in jobs:
        marker = read_json(job["marker"])
        if marker["identity"] != job["identity"]:
            raise ValueError("Stale mining shard")
        assert_file_hashes(marker["artifacts"])
    parts = [read_json(job["output"].with_suffix(".stats.json")) for job in jobs]
    result = merge_stats(parts, output, workers)
    exclusions = [read_json(job["output"].with_suffix(".excluded_text_ids.json")) for job in jobs]
    if any(part != exclusions[0] for part in exclusions):
        raise ValueError("Mining shards disagree on corpus exclusions")
    atomic_jsonl(output, (row for job in jobs for row in iter_json_records(job["output"])))
    atomic_json(output.with_suffix(".excluded_text_ids.json"), exclusions[0])
    atomic_json(output.with_suffix(".stats.json"), result)
    print(f"Mining merged {len(jobs)} shards: {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--num-gpus", type=int, required=True)
    args, miner_args = parser.parse_known_args()
    run_sharded(miner_args, args.num_gpus)


if __name__ == "__main__":
    main()
