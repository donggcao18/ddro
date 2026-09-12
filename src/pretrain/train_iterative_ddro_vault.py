#!/usr/bin/env python3
"""Round-based Vault DPO with fixed, model-mined preferences per round."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parents[1]
MINING = SRC / "scripts/bm25/the_vault"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(MINING))
from common import document_targets, iter_json_records, normalize_query
from data.data_prep.prepare_vault_dpo_metadata import prepare
from pretrain.iterative_dpo_utils import (
    assert_file_hashes, atomic_json, atomic_jsonl, checkpoint_hash, file_hash,
    latest_resumable_checkpoint, partition_queries, read_json, round_queries,
)
from pretrain.update_dpo_reference import reference_identity


DEFAULTS = {
    "rounds": None, "queries_per_round": None, "steps_per_round": 1000,
    "epochs_per_round": None, "seed": 42, "validation_fraction": 0.05,
    "target_type": "text_id", "id_mode": "auto", "structure_id_sources": [],
    "input_format": "legacy", "corpus_files": [],
    "reference_update": "replace", "reference_ema_decay": 0.9,
    "num_gpus": 1, "precision": "bf16", "device": "auto",
    "num_beams": 16, "negatives_per_query": 4, "mining_batch_size": 16,
    "max_prompt_length": 256, "max_target_length": 32, "length_penalty": 1.0,
    "selection_metric": "mrr@10",
    "training": {},
}
TRAIN_DEFAULTS = {
    "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8,
    "learning_rate": 1e-6, "beta": 0.4, "warmup_ratio": 0.1,
    "weight_decay": 0.0, "max_grad_norm": 0.5, "save_steps": 500,
    "logging_steps": 10, "save_total_limit": 2, "dataset_num_proc": 1,
    "dataloader_num_workers": 0,
}


def load_config(path: Path) -> dict:
    raw = read_json(path)
    required = {"checkpoint_path", "query_file", "output_dir"}
    unknown = raw.keys() - required - DEFAULTS.keys()
    if unknown or required - raw.keys():
        raise ValueError(f"Invalid config keys: unknown={unknown}, missing={required - raw.keys()}")
    cfg = {**DEFAULTS, **raw}
    if cfg["reference_update"] not in {"replace", "ema"}:
        raise ValueError("reference_update must be replace or ema")
    if (type(cfg["reference_ema_decay"]) not in {int, float}
            or not math.isfinite(cfg["reference_ema_decay"])
            or not 0 <= cfg["reference_ema_decay"] <= 1):
        raise ValueError("reference_ema_decay must be between zero and one")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if "epochs_per_round" in raw and "steps_per_round" not in raw:
        cfg["steps_per_round"] = None
    if (cfg["steps_per_round"] is None) == (cfg["epochs_per_round"] is None):
        raise ValueError("Specify exactly one of steps_per_round or epochs_per_round")
    if cfg["input_format"] not in {"legacy", "multilabel"}:
        raise ValueError("Invalid input_format")
    if cfg["input_format"] == "multilabel":
        if "target_type" not in raw or not isinstance(cfg["target_type"], str) or not cfg["target_type"].strip():
            raise ValueError("multilabel requires target_type to name the document ID column")
        if cfg["target_type"] != cfg["target_type"].strip():
            raise ValueError("target_type must not contain surrounding whitespace")
        if cfg["id_mode"] != "auto" or cfg["structure_id_sources"]:
            raise ValueError("multilabel reads IDs directly; omit id_mode and structure_id_sources")
    elif cfg["target_type"] not in {"text_id", "url", "structure_id_v3"}:
        raise ValueError("Invalid target_type for legacy inputs")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Invalid precision")
    if cfg["id_mode"] not in {"auto", "numeric", "text_id"}:
        raise ValueError("Invalid id_mode")
    training = cfg["training"]
    if training.keys() - TRAIN_DEFAULTS.keys():
        raise ValueError(f"Unsupported training options: {training.keys() - TRAIN_DEFAULTS.keys()}")
    cfg["training"] = {**TRAIN_DEFAULTS, **training}
    for key in ("rounds", "num_gpus", "num_beams", "negatives_per_query", "mining_batch_size",
                "max_prompt_length", "max_target_length", "queries_per_round", "steps_per_round"):
        value = cfg[key]
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{key} must be a positive integer")
    if cfg["epochs_per_round"] is not None and not (0 < cfg["epochs_per_round"] < float("inf")):
        raise ValueError("epochs_per_round must be finite and positive")
    if not 0 < cfg["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not math.isfinite(cfg["length_penalty"]):
        raise ValueError("length_penalty must be finite")
    metrics = {f"{name}@{k}" for name in ("mrr", "recall")
               for k in {1, min(5, cfg["num_beams"]), min(10, cfg["num_beams"]), cfg["num_beams"]}}
    if cfg["selection_metric"] not in metrics:
        raise ValueError(f"selection_metric must be one of {sorted(metrics)}")
    for key, value in cfg["training"].items():
        if key in {"warmup_ratio", "weight_decay", "dataloader_num_workers"}:
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{key} must be finite and nonnegative")
        elif value <= 0 or not math.isfinite(value):
            raise ValueError(f"{key} must be finite and positive")
        if key not in {"learning_rate", "beta", "warmup_ratio", "weight_decay", "max_grad_norm"} and type(value) is not int:
            raise ValueError(f"{key} must be an integer")
    if cfg["training"]["warmup_ratio"] >= 1:
        raise ValueError("warmup_ratio must be less than one")
    for key in ("checkpoint_path", "query_file", "output_dir"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / value).resolve())
    for key in ("corpus_files", "structure_id_sources"):
        if not isinstance(cfg[key], list):
            raise ValueError(f"{key} must be a list of paths")
        cfg[key] = [str((path.parent / Path(p).expanduser()).resolve()) for p in cfg[key]]
    if not cfg["corpus_files"] and cfg["input_format"] == "legacy":
        raise ValueError("corpus_files cannot be empty for legacy inputs")
    return cfg


@contextmanager
def run_lock(directory: Path):
    """OS-released lock prevents concurrent controllers, including after a crash."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".controller.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def generation_command(cfg: dict, checkpoint: str, queries: Path, documents: Path,
                       output: Path, fingerprint: str, round_id: int, evaluate=False) -> list[str]:
    command = [sys.executable, str(MINING / "mine_model_confusion_negatives.py")]
    options = {
        "checkpoint-path": checkpoint, "checkpoint-fingerprint": fingerprint,
        "query-metadata": queries, "document-metadata": documents, "output": output,
        "target-type": cfg["target_type"], "num-beams": cfg["num_beams"],
        "negatives-per-query": cfg["negatives_per_query"], "batch-size": cfg["mining_batch_size"],
        "max-prompt-length": cfg["max_prompt_length"], "max-target-length": cfg["max_target_length"],
        "length-penalty": cfg["length_penalty"], "seed": cfg["seed"] + max(round_id, 0),
        "device": cfg["device"], "round-id": round_id,
        "target-collision-policy": "error", "target-length-policy": "error",
    }
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    command.append("--strict-corpus")
    if cfg["precision"] != "fp32":
        command.append(f"--{cfg['precision']}")
    if evaluate:
        command.append("--evaluation-only")
    return command


def training_options(cfg: dict, round_id: int) -> dict:
    return {**cfg["training"], "seed": cfg["seed"] + round_id,
            "max_prompt_length": cfg["max_prompt_length"], "max_target_length": cfg["max_target_length"],
            "max_steps": cfg["steps_per_round"] or -1,
            "num_train_epochs": cfg["epochs_per_round"] or 1.0}


def training_command(cfg: dict, manifest_path: Path, directory: Path,
                     resume_checkpoint: Path | None) -> list[str]:
    manifest = read_json(manifest_path)
    command = [sys.executable]
    if cfg["num_gpus"] > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={cfg['num_gpus']}"]
    command += [str(SRC / "pretrain/train_ddro_vault.py")]
    options = {**manifest["training"], "checkpoint_path": manifest["policy_checkpoint"],
               "reference_checkpoint_path": manifest["reference_checkpoint"], "train_file": manifest["pairs_file"],
               "round_manifest": manifest_path, "output_dir": directory, "validation_split": 0}
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    command += ["--no-load_best_model_at_end", "--export_latest", "--strict_targets", "--gradient_checkpointing"]
    if cfg["precision"] != "fp32":
        command.append(f"--{cfg['precision']}")
    if resume_checkpoint:
        command.extend(["--resume_from_checkpoint", str(resume_checkpoint)])
    return command


def audit_pairs(path: Path, queries: list[dict], documents: Path, target_type: str,
                round_id: int, fingerprint: str, quota: int, previous: Path | None) -> dict:
    query_map = {r["query_key"]: r for r in queries}
    targets = {r["text_id"]: document_targets(r, target_type) for r in iter_json_records(documents)}
    previous_keys = set()
    if previous:
        previous_keys = {(r["query_key"], r["chosen_text_id"], r["rejected_text_id"])
                         for r in iter_json_records(previous)}
    seen, counts = set(), Counter()
    for row in iter_json_records(path):
        query = query_map.get(row.get("query_key"))
        if query is None or row.get("prompt") != query["prompt"]:
            raise ValueError("Mined pair belongs to an unselected query or changed prompt")
        chosen, rejected = row["chosen_text_id"], row["rejected_text_id"]
        if (chosen != query["target_text_id"] or rejected in query["positive_text_ids"]
                or set(row["positive_text_ids"]) != set(query["positive_text_ids"])):
            raise ValueError("Invalid ground-truth preference")
        if targets.get(chosen) != [row["chosen"]] or targets.get(rejected) != [row["rejected"]]:
            raise ValueError("Preference has an invalid/ambiguous decoder target")
        if row.get("round_id") != round_id or row.get("policy_fingerprint") != fingerprint:
            raise ValueError("Stale mining provenance")
        if row.get("negative_source") != "model_confusion":
            raise ValueError("Only model-confusion negatives are allowed")
        key = (row["query_key"], chosen, rejected)
        if key in seen or row["chosen"] == row["rejected"]:
            raise ValueError("Duplicate or indistinguishable preference")
        seen.add(key)
        counts[row["query_key"]] += 1
        if counts[row["query_key"]] > quota:
            raise ValueError("Negative quota exceeded")
    return {"pairs": len(seen), "selected_queries": len(queries), "queries_with_pairs": len(counts),
            "queries_without_negatives": len(queries) - len(counts),
            "query_coverage": len(counts) / max(1, len(queries)),
            "pairs_per_query": dict(Counter(counts.get(key, 0) for key in query_map)),
            "overlap_with_previous_pairs": len(seen & previous_keys),
            "previous_overlap_fraction": len(seen & previous_keys) / max(1, len(seen))}


def run_pipeline(cfg: dict, resume=False, prepare_only=False, runner=run_command) -> dict:
    output = Path(cfg["output_dir"])
    with run_lock(output):
        return _run_pipeline(cfg, output, resume, prepare_only, runner)


def _run_pipeline(cfg, output, resume, prepare_only, runner):
    manifest_path = output / "run_manifest.json"
    sources = [cfg["query_file"], *cfg["corpus_files"], *cfg["structure_id_sources"]]
    code = [Path(__file__), SRC / "pretrain/train_ddro_vault.py",
            SRC / "pretrain/update_dpo_reference.py",
            SRC / "pretrain/iterative_dpo_utils.py", SRC / "data/data_prep/prepare_vault_dpo_metadata.py",
            MINING / "mine_model_confusion_negatives.py", MINING / "prepare_bm25.py", MINING / "common.py",
            SRC / "utils/trie.py"]
    versions = {}
    for package in ("torch", "transformers", "trl", "datasets", "accelerate", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    identity = {"config": cfg, "inputs": {str(p): file_hash(p) for p in sources},
                "code": {str(p): file_hash(p) for p in code}, "versions": versions,
                "initial_checkpoint": checkpoint_hash(cfg["checkpoint_path"])}
    if manifest_path.exists():
        if not resume:
            raise ValueError("Run exists; use --resume with the same config or a new output_dir")
        state = read_json(manifest_path)
        if state["identity"] != identity:
            raise ValueError("Resume config, source data, code, environment, or SFT checkpoint changed")
    else:
        if resume:
            raise ValueError("No run_manifest.json to resume")
        if any(p.name != ".controller.lock" for p in output.iterdir()):
            raise ValueError("New run requires an empty output directory")
        state = {"identity": identity, "stages": {}, "rounds": [], "complete": False}
        atomic_json(manifest_path, state)

    def stage(name, artifacts, action):
        if name in state["stages"]:
            assert_file_hashes(state["stages"][name])
            return
        print(f"Stage: {name}", flush=True)
        action()
        state["stages"][name] = {str(p): file_hash(p) for p in artifacts}
        atomic_json(manifest_path, state)

    metadata = output / "metadata"
    documents = metadata / "document_metadata.jsonl"
    training = metadata / "train_queries.jsonl"
    validation = metadata / "validation_queries.jsonl"
    stage("prepare", [documents, training, validation, metadata / "split_manifest.json"], lambda: prepare(
        cfg["corpus_files"], cfg["query_file"], str(metadata), cfg["validation_fraction"], cfg["seed"],
        cfg["structure_id_sources"], cfg["id_mode"], cfg["input_format"], cfg["target_type"]))
    queries = list(iter_json_records(training))
    heldout = list(iter_json_records(validation))
    if ({normalize_query(r["prompt"]) for r in queries} & {normalize_query(r["prompt"]) for r in heldout}
            or {r["family_id"] for r in queries} & {r["family_id"] for r in heldout}):
        raise ValueError("Training and validation query families overlap")
    partitions = partition_queries(queries, cfg["queries_per_round"], cfg["seed"]) if cfg["rounds"] is None else None
    total_rounds = len(partitions) if partitions is not None else cfg["rounds"]
    plan = {"mode": "partition_all" if partitions is not None else "fixed_rounds",
            "total_rounds": total_rounds, "training_queries": len(queries), "validation_queries": len(heldout),
            "queries_per_round": cfg["queries_per_round"], "seed": cfg["seed"],
            "partition_sizes": [len(part) for part in partitions] if partitions is not None else None}
    plan_path = output / "round_plan.json"
    stage("round-plan", [plan_path], lambda: atomic_json(plan_path, plan))
    if read_json(plan_path) != plan:
        raise ValueError("Round partition plan changed")
    state["planned_rounds"] = total_rounds
    atomic_json(manifest_path, state)
    print(f"Round plan: {total_rounds} rounds for {len(queries)} training queries ({plan['mode']}).", flush=True)
    if prepare_only:
        return state

    def evaluate(name, checkpoint, fingerprint, round_id, directory):
        predictions = directory / "validation_predictions.jsonl"
        stats = predictions.with_suffix(".stats.json")
        stage(name, [predictions, stats], lambda: runner(generation_command(
            cfg, checkpoint, validation, documents, predictions, fingerprint, round_id, evaluate=True)))
        return read_json(stats)["metrics"]

    checkpoint = cfg["checkpoint_path"]
    fingerprint = identity["initial_checkpoint"]
    reference_checkpoint, reference_fingerprint = checkpoint, fingerprint
    baseline = evaluate("baseline", checkpoint, fingerprint, -1, output / "baseline")
    best = {"round_id": -1, "checkpoint": checkpoint, "metrics": baseline,
            "checkpoint_fingerprint": fingerprint}
    previous = None
    results = []
    for round_id in range(total_rounds):
        directory = output / f"round-{round_id:03d}"
        selected = directory / "queries.jsonl"
        pairs = directory / "preferences.jsonl"
        audit = directory / "pair_audit.json"
        selected_rows = (partitions[round_id] if partitions is not None else
                         round_queries(queries, cfg["queries_per_round"], round_id, cfg["seed"]))
        stage(f"select-{round_id}", [selected], lambda: atomic_jsonl(selected, selected_rows))
        mining_stats = pairs.with_suffix(".stats.json")
        def mine_round():
            runner(generation_command(cfg, checkpoint, selected, documents, pairs, fingerprint, round_id))
            report = audit_pairs(pairs, selected_rows, documents, cfg["target_type"],
                                 round_id, fingerprint, cfg["negatives_per_query"], previous)
            if cfg["steps_per_round"]:
                presentations = (cfg["steps_per_round"] * cfg["num_gpus"] *
                                 cfg["training"]["per_device_train_batch_size"] *
                                 cfg["training"]["gradient_accumulation_steps"])
                report["nominal_pair_presentations"] = presentations if report["pairs"] else 0
                report["nominal_dataset_passes"] = presentations / report["pairs"] if report["pairs"] else 0
            else:
                report["nominal_pair_presentations"] = report["pairs"] * cfg["epochs_per_round"]
                report["nominal_dataset_passes"] = cfg["epochs_per_round"] if report["pairs"] else 0
            atomic_json(audit, report)
            print("Round preferences:", json.dumps(report), flush=True)
        stage(f"mine-{round_id}", [pairs, mining_stats, audit], mine_round)
        round_manifest = directory / "round_inputs.json"
        payload = {"round_id": round_id, "policy_checkpoint": checkpoint,
                   "policy_fingerprint": fingerprint, "pairs_file": str(pairs),
                   "reference_checkpoint": reference_checkpoint, "reference_fingerprint": reference_fingerprint,
                   "inputs": {str(p): file_hash(p) for p in [selected, pairs, documents, validation]},
                   "training": training_options(cfg, round_id), "precision": cfg["precision"],
                   "num_gpus": cfg["num_gpus"]}
        stage(f"round-inputs-{round_id}", [round_manifest], lambda: atomic_json(round_manifest, payload))
        if read_json(round_manifest) != payload:
            raise ValueError("Round snapshot or data provenance changed")
        round_identity = file_hash(round_manifest)
        train_dir = directory / "training"
        latest = train_dir / "latest"
        completion = latest / "round_complete.json"
        def train():
            if completion.is_file():
                info = read_json(completion)
                if info["identity"] != round_identity or checkpoint_hash(latest) != info["checkpoint_fingerprint"]:
                    raise ValueError("Completed policy snapshot does not match the round")
                return
            resume_checkpoint = latest_resumable_checkpoint(
                train_dir, round_identity, cfg["num_gpus"], cfg["precision"] == "fp16")
            runner(training_command(cfg, round_manifest, train_dir, resume_checkpoint))
        report = read_json(audit)
        if report["pairs"] == 0:
            skipped = directory / "round_skipped.json"
            marker = {"identity": round_identity, "reason": "no_model_negatives",
                      "checkpoint": checkpoint, "checkpoint_fingerprint": fingerprint, "global_step": 0}
            stage(f"skip-{round_id}", [skipped], lambda: atomic_json(skipped, marker))
            if read_json(skipped) != marker:
                raise ValueError("Skipped round provenance changed")
            print(f"Round {round_id}: no usable negatives; skipping training and keeping the policy.", flush=True)
            info = marker
        else:
            stage(f"train-{round_id}", [completion, train_dir / "training_metrics.json"], train)
            info = read_json(completion)
            if info["identity"] != round_identity or checkpoint_hash(latest) != info["checkpoint_fingerprint"]:
                raise ValueError("Round output snapshot changed or is incomplete")
            checkpoint, fingerprint = str(latest), info["checkpoint_fingerprint"]
            if cfg["reference_update"] == "ema":
                reference_output = directory / "reference"
                reference_marker = reference_output / "reference_complete.json"
                expected = reference_identity(reference_checkpoint, checkpoint, reference_fingerprint,
                                              fingerprint, cfg["reference_ema_decay"])
                command = [sys.executable, str(SRC / "pretrain/update_dpo_reference.py")]
                for key, value in {**expected, "output": str(reference_output)}.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                stage(f"reference-{round_id}", [reference_marker], lambda: runner(command))
                reference_info = read_json(reference_marker)
                if (reference_info["identity"] != expected
                        or checkpoint_hash(reference_output) != reference_info["checkpoint_fingerprint"]):
                    raise ValueError("EMA reference snapshot or provenance changed")
                reference_checkpoint = str(reference_output)
                reference_fingerprint = reference_info["checkpoint_fingerprint"]
            else:
                reference_checkpoint, reference_fingerprint = checkpoint, fingerprint
        metrics = evaluate(f"evaluate-{round_id}", checkpoint, fingerprint, round_id, directory)
        result = {"round_id": round_id, "checkpoint": checkpoint, "checkpoint_fingerprint": fingerprint,
                  "metrics": metrics, "pair_audit": report, "optimizer_steps": info["global_step"],
                  "reference_checkpoint": payload["reference_checkpoint"],
                  "next_reference_checkpoint": reference_checkpoint,
                  "status": "trained" if report["pairs"] else "skipped_no_negatives"}
        results.append(result)
        if metrics[cfg["selection_metric"]] > best["metrics"][cfg["selection_metric"]]:
            best = {key: result[key] for key in ("round_id", "checkpoint", "checkpoint_fingerprint", "metrics")}
        state.update({"rounds": results, "best": best, "latest": checkpoint,
                      "latest_reference": reference_checkpoint})
        atomic_json(manifest_path, state)
        atomic_json(output / "best_checkpoint.json", best)
        previous = pairs
    state["complete"] = True
    atomic_json(manifest_path, state)
    print(f"Completed {len(results)} rounds. Latest: {checkpoint}. Best: {best['checkpoint']}", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare fixed query splits, then exit before generation")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch the controller with python; it launches torchrun for each training round")
    run_pipeline(load_config(args.config.resolve()), args.resume, args.prepare_only)


if __name__ == "__main__":
    main()
