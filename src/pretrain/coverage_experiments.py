#!/usr/bin/env python3
"""Prepare a fixed-catalog comparison and summarize its held-out results."""

import argparse
import csv
from hashlib import sha256
from itertools import zip_longest
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bm25.the_vault.common import iter_json_records, write_jsonl
from utils.grpo_data import load_corpus, prepare_url_file, split_queries, validate_query_sets
from utils.retrieval_metrics import METRIC_NAMES


EXPERIMENTS = ("multi_positive_dpo", "missed_positive_dpo", "multi_negative_dpo")
AVAILABLE_EXPERIMENTS = (*EXPERIMENTS, "tca_grpo")


def baseline_signature(args, metrics):
    catalog = sorted(set(load_corpus(args.corpus_file)))
    catalog_hash = sha256(json.dumps(catalog, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    return {"checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "test_sha256": fingerprint(args.test_input_file)["sha256"], "catalog_sha256": catalog_hash,
            "max_prompt_length": args.max_prompt_length, "max_target_length": args.max_target_length,
            "num_beams": min(args.num_beams, metrics["corpus_targets"]), "precision": args.precision,
            "invalid_target_policy": args.invalid_target_policy, "score_normalization": "trie", "catalog_mode": "fixed"}


def legacy_baseline_signature(source, metrics):
    """Recover provenance for completed suites created before reuse metadata existed."""
    root = source.parent
    suite = json.loads((root / "suite_data.json").read_text(encoding="utf-8"))
    manifests = sorted(root.glob("*/grpo_run.json"))
    if not manifests:
        raise ValueError("Legacy baseline needs its suite_data.json, data/corpus.jsonl and an experiment grpo_run.json")
    arguments = json.loads(manifests[0].read_text(encoding="utf-8"))["arguments"]
    catalog = sorted(set(load_corpus(root / "data/corpus.jsonl")))
    return {"checkpoint_path": str(Path(metrics["model_path"]).resolve()),
            "test_sha256": suite["test"]["sha256"],
            "catalog_sha256": sha256(json.dumps(catalog, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest(),
            "max_prompt_length": arguments["max_prompt_length"], "max_target_length": arguments["max_target_length"],
            "num_beams": metrics["num_beams"], "precision": "fp16" if arguments.get("fp16") else "fp32",
            "invalid_target_policy": metrics["target_filtering"]["policy"],
            "score_normalization": metrics["score_normalization"], "catalog_mode": metrics["catalog_mode"]}


def baseline_artifacts(args):
    """Record or reuse finished baseline inference; never run model generation."""
    destination = Path(args.destination_dir)
    source = Path(args.source_dir) if args.source_dir else destination
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    missing_metrics = [name for name in METRIC_NAMES if name not in metrics]
    if missing_metrics:
        raise ValueError(f"Completed baseline is missing coverage metrics: {missing_metrics}")
    predictions = source / "predictions.jsonl"
    if not predictions.is_file():
        raise ValueError("Baseline reuse requires completed predictions.jsonl, not temporary per-rank shards")
    with predictions.open(encoding="utf-8") as handle:
        if sum(bool(line.strip()) for line in handle) != metrics["num_queries"]:
            raise ValueError("Baseline predictions are incomplete: row count does not match num_queries")
    expected = baseline_signature(args, metrics)
    for name in ("model_path", "num_beams", "score_normalization", "catalog_mode"):
        actual = str(Path(metrics[name]).resolve()) if name == "model_path" else metrics[name]
        wanted = expected["checkpoint_path"] if name == "model_path" else expected[name]
        if actual != wanted:
            raise ValueError(f"Baseline mismatch for {name}: {actual!r} != {wanted!r}")
    if args.source_dir:
        provenance = source / "baseline_provenance.json"
        if provenance.is_file():
            recorded = json.loads(provenance.read_text(encoding="utf-8"))
            for name, key in (("metrics.json", "metrics_sha256"), ("predictions.jsonl", "predictions_sha256")):
                if fingerprint(source / name)["sha256"] != recorded[key]:
                    raise ValueError(f"Cached baseline {name} changed since its provenance was recorded")
            previous = recorded["signature"]
        else:
            previous = legacy_baseline_signature(source, metrics)
        mismatched = [name for name in expected if previous.get(name) != expected[name]]
        if mismatched:
            raise ValueError(f"Cannot reuse baseline: mismatched {', '.join(mismatched)}. Use RUN_BASELINE_INFERENCE=1 to recompute.")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Baseline destination must be empty")
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "predictions.jsonl", "target_filtering.json", "excluded_targets.jsonl"):
            if (source / name).is_file():
                shutil.copy2(source / name, destination / name)
    provenance = {"signature": expected, "reused_from": str(source.resolve()) if args.source_dir else None,
                  "metrics_sha256": fingerprint(destination / "metrics.json")["sha256"],
                  "predictions_sha256": fingerprint(destination / "predictions.jsonl")["sha256"]}
    (destination / "baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"{'Reused' if args.source_dir else 'Recorded'} completed SFT baseline: {destination}", flush=True)


def fingerprint(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def prepare(args):
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Suite directory must be empty; choose a fresh output_dir")
    train, catalog = prepare_url_file(args.train_input_file)
    if args.eval_input_file:
        validation, validation_catalog = prepare_url_file(args.eval_input_file, include_positive_targets_in_corpus=False)
        catalog = sorted(set(catalog) | set(validation_catalog))
    else:
        train, validation = split_queries(train, args.validation_split, args.seed)
    # The inventory is fixed before all experiments. Test query text and test
    # relevance labels never enter training; only document URLs extend inventory.
    test_rows, test_catalog = prepare_url_file(args.test_input_file, include_positive_targets_in_corpus=False)
    catalog = sorted(set(catalog) | set(test_catalog))
    if args.corpus_file:
        catalog = load_corpus(args.corpus_file)
    validate_query_sets(train, validation, set(catalog))
    if not validation:
        raise ValueError("A nonempty validation split is required")
    data = root / "data"
    data.mkdir(parents=True)
    write_jsonl(data / "train.jsonl", train, ensure_ascii=True)
    write_jsonl(data / "validation.jsonl", validation, ensure_ascii=True)
    write_jsonl(data / "corpus.jsonl", ({"target": target} for target in catalog), ensure_ascii=True)
    report = {"train": fingerprint(args.train_input_file), "test": fingerprint(args.test_input_file),
              "validation": fingerprint(args.eval_input_file) if args.eval_input_file else None,
              "corpus": fingerprint(args.corpus_file) if args.corpus_file else None,
              "seed": args.seed, "validation_split": args.validation_split,
              "train_queries": len(train), "validation_queries": len(validation), "corpus_targets": len(catalog),
              "experiments": list(getattr(args, "experiments", EXPERIMENTS)),
              "positive_label_histograms": {name: dict(sorted(Counter(len(row["positive_targets"]) for row in rows).items()))
                                            for name, rows in (("train", train), ("validation", validation), ("test", test_rows))},
              "inventory_rule": "explicit corpus, or training inventory plus validation/test record document URLs; no test-label-derived IDs"}
    (root / "suite_data.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def transitions(baseline_path, candidate_path):
    result = {"hit1_lost_queries": 0, "hit1_recovered_queries": 0,
              "coverage_improved_queries": 0, "coverage_worsened_queries": 0}
    for baseline, candidate in zip_longest(iter_json_records(baseline_path), iter_json_records(candidate_path)):
        if baseline is None or candidate is None or any(baseline[key] != candidate[key] for key in
                                                       ("query_index", "query", "positive_url_based_id")):
            raise ValueError("Baseline and candidate predictions are not aligned")
        result["hit1_lost_queries"] += int(baseline["hit1"] > candidate["hit1"])
        result["hit1_recovered_queries"] += int(baseline["hit1"] < candidate["hit1"])
        result["coverage_improved_queries"] += int(baseline["relevant_count10"] < candidate["relevant_count10"])
        result["coverage_worsened_queries"] += int(baseline["relevant_count10"] > candidate["relevant_count10"])
    return result


def summarize(args):
    root = Path(args.output_dir)
    baseline = json.loads((root / "sft_baseline/metrics.json").read_text(encoding="utf-8"))
    report = {"baseline": baseline, "experiments": {}, "selection_uses_test_metrics": False}
    rows = [{"experiment": "sft_baseline", "model": "baseline", **{k: baseline[k] for k in METRIC_NAMES}}]
    for name in args.experiments:
        path = root / name
        selection = json.loads((path / "selection.json").read_text(encoding="utf-8"))
        candidate = json.loads((path / "candidate_inference/metrics.json").read_text(encoding="utf-8"))
        if any(candidate[key] != baseline[key] for key in ("num_queries", "corpus_targets", "num_beams", "score_normalization", "catalog_mode")):
            raise ValueError("Inference settings differ between baseline and candidate")
        selected = candidate if selection["selected"] == "candidate" else baseline
        report["experiments"][name] = {
            "validation_selection": selection, "candidate_test": candidate, "selected_test": selected,
            "candidate_delta": {k: candidate[k] - baseline[k] for k in METRIC_NAMES},
            "selected_delta": {k: selected[k] - baseline[k] for k in METRIC_NAMES},
            "query_transitions": transitions(root / "sft_baseline/predictions.jsonl", path / "candidate_inference/predictions.jsonl")}
        rows.extend([{"experiment": name, "model": role, **{k: metrics[k] for k in METRIC_NAMES}}
                     for role, metrics in (("trained_candidate", candidate), ("selected_" + selection["selected"], selected))])
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "model", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['experiment']:24s} {row['model']:20s} Hit1={row['hit1']:.4f} "
              f"Recall10={row['recall10']:.4f} nDCG10={row['ndcg10']:.4f} positives@10={row['relevant_count10']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparation = subparsers.add_parser("prepare")
    preparation.add_argument("--train_input_file", required=True)
    preparation.add_argument("--test_input_file", required=True)
    preparation.add_argument("--eval_input_file")
    preparation.add_argument("--corpus_file")
    preparation.add_argument("--validation_split", type=float, default=.01)
    preparation.add_argument("--seed", type=int, default=42)
    preparation.add_argument("--output_dir", required=True)
    preparation.add_argument("--experiments", nargs="+", choices=AVAILABLE_EXPERIMENTS, default=list(EXPERIMENTS))
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--output_dir", required=True)
    summary.add_argument("--experiments", nargs="+", choices=AVAILABLE_EXPERIMENTS, default=list(EXPERIMENTS))
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--source_dir", help="Existing completed sft_baseline directory; omit to record newly generated results.")
    for name in ("destination_dir", "checkpoint_path", "test_input_file", "corpus_file"):
        baseline.add_argument("--" + name, required=True)
    baseline.add_argument("--max_prompt_length", type=int, default=256)
    baseline.add_argument("--max_target_length", type=int, default=128)
    baseline.add_argument("--num_beams", type=int, default=10)
    baseline.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    baseline.add_argument("--invalid_target_policy", choices=("skip", "error"), default="skip")
    args = parser.parse_args()
    {"prepare": prepare, "summarize": summarize, "baseline": baseline_artifacts}[args.command](args)


if __name__ == "__main__":
    main()
