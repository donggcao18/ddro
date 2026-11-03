#!/usr/bin/env python3
"""Prepare a fixed-catalog comparison and summarize its held-out results."""

import argparse
import csv
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bm25.the_vault.common import iter_json_records, write_jsonl
from utils.grpo_data import load_corpus, prepare_url_file, split_queries, validate_query_sets
from utils.retrieval_metrics import METRIC_NAMES


EXPERIMENTS = ("multi_positive_dpo", "missed_positive_dpo", "multi_negative_dpo")


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
    _, test_catalog = prepare_url_file(args.test_input_file, include_positive_targets_in_corpus=False)
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
              "experiments": list(EXPERIMENTS),
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
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--output_dir", required=True)
    summary.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args()
    (prepare if args.command == "prepare" else summarize)(args)


if __name__ == "__main__":
    main()
