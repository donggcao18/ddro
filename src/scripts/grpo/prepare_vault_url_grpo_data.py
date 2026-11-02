#!/usr/bin/env python3
"""Prepare URL GRPO from one native Vault JSON/JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bm25.the_vault.common import write_jsonl
from utils.grpo_data import prepare_url_file, split_queries, validate_query_sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-split", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, targets = prepare_url_file(args.input_file)
    train, validation = split_queries(rows, args.validation_split, args.seed)
    validate_query_sets(train, validation, set(targets))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # Preserve raw identifiers losslessly, even malformed Unicode. The model's
    # target-index validation can then report/exclude them using the tokenizer.
    write_jsonl(destination / "train.jsonl", train, ensure_ascii=True)
    write_jsonl(destination / "validation.jsonl", validation, ensure_ascii=True)
    write_jsonl(destination / "corpus.jsonl", ({"target": target} for target in targets), ensure_ascii=True)
    stats = {
        "input_file": args.input_file, "target_type": "url", "seed": args.seed,
        "query_field": "text", "positive_field": "positive_url_based_id",
        "train_queries": len(train), "validation_queries": len(validation),
        "corpus_targets": len(targets),
        "multi_positive_queries": sum(len(row["positive_targets"]) > 1 for row in rows),
    }
    (destination / "prepare_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
