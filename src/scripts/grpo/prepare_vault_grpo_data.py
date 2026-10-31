#!/usr/bin/env python3
"""Convert existing Vault metadata into query-level GRPO data and a full corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bm25.the_vault.common import write_jsonl
from utils.grpo_data import TARGET_TYPES, prepare_metadata, split_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-metadata", required=True)
    parser.add_argument("--document-metadata", required=True)
    parser.add_argument("--target-type", choices=TARGET_TYPES, default="structure_id_v6")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-split", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, targets = prepare_metadata(args.query_metadata, args.document_metadata, args.target_type)
    train, validation = split_queries(rows, args.validation_split, args.seed)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "train.jsonl", train)
    # Write an empty file when disabled, avoiding stale validation data on reruns.
    write_jsonl(destination / "validation.jsonl", validation)
    write_jsonl(destination / "corpus.jsonl", ({"target": target} for target in targets))
    stats = {
        "target_type": args.target_type, "seed": args.seed,
        "train_queries": len(train), "validation_queries": len(validation),
        "corpus_targets": len(targets),
        "multi_positive_queries": sum(len(row["positive_targets"]) > 1 for row in rows),
    }
    (destination / "prepare_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
