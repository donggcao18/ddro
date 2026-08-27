#!/usr/bin/env python3
"""Join structure_id_v3 targets into Vault SFT data through numeric_id."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            character = handle.read(1)
            if not character:
                return
            if not character.isspace():
                first = character
                break
        handle.seek(0)
        if first == "[":
            rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError(f"{source} must contain a JSON list or JSONL records")
            yield from (row for row in rows if isinstance(row, dict))
            return
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            yield row


def string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        raise TypeError(f"Expected a scalar ID, received {type(value).__name__}")
    return str(value).strip()


def id_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    output: list[str] = []
    for value in values:
        item = string_value(value)
        if item and item not in output:
            output.append(item)
    return output


def load_numeric_mapping(path: str, field: str) -> tuple[dict[str, str], dict[str, int]]:
    mapping: dict[str, str] = {}
    stats = defaultdict(int)
    for row_number, row in enumerate(iter_json_records(path), start=1):
        stats["source_rows"] += 1
        numeric_id = string_value(row.get("numeric_id"))
        target = string_value(row.get(field))
        if not numeric_id:
            stats["source_missing_numeric_id"] += 1
            continue
        if not target:
            stats["source_missing_target"] += 1
            continue
        previous = mapping.get(numeric_id)
        if previous is not None and previous != target:
            raise ValueError(
                f"numeric_id {numeric_id!r} maps to both {previous!r} and "
                f"{target!r} at {path}:{row_number}"
            )
        mapping[numeric_id] = target
    stats["source_numeric_ids"] = len(mapping)
    return mapping, dict(stats)


def load_legacy_bridge(
    original_paths: list[str], numeric_to_structure: dict[str, str]
) -> tuple[dict[str, str], dict[str, set[str]]]:
    numeric_to_legacy: dict[str, str] = {}
    legacy_to_structures: dict[str, set[str]] = defaultdict(set)
    for path in original_paths:
        for row_number, row in enumerate(iter_json_records(path), start=1):
            numeric_id = string_value(row.get("numeric_id"))
            legacy_id = string_value(row.get("text_id"))
            if not numeric_id or not legacy_id:
                continue
            previous = numeric_to_legacy.get(numeric_id)
            if previous is not None and previous != legacy_id:
                raise ValueError(
                    f"numeric_id {numeric_id!r} maps to both {previous!r} and "
                    f"{legacy_id!r} at {path}:{row_number}"
                )
            numeric_to_legacy[numeric_id] = legacy_id
            structure_id = numeric_to_structure.get(numeric_id)
            if structure_id:
                legacy_to_structures[legacy_id].add(structure_id)
    return numeric_to_legacy, legacy_to_structures


def merge(args: argparse.Namespace) -> dict[str, Any]:
    numeric_to_structure, stats = load_numeric_mapping(
        args.structure_source, args.structure_id_field
    )
    numeric_to_legacy, legacy_to_structures = load_legacy_bridge(
        args.original, numeric_to_structure
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counters = defaultdict(int, stats)

    with output_path.open("w", encoding="utf-8") as output_handle:
        for row_number, row in enumerate(iter_json_records(args.input), start=1):
            counters["input_rows"] += 1
            numeric_id = string_value(row.get("numeric_id"))
            target_structure = numeric_to_structure.get(numeric_id, "")
            if not target_structure:
                counters["rows_missing_structure_id"] += 1
                if args.on_missing == "error":
                    raise ValueError(
                        f"No {args.structure_id_field} mapping for numeric_id="
                        f"{numeric_id!r} at {args.input}:{row_number}"
                    )
                continue

            positive_legacy_ids = id_list(row.get("text_id"))
            current_legacy_id = numeric_to_legacy.get(numeric_id, "")
            if current_legacy_id and current_legacy_id not in positive_legacy_ids:
                positive_legacy_ids.append(current_legacy_id)

            positive_structures = sorted(
                {
                    structure_id
                    for legacy_id in positive_legacy_ids
                    for structure_id in legacy_to_structures.get(legacy_id, set())
                }
                | {target_structure}
            )
            unmapped_legacy_ids = sorted(
                legacy_id
                for legacy_id in positive_legacy_ids
                if legacy_id not in legacy_to_structures
            )
            if unmapped_legacy_ids:
                counters["rows_with_unmapped_positive_text_ids"] += 1
                counters["unmapped_positive_text_ids"] += len(unmapped_legacy_ids)
                if args.on_missing == "error":
                    raise ValueError(
                        f"No {args.structure_id_field} mapping for positive text IDs "
                        f"{unmapped_legacy_ids!r} at {args.input}:{row_number}"
                    )

            targets = positive_structures if args.expand_multilabel else [target_structure]
            for target in targets:
                output_row = dict(row)
                output_row["legacy_text_ids"] = positive_legacy_ids
                output_row["legacy_target_text_id"] = current_legacy_id
                output_row[args.structure_id_field] = target
                output_row["positive_structure_id_v3s"] = positive_structures
                output_row[args.output_target_field] = target
                output_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                counters["output_rows"] += 1
            if len(positive_structures) > 1:
                counters["multilabel_input_rows"] += 1

    counters.update(
        {
            "legacy_numeric_ids": len(numeric_to_legacy),
            "legacy_text_ids_with_structure": len(legacy_to_structures),
            "output": str(output_path),
        }
    )
    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(counters), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return dict(counters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge structure_id_v3 into Vault training data by numeric_id."
    )
    parser.add_argument("--input", required=True, help="Current ready-to-feed JSON/JSONL")
    parser.add_argument(
        "--structure-source",
        default=(
            "/home/users/congthanh_le/scratch/veil/CodeGR/data/augmented_dsi/"
            "Ruby_merged.jsonl"
        ),
    )
    parser.add_argument(
        "--original",
        action="append",
        required=True,
        help="Original indexed train/test data used to bridge numeric_id to legacy text_id.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--structure-id-field", default="structure_id_v3")
    parser.add_argument(
        "--output-target-field",
        default="text_id",
        help="Scalar field consumed by the SFT loader (default: text_id).",
    )
    parser.add_argument(
        "--expand-multilabel",
        action="store_true",
        help="Write one training row per mapped positive structure target.",
    )
    parser.add_argument("--on-missing", choices=["error", "skip"], default="error")
    return parser


def main() -> None:
    print(json.dumps(merge(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
