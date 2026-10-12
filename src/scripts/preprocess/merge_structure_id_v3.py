#!/usr/bin/env python3
"""Join structure_id_v3 targets into Vault SFT data by stable document key."""

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


def join_value(row: dict[str, Any], join_key: str) -> str:
    if join_key == "url_based_id":
        return string_value(row.get("url_based_id", row.get("url_id", "")))
    return string_value(row.get(join_key))


def load_structure_mapping(
    path: str, field: str, join_key: str
) -> tuple[dict[str, str], dict[str, int | str]]:
    mapping: dict[str, str] = {}
    stats = defaultdict(int)
    for row_number, row in enumerate(iter_json_records(path), start=1):
        stats["source_rows"] += 1
        key = join_value(row, join_key)
        target = string_value(row.get(field))
        if not key:
            stats["source_missing_join_key"] += 1
            continue
        if not target:
            stats["source_missing_target"] += 1
            continue
        previous = mapping.get(key)
        if previous is not None and previous != target:
            raise ValueError(
                f"{join_key} {key!r} maps to both {previous!r} and "
                f"{target!r} at {path}:{row_number}"
            )
        mapping[key] = target
    stats["join_key"] = join_key
    stats["source_join_keys"] = len(mapping)
    return mapping, dict(stats)


def load_legacy_bridge(
    original_paths: list[str], key_to_structure: dict[str, str], join_key: str
) -> tuple[dict[str, str], dict[str, set[str]]]:
    key_to_legacy: dict[str, str] = {}
    legacy_to_structures: dict[str, set[str]] = defaultdict(set)
    for path in original_paths:
        for row_number, row in enumerate(iter_json_records(path), start=1):
            key = join_value(row, join_key)
            legacy_id = string_value(row.get("text_id"))
            if not key or not legacy_id:
                continue
            previous = key_to_legacy.get(key)
            if previous is not None and previous != legacy_id:
                raise ValueError(
                    f"{join_key} {key!r} maps to both {previous!r} and "
                    f"{legacy_id!r} at {path}:{row_number}"
                )
            key_to_legacy[key] = legacy_id
            structure_id = key_to_structure.get(key)
            if structure_id:
                legacy_to_structures[legacy_id].add(structure_id)
    return key_to_legacy, legacy_to_structures


def merge(args: argparse.Namespace) -> dict[str, Any]:
    join_key = getattr(args, "join_key", "url_based_id")
    key_to_structure, stats = load_structure_mapping(
        args.structure_source, args.structure_id_field, join_key
    )
    key_to_legacy, legacy_to_structures = load_legacy_bridge(
        args.original, key_to_structure, join_key
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_log_path = output_path.with_suffix(".missing.jsonl")
    counters = defaultdict(int, stats)
    unique_unmapped_positive_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output_handle, missing_log_path.open(
        "w", encoding="utf-8"
    ) as missing_handle:
        for row_number, row in enumerate(iter_json_records(args.input), start=1):
            counters["input_rows"] += 1
            numeric_id = string_value(row.get("numeric_id"))
            key = join_value(row, join_key)
            target_structure = key_to_structure.get(key, "")
            if not target_structure:
                counters["rows_missing_structure_id"] += 1
                if args.on_missing == "error":
                    raise ValueError(
                        f"No {args.structure_id_field} mapping for {join_key}="
                        f"{key!r} at {args.input}:{row_number}"
                    )
                missing_handle.write(
                    json.dumps(
                        {
                            "row_number": row_number,
                            "numeric_id": numeric_id,
                            join_key: key,
                            "reason": "missing_target_structure_id_v3",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            positive_legacy_ids = id_list(row.get("text_id"))
            current_legacy_id = key_to_legacy.get(key, "")
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
                unique_unmapped_positive_ids.update(unmapped_legacy_ids)
                if args.on_missing == "error":
                    raise ValueError(
                        f"No {args.structure_id_field} mapping for positive text IDs "
                        f"{unmapped_legacy_ids!r} at {args.input}:{row_number}"
                    )
                missing_handle.write(
                    json.dumps(
                        {
                            "row_number": row_number,
                            "numeric_id": numeric_id,
                            join_key: key,
                            "reason": "unmapped_positive_text_ids",
                            "text_ids": unmapped_legacy_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
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
            "legacy_join_keys": len(key_to_legacy),
            "legacy_text_ids_with_structure": len(legacy_to_structures),
            "unique_unmapped_positive_text_ids": len(unique_unmapped_positive_ids),
            "missing_log": str(missing_log_path),
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
        description="Merge structure_id_v3 into Vault training data by stable document key."
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
        help="Original indexed train/test data used to bridge the join key to legacy text_id.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--structure-id-field", default="structure_id_v3")
    parser.add_argument(
        "--join-key",
        choices=["url_based_id", "numeric_id"],
        default="url_based_id",
        help="Cross-file identity key (default: url_based_id).",
    )
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
