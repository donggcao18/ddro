#!/usr/bin/env python3
"""Convert already-mined Vault DPO pairs from text_id targets to URL targets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from common import as_text_id, iter_json_records


class UrlMappingError(ValueError):
    """Raised when a text_id does not have exactly one usable URL target."""


def build_url_lookup(
    document_metadata: str | Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load valid URL mappings and retain per-text_id validation errors."""
    lookup: dict[str, str] = {}
    invalid: dict[str, str] = {}

    for row_number, row in enumerate(iter_json_records(document_metadata), start=1):
        text_id = as_text_id(row.get("text_id"))
        if not text_id:
            raise UrlMappingError(
                f"Missing text_id in document metadata row {row_number}"
            )

        raw_urls = row.get("url_based_ids")
        if raw_urls is None:
            raw_urls = row.get("url_based_id", row.get("url_id", ""))
        if not isinstance(raw_urls, list):
            raw_urls = [raw_urls]

        urls: list[str] = []
        for raw_url in raw_urls:
            url = str(raw_url or "").strip()
            if url and url not in urls:
                urls.append(url)

        if not urls:
            invalid[text_id] = f"No url_based_id is available for text_id={text_id!r}"
            lookup.pop(text_id, None)
            continue
        if len(urls) != 1:
            invalid[text_id] = (
                f"text_id={text_id!r} maps to multiple URL IDs: {urls!r}"
            )
            lookup.pop(text_id, None)
            continue
        if text_id in invalid:
            continue

        previous = lookup.get(text_id)
        if previous is not None and previous != urls[0]:
            invalid[text_id] = (
                f"text_id={text_id!r} maps to both {previous!r} and {urls[0]!r}"
            )
            lookup.pop(text_id, None)
            continue
        lookup[text_id] = urls[0]

    return lookup, invalid


def resolve_url(
    text_id: str,
    url_lookup: dict[str, str],
    invalid_mappings: dict[str, str],
    row_number: int,
) -> str:
    """Resolve one referenced target and report its precise mapping error."""
    if text_id in invalid_mappings:
        raise UrlMappingError(
            f"{invalid_mappings[text_id]} (DPO input row {row_number})"
        )
    try:
        return url_lookup[text_id]
    except KeyError as exc:
        raise UrlMappingError(
            f"No document metadata mapping for text_id={text_id!r} "
            f"in DPO input row {row_number}"
        ) from exc


def get_pair_text_id(row: dict[str, Any], role: str, row_number: int) -> str:
    """Read the audit text_id, with the old chosen/rejected field as fallback."""
    value = row.get(f"{role}_text_id", row.get(role))
    text_id = as_text_id(value)
    if not text_id:
        raise UrlMappingError(
            f"Missing {role}_text_id in DPO input row {row_number}"
        )
    return text_id


def convert(args: argparse.Namespace) -> dict[str, Any]:
    """Stream text_id DPO pairs to an atomic URL-target JSONL output."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files")

    url_lookup, invalid_mappings = build_url_lookup(args.document_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    output_rows = 0
    skipped_mapping_errors = 0
    skipped_same_url_pairs = 0
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)

            for row_number, row in enumerate(iter_json_records(input_path), start=1):
                input_rows += 1
                try:
                    chosen_text_id = get_pair_text_id(row, "chosen", row_number)
                    rejected_text_id = get_pair_text_id(row, "rejected", row_number)
                    chosen_url = resolve_url(
                        chosen_text_id, url_lookup, invalid_mappings, row_number
                    )
                    rejected_url = resolve_url(
                        rejected_text_id, url_lookup, invalid_mappings, row_number
                    )
                except UrlMappingError:
                    if args.on_mapping_error == "skip":
                        skipped_mapping_errors += 1
                        continue
                    raise

                # A preference between identical decoder targets is invalid even
                # when those URLs originated from two different quantized IDs.
                if chosen_url == rejected_url:
                    skipped_same_url_pairs += 1
                    continue

                output_row = dict(row)
                output_row["chosen"] = chosen_url
                output_row["rejected"] = rejected_url
                output_row["chosen_text_id"] = chosen_text_id
                output_row["rejected_text_id"] = rejected_text_id
                output_row["chosen_url_based_id"] = chosen_url
                output_row["rejected_url_based_id"] = rejected_url
                output_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                output_rows += 1

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    stats = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "skipped_mapping_errors": skipped_mapping_errors,
        "skipped_same_url_pairs": skipped_same_url_pairs,
        "document_url_mappings": len(url_lookup),
        "invalid_document_url_mappings": len(invalid_mappings),
        "input": str(input_path),
        "output": str(output_path),
    }
    stats_path = (
        Path(args.stats_output)
        if args.stats_output
        else output_path.with_suffix(".stats.json")
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as stats_handle:
        json.dump(stats, stats_handle, indent=2)
        stats_handle.write("\n")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an already-mined Vault DPO JSON/JSONL file from quantized "
            "text_id targets to url_based_id targets without rerunning BM25."
        )
    )
    parser.add_argument("--input", required=True, help="Existing text_id DPO JSONL")
    parser.add_argument(
        "--document-metadata",
        required=True,
        help="document_metadata.jsonl produced by prepare_bm25.py",
    )
    parser.add_argument("--output", required=True, help="URL-target DPO JSONL")
    parser.add_argument("--stats-output", help="Optional conversion statistics path")
    parser.add_argument(
        "--on-mapping-error",
        choices=("error", "skip"),
        default="error",
        help="Fail on a missing/ambiguous mapping (default), or skip that DPO row.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = convert(args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
