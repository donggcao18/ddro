"""Shared helpers for the Vault BM25 data pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


QUERY_PREFIX_RE = re.compile(r"^\s*Query\s*:\s*", re.IGNORECASE)
CODE_PREFIX_RE = re.compile(r"^\s*Code\s*:\s*", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read either JSONL or a JSON file containing a top-level list."""
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
            for row in rows:
                if isinstance(row, dict):
                    yield row
            return

        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {source}:{line_number}")
            yield row


def normalize_query(text: Any) -> str:
    """Normalize original query text exactly as build_multilable.py does."""
    without_prefix = QUERY_PREFIX_RE.sub("", str(text))
    return SPACE_RE.sub(" ", without_prefix.strip().lower())


def clean_prompt(text: Any) -> str:
    """Remove an optional Query: prefix while preserving prompt casing."""
    return SPACE_RE.sub(" ", QUERY_PREFIX_RE.sub("", str(text)).strip())


def clean_code(text: Any) -> str:
    """Remove an optional Code: prefix from a code document."""
    return CODE_PREFIX_RE.sub("", str(text)).strip()


def as_text_id(value: Any) -> str:
    """Convert a scalar Vault identifier to its stable string form."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        raise TypeError(f"Expected a scalar identifier, received {type(value).__name__}")
    return str(value)


def as_text_id_list(value: Any) -> list[str]:
    """Convert a scalar or multi-label identifier value to unique strings."""
    raw_values: Iterable[Any]
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]

    output: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        text_id = as_text_id(raw_value)
        if text_id and text_id not in seen:
            seen.add(text_id)
            output.append(text_id)
    return output


def unique_strings(value: Any) -> list[str]:
    """Convert a scalar/list metadata value to unique non-empty strings."""
    raw_values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        item = str(raw_value or "").strip()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def document_targets(row: dict[str, Any], target_type: str) -> list[str]:
    """Read decoder targets for one canonical text_id document."""
    if target_type == "text_id":
        return unique_strings(row.get("text_id"))
    if target_type == "url":
        return unique_strings(
            row.get("url_based_ids", row.get("url_based_id", row.get("url_id", "")))
        )
    if target_type == "structure_id_v3":
        return unique_strings(
            row.get("structure_id_v3s", row.get("structure_id_v3", ""))
        )
    raise ValueError(f"Unsupported target type: {target_type!r}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write JSONL records and return the number written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
