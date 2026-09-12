#!/usr/bin/env python3
"""Prepare model-only Vault DPO metadata, without retrieval dependencies.

Multilabel inputs supply document IDs and per-query positive lists directly.
Optional corpus files add identities only. No transitive relevance expansion
is performed; the legacy input path retains its exact-query label aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import as_text_id_list, clean_prompt, iter_json_records, normalize_query
from prepare_bm25 import load_structure_id_map
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash


def multilabel_rows(corpus_files: list[str], query_file: str, doc_id_type: str) -> tuple[dict, list]:
    """Read type A and positive_A directly, without mapping or expanding labels."""
    documents, seen = {}, {}
    positive_field = f"positive_{doc_id_type}"

    def doc_id(value, field):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{field} must be a nonempty document ID string without surrounding whitespace")
        return value

    def add_document(target):
        # text_id is the pipeline's internal key, containing the selected ID verbatim.
        document = {"text_id": target, "doc_id_type": doc_id_type}
        if doc_id_type == "url_based_id":
            document["url_based_ids"] = [target]
        documents.setdefault(target, document)

    # Optional additional collection: read identities only, never its query labels.
    for source in corpus_files:
        for raw in iter_json_records(source):
            add_document(doc_id(raw.get(doc_id_type), f"corpus {doc_id_type}"))

    for line, raw in enumerate(iter_json_records(query_file), 1):
        prompt_value = raw.get("prompt", raw.get("text", ""))
        if not isinstance(prompt_value, str):
            raise ValueError(f"Training query must be a string at record {line}")
        prompt = clean_prompt(prompt_value)
        if not prompt:
            raise ValueError(f"Empty training query at record {line}")
        source = doc_id(raw.get(doc_id_type), f"{doc_id_type} at record {line}")
        labels = raw.get(positive_field)
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"{positive_field} must be a nonempty list at record {line}")
        positives = sorted({doc_id(value, f"{positive_field} at record {line}") for value in labels})
        for target in [source, *positives]:
            add_document(target)
        # The source is provenance, not an implicit label for a pseudo-query.
        chosen = source if source in positives else positives[0]
        family = str(raw.get("family_id", raw.get("augmentation_family", "")))
        identity = [normalize_query(prompt), source, chosen, positives]
        key = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        if key in seen:
            if family:
                seen[key]["source_family_ids"].append(family)
            continue
        seen[key] = {"query_key": key, "prompt": prompt, "target_text_id": chosen,
                     "positive_text_ids": positives, "source_doc_id": source,
                     "family_id": family, "source_family_ids": [family] if family else [],
                     "numeric_id": str(raw.get("numeric_id", ""))}
    if not seen:
        raise ValueError("Empty multilabel query file")
    return documents, list(seen.values())


def prepare(corpus_files: list[str], query_file: str, output_dir: str,
            validation_fraction: float = 0.05, seed: int = 42,
            structure_sources: list[str] | None = None,
            id_mode: str = "auto", input_format: str = "legacy",
            doc_id_type: str = "url_based_id") -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if input_format not in {"legacy", "multilabel"}:
        raise ValueError("Invalid input_format")
    if input_format == "multilabel":
        if not isinstance(doc_id_type, str) or not doc_id_type.strip() or doc_id_type != doc_id_type.strip():
            raise ValueError("doc_id_type must name a document ID column")
        if id_mode != "auto" or structure_sources:
            raise ValueError("multilabel reads IDs directly; omit id_mode and structure_id_sources")
        documents, rows = multilabel_rows(corpus_files, query_file, doc_id_type)
    else:
        documents, rows = legacy_rows(corpus_files, query_file, structure_sources, id_mode)

    # Connected components are used ONLY for split membership, not relevance.
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for index, row in enumerate(rows):
        normalized = normalize_query(row["prompt"])
        keys = [("prompt", normalized)] + [("document", p) for p in row["positive_text_ids"]]
        if "source_doc_id" in row:
            keys.append(("document", row["source_doc_id"]))
        keys.extend(("family", family) for family in row.pop("source_family_ids"))
        for key in keys:
            if key in owners:
                parent[root(index)] = root(owners[key])
            else:
                owners[key] = index
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    components = sorted(groups.values(), key=lambda group: min(r["query_key"] for r in group))
    if len(components) < 2:
        raise ValueError("Need at least two independent query families for train/validation")
    random.Random(seed).shuffle(components)
    validation, train = [], []
    wanted = max(1, round(len(rows) * validation_fraction))
    for index, group in enumerate(components):
        destination = validation if len(validation) < wanted and index < len(components) - 1 else train
        family_id = min(r["query_key"] for r in group)
        for row in group:
            row["family_id"] = family_id
        destination.extend(group)
    output = Path(output_dir)
    artifacts = {
        "documents": output / "document_metadata.jsonl",
        "train": output / "train_queries.jsonl",
        "validation": output / "validation_queries.jsonl",
    }
    atomic_jsonl(artifacts["documents"], sorted(documents.values(), key=lambda r: r["text_id"]))
    atomic_jsonl(artifacts["train"], sorted(train, key=lambda r: r["query_key"]))
    atomic_jsonl(artifacts["validation"], sorted(validation, key=lambda r: r["query_key"]))
    manifest = {
        "seed": seed, "validation_fraction": validation_fraction,
        "input_format": input_format, "candidate_documents": len(documents),
        "train_queries": len(train), "validation_queries": len(validation),
        "relevance_policy": (f"per-row positive_{doc_id_type} only; no label expansion"
                             if input_format == "multilabel" else
                             "explicit labels plus exact normalized-query positives; no transitive expansion"),
        "inputs": {str(Path(p).resolve()): file_hash(p) for p in [*corpus_files, query_file, *(structure_sources or [])]},
        "artifacts": {str(p.resolve()): file_hash(p) for p in artifacts.values()},
    }
    atomic_json(output / "split_manifest.json", manifest)
    return manifest


def legacy_rows(corpus_files, query_file, structure_sources, id_mode):
    documents: dict[str, dict] = {}
    numeric_map: dict[str, str] = {}
    structures, _ = load_structure_id_map(
        [Path(p) for p in structure_sources or []], "structure_id_v3", "url_based_id"
    )
    for source in corpus_files:
        for row in iter_json_records(source):
            targets = as_text_id_list(row.get("text_id"))
            if len(targets) != 1:
                raise ValueError("Corpus rows must have exactly one canonical text_id")
            target = targets[0]
            doc = documents.setdefault(target, {"text_id": target, "url_based_ids": [],
                                                 "structure_id_v3s": []})
            numeric = str(row.get("numeric_id", ""))
            if numeric:
                if numeric in numeric_map and numeric_map[numeric] != target:
                    raise ValueError(f"Ambiguous numeric_id: {numeric}")
                numeric_map[numeric] = target
            urls = as_text_id_list(row.get("url_based_ids", row.get("url_based_id", row.get("url_id"))))
            doc["url_based_ids"] = sorted(set(doc["url_based_ids"]) | set(urls))
            values = as_text_id_list(row.get("structure_id_v3s", row.get("structure_id_v3")))
            values.extend(structures[url] for url in urls if url in structures)
            doc["structure_id_v3s"] = sorted(set(doc["structure_id_v3s"]) | set(values))
    if not documents:
        raise ValueError("Empty document corpus")

    rows = []
    prompt_positives: dict[str, set[str]] = defaultdict(set)
    seen = {}
    for raw in iter_json_records(query_file):
        prompt = clean_prompt(raw.get("prompt", raw.get("text", "")))
        if not prompt:
            raise ValueError("Empty training query")
        explicit = as_text_id_list(raw.get("positive_text_ids"))
        targets = as_text_id_list(raw.get("target_text_id", raw.get("text_id")))
        numeric = str(raw.get("numeric_id", ""))
        if raw.get("target_text_id") is not None:
            chosen = str(raw["target_text_id"])
        elif numeric in numeric_map:
            chosen = numeric_map[numeric]
            canonical_targets = set(targets) & documents.keys()
            if canonical_targets and chosen not in canonical_targets:
                raise ValueError("numeric_id and canonical text_id disagree")
        elif len(targets) == 1 and id_mode != "text_id" and targets[0] in numeric_map:
            if id_mode == "auto" and targets[0] in documents and numeric_map[targets[0]] != targets[0]:
                raise ValueError("Ambiguous query ID; specify --id-mode numeric or text_id")
            chosen = numeric_map[targets[0]]
            targets = []  # Raw q10 IDs are numeric IDs, not relevance labels.
        elif id_mode != "numeric" and targets:
            chosen = targets[0]
        else:
            raise ValueError(f"Cannot resolve target for query: {prompt!r}")
        positives = set(explicit) | {chosen}
        if len(targets) > 1:
            positives.update(targets)
        if not positives <= documents.keys():
            raise ValueError(f"Query references unknown documents: {positives - documents.keys()}")
        normalized = normalize_query(prompt)
        prompt_positives[normalized].update(positives)
        # All augmentations for the same source document stay together. Explicit
        # family IDs join additional rows, but never replace this protection.
        family = str(raw.get("family_id", raw.get("augmentation_family", "")))
        key = hashlib.sha256(f"{normalized}\0{chosen}".encode()).hexdigest()
        if key in seen:
            if family:
                seen[key]["source_family_ids"].append(family)
            continue
        prepared = {"query_key": key, "prompt": prompt, "target_text_id": chosen,
                    "positive_text_ids": sorted(positives), "family_id": family,
                    "source_family_ids": [family] if family else [], "numeric_id": numeric}
        seen[key] = prepared
        rows.append(prepared)

    for row in rows:
        normalized = normalize_query(row["prompt"])
        row["positive_text_ids"] = sorted(prompt_positives[normalized])
    return documents, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", action="append", default=[],
                        help="Required for legacy inputs; optional extra candidates for multilabel")
    parser.add_argument("--input-format", choices=["legacy", "multilabel"], default="legacy")
    parser.add_argument("--doc-id-type", default="url_based_id", help="Multilabel ID column A, with labels in positive_A")
    parser.add_argument("--query-file", required=True, help="Training queries/augmentations with verified target IDs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structure-id-source", action="append", default=[])
    parser.add_argument("--id-mode", choices=["auto", "numeric", "text_id"], default="auto")
    args = parser.parse_args()
    print(prepare(args.corpus_file, args.query_file, args.output_dir,
                  args.validation_fraction, args.seed, args.structure_id_source, args.id_mode,
                  args.input_format, args.doc_id_type))


if __name__ == "__main__":
    main()
