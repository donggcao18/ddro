"""Prepare final-only retrieval queries without changing training data or corpus."""

import hashlib
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import clean_prompt, iter_json_records
from mine_model_confusion_negatives import build_document_targets
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash


def prepare_inference_queries(infer_file, document_metadata, output, target_type, input_format):
    sources = {str(p): file_hash(p) for p in (infer_file, document_metadata)}
    documents = {row["text_id"]: row for row in iter_json_records(document_metadata)}
    _, target_to_id, invalid = build_document_targets(documents, target_type)
    if invalid:
        raise ValueError("Inference corpus has missing or ambiguous document targets")
    canonical_ids = {key: key for key in documents} if input_format != "multilabel" else None

    def checked_id(value, field):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"Inference {field} must be a nonempty ID string without surrounding whitespace")
        return value

    rows, fallback_rows = [], 0
    for index, raw in enumerate(iter_json_records(infer_file), 1):
        prompt = raw.get("prompt", raw.get("text", raw.get("query", "")))
        if not isinstance(prompt, str) or not clean_prompt(prompt):
            raise ValueError(f"Empty or invalid inference query at record {index}")
        prompt = clean_prompt(prompt)
        if input_format == "multilabel":
            positive_field = f"positive_{target_type}"
            explicit = positive_field in raw
            labels = raw[positive_field] if explicit else [raw.get(target_type)]
            mapping = target_to_id
        else:
            positive_field = "positive_text_ids"
            explicit = positive_field in raw
            labels = raw[positive_field] if explicit else [raw.get("target_text_id", raw.get("text_id"))]
            mapping = canonical_ids
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Inference {positive_field} must be a nonempty list at record {index}")
        targets = sorted({checked_id(value, positive_field) for value in labels})
        missing = set(targets) - mapping.keys()
        if missing:
            raise ValueError(f"Inference positives are absent from the retrieval corpus at record {index}: "
                             f"{sorted(missing)[:3]}. Supply the full collection with --corpus-file; "
                             "INFER_FILE does not expand the training corpus.")
        positives = sorted({mapping[target] for target in targets})
        # Preserve every evaluation record, including repeated queries. Its
        # index makes shard keys unique without merging relevance labels.
        key = hashlib.sha256(json.dumps([index, prompt, positives], ensure_ascii=False).encode()).hexdigest()
        rows.append({"query_key": key, "prompt": prompt, "target_text_id": positives[0],
                     "positive_text_ids": positives, "infer_record": index})
        fallback_rows += not explicit
    if not rows:
        raise ValueError("INFER_FILE contains no evaluation queries")
    if any(file_hash(path) != expected for path, expected in sources.items()):
        raise ValueError("Inference inputs changed while preparing queries")
    report = {"queries": len(rows), "single_label_fallback_rows": fallback_rows,
              "target_type": target_type, "input_format": input_format, "sources": sources}
    atomic_jsonl(output, rows)
    atomic_json(Path(output).with_suffix(".audit.json"), report)
    return report
