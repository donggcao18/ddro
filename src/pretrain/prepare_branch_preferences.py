#!/usr/bin/env python3
"""Annotate committed model-mined sequence pairs with safe local branch preferences."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import iter_json_records
from mine_model_confusion_negatives import build_document_targets, build_target_index
from pretrain.branch_preference_utils import (
    assign_branch_weights, describe_branch, tokenizer_fingerprint,
)
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash
from utils.trie import Trie


def prepare_branch_preferences(pairs_file, query_metadata, document_metadata, checkpoint_path,
                               output, target_type, max_target_length,
                               target_token_policy="error", audit_output=None):
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
    if not config.is_encoder_decoder or config.decoder_start_token_id is None:
        raise ValueError("Branch preferences require an encoder-decoder checkpoint")
    fingerprint = tokenizer_fingerprint(tokenizer, config)
    sources = {str(p): file_hash(p) for p in (pairs_file, query_metadata, document_metadata)}
    documents = {row["text_id"]: row for row in iter_json_records(document_metadata)}
    targets, _, invalid = build_document_targets(documents, target_type)
    if invalid:
        raise ValueError("Corpus has missing or ambiguous document targets")
    sequences, _, _, _, _ = build_target_index(tokenizer, targets.values(), max_target_length,
                                               collision_policy="error", length_policy="error")
    for sequence in sequences:
        if (tokenizer.eos_token_id is None or sequence[-1] != tokenizer.eos_token_id
                or sequence.count(tokenizer.eos_token_id) != 1):
            raise ValueError("Every branch target must contain exactly one terminal EOS")
        if target_token_policy == "error" and (
            tokenizer.pad_token_id in sequence or tokenizer.unk_token_id in sequence
        ):
            raise ValueError("Branch corpus contains padding/unknown target tokens")
    trie = Trie([[config.decoder_start_token_id] + sequence for sequence in sequences])

    @lru_cache(maxsize=None)
    def encode(target):
        return tuple(tokenizer(target, add_special_tokens=True, truncation=False)["input_ids"])

    queries = {row["query_key"]: row for row in iter_json_records(query_metadata)}
    rows = []
    for raw in iter_json_records(pairs_file):
        query = queries.get(raw.get("query_key"))
        if query is None or raw.get("prompt") != query["prompt"]:
            raise ValueError("Preference belongs to an unselected query or changed prompt")
        positives = query["positive_text_ids"]
        if (set(raw.get("positive_text_ids", [])) != set(positives)
                or raw.get("chosen_text_id") != query["target_text_id"]
                or raw.get("rejected_text_id") in positives):
            raise ValueError("Preference has inconsistent positive/negative labels")
        if (targets.get(raw["chosen_text_id"]) != raw["chosen"]
                or targets.get(raw["rejected_text_id"]) != raw["rejected"]):
            raise ValueError("Preference target differs from corpus mapping")
        positive_targets = sorted({targets[p] for p in positives})
        branch = describe_branch(raw["query_key"], encode(raw["chosen"]), encode(raw["rejected"]),
                                 [encode(p) for p in positive_targets],
                                 lambda prefix: trie.get([config.decoder_start_token_id] + prefix))
        rows.append({**raw, **branch, "branch_positive_targets": positive_targets,
                     "branch_tokenizer_fingerprint": fingerprint})
    audit = assign_branch_weights(rows)
    if any(file_hash(path) != expected for path, expected in sources.items()):
        raise ValueError("Branch preparation inputs changed while reading")
    audit.update(schema_version=1, sources=sources, tokenizer_fingerprint=fingerprint,
                 checkpoint=str(checkpoint_path), target_type=target_type,
                 max_target_length=max_target_length, output=str(output))
    atomic_jsonl(output, rows)
    atomic_json(audit_output or Path(output).with_suffix(".audit.json"), audit)
    print(f"Branch preferences: {audit['eligible_rows']}/{len(rows)} rows, "
          f"{audit['distinct_decisions']} distinct decisions, "
          f"{audit['eligible_queries']}/{audit['query_count']} queries", flush=True)
    return audit


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs-file", "query-metadata", "document-metadata", "checkpoint-path", "output", "target-type"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-target-length", type=int, required=True)
    parser.add_argument("--target-token-policy", choices=["error", "allow"], default="error")
    parser.add_argument("--audit-output")
    return parser


if __name__ == "__main__":
    prepare_branch_preferences(**vars(build_parser().parse_args()))
