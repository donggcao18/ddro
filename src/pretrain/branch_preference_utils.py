"""Token-level branch annotations; no model or torch dependency."""

from collections import Counter, defaultdict
import hashlib
import json
import math

SCHEMA_VERSION = 1
BRANCH_COLUMNS = (
    "query_key", "branch_schema_version", "branch_active", "branch_skip_reason",
    "branch_position", "branch_prefix_tokens", "branch_preferred_token",
    "branch_rejected_token", "branch_positive_children", "branch_positive_targets",
    "branch_key", "branch_loss_weight", "branch_tokenizer_fingerprint",
)


def encode_branch_target(tokenizer, target, limit, length_policy="error", token_policy="error"):
    """Use the same effective target tokens as constrained mining and TRL."""
    options = {"add_special_tokens": True, "truncation": length_policy == "truncate"}
    if length_policy == "truncate":
        options["max_length"] = limit
    tokens = tuple(tokenizer(target, **options)["input_ids"])
    if not tokens or len(tokens) > limit:
        raise ValueError(f"Branch target exceeds the configured length policy: {target!r}")
    if tokenizer.eos_token_id is not None and tokens[-1] != tokenizer.eos_token_id:
        raise ValueError(f"Branch target does not end in EOS: {target!r}")
    if token_policy != "allow" and (tokenizer.pad_token_id in tokens or (
        tokenizer.unk_token_id is not None and tokenizer.unk_token_id in tokens
    )):
        raise ValueError(f"Branch target contains padding/unknown tokens: {target!r}")
    return tokens


def tokenizer_fingerprint(tokenizer, config):
    identity = {"vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map,
                "decoder_start": config.decoder_start_token_id,
                "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id}
    # Include normalization/pre-tokenization, not just vocabulary assignments.
    if getattr(tokenizer, "is_fast", False):
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        # These change during ordinary calls but do not change ID segmentation.
        backend.pop("padding", None)
        backend.pop("truncation", None)
        identity["backend"] = backend
    else:
        identity["init"] = {k: v for k, v in tokenizer.init_kwargs.items()
                            if k not in {"name_or_path", "tokenizer_file"}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def first_divergence(chosen, rejected):
    for index, (positive, negative) in enumerate(zip(chosen, rejected)):
        if positive != negative:
            return index
    return None


def decision_key(query_key, prefix, positive, negative):
    return hashlib.sha256(json.dumps([query_key, list(prefix), positive, negative],
                                     separators=(",", ":")).encode()).hexdigest()


def describe_branch(query_key, chosen, rejected, positives, legal_children=None):
    """Compare only the first divergent actions under one identical history."""
    result = {"branch_schema_version": SCHEMA_VERSION, "branch_active": False,
              "branch_skip_reason": "identical_tokens", "branch_position": -1,
              "branch_prefix_tokens": [], "branch_preferred_token": -1,
              "branch_rejected_token": -1, "branch_positive_children": [],
              "branch_key": "", "branch_loss_weight": 0.0}
    if tuple(chosen) not in {tuple(p) for p in positives}:
        raise ValueError("Chosen target is not among the query's known positives")
    position = first_divergence(chosen, rejected)
    if position is None:
        if list(chosen) != list(rejected):
            raise ValueError("Complete targets must include EOS; one target ends inside another")
        return result
    prefix = list(chosen[:position])
    positive, negative = int(chosen[position]), int(rejected[position])
    children = sorted({int(p[position]) for p in positives
                       if len(p) > position and list(p[:position]) == prefix})
    if legal_children is not None:
        allowed = set(legal_children(prefix))
        if positive not in allowed or negative not in allowed or not set(children) <= allowed:
            raise ValueError("Branch action is absent from the corpus trie")
    result.update(branch_position=position, branch_prefix_tokens=prefix,
                  branch_preferred_token=positive, branch_rejected_token=negative,
                  branch_positive_children=children)
    if negative in children:
        result["branch_skip_reason"] = "negative_branch_contains_positive"
    else:
        result.update(branch_active=True, branch_skip_reason="",
                      branch_key=decision_key(query_key, prefix, positive, negative))
    return result


def assign_branch_weights(rows):
    """Uniform row sampling estimates a mean over queries and distinct decisions."""
    queries = {row["query_key"] for row in rows}
    multiplicities = Counter(row["branch_key"] for row in rows if row["branch_active"])
    decisions = defaultdict(set)
    for row in rows:
        if row["branch_active"]:
            decisions[row["query_key"]].add(row["branch_key"])
    for row in rows:
        row["branch_loss_weight"] = (
            len(rows) / (len(queries) * len(decisions[row["query_key"]])
                         * multiplicities[row["branch_key"]]) if row["branch_active"] else 0.0
        )
    return {"sequence_rows": len(rows), "query_count": len(queries),
            "eligible_rows": sum(multiplicities.values()), "eligible_queries": len(decisions),
            "distinct_decisions": len(multiplicities),
            "skip_reasons": dict(Counter(row["branch_skip_reason"] for row in rows
                                         if not row["branch_active"])),
            "divergence_positions": dict(Counter(str(row["branch_position"]) for row in rows
                                                if row["branch_active"])),
            "max_branch_weight": max((row["branch_loss_weight"] for row in rows), default=0.0)}


def validate_branch_rows(rows, encode, fingerprint):
    """Reject stale annotations and weights before any training/tokenization."""
    checked = []
    for row in rows:
        missing = set(BRANCH_COLUMNS) - row.keys()
        if missing:
            raise ValueError(f"Missing branch columns: {sorted(missing)}")
        if (row["branch_schema_version"] != SCHEMA_VERSION
                or type(row["branch_schema_version"]) is not int
                or type(row["branch_active"]) is not bool
                or not isinstance(row["query_key"], str) or not row["query_key"]):
            raise ValueError("Invalid branch schema/query key")
        if row["branch_tokenizer_fingerprint"] != fingerprint:
            raise ValueError("Branch annotations use a different tokenizer/configuration")
        targets = row["branch_positive_targets"]
        if (not isinstance(targets, list) or not targets
                or any(not isinstance(p, str) or not p for p in targets)):
            raise ValueError("branch_positive_targets must be a nonempty list of IDs")
        chosen, rejected = encode(row["chosen"]), encode(row["rejected"])
        expected = describe_branch(row["query_key"], chosen, rejected, [encode(p) for p in targets])
        for key, value in expected.items():
            if key != "branch_loss_weight" and row[key] != value:
                raise ValueError(f"Branch metadata disagrees with tokenized targets: {key}")
        for key in ("branch_position", "branch_preferred_token", "branch_rejected_token"):
            if type(row[key]) is not int:
                raise ValueError(f"Invalid integer branch field: {key}")
        weight = row["branch_loss_weight"]
        if type(weight) not in {float, int} or not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid branch loss weight")
        checked.append(dict(row))
    assign_branch_weights(checked)
    for original, expected in zip(rows, checked):
        if not math.isclose(original["branch_loss_weight"], expected["branch_loss_weight"],
                            rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Branch weights disagree with query/decision multiplicities")
