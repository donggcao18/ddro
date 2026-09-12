#!/usr/bin/env python3
"""Mine Vault DPO negatives from a frozen policy snapshot's constrained predictions.

This is intentionally independent from the existing BM25 miner.  It loads the
same Hugging Face checkpoint format as ``train_ddro_vault.py``, constrains beam
search to tokenizer encodings of known Vault decoder targets, filters every
known positive, and writes zero to ``--negatives-per-query`` model-confusion
pairs for each query. A later combiner can fill any shortfall with BM25 pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from common import as_text_id, as_text_id_list, document_targets, iter_json_records


SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.trie import Trie  # noqa: E402


def validate_checkpoint(path_value: str) -> Path:
    """Validate the minimum portable Hugging Face checkpoint layout."""
    path = Path(path_value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    weight_candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    tokenizer_candidates = (
        "tokenizer.json",
        "spiece.model",
        "sentencepiece.bpe.model",
    )
    missing: list[str] = []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    if not any((path / name).is_file() for name in weight_candidates):
        missing.append("model weights")
    if not any((path / name).is_file() for name in tokenizer_candidates):
        missing.append("tokenizer files")
    if missing:
        raise FileNotFoundError(
            f"{path} is not a complete Hugging Face checkpoint; missing: "
            + ", ".join(missing)
        )
    return path


def batched(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield input records in fixed-size lists."""
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_target(tokenizer: Any, text_id: str) -> tuple[int, ...]:
    """Tokenize a DocID exactly once without truncation."""
    encoded = tokenizer(
        text_id,
        add_special_tokens=True,
        truncation=False,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return tuple(int(token_id) for token_id in token_ids)


def build_target_index(
    tokenizer: Any,
    targets: Iterable[str],
    max_target_length: int,
    collision_policy: str = "error",
    length_policy: str = "error",
) -> tuple[
    list[list[int]],
    dict[tuple[int, ...], str],
    set[str],
    list[list[str]],
    list[tuple[int, str]],
]:
    """Build unique target sequences and identify unusable decoder targets."""
    if length_policy not in {"error", "skip", "truncate"}:
        raise ValueError("Invalid target length policy")
    if collision_policy not in {"error", "skip", "allow"}:
        raise ValueError("Invalid target collision policy")
    sequence_groups: dict[tuple[int, ...], list[str]] = {}
    overlong_targets: list[tuple[int, str]] = []
    for target in sorted(set(targets)):
        sequence = encode_target(tokenizer, target)
        if not sequence:
            raise ValueError(f"Tokenizer produced no target tokens for target={target!r}")
        if len(sequence) > max_target_length:
            overlong_targets.append((len(sequence), target))
            if length_policy != "truncate":
                continue
            # Match encoder-decoder DPO tokenization, including special tokens.
            sequence = tuple(tokenizer(target, add_special_tokens=True, truncation=True,
                                       max_length=max_target_length)["input_ids"])
            if not sequence or len(sequence) > max_target_length:
                raise ValueError(f"Tokenizer did not truncate target to the requested length: {target!r}")
        sequence_groups.setdefault(sequence, []).append(target)
    overlong_targets.sort(reverse=True)
    if overlong_targets and length_policy == "error":
        maximum_length = overlong_targets[0][0]
        examples = "; ".join(
            f"{length} tokens: {target!r}"
            for length, target in overlong_targets[:3]
        )
        raise ValueError(
            f"{len(overlong_targets)} decoder targets exceed "
            f"--max-target-length={max_target_length}; the corpus maximum is "
            f"{maximum_length} tokens. Targets must not be truncated. Increase "
            f"--max-target-length to at least {maximum_length}. Longest examples: "
            f"{examples}"
        )

    collision_groups = [
        group for group in sequence_groups.values() if len(group) > 1
    ]
    if collision_groups and collision_policy == "error":
        first_group = collision_groups[0]
        raise ValueError(
            "Two Vault targets have the same tokenizer target sequence: "
            f"{first_group[0]!r} and {first_group[1]!r}. They cannot form "
            "distinguishable DPO targets. Use --target-collision-policy skip "
            "to exclude all affected DocIDs from mining and hybrid data."
        )

    collision_targets = {
        target for group in collision_groups for target in group
    }
    mining_excluded_targets = (collision_targets if collision_policy != "allow" else set()) | (
        {target for _, target in overlong_targets} if length_policy != "truncate" else set())
    sequence_to_target = {
        sequence: group[0]
        for sequence, group in sequence_groups.items()
        if len(group) == 1 or collision_policy == "allow"
    }
    encoded_targets = [list(sequence) for sequence in sequence_to_target]
    if not encoded_targets:
        raise ValueError("No valid document targets were found")
    return (
        encoded_targets,
        sequence_to_target,
        mining_excluded_targets,
        collision_groups,
        overlong_targets,
    )


def build_document_targets(
    document_rows: dict[str, dict[str, Any]],
    target_type: str,
) -> tuple[dict[str, str], dict[str, str], int]:
    """Map corpus text IDs to unique decoder targets and back."""
    text_id_to_target: dict[str, str] = {}
    target_to_text_id: dict[str, str] = {}
    invalid_target_mappings = 0

    for text_id, row in document_rows.items():
        targets = document_targets(row, target_type)
        if len(targets) != 1:
            invalid_target_mappings += 1
            continue
        target = targets[0]

        previous_text_id = target_to_text_id.get(target)
        if previous_text_id is not None and previous_text_id != text_id:
            raise ValueError(
                f"Decoder target {target!r} maps to multiple text IDs: "
                f"{previous_text_id!r} and {text_id!r}"
            )
        text_id_to_target[text_id] = target
        target_to_text_id[target] = text_id

    if not text_id_to_target:
        raise ValueError(f"No usable {target_type} decoder targets were found")
    return text_id_to_target, target_to_text_id, invalid_target_mappings


def canonical_generated_tokens(
    sequence: Sequence[int],
    decoder_start_token_id: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[int, ...]:
    """Normalize one generated encoder-decoder sequence for exact target lookup."""
    tokens = [int(token_id) for token_id in sequence]
    if tokens and tokens[0] == decoder_start_token_id:
        tokens = tokens[1:]

    canonical: list[int] = []
    for token_id in tokens:
        if eos_token_id is not None and token_id == eos_token_id:
            canonical.append(token_id)
            break
        if pad_token_id is not None and token_id == pad_token_id:
            break
        canonical.append(token_id)
    return tuple(canonical)


def select_model_candidates(
    sequences: Sequence[Sequence[int]],
    sequence_scores: Sequence[float | None],
    sequence_to_target: dict[tuple[int, ...], str],
    positive_targets: set[str],
    limit: int,
    decoder_start_token_id: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Map beams to unique valid non-positive targets in model rank order."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    counters: Counter[str] = Counter()

    for rank, (sequence, score) in enumerate(
        zip(sequences, sequence_scores), start=1
    ):
        canonical = canonical_generated_tokens(
            sequence,
            decoder_start_token_id,
            pad_token_id,
            eos_token_id,
        )
        target = sequence_to_target.get(canonical)
        if target is None:
            counters["unmapped_generation_sequences"] += 1
            continue
        if rank == 1:
            counters[
                "queries_with_top1_positive"
                if target in positive_targets
                else "queries_with_top1_incorrect"
            ] += 1
        if target in positive_targets:
            counters["positive_beams_filtered"] += 1
            continue
        if target in seen:
            counters["duplicate_beams_filtered"] += 1
            continue
        seen.add(target)
        selected.append(
            {
                "target": target,
                "rank": rank,
                "score": None if score is None else float(score),
            }
        )
        if len(selected) == limit:
            break

    return selected, counters


def retrieval_metrics(targets: Sequence[str | None], positives: set[str],
                      cutoffs: Sequence[int]) -> dict[str, float]:
    """Score the original ranked beams; invalid beams still occupy a rank."""
    if not positives:
        raise ValueError("Retrieval evaluation requires known positives")
    result = {}
    for k in cutoffs:
        prefix = targets[:k]
        result[f"recall@{k}"] = len(set(prefix) & positives) / len(positives)
        result[f"mrr@{k}"] = next(
            (1.0 / rank for rank, target in enumerate(prefix, 1) if target in positives), 0.0
        )
    result["valid_generation_rate"] = sum(t is not None for t in targets) / max(1, len(targets))
    return result


def resolve_device(value: str) -> Any:
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def atomic_output_path(path: Path) -> Path:
    """Return a same-directory temporary path suitable for atomic replacement."""
    return path.with_name(path.name + f".tmp.{os.getpid()}")


def mine(args: argparse.Namespace) -> dict[str, Any]:
    """Run constrained model-confusion mining and return audit statistics."""
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

    set_seed(getattr(args, "seed", 42))

    if args.negatives_per_query <= 0:
        raise ValueError("--negatives-per-query must be greater than zero")
    if args.num_beams <= 0:
        raise ValueError("--num-beams must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.max_prompt_length <= 0 or args.max_target_length <= 0:
        raise ValueError("Token length limits must be greater than zero")
    if args.limit_queries is not None and args.limit_queries <= 0:
        raise ValueError("--limit-queries must be greater than zero")

    checkpoint_path = validate_checkpoint(args.checkpoint_path)
    device = resolve_device(args.device)
    if (args.bf16 or args.fp16) and device.type != "cuda":
        raise ValueError("--bf16 and --fp16 require a CUDA device")

    config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
    if not config.is_encoder_decoder:
        raise ValueError("Model-confusion mining requires an encoder-decoder checkpoint")
    if config.decoder_start_token_id is None:
        raise ValueError("Checkpoint config has no decoder_start_token_id")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Checkpoint tokenizer has no pad_token_id")

    document_rows: dict[str, dict[str, Any]] = {}
    for row in iter_json_records(args.document_metadata):
        text_id = as_text_id(row.get("text_id"))
        if text_id:
            document_rows[text_id] = row
    text_id_to_target, target_to_text_id, invalid_target_mappings = (
        build_document_targets(document_rows, args.target_type)
    )
    if getattr(args, "strict_corpus", False) and invalid_target_mappings:
        raise ValueError(f"{invalid_target_mappings} corpus documents have missing/ambiguous decoder targets")
    (
        encoded_targets,
        sequence_to_target,
        mining_excluded_targets,
        collision_groups,
        overlong_targets,
    ) = build_target_index(
        tokenizer,
        text_id_to_target.values(),
        args.max_target_length,
        args.target_collision_policy,
        args.target_length_policy,
    )
    if getattr(args, "strict_corpus", False):
        for sequence in encoded_targets:
            if tokenizer.eos_token_id is not None and sequence[-1] != tokenizer.eos_token_id:
                raise ValueError("Every corpus target must end in EOS")
            if tokenizer.pad_token_id in sequence or (
                tokenizer.unk_token_id is not None and tokenizer.unk_token_id in sequence
            ):
                raise ValueError("Corpus target contains padding or unknown tokens; correct the tokenizer/target mapping")
    collision_targets = ({target for group in collision_groups for target in group}
                         if args.target_collision_policy != "allow" else set())
    target_aliases = ({group[0]: group for group in collision_groups}
                      if args.target_collision_policy == "allow" else {})
    target_representatives = {target: representative for representative, group in target_aliases.items()
                              for target in group}
    overlength_targets = ({target for _, target in overlong_targets}
                          if args.target_length_policy != "truncate" else set())
    collision_excluded_text_ids = {
        target_to_text_id[target]
        for target in collision_targets
        if target in target_to_text_id
    }
    overlength_excluded_text_ids = {
        target_to_text_id[target]
        for target in overlength_targets
        if target in target_to_text_id
    }
    mining_excluded_text_ids = {
        target_to_text_id[target]
        for target in mining_excluded_targets
        if target in target_to_text_id
    }
    docid_trie = Trie(
        [
            [int(config.decoder_start_token_id)] + target_tokens
            for target_tokens in encoded_targets
        ]
    )

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, **model_kwargs)
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        raise ValueError(
            "Checkpoint/tokenizer vocabulary mismatch: model has "
            f"{model.get_input_embeddings().num_embeddings} embeddings but tokenizer "
            f"has {len(tokenizer)} tokens. The miner will not resize the SFT checkpoint."
        )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    def prefix_allowed_tokens_fn(batch_id: int, sent_ids: Any) -> list[int]:
        allowed = docid_trie.get(sent_ids.tolist())
        return allowed if allowed else [tokenizer.pad_token_id]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path = output_path.with_suffix(".excluded_text_ids.json")
    with exclusions_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "target_type": args.target_type,
                "collision_policy": args.target_collision_policy,
                "length_policy": args.target_length_policy,
                "max_target_length": args.max_target_length,
                "truncated_targets": [target for _, target in overlong_targets]
                    if args.target_length_policy == "truncate" else [],
                "collision_groups": [
                    {
                        "targets": group,
                        "text_ids": [target_to_text_id[target] for target in group],
                    }
                    for group in collision_groups
                ],
                "overlength_targets": [
                    {
                        "target": target,
                        "text_id": target_to_text_id[target],
                        "token_length": length,
                    }
                    for length, target in overlong_targets
                ],
                # Legacy hybrid paths may retain overlength chosen targets.
                # Strict iterative mining also skips queries whose chosen
                # target is overlength; no truncated ID enters training.
                "strict_training_excluded_text_ids": sorted(mining_excluded_text_ids),
                "excluded_targets": sorted(collision_targets),
                "excluded_text_ids": sorted(collision_excluded_text_ids),
                "model_mining_excluded_targets": sorted(
                    mining_excluded_targets
                ),
                "model_mining_excluded_text_ids": sorted(
                    mining_excluded_text_ids
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    temporary_output = atomic_output_path(output_path)
    stats: Counter[str] = Counter()
    candidate_count_distribution: Counter[int] = Counter()
    evaluation_only = getattr(args, "evaluation_only", False)
    metric_totals: Counter[str] = Counter()
    cutoffs = sorted({1, min(5, args.num_beams), min(10, args.num_beams), args.num_beams})

    query_iterator: Iterable[dict[str, Any]] = iter_json_records(args.query_metadata)
    if args.limit_queries is not None:
        import itertools

        query_iterator = itertools.islice(query_iterator, args.limit_queries)

    try:
        with temporary_output.open("w", encoding="utf-8") as output_handle:
            for query_batch in tqdm(
                batched(query_iterator, args.batch_size),
                desc="Mining model-confusion negatives",
            ):
                eligible_queries: list[dict[str, Any]] = []
                for query in query_batch:
                    stats["queries_seen"] += 1
                    query_target_id = as_text_id(query.get("target_text_id"))
                    query_positive_ids = set(as_text_id_list(query.get("positive_text_ids"))) | {query_target_id}
                    if query_positive_ids & overlength_excluded_text_ids:
                        stats["queries_with_overlength_positive"] += 1
                    if query_target_id in collision_excluded_text_ids:
                        stats["queries_skipped_target_collision"] += 1
                        continue
                    if query_target_id in overlength_excluded_text_ids:
                        if getattr(args, "strict_corpus", False) and not evaluation_only:
                            stats["queries_skipped_overlength_target"] += 1
                            continue
                        stats["queries_with_overlength_target_processed"] += 1
                    eligible_queries.append(query)
                query_batch = eligible_queries
                if not query_batch:
                    continue

                prompts = [str(row.get("prompt", "")).strip() for row in query_batch]
                if any(not prompt for prompt in prompts):
                    raise ValueError("query_metadata contains an empty prompt")
                tokenized = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_prompt_length,
                    return_token_type_ids=False,
                    return_tensors="pt",
                )
                tokenized = {
                    key: value.to(device)
                    for key, value in tokenized.items()
                    if isinstance(value, torch.Tensor)
                }
                with torch.inference_mode():
                    generated = model.generate(
                        **tokenized,
                        max_length=args.max_target_length + 1,
                        num_beams=args.num_beams,
                        num_return_sequences=args.num_beams,
                        do_sample=False,
                        early_stopping=True,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                        return_dict_in_generate=True,
                        output_scores=True,
                        length_penalty=getattr(args, "length_penalty", 1.0),
                    )

                all_sequences = generated.sequences.detach().cpu().tolist()
                generated_scores = getattr(generated, "sequences_scores", None)
                if generated_scores is None:
                    all_scores: list[float | None] = [None] * len(all_sequences)
                else:
                    all_scores = generated_scores.detach().float().cpu().tolist()

                for batch_index, query in enumerate(query_batch):
                    stats["queries_processed"] += 1
                    start = batch_index * args.num_beams
                    end = start + args.num_beams
                    positive_ids = {
                        as_text_id(value)
                        for value in query.get("positive_text_ids", [])
                        if as_text_id(value)
                    }
                    target_text_id = as_text_id(query.get("target_text_id"))
                    if not target_text_id:
                        raise ValueError(
                            f"Missing target_text_id for query {query.get('query_key')!r}"
                        )
                    positive_ids.add(target_text_id)
                    positive_targets = {
                        text_id_to_target[positive_id]
                        for positive_id in positive_ids
                        if positive_id in text_id_to_target
                        and text_id_to_target[positive_id]
                        not in mining_excluded_targets
                    }
                    if evaluation_only:
                        missing = positive_ids - text_id_to_target.keys()
                        if missing:
                            raise ValueError("Evaluation positives must all map to corpus targets")
                        positive_targets = {text_id_to_target[p] for p in positive_ids}
                        if positive_targets & (mining_excluded_targets - overlength_targets):
                            raise ValueError("Evaluation positives contain ambiguous targets")
                        # Keep long positives in the ground truth. Since they
                        # cannot be generated, they count as misses, rather
                        # than silently shrinking recall's denominator.
                        ranked_targets = [sequence_to_target.get(canonical_generated_tokens(
                            sequence, int(config.decoder_start_token_id),
                            tokenizer.pad_token_id, tokenizer.eos_token_id,
                        )) for sequence in all_sequences[start:end]]
                        metric_totals.update(retrieval_metrics(ranked_targets, positive_targets, cutoffs))
                        stats["ambiguous_generation_sequences"] += sum(t in target_aliases for t in ranked_targets)
                        output_handle.write(json.dumps({
                            "query_key": query["query_key"],
                            "positive_text_ids": sorted(positive_ids),
                            "predicted_text_ids": [target_to_text_id.get(t) for t in ranked_targets],
                            "predicted_text_id_groups": [
                                [target_to_text_id[alias] for alias in target_aliases.get(t, [t])]
                                if t is not None else [] for t in ranked_targets],
                            "round_id": getattr(args, "round_id", None),
                        }) + "\n")
                        continue
                    if target_text_id not in text_id_to_target:
                        raise ValueError(
                            f"Query {query.get('query_key')!r} targets text_id="
                            f"{target_text_id!r}, which has no usable "
                            f"{args.target_type} decoder target"
                        )
                    # Any sequence shared with a positive is ineligible as a
                    # negative, even if its representative has another full ID.
                    positive_targets = {target_representatives.get(t, t) for t in positive_targets}
                    candidates, selection_stats = select_model_candidates(
                        all_sequences[start:end],
                        all_scores[start:end],
                        sequence_to_target,
                        positive_targets,
                        args.negatives_per_query,
                        int(config.decoder_start_token_id),
                        tokenizer.pad_token_id,
                        tokenizer.eos_token_id,
                    )
                    stats.update(selection_stats)
                    candidate_count_distribution[len(candidates)] += 1
                    if not candidates:
                        stats["queries_with_no_model_negatives"] += 1
                    if len(candidates) == args.negatives_per_query:
                        stats["queries_with_full_model_quota"] += 1
                    else:
                        stats["queries_with_model_shortfall"] += 1

                    chosen_ids = (
                        sorted(positive_ids)
                        if args.pair_all_positives
                        else [target_text_id]
                    )
                    for chosen_id in chosen_ids:
                        chosen_target = text_id_to_target.get(chosen_id)
                        if chosen_id in collision_excluded_text_ids:
                            stats["chosen_targets_skipped_collision"] += 1
                            continue
                        if getattr(args, "strict_corpus", False) and chosen_id in overlength_excluded_text_ids:
                            stats["chosen_targets_skipped_overlength"] += 1
                            continue
                        if chosen_target is None:
                            raise ValueError(
                                f"Chosen text_id={chosen_id!r} has no usable "
                                f"{args.target_type} decoder target"
                            )
                        for candidate in candidates:
                            rejected_target = candidate["target"]
                            rejected_id = target_to_text_id[rejected_target]
                            rejected_metadata = document_rows[rejected_id]
                            output_row = {
                                "prompt": str(query["prompt"]),
                                "chosen": chosen_target,
                                "rejected": rejected_target,
                                "query_key": str(query["query_key"]),
                                "numeric_id": query.get("numeric_id", ""),
                                "chosen_text_id": chosen_id,
                                "rejected_text_id": rejected_id,
                                "positive_text_ids": sorted(positive_ids),
                                "negative_source": "model_confusion",
                                "target_type": args.target_type,
                                "model_rank": candidate["rank"],
                                "model_score": candidate["score"],
                                "rejected_url_based_ids": rejected_metadata.get(
                                    "url_based_ids", []
                                ),
                            }
                            if getattr(args, "round_id", None) is not None:
                                output_row.update({
                                    "round_id": args.round_id,
                                    "policy_fingerprint": getattr(args, "checkpoint_fingerprint", None),
                                    "family_id": query.get("family_id", ""),
                                })
                            if args.target_type == "structure_id_v3":
                                output_row["chosen_structure_id_v3"] = chosen_target
                                output_row["rejected_structure_id_v3"] = rejected_target
                                output_row["positive_structure_id_v3s"] = sorted(
                                    positive_targets
                                )
                            output_handle.write(
                                json.dumps(output_row, ensure_ascii=False) + "\n"
                            )
                            stats["model_pairs_written"] += 1
        temporary_output.replace(output_path)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    result: dict[str, Any] = dict(stats)
    result["candidate_count_distribution"] = {
        str(count): queries
        for count, queries in sorted(candidate_count_distribution.items())
    }
    result.update(
        {
            "documents": len(document_rows),
            "usable_decoder_targets": len(sequence_to_target),
            "invalid_document_target_mappings": invalid_target_mappings,
            "invalid_document_url_mappings": (
                invalid_target_mappings if args.target_type == "url" else 0
            ),
            "tokenized_docid_sequences": len(sequence_to_target),
            "target_type": args.target_type,
            "target_collision_policy": args.target_collision_policy,
            "target_length_policy": args.target_length_policy,
            "max_target_length": args.max_target_length,
            "tokenizer_collision_groups": len(collision_groups),
            "allowed_collision_groups": len(target_aliases),
            "excluded_collision_targets": len(collision_targets),
            "excluded_collision_text_ids": len(collision_excluded_text_ids),
            "excluded_overlength_targets": len(overlength_targets),
            "truncated_targets": len(overlong_targets) if args.target_length_policy == "truncate" else 0,
            "excluded_overlength_text_ids": len(overlength_excluded_text_ids),
            "model_mining_excluded_targets": len(mining_excluded_targets),
            "model_mining_excluded_text_ids": len(mining_excluded_text_ids),
            "exclusions_output": str(exclusions_path),
            "collision_exclusions_output": str(exclusions_path),
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "num_beams": args.num_beams,
            "negatives_per_query": args.negatives_per_query,
            "output": str(output_path),
            "seed": getattr(args, "seed", 42),
            "round_id": getattr(args, "round_id", None),
            "policy_fingerprint": getattr(args, "checkpoint_fingerprint", None),
            "length_penalty": getattr(args, "length_penalty", 1.0),
        }
    )
    if evaluation_only:
        if not stats["queries_processed"]:
            raise ValueError("No validation queries were evaluated")
        result["metrics"] = {key: value / stats["queries_processed"] for key, value in metric_totals.items()}
    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine Vault DPO negatives from constrained SFT model predictions."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-fingerprint")
    parser.add_argument("--round-id", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--evaluation-only", action="store_true",
                        help="Write ranked predictions and retrieval metrics instead of preference pairs")
    parser.add_argument("--strict-corpus", action="store_true",
                        help="Reject missing/ambiguous mappings, padding, and unknown target tokens")
    parser.add_argument("--query-metadata", required=True)
    parser.add_argument("--document-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-type",
        default="text_id",
        help="Legacy namespace (text_id, url, structure_id_v3), or exact ID column from prepared multilabel metadata.",
    )
    parser.add_argument(
        "--target-collision-policy",
        choices=["error", "skip", "allow"],
        default="error",
        help=(
            "Fail on tokenizer-identical targets (default), or exclude every "
            "affected target, or allow shared token sequences with a deterministic "
            "representative and exclude every positive's aliases from negatives."
        ),
    )
    parser.add_argument(
        "--target-length-policy",
        choices=["error", "skip", "truncate"],
        default="error",
        help=(
            "How to handle decoder targets longer than --max-target-length. "
            "Use skip to remove them from the candidate trie, or truncate to "
            "use tokenizer-truncated sequences while preserving original ID labels."
        ),
    )
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--pair-all-positives", action="store_true")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    return parser


def main() -> None:
    stats = mine(build_parser().parse_args())
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
