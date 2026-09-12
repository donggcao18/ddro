#!/usr/bin/env python3
"""Build an immutable FP32 EMA reference on CPU between DPO rounds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

try:
    from .iterative_dpo_utils import atomic_json, checkpoint_hash, read_json
except ImportError:
    from iterative_dpo_utils import atomic_json, checkpoint_hash, read_json


def reference_identity(reference_checkpoint, policy_checkpoint, reference_fingerprint,
                       policy_fingerprint, decay):
    return {"reference_checkpoint": str(Path(reference_checkpoint).resolve()),
            "policy_checkpoint": str(Path(policy_checkpoint).resolve()),
            "reference_fingerprint": reference_fingerprint,
            "policy_fingerprint": policy_fingerprint, "decay": decay}


def ema_parameters(reference, policy, decay):
    """Blend each unique parameter once, including T5's shared embeddings."""
    import torch

    if not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError("EMA decay must be between zero and one")
    pairs = []
    for method in ("named_parameters", "named_buffers"):
        old, new = dict(getattr(reference, method)()), dict(getattr(policy, method)())
        if old.keys() != new.keys() or any(old[k].shape != new[k].shape for k in old):
            raise ValueError("EMA policy/reference parameter or buffer layout differs")
        pairs.extend((old[k], new[k]) for k in old)
    with torch.no_grad():
        for old, new in pairs:
            if old.is_floating_point():
                if old.dtype != torch.float32 or new.dtype != torch.float32:
                    raise ValueError("EMA requires FP32 policy/reference tensors")
                if decay == 0:
                    old.copy_(new)
                elif decay != 1:
                    old.mul_(decay).add_(new, alpha=1 - decay)
            else:
                old.copy_(new)


def update_reference(reference_checkpoint, policy_checkpoint, reference_fingerprint,
                     policy_fingerprint, decay, output):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError("EMA decay must be between zero and one")
    identity = reference_identity(reference_checkpoint, policy_checkpoint,
                                  reference_fingerprint, policy_fingerprint, decay)
    for role in ("reference", "policy"):
        if checkpoint_hash(identity[f"{role}_checkpoint"]) != identity[f"{role}_fingerprint"]:
            raise ValueError(f"EMA {role} checkpoint changed")
    output = Path(output).resolve()
    marker = output / "reference_complete.json"
    if output.exists():
        if not marker.is_file():
            raise ValueError("EMA output exists without a completion marker")
        info = read_json(marker)
        if info["identity"] != identity or checkpoint_hash(output) != info["checkpoint_fingerprint"]:
            raise ValueError("Existing EMA reference or provenance changed")
        return info

    old_tokenizer = AutoTokenizer.from_pretrained(reference_checkpoint, local_files_only=True)
    new_tokenizer = AutoTokenizer.from_pretrained(policy_checkpoint, local_files_only=True)
    if (type(old_tokenizer) is not type(new_tokenizer)
            or old_tokenizer.get_vocab() != new_tokenizer.get_vocab()
            or old_tokenizer.special_tokens_map != new_tokenizer.special_tokens_map):
        raise ValueError("EMA policy/reference tokenizers differ")
    if old_tokenizer.is_fast:
        tokenizer_states = []
        for tokenizer in (old_tokenizer, new_tokenizer):
            state = json.loads(tokenizer.backend_tokenizer.to_str())
            # These reflect the last batching call, not the ID encoding rules.
            state.pop("padding", None)
            state.pop("truncation", None)
            tokenizer_states.append(state)
        if tokenizer_states[0] != tokenizer_states[1]:
            raise ValueError("EMA policy/reference tokenizer encoding rules differ")
    elif hasattr(old_tokenizer, "sp_model"):
        if old_tokenizer.sp_model.serialized_model_proto() != new_tokenizer.sp_model.serialized_model_proto():
            raise ValueError("EMA policy/reference SentencePiece models differ")
    reference = AutoModelForSeq2SeqLM.from_pretrained(
        reference_checkpoint, local_files_only=True, torch_dtype=torch.float32).cpu()
    policy = AutoModelForSeq2SeqLM.from_pretrained(
        policy_checkpoint, local_files_only=True, torch_dtype=torch.float32).cpu()
    for key in ("model_type", "vocab_size", "decoder_start_token_id", "eos_token_id", "pad_token_id",
                "tie_word_embeddings", "tie_encoder_decoder"):
        if getattr(reference.config, key, None) != getattr(policy.config, key, None):
            raise ValueError(f"EMA policy/reference configuration differs: {key}")
    reference.requires_grad_(False).eval()
    policy.requires_grad_(False).eval()
    ema_parameters(reference, policy, decay)
    reference.config.use_cache = True
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the weights and completion marker together. Interrupted writes
    # cannot be loaded as a round reference, and input snapshots stay untouched.
    with tempfile.TemporaryDirectory(prefix=".ema-", dir=output.parent) as temporary:
        snapshot = Path(temporary) / "snapshot"
        reference.save_pretrained(snapshot, safe_serialization=True)
        old_tokenizer.save_pretrained(snapshot)
        info = {"identity": identity, "checkpoint_fingerprint": checkpoint_hash(snapshot)}
        atomic_json(snapshot / "reference_complete.json", info)
        snapshot.replace(output)
    return info


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--reference-fingerprint", required=True)
    parser.add_argument("--policy-fingerprint", required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    print(update_reference(**vars(build_parser().parse_args())))
