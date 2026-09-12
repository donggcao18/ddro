"""Dependency-free artifact and sampling helpers for iterative Vault DPO."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path: str | Path) -> str:
    """Hash portable model/tokenizer files, excluding optimizer and run state."""
    path = Path(path)
    names = {"config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "spiece.model", "sentencepiece.bpe.model", "vocab.json", "merges.txt"}
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.name in names or p.name.startswith("model") and (
            p.name.endswith(".safetensors") or p.name.endswith(".index.json"))
        or p.name.startswith("pytorch_model") and (
            p.name.endswith(".bin") or p.name.endswith(".index.json"))))
    if not (path / "config.json").is_file() or not any(
        p.suffix in {".bin", ".safetensors"} for p in files
    ):
        raise ValueError(f"Not a portable model checkpoint: {path}")
    if not any((path / name).is_file() for name in (
        "tokenizer.json", "spiece.model", "sentencepiece.bpe.model"
    )):
        raise ValueError(f"Missing checkpoint tokenizer: {path}")
    return hashlib.sha256(json.dumps(
        [(p.name, file_hash(p)) for p in files], separators=(",", ":")
    ).encode()).hexdigest()


def _query_order(rows: list[dict], count: int | None, seed: int) -> tuple[list[dict], int]:
    if not rows:
        raise ValueError("No training queries")
    if count is not None and count <= 0:
        raise ValueError("queries_per_round must be positive")
    order = sorted(rows, key=lambda row: row["query_key"])
    random.Random(seed).shuffle(order)
    size = min(count or len(order), len(order))
    return order, size


def partition_queries(rows: list[dict], count: int | None, seed: int) -> list[list[dict]]:
    """Shuffle once and cover all queries once, keeping a smaller final partition."""
    order, size = _query_order(rows, count, seed)
    return [order[start:start + size] for start in range(0, len(order), size)]


def round_queries(rows: list[dict], count: int | None, round_id: int, seed: int) -> list[dict]:
    """Legacy fixed-round schedule: take rotating windows, wrapping when needed."""
    order, size = _query_order(rows, count, seed)
    start = (round_id * size) % len(order)
    return [order[(start + i) % len(order)] for i in range(size)]


def assert_file_hashes(hashes: dict[str, str]) -> None:
    for path, expected in hashes.items():
        if not Path(path).is_file() or file_hash(path) != expected:
            raise ValueError(f"Immutable run input/artifact changed: {path}")


def verify_round_inputs(manifest: dict) -> None:
    assert_file_hashes(manifest["inputs"])
    if checkpoint_hash(manifest["policy_checkpoint"]) != manifest["policy_fingerprint"]:
        raise ValueError("Round starting checkpoint changed")
    if checkpoint_hash(manifest["reference_checkpoint"]) != manifest["reference_fingerprint"]:
        raise ValueError("Round reference checkpoint changed")


def latest_resumable_checkpoint(directory: Path, identity: str,
                                world_size: int = 1, fp16: bool = False) -> Path | None:
    candidates = []
    for path in directory.glob("checkpoint-*"):
        marker = path / "round_checkpoint.json"
        if not marker.is_file():
            continue
        info = read_json(marker)
        if info.get("identity") != identity:
            raise ValueError(f"Checkpoint belongs to a different round: {path}")
        if info.get("world_size", 1) != world_size:
            raise ValueError(f"Checkpoint process count changed: {path}")
        required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
        required.extend(["rng_state.pth"] if world_size == 1 else
                        [f"rng_state_{rank}.pth" for rank in range(world_size)])
        if fp16:
            required.append("scaler.pt")
        if not all((path / name).is_file() for name in required):
            continue
        if read_json(path / "trainer_state.json")["global_step"] != info["global_step"]:
            raise ValueError(f"Checkpoint step mismatch: {path}")
        checkpoint_hash(path)
        candidates.append((int(info["global_step"]), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]
