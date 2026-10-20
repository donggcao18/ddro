"""Prune only checkpoint payload files; retain round data and audit metadata."""

from pathlib import Path
import re
import stat

from pretrain.iterative_dpo_utils import atomic_json


PAYLOAD = re.compile(
    r"(?:model(?:-\d+-of-\d+)?\.safetensors|pytorch_model(?:-\d+-of-\d+)?\.bin|"
    r"adapter_model\.(?:bin|safetensors)|optimizer\.(?:pt|bin)|scheduler\.pt|scaler\.pt|"
    r"rng_state(?:_\d+)?\.pth|training_args\.bin)\Z")


def checkpoint_pruned(state, checkpoint):
    """A pending deletion is also unavailable for historical replay."""
    target = str(Path(checkpoint).resolve())
    return any(str(Path(item["checkpoint"]).resolve()) == target
               for item in state.get("checkpoint_pruning", []))


def resume_after_pruning(state, requested_start):
    if requested_start == 0 and state.get("checkpoint_pruning"):
        return len(state["rounds"]), True
    return requested_start, False


def _safe_path(path, output, round_id):
    """Check lexical AND resolved containment, rejecting links/junctions."""
    output = Path(output).resolve()
    round_root = output / f"round-{round_id:03d}"
    path = Path(path).absolute()
    try:
        relative = path.relative_to(round_root)
    except ValueError:
        raise ValueError(f"Checkpoint pruning path is outside its round: {path}") from None
    if not relative.parts or relative.parts[0] not in {"training", "reference"}:
        raise ValueError(f"Checkpoint pruning path is not under training/reference: {path}")
    current = round_root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if current.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError(f"Checkpoint pruning refuses a linked path: {current}")
    if not path.resolve().is_relative_to(round_root):
        raise ValueError(f"Checkpoint pruning path resolves outside its round: {path}")
    return path


def _checkpoint_roots(output, round_id):
    directory = Path(output).resolve() / f"round-{round_id:03d}"
    training = directory / "training"
    roots = [training, training / "latest", training / "final", directory / "reference"]
    _safe_path(training, output, round_id)
    if training.is_dir():
        roots.extend(child for child in training.iterdir() if re.fullmatch(r"checkpoint-\d+", child.name))
    return roots


def prune_round_checkpoints(state, output, keep_last, protected=()):
    """Persist deletion intent first, so interruption can safely resume cleanup.

    Keep the newest checkpoint-bearing completed rounds. Skipped rounds do not
    evict a still-needed policy. No directory is recursively deleted.
    """
    if not keep_last:
        return
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last_round_checkpoints must be a positive integer or zero to disable")
    output = Path(output).resolve()
    manifest = output / "run_manifest.json"
    protected = {Path(p).resolve() for p in protected if p}
    protected.update(Path(state[key]).resolve() for key in ("latest", "latest_reference") if state.get(key))
    rounds = [row for row in state["rounds"] if row["status"] == "trained"]
    candidates = rounds[:-keep_last]
    ledger = state.setdefault("checkpoint_pruning", [])
    recorded = {item["round_id"] for item in ledger}
    for row in candidates:
        round_id = row["round_id"]
        if round_id in recorded:
            continue
        expected = output / f"round-{round_id:03d}" / "training/latest"
        if Path(row["checkpoint"]).absolute() != expected:
            raise ValueError(f"Unexpected round checkpoint path for pruning: {row['checkpoint']}")
        round_root = expected.parent.parent
        if any(p.is_relative_to(round_root) or round_root.is_relative_to(p) for p in protected):
            continue
        files = []
        for root in _checkpoint_roots(output, round_id):
            _safe_path(root, output, round_id)
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if PAYLOAD.fullmatch(path.name):
                    _safe_path(path, output, round_id)
                    if path.is_file():
                        files.append({"path": str(path), "bytes": path.stat().st_size})
        if files:
            ledger.append({"round_id": round_id, "checkpoint": str(expected),
                           "status": "pending", "files": files, "bytes": sum(p["bytes"] for p in files)})
    if not ledger:
        state.pop("checkpoint_pruning", None)
        return
    # Complete a pending plan even if deletion was interrupted after some files
    # disappeared. Every target is validated before any unlink in that plan.
    for item in ledger:
        if item["status"] == "complete":
            continue
        round_id = item["round_id"]
        if round_id in {r["round_id"] for r in rounds[-keep_last:]}:
            raise ValueError("Cannot retain a checkpoint whose pruning was already started")
        paths = [_safe_path(p["path"], output, round_id) for p in item["files"]]
        for path in paths:
            if not PAYLOAD.fullmatch(path.name) or not any(
                    path.parent == root for root in _checkpoint_roots(output, round_id)):
                raise ValueError(f"Unexpected checkpoint payload in pruning plan: {path}")
            if any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
                raise ValueError(f"Checkpoint pruning would remove an active/input checkpoint: {path}")
        if state.get("best") and checkpoint_pruned(state, state["best"]["checkpoint"]):
            state["best"]["checkpoint_available"] = False
        atomic_json(manifest, state)
        for path in paths:
            _safe_path(path, output, round_id)
            path.unlink(missing_ok=True)
        item["status"] = "complete"
        atomic_json(manifest, state)
        print(f"Pruned checkpoint payloads from round {round_id}; retained its data and metadata "
              f"({item['bytes'] / 1024**3:.2f} GiB; keeping {keep_last} latest checkpoint rounds).", flush=True)
    if state.get("best") and checkpoint_pruned(state, state["best"]["checkpoint"]):
        state["best"]["checkpoint_available"] = False
        atomic_json(manifest, state)
    if state.get("best"):
        atomic_json(output / "best_checkpoint.json", state["best"])
