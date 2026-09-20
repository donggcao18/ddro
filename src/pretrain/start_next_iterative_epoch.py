#!/usr/bin/env python3
"""Start another full query pass from a completed EMA run's policy and reference."""

import argparse
import os
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from pretrain.iterative_dpo_utils import assert_file_hashes, atomic_json, read_json
from pretrain.train_iterative_ddro_vault import load_config, run_pipeline, verify_start_checkpoint


def next_epoch_config(previous_output: Path, output: Path) -> dict:
    previous_output, output = previous_output.resolve(), output.resolve()
    if output == previous_output or previous_output in output.parents or output in previous_output.parents:
        raise ValueError("The next epoch must use a separate output directory, not a parent or child of the old run")
    state = read_json(previous_output / "run_manifest.json")
    rounds = state.get("rounds", [])
    if (not state.get("complete") or not rounds
            or [row["round_id"] for row in rounds] != list(range(state.get("planned_rounds", -1)))):
        raise ValueError("The previous run must have completed all planned rounds")
    cfg = dict(state["identity"]["config"])
    if cfg["reference_update"] != "ema" or cfg["rounds"] is not None or cfg["epochs_per_round"] != 1:
        raise ValueError("Expected a completed partition_all EMA run with epochs_per_round=1")
    assert_file_hashes(state["identity"]["inputs"])
    # Preserve the split algorithm as well as the seed and source data.
    split_code = (SRC / "data/data_prep/prepare_vault_dpo_metadata.py",
                  SRC / "scripts/bm25/the_vault/common.py")
    assert_file_hashes({str(p): state["identity"]["code"][str(p)] for p in split_code})
    assert_file_hashes(state["stages"]["prepare"])
    last = rounds[-1]
    policy, reference = last["checkpoint"], last["next_reference_checkpoint"]
    if state["latest"] != policy or state["latest_reference"] != reference:
        raise ValueError("Latest policy/reference pointers disagree with the final round")
    verify_start_checkpoint(policy, last["checkpoint_fingerprint"], "policy", 0)
    if Path(reference).resolve() == Path(cfg["checkpoint_path"]).resolve():
        reference_hash = state["identity"]["initial_checkpoint"]
    elif (cfg.get("initial_reference_checkpoint") and Path(reference).resolve()
          == Path(cfg["initial_reference_checkpoint"]).resolve()):
        reference_hash = state["identity"]["initial_reference_checkpoint"]
    else:
        marker = Path(reference) / "reference_complete.json"
        recorded = {p: h for name, artifacts in state["stages"].items()
                    if name.startswith("reference-") for p, h in artifacts.items()
                    if Path(p).resolve() == marker.resolve()}
        if not recorded:
            raise ValueError("Final EMA reference has no committed completion marker")
        assert_file_hashes(recorded)
        reference_hash = read_json(marker)["checkpoint_fingerprint"]
    verify_start_checkpoint(reference, reference_hash, "reference", 0)
    cfg.update(checkpoint_path=policy, initial_reference_checkpoint=reference, output_dir=str(output))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume the new epoch, not the source run")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch this script once; the controller starts the GPU workers")
    previous = args.previous_output.expanduser().resolve()
    output = (args.output_dir.expanduser().resolve() if args.output_dir
              else previous.with_name(previous.name + "-epoch2"))
    cfg = next_epoch_config(previous, output)
    # Keep the config beside the output: new pipeline outputs must start empty.
    path = output.with_name(output.name + ".config.json")
    if path.exists():
        if read_json(path) != cfg:
            raise ValueError(f"Existing next-epoch configuration differs: {path}")
    else:
        atomic_json(path, cfg)
    cfg = load_config(path)
    print(f"Starting next dataset pass. Policy: {cfg['checkpoint_path']}; "
          f"EMA reference: {cfg['initial_reference_checkpoint']}; output: {output}", flush=True)
    run_pipeline(cfg, resume=args.resume, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
