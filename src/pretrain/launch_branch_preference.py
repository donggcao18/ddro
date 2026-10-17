#!/usr/bin/env python3
"""Create a reproducible continuation config and run the existing round controller."""

import argparse
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from pretrain.iterative_dpo_utils import atomic_json, read_json
from pretrain.train_iterative_ddro_vault import load_config, run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Initial policy/reference checkpoint, e.g. your completed normal-DPO model")
    parser.add_argument("--query-file", required=True, help="Multilabel query JSON/JSONL")
    parser.add_argument("--output-dir", required=True, help="Separate, initially empty directory for this arm")
    parser.add_argument("--target-type", default="url_based_id", help="ID column; requires positive_<column> labels")
    parser.add_argument("--corpus-file", action="append", default=[], help="Optional additional full-corpus metadata")
    parser.add_argument("--arm", choices=["dpo", "zero", "branch"], default="branch")
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--branch-beta", type=float)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--mining-num-gpus", type=int)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rounds", type=lambda value: None if value == "all" else int(value), default=None,
                        help="Total rounds or 'all' (default: cover every training query once without wrapping)")
    parser.add_argument("--eval-every-rounds", type=int, default=8,
                        help="Evaluate retrieval every N completed rounds")
    parser.add_argument("--eval-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--queries-per-round", type=int, default=10000)
    parser.add_argument("--steps-per-round", type=int, default=1000)
    parser.add_argument("--num-beams", type=int, default=15)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--mining-batch-size", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16, help="Training pairs per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--target-length-policy", choices=["error", "skip", "truncate"], default="truncate")
    parser.add_argument("--target-collision-policy", choices=["error", "skip", "allow"], default="allow")
    parser.add_argument("--target-token-policy", choices=["error", "allow"], default="allow")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def experiment_config(args):
    training = {"per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation,
                "learning_rate": args.learning_rate, "beta": args.beta,
                "warmup_ratio": args.warmup_ratio, "save_steps": args.save_steps,
                "logging_steps": args.logging_steps, "dataset_num_proc": 1,
                "dataloader_num_workers": 0}
    if args.arm != "dpo":
        training.update(preference_objective="branch-dpo",
                        branch_loss_weight=0.0 if args.arm == "zero" else args.branch_loss_weight,
                        branch_beta=args.branch_beta if args.branch_beta is not None else args.beta)
    return {"checkpoint_path": str(Path(args.checkpoint).resolve()),
            "initial_reference_checkpoint": str(Path(args.checkpoint).resolve()),
            "query_file": str(Path(args.query_file).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
            "corpus_files": [str(Path(p).resolve()) for p in args.corpus_file],
            "input_format": "multilabel", "target_type": args.target_type,
            "rounds": args.rounds, "queries_per_round": args.queries_per_round,
            "steps_per_round": args.steps_per_round, "reference_update": "ema",
            "reference_ema_decay": 1.0, "seed": args.seed,
            "validation_fraction": args.validation_fraction, "num_gpus": args.num_gpus,
            "mining_num_gpus": args.mining_num_gpus or args.num_gpus,
            "precision": args.precision, "device": args.device,
            "num_beams": args.num_beams, "negatives_per_query": args.negatives_per_query,
            "mining_batch_size": args.mining_batch_size,
            "max_prompt_length": args.max_prompt_length, "max_target_length": args.max_target_length,
            "length_penalty": args.length_penalty, "target_length_policy": args.target_length_policy,
            "target_collision_policy": args.target_collision_policy, "target_token_policy": args.target_token_policy,
            "selection_metric": "mrr@10", "eval_every_rounds": args.eval_every_rounds,
            "eval_at_end": args.eval_at_end, "training": training}


def main():
    args = build_parser().parse_args()
    if args.num_beams < 10:
        raise ValueError("This launcher selects MRR@10; --num-beams must be at least 10")
    config = experiment_config(args)
    output = Path(config["output_dir"])
    config_path = output.parent / (output.name + ".config.json")
    if (config_path.exists() and read_json(config_path) != config
            and (args.resume or (output / "run_manifest.json").exists())):
        raise ValueError(f"Resolved config changed: {config_path}; use a separate output directory for another arm")
    if args.resume and not (output / "run_manifest.json").is_file():
        raise ValueError("No existing run_manifest.json to resume")
    atomic_json(config_path, config)
    print(f"Resolved experiment config: {config_path}", flush=True)
    run_pipeline(load_config(config_path), resume=args.resume, prepare_only=args.prepare_only,
                 start_round=args.start_round)


if __name__ == "__main__":
    main()
