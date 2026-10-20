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
    parser.add_argument("--infer-file", help="Optional final-only evaluation file; beam early stopping is disabled")
    parser.add_argument("--bm25-file", help="Optional BM25 preference JSONL, matched by prompt and chosen_<target-type>")
    parser.add_argument("--bm25-training-negatives-per-query", "--bm25-negatives-per-query",
                        dest="bm25_negatives_per_query", type=int, default=2,
                        help="BM25 target share of the final training quota; other sources may backfill shortfalls (default: 2)")
    parser.add_argument("--output-dir", required=True, help="Separate, initially empty directory for this arm")
    parser.add_argument("--target-type", default="url_based_id", help="ID column; requires positive_<column> labels")
    parser.add_argument("--corpus-file", action="append", default=[], help="Optional additional full-corpus metadata")
    parser.add_argument("--arm", choices=["dpo", "zero", "branch"], default="branch")
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--branch-beta", type=float)
    parser.add_argument("--reference-ema-decay", type=float, default=1.0,
                        help="Old-reference weight in EMA: ref <- decay*ref + (1-decay)*policy; 1 freezes the reference")
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
    parser.add_argument("--total-training-negatives-per-query", "--negatives-per-query",
                        dest="negatives_per_query", type=int, default=4,
                        help="Maximum final training negatives per query, summed across sources (default: 4)")
    parser.add_argument("--policy-mining-negatives-per-query", type=int,
                        help="Maximum model-confusion candidates retained from beams before mixing; defaults to the total training quota")
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
    parser.add_argument("--keep-last-round-checkpoints", type=int, default=3,
                        help="Keep checkpoint payloads for the N latest completed training rounds; 0 keeps all (default: 3)")
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
    config = {"checkpoint_path": str(Path(args.checkpoint).resolve()),
            "initial_reference_checkpoint": str(Path(args.checkpoint).resolve()),
            "query_file": str(Path(args.query_file).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
            "corpus_files": [str(Path(p).resolve()) for p in args.corpus_file],
            "input_format": "multilabel", "target_type": args.target_type,
            "rounds": args.rounds, "queries_per_round": args.queries_per_round,
            "steps_per_round": args.steps_per_round, "reference_update": "ema",
            "reference_ema_decay": args.reference_ema_decay, "seed": args.seed,
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
    if args.infer_file:
        config["infer_file"] = str(Path(args.infer_file).resolve())
    config["keep_last_round_checkpoints"] = args.keep_last_round_checkpoints
    if args.policy_mining_negatives_per_query is not None and args.policy_mining_negatives_per_query != args.negatives_per_query:
        config["policy_mining_negatives_per_query"] = args.policy_mining_negatives_per_query
    if args.bm25_file:
        config.update(bm25_file=str(Path(args.bm25_file).resolve()),
                      bm25_negatives_per_query=args.bm25_negatives_per_query)
    return config


def main():
    args = build_parser().parse_args()
    if args.num_beams < 10:
        raise ValueError("This launcher selects MRR@10; --num-beams must be at least 10")
    config = experiment_config(args)
    output = Path(config["output_dir"])
    config_path = output.parent / (output.name + ".config.json")
    if config_path.exists():
        previous = read_json(config_path)
        if args.resume and not config.get("infer_file") and previous.get("infer_file"):
            config["infer_file"] = previous["infer_file"]
        if previous != config and (args.resume or (output / "run_manifest.json").exists()):
            runtime_keys = {"infer_file", "keep_last_round_checkpoints"}
            same_training = ({k: v for k, v in previous.items() if k not in runtime_keys}
                             == {k: v for k, v in config.items() if k not in runtime_keys})
            adding_inference = args.resume and not previous.get("infer_file") and config.get("infer_file")
            same_inference = previous.get("infer_file") == config.get("infer_file")
            if not (args.resume and same_training and (same_inference or adding_inference)):
                raise ValueError(f"Resolved config changed: {config_path}; use a separate output directory for another arm")
    if args.resume and not (output / "run_manifest.json").is_file():
        raise ValueError("No existing run_manifest.json to resume")
    atomic_json(config_path, config)
    print(f"Resolved experiment config: {config_path}", flush=True)
    run_pipeline(load_config(config_path), resume=args.resume, prepare_only=args.prepare_only,
                 start_round=args.start_round)


if __name__ == "__main__":
    main()
