#!/usr/bin/env python3
"""Train the multi-positive, token-trajectory GRPO experiment for T5/CodeT5."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pretrain.train_grpo_vault import build_trainer, make_parser
from pretrain.train_coverage_vault import selection_score
from utils.grpo_trainer import retrieval_metrics
from utils.tca_grpo_trainer import TCASettings, TCAGrpoTrainer


def parse_args():
    parser = make_parser()
    parser.description = __doc__
    parser.set_defaults(objective="tca_grpo", beta=.04, do_sample="true", max_steps=100,
                        per_device_train_batch_size=1, gradient_accumulation_steps=16,
                        learning_rate=1e-6, eval_steps=50, save_steps=50)
    parser.add_argument("--hit_bonus", type=float, default=2.)
    parser.add_argument("--overflow_penalty", type=float, default=-.5)
    parser.add_argument("--reward_std_floor", type=float, default=.001)
    parser.add_argument("--gold_candidates_per_query", type=int, default=2)
    parser.add_argument("--gold_cache_mb", type=float, default=128.)
    parser.add_argument("--reference_batch_size", type=int, default=16)
    parser.add_argument("--all_zero_ranking_weight", type=float, default=.1)
    parser.add_argument("--sft_anchor_weight", type=float, default=.1)
    parser.add_argument("--negatives_per_query", type=int, default=4)
    parser.add_argument("--trajectory_alignment", choices=("pre_token", "post_token"), default="pre_token",
                        help="pre_token follows author alignment; post_token compares states after each chosen token.")
    parser.add_argument("--hit1_tolerance", type=float, default=0.)
    parser.add_argument("--min_validation_recall_gain", type=float, default=.005,
                        help="Minimum absolute validation Recall@10 gain for final/ selection; .005 = 0.5 pp.")
    return parser.parse_args()


def settings_from_args(args):
    values = {key: getattr(args, key) for key in TCASettings.__dataclass_fields__}
    values["do_sample"] = args.do_sample == "true"
    return TCASettings(**values)


def main():
    args = parse_args()
    if not 0 <= args.hit1_tolerance <= 1 or not 0 <= args.min_validation_recall_gain <= 1:
        raise ValueError("Validation tolerance and minimum gain must be between 0 and 1")
    destination = Path(args.output_dir)
    if not args.resume_from_checkpoint and ((destination / "candidate_final").exists() or list(destination.glob("checkpoint-*"))):
        raise ValueError("Use a fresh output_dir or explicit resume_from_checkpoint for an existing run")
    settings = settings_from_args(args)
    args.metric_for_best_model = "coverage_selection"
    trainer = build_trainer(args, settings=settings, trainer_class=TCAGrpoTrainer, require_reference=True)
    if args.dry_run:
        inputs = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))
        with torch.no_grad(), trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(trainer.model, inputs)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite TCA dry-run loss")
        trainer.log({"loss": float(loss)})
        print("TCA dry run successful: token rewards and auxiliary losses evaluated; no optimizer update.", flush=True)
        return
    if trainer.eval_dataset is None:
        raise ValueError("TCA experiment requires validation data for checkpoint selection")
    print("=== Evaluating starting SFT checkpoint on validation ===", flush=True)
    baseline = trainer.evaluate(metric_key_prefix="baseline")

    def guarded_metrics(prediction):
        metrics = retrieval_metrics(prediction)
        metrics["coverage_selection"] = selection_score(metrics, baseline["baseline_hit@1"], args.hit1_tolerance)
        return metrics

    trainer.compute_metrics = guarded_metrics
    trainer._rollout_metrics.clear()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    candidate = trainer.evaluate()
    trainer.model.config.use_cache = True
    trainer.save_model(str(destination / "candidate_final"))
    trainer.save_state()
    gain = candidate["eval_recall@10"] - baseline["baseline_recall@10"]
    improved = candidate["eval_coverage_selection"] >= 0 and gain > 1e-10 and gain + 1e-10 >= args.min_validation_recall_gain
    if not improved:
        trainer.model.load_state_dict(trainer.accelerator.unwrap_model(trainer.reference_model).state_dict())
    trainer.save_model(str(destination / "final"))
    if trainer.is_world_process_zero():
        report = {"objective": "tca_grpo", "baseline": baseline, "candidate": candidate,
                  "selected": "candidate" if improved else "sft", "validation_recall_gain": gain,
                  "hit1_tolerance": args.hit1_tolerance, "min_validation_recall_gain": args.min_validation_recall_gain,
                  "validation_hit1_floor": baseline["baseline_hit@1"] - args.hit1_tolerance,
                  "best_model_checkpoint": trainer.state.best_model_checkpoint, "global_step": trainer.state.global_step,
                  "candidate_model": str(destination / "candidate_final"), "selected_model": str(destination / "final"),
                  "method_reference": "https://arxiv.org/abs/2608.12049",
                  "adaptations": ["multi-positive whole-path choice and hit-set bonus", "online bounded gold cache",
                                  "reward variance floor", "separate all-wrong raw-score ranking and positive SFT loss"],
                  "validation_note": "Post-training exclusion does not establish exclusion from SFT; verify query provenance."}
        (destination / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
