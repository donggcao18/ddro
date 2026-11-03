#!/usr/bin/env python3
"""Train one of three controlled multi-positive URL preference experiments."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pretrain.train_grpo_vault import build_trainer, make_parser
from utils.coverage_trainer import CoveragePreferenceTrainer, CoverageSettings, OBJECTIVES
from utils.grpo_trainer import retrieval_metrics


def selection_score(metrics: dict[str, float], baseline_hit1: float, tolerance: float) -> float:
    """Prefer coverage only within the allowed absolute Hit@1 regression."""
    return metrics["recall@10"] if metrics["hit@1"] + 1e-8 >= baseline_hit1 - tolerance else -1.


def main() -> None:
    parser = make_parser()
    parser.description = __doc__
    parser.set_defaults(do_sample="false", num_generations=10, beta=0., learning_rate=2e-6,
                        per_device_train_batch_size=4, gradient_accumulation_steps=4)
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--preference_beta", type=float, default=.1)
    parser.add_argument("--sft_anchor_weight", type=float, default=.1)
    parser.add_argument("--positives_per_query", type=int, default=2)
    parser.add_argument("--negatives_per_query", type=int, default=4)
    parser.add_argument("--missed_positive_weight", type=float, default=2.)
    parser.add_argument("--hit1_tolerance", type=float, default=.005,
                        help="Allowed absolute validation Hit@1 drop, .005 = 0.5 percentage points.")
    args = parser.parse_args()
    if not 0 <= args.hit1_tolerance <= 1:
        parser.error("hit1_tolerance must be between 0 and 1")
    destination = Path(args.output_dir)
    if not args.resume_from_checkpoint and (
        (destination / "candidate_final").exists() or list(destination.glob("checkpoint-*"))
    ):
        raise ValueError("Output contains a previous training run; use a new output_dir or explicit resume_from_checkpoint")
    values = {key: getattr(args, key) for key in CoverageSettings.__dataclass_fields__}
    values["do_sample"] = args.do_sample == "true"
    settings = CoverageSettings(**values)
    args.metric_for_best_model = "coverage_selection"
    trainer = build_trainer(args, settings=settings, trainer_class=CoveragePreferenceTrainer, require_reference=True)
    if args.dry_run:
        inputs = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))
        with torch.no_grad(), trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(trainer.model, inputs)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite preference loss in dry run")
        trainer.log({"loss": float(loss)})
        print("Preference dry run successful; annotated positives and mined negatives scored; no optimizer update.")
        return
    if trainer.eval_dataset is None:
        raise ValueError("These experiments require validation data for the Hit@1 guard and coverage selection")
    # Evaluate the immutable starting checkpoint, including on resume. Evaluating
    # the resumed policy instead would silently move the Hit@1 floor.
    baseline = trainer.evaluate(metric_key_prefix="baseline")
    baseline_hit1 = baseline["baseline_hit@1"]

    def guarded_metrics(prediction):
        metrics = retrieval_metrics(prediction)
        metrics["coverage_selection"] = selection_score(metrics, baseline_hit1, args.hit1_tolerance)
        return metrics

    trainer.compute_metrics = guarded_metrics
    trainer._rollout_metrics.clear()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    candidate_metrics = trainer.evaluate()
    trainer.model.config.use_cache = True
    trainer.save_model(str(destination / "candidate_final"))
    trainer.save_state()
    candidate_score = candidate_metrics["eval_coverage_selection"]
    baseline_score = baseline["baseline_recall@10"]
    improved = candidate_score > baseline_score + 1e-10
    if not improved:
        reference = trainer.accelerator.unwrap_model(trainer.reference_model)
        trainer.model.load_state_dict(reference.state_dict())
    trainer.save_model(str(destination / "final"))
    if trainer.is_world_process_zero():
        result = {"objective": settings.objective, "settings": asdict(settings), "baseline": baseline,
                  "candidate": candidate_metrics, "selected": "candidate" if improved else "sft",
                  "best_model_checkpoint": trainer.state.best_model_checkpoint, "global_step": trainer.state.global_step,
                  "hit1_tolerance": args.hit1_tolerance, "validation_hit1_floor": baseline_hit1 - args.hit1_tolerance,
                  "selection_rule": "max validation recall@10 within Hit@1 floor; SFT fallback without strict recall improvement",
                  "candidate_model": str(destination / "candidate_final"), "selected_model": str(destination / "final")}
        (destination / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
