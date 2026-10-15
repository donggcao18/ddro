"""Real local T5: generation, annotation, training, frozen reference, and resume."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))


@unittest.skipUnless(os.environ.get("RUN_DPO_INTEGRATION") == "1", "Set RUN_DPO_INTEGRATION=1")
class BranchPipelineTest(unittest.TestCase):
    def test_real_generation_training_interruption_and_resume(self):
        self.check_pipeline()

    def test_truncated_aliases_generation_training_and_resume(self):
        self.check_pipeline(compatibility=True)

    def check_pipeline(self, compatibility=False):
        import torch
        from transformers import T5ForConditionalGeneration, TrainerCallback
        import mine_model_confusion_negatives as miner
        from pretrain import train_ddro_vault as training
        from pretrain import branch_preference_trainer as branch_training
        from pretrain import update_dpo_reference as reference
        from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash, read_json
        from pretrain.train_iterative_ddro_vault import load_config, run_pipeline
        from test_branch_preference_trainer import tiny_checkpoint

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial, _ = tiny_checkpoint(root / "dpo")
            initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
            ids = [f"A B C{i} D{i}" for i in range(8)]
            aliases = [target + " unknown_suffix" for target in ids] if compatibility else []
            if compatibility:
                atomic_jsonl(root / "corpus.jsonl", [{"semantic_id": target} for target in aliases])
            atomic_jsonl(root / "queries.jsonl", [
                {"text": f"find C{i}", "semantic_id": target,
                 "positive_semantic_id": [target, ids[i ^ 1]] + ([aliases[i]] if compatibility else []),
                 "family_id": f"family-{i // 2}"}
                for i, target in enumerate(ids)])
            config_path = root / "config.json"
            atomic_json(config_path, {
                "checkpoint_path": "dpo", "initial_reference_checkpoint": "dpo",
                "query_file": "queries.jsonl", "output_dir": "straight", "input_format": "multilabel",
                "target_type": "semantic_id", "rounds": 2, "queries_per_round": 4, "steps_per_round": 2,
                "reference_update": "ema", "reference_ema_decay": 1.0,
                "precision": "fp32", "device": "cpu", "num_beams": 8,
                "negatives_per_query": 2, "mining_batch_size": 2, "max_prompt_length": 8,
                "max_target_length": 4 if compatibility else 8,
                "target_length_policy": "truncate" if compatibility else "error",
                "target_collision_policy": "allow" if compatibility else "error",
                "target_token_policy": "allow" if compatibility else "error",
                "corpus_files": ["corpus.jsonl"] if compatibility else [],
                "selection_metric": "mrr@8", "training": {
                    "preference_objective": "branch-dpo", "branch_loss_weight": 0.1,
                    "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-3, "warmup_ratio": 0, "save_steps": 1,
                    "logging_steps": 1, "dataset_num_proc": 1, "dataloader_num_workers": 0,
                }})
            interrupt = [False]
            interrupted = [False]
            actual_trainer = branch_training.BranchDPOTrainer

            class InterruptOnce(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    if interrupt[0] and not interrupted[0]:
                        interrupted[0] = True
                        raise RuntimeError("branch test interruption after save")

            def trainer_factory(*args, **kwargs):
                kwargs["callbacks"].append(InterruptOnce())
                return actual_trainer(*args, **kwargs)

            calls = []

            def runner(command):
                calls.append(command)
                if "--decay" in command:
                    options = vars(reference.build_parser().parse_args(command[2:]))
                    reference.update_reference(**options)
                    weights = T5ForConditionalGeneration.from_pretrained(options["output"]).state_dict()
                    for key, expected in initial_state.items():
                        torch.testing.assert_close(weights[key], expected, rtol=0, atol=0)
                elif "--checkpoint-path" in command:
                    miner.mine(miner.build_parser().parse_args(command[2:]))
                else:
                    with patch.object(sys, "argv", command[1:]), \
                         patch.object(branch_training, "BranchDPOTrainer", trainer_factory):
                        training.train_round(training.parse_args())

            cfg = load_config(config_path)
            straight = run_pipeline(cfg, runner=runner)
            self.assertTrue(straight["complete"])
            self.assertTrue(all(r["branch_audit"]["eligible_rows"] > 0 for r in straight["rounds"]))
            if compatibility:
                for result in straight["rounds"]:
                    self.assertEqual(result["branch_audit"]["overlength_targets"], 16)
                    self.assertEqual(result["branch_audit"]["collision_groups"], 8)
                    self.assertEqual(result["branch_audit"]["excluded_targets"], 0)
            for index in range(2):
                from common import iter_json_records
                directory = root / f"straight/round-{index:03d}"
                raw = list(iter_json_records(directory / "preferences.jsonl"))
                annotated = list(iter_json_records(directory / "branch_preferences.jsonl"))
                self.assertEqual(len(raw), len(annotated))
                for before, after in zip(raw, annotated):
                    self.assertTrue(all(after[key] == value for key, value in before.items()))
                manifest = read_json(directory / "round_inputs.json")
                self.assertEqual(manifest["inputs"][str(directory / "branch_preferences.jsonl")],
                                 file_hash(directory / "branch_preferences.jsonl"))
                args = torch.load(directory / "training/latest/training_args.bin", weights_only=False)
                self.assertEqual(args.branch_loss_weight, 0.1)
                if compatibility:
                    self.assertTrue(all(row["branch_position"] < 4 for row in annotated))
                    self.assertTrue(all(row["rejected_text_id"] not in row["positive_text_ids"]
                                        for row in annotated))

            # Extend a completed pilot through the remaining queries without
            # retraining round zero or replacing the initial frozen reference.
            pilot_cfg = {**cfg, "output_dir": str(root / "extended"), "rounds": 1}
            run_pipeline(pilot_cfg, runner=runner)
            preserved = {p: file_hash(p) for p in (root / "extended/round-000").rglob("*") if p.is_file()}
            calls.clear()
            interrupt[0] = True
            with self.assertRaisesRegex(RuntimeError, "branch test interruption"):
                run_pipeline(pilot_cfg, resume=True, runner=runner, start_round=1, extend_rounds="all")
            self.assertEqual({p: file_hash(p) for p in preserved}, preserved)
            extended = run_pipeline(pilot_cfg, resume=True, runner=runner, start_round=1, extend_rounds="all")
            self.assertTrue(extended["complete"])
            self.assertEqual(extended["full_query_pass"]["partition_sizes"], [4, 2])
            self.assertEqual({p: file_hash(p) for p in preserved}, preserved)
            self.assertFalse(any("--round-id" in c and c[c.index("--round-id") + 1] in {"-1", "0"}
                                 for c in calls))
            # The full-pass continuation uses only two remaining queries in
            # its final round, whereas the fixed two-round run wraps to four.
            # Assert real updates and the original reference, rather than
            # expecting identical weights from different preference datasets.
            final_weights = T5ForConditionalGeneration.from_pretrained(extended["latest"]).state_dict()
            pilot_weights = T5ForConditionalGeneration.from_pretrained(
                root / "extended/round-000/training/latest").state_dict()
            self.assertTrue(any(not torch.equal(final_weights[k], pilot_weights[k]) for k in final_weights))

            interrupted_cfg = {**cfg, "output_dir": str(root / "resumed")}
            interrupt[0] = True
            interrupted[0] = False
            with self.assertRaisesRegex(RuntimeError, "branch test interruption"):
                run_pipeline(interrupted_cfg, runner=runner)
            committed_pairs = root / "resumed/round-000/branch_preferences.jsonl"
            before_hash = file_hash(committed_pairs)
            calls.clear()
            resumed = run_pipeline(interrupted_cfg, resume=True, runner=runner)
            self.assertTrue(resumed["complete"])
            self.assertEqual(file_hash(committed_pairs), before_hash)
            self.assertTrue(any("--resume_from_checkpoint" in call for call in calls))
            straight_weights = T5ForConditionalGeneration.from_pretrained(straight["latest"]).state_dict()
            resumed_weights = T5ForConditionalGeneration.from_pretrained(resumed["latest"]).state_dict()
            for key, value in straight_weights.items():
                other = resumed_weights[key]
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "changed"):
                run_pipeline({**interrupted_cfg, "training": {**cfg["training"], "branch_loss_weight": 0.3}},
                             resume=True, runner=runner)
            committed_pairs.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                run_pipeline(interrupted_cfg, resume=True, runner=runner)


if __name__ == "__main__":
    unittest.main()
