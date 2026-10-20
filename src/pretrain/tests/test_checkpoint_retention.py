"""Payload-only checkpoint pruning, historical integrity and resumable cleanup."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from pretrain.checkpoint_retention import PAYLOAD, checkpoint_pruned, prune_round_checkpoints
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, assert_file_hashes, checkpoint_hash, file_hash, read_json
from pretrain.launch_branch_preference import build_parser, experiment_config
from pretrain.train_iterative_ddro_vault import load_config, run_pipeline
from test_iterative_dpo import FixtureWorker, fake_checkpoint, fixture


class PayloadWorker(FixtureWorker):
    def __call__(self, command):
        super().__call__(command)
        if "--round_manifest" in command:
            directory = Path(command[command.index("--output_dir") + 1])
            for name in ("checkpoint-1", "final"):
                fake_checkpoint(directory / name)
                for payload in ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state_0.pth"):
                    (directory / name / payload).write_bytes(b"optimizer state")
                atomic_json(directory / name / "trainer_state.json", {"keep": "metadata"})
            (directory / "events.out.tfevents").write_bytes(b"keep logs")
            (directory.parent / "other_round_file.bin").write_bytes(b"keep non-checkpoint data")
            (directory / "latest/training_args.bin").write_bytes(b"checkpoint arguments")


class CheckpointRetentionTest(unittest.TestCase):
    def config(self, root, rounds=6, mode="ema", decay=0.9):
        path = fixture(root)
        atomic_json(path, {**read_json(path), "rounds": rounds, "reference_update": mode,
                           "reference_ema_decay": decay, "keep_last_round_checkpoints": 3})
        return load_config(path)

    def assert_window(self, root, kept, total):
        for i in range(total):
            directory = root / f"run/round-{i:03d}"
            self.assertTrue((directory / "preferences.jsonl").is_file())
            self.assertTrue((directory / "pair_audit.json").is_file())
            self.assertTrue((directory / "training/training_metrics.json").is_file())
            self.assertTrue((directory / "training/latest/round_complete.json").is_file())
            self.assertTrue((directory / "training/latest/config.json").is_file())
            self.assertTrue((directory / "training/events.out.tfevents").is_file())
            self.assertEqual((directory / "other_round_file.bin").read_bytes(), b"keep non-checkpoint data")
            self.assertTrue((directory / "training/checkpoint-1/trainer_state.json").is_file())
            self.assertEqual((directory / "training/latest/model.safetensors").exists(), i in kept)
            self.assertEqual((directory / "training/final/model.safetensors").exists(), i in kept)
            self.assertEqual((directory / "training/checkpoint-1/optimizer.pt").exists(), i in kept)

    def test_keeps_three_trained_rounds_with_ema_and_replace_and_resumes(self):
        for mode, decay in (("ema", 0.9), ("ema", 1.0), ("replace", 0.9)):
            with self.subTest(mode=mode, decay=decay), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, worker = self.config(root, mode=mode, decay=decay), PayloadWorker()
                state = run_pipeline(cfg, runner=worker)
                self.assert_window(root, {3, 4, 5}, 6)
                if mode == "ema":
                    for i in range(6):
                        reference = root / f"run/round-{i:03d}/reference"
                        self.assertEqual((reference / "model.safetensors").exists(), i >= 3)
                        self.assertTrue((reference / "reference_complete.json").is_file())
                self.assertEqual([r["round_id"] for r in state["checkpoint_pruning"]], [0, 1, 2])
                self.assertTrue(all(r["status"] == "complete" for r in state["checkpoint_pruning"]))
                for artifacts in state["stages"].values():
                    assert_file_hashes(artifacts)
                initial_hash = checkpoint_hash(root / "sft")
                worker.calls.clear()
                resumed = run_pipeline(cfg, runner=worker, resume=True)
                self.assertTrue(resumed["complete"])
                self.assertFalse(worker.calls)
                self.assertEqual(checkpoint_hash(root / "sft"), initial_hash)

    def test_mining_failure_after_pruning_resumes_from_last_completed_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, worker = self.config(root), PayloadWorker()

            def fail(command):
                if "--round-id" in command and command[command.index("--round-id") + 1] == "4":
                    raise RuntimeError("mining interrupted")
                worker(command)

            with self.assertRaisesRegex(RuntimeError, "mining interrupted"):
                run_pipeline(cfg, runner=fail)
            self.assertFalse((root / "run/round-000/training/latest/model.safetensors").exists())
            worker.calls.clear()
            self.assertTrue(run_pipeline(cfg, runner=worker, resume=True)["complete"])
            train_ids = [read_json(c[c.index("--round_manifest") + 1])["round_id"]
                         for c in worker.calls if "--round_manifest" in c]
            self.assertEqual(train_ids, [4, 5])
            self.assert_window(root, {3, 4, 5}, 6)

    def test_pending_partial_deletion_is_retried_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, worker = self.config(root, rounds=4), PayloadWorker()
            actual = Path.unlink
            interrupted = [False]

            def fail(path, *args, **kwargs):
                if path == root / "run/round-000/training/latest/model.safetensors" and not interrupted[0]:
                    interrupted[0] = True
                    raise OSError("interrupted pruning")
                return actual(path, *args, **kwargs)

            with patch.object(Path, "unlink", fail):
                with self.assertRaisesRegex(OSError, "interrupted pruning"):
                    run_pipeline(cfg, runner=worker)
            state = read_json(root / "run/run_manifest.json")
            self.assertEqual(state["checkpoint_pruning"][0]["status"], "pending")
            self.assertEqual(len(state["rounds"]), 4)
            worker.calls.clear()
            resumed = run_pipeline(cfg, runner=worker, resume=True)
            self.assertFalse(worker.calls)
            self.assertEqual(resumed["checkpoint_pruning"][0]["status"], "complete")
            self.assert_window(root, {1, 2, 3}, 4)

    def test_skipped_rounds_do_not_evict_active_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, worker = self.config(root, rounds=8), PayloadWorker()
            worker.empty_mining_rounds = {3, 4, 5, 6}
            state = run_pipeline(cfg, runner=worker)
            self.assertFalse((root / "run/round-000/training/latest/model.safetensors").exists())
            for i in (1, 2, 7):
                self.assertTrue((root / f"run/round-{i:03d}/training/latest/model.safetensors").is_file())
            self.assertEqual([r["round_id"] for r in state["checkpoint_pruning"]], [0])
            worker.calls.clear()
            self.assertTrue(run_pipeline(cfg, runner=worker, resume=True)["complete"])
            self.assertFalse(worker.calls)

    def test_old_best_keeps_metrics_and_marks_weights_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, worker = self.config(root), PayloadWorker()

            def scored(command):
                worker(command)
                if "--evaluation-only" in command and command[command.index("--round-id") + 1] == "0":
                    output = Path(command[command.index("--output") + 1])
                    atomic_json(output.with_suffix(".stats.json"), {"metrics": {"mrr@8": 0.99}})

            state = run_pipeline(cfg, runner=scored)
            self.assertEqual(state["best"]["round_id"], 0)
            self.assertEqual(state["best"]["metrics"]["mrr@8"], 0.99)
            self.assertFalse(read_json(root / "run/best_checkpoint.json")["checkpoint_available"])
            worker.calls.clear()
            resumed = run_pipeline(cfg, runner=scored, resume=True)
            self.assertEqual(resumed["best"]["round_id"], 0)
            self.assertFalse(worker.calls)
            with self.assertRaisesRegex(ValueError, "required policy checkpoint is missing"):
                run_pipeline(cfg, runner=scored, resume=True, start_round=1)

    def test_existing_run_can_enable_retention_with_known_controller_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, worker = self.config(root, rounds=4), PayloadWorker()
            cfg["keep_last_round_checkpoints"] = None
            run_pipeline(cfg, runner=worker)
            manifest = root / "run/run_manifest.json"
            state = read_json(manifest)
            state["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = (
                "f0eeeb3fd73ab76a4f8c8327c3d3397e13767c960da86a3f78317f0281f550ca")
            del state["identity"]["code"][str(SRC / "pretrain/checkpoint_retention.py")]
            atomic_json(manifest, state)
            worker.calls.clear()
            self.assertTrue(run_pipeline(cfg, runner=worker, resume=True, keep_last_round_checkpoints=3)["complete"])
            self.assertFalse(worker.calls)
            self.assert_window(root, {1, 2, 3}, 4)

    def test_no_pruning_when_disabled_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, worker = self.config(root, rounds=4), PayloadWorker()
            cfg["keep_last_round_checkpoints"] = 0
            state = run_pipeline(cfg, runner=worker)
            self.assertNotIn("checkpoint_pruning", state)
            self.assert_window(root, {0, 1, 2, 3}, 4)
            path = root / "config.json"
            for invalid in (-1, True, 1.5, "3"):
                atomic_json(path, {**cfg, "keep_last_round_checkpoints": invalid})
                with self.assertRaisesRegex(ValueError, "keep_last_round_checkpoints"):
                    load_config(path)
        args = build_parser().parse_args(["--checkpoint", "model", "--query-file", "q", "--output-dir", "run"])
        self.assertEqual(experiment_config(args)["keep_last_round_checkpoints"], 3)

    def test_pending_plan_cannot_delete_an_external_or_non_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, worker = self.config(root, rounds=4), PayloadWorker()
            state = run_pipeline(cfg, runner=worker)
            protected = root / "outside/model.safetensors"
            protected.parent.mkdir()
            protected.write_bytes(b"untouched")
            record = state["checkpoint_pruning"][0]
            for target in (protected, root / "run/round-000/preferences.jsonl"):
                record.update(status="pending", files=[{"path": str(target), "bytes": 1}])
                original = file_hash(target)
                with self.assertRaises(ValueError):
                    prune_round_checkpoints(state, root / "run", 3)
                self.assertEqual(file_hash(target), original)


@unittest.skipUnless(os.environ.get("RUN_DPO_INTEGRATION") == "1", "Set RUN_DPO_INTEGRATION=1")
class RealRetentionTest(unittest.TestCase):
    def test_real_ema_training_prunes_weights_and_preserves_final_inference(self):
        import torch
        import mine_model_confusion_negatives as miner
        from pretrain import train_ddro_vault as training
        from pretrain import update_dpo_reference as reference
        from test_branch_preference_trainer import tiny_checkpoint

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_checkpoint(root / "model")
            ids = [f"A B C{i} D{i}" for i in range(8)]
            atomic_jsonl(root / "queries.jsonl", [{"text": f"find C{i}", "url_based_id": target,
                                                   "positive_url_based_id": [target]} for i, target in enumerate(ids)])
            atomic_jsonl(root / "infer.jsonl", [{"text": "find C0", "url_based_id": ids[0]}])
            path = root / "config.json"
            atomic_json(path, {"checkpoint_path": "model", "query_file": "queries.jsonl", "infer_file": "infer.jsonl",
                               "input_format": "multilabel", "target_type": "url_based_id", "output_dir": "run",
                               "rounds": 5, "queries_per_round": 2, "steps_per_round": 1, "num_beams": 4,
                               "negatives_per_query": 2, "mining_batch_size": 2, "max_prompt_length": 8,
                               "max_target_length": 8, "precision": "fp32", "device": "cpu", "selection_metric": "mrr@4",
                               "keep_last_round_checkpoints": 3, "reference_update": "ema", "reference_ema_decay": 0.9,
                               "eval_every_rounds": 4, "training": {"preference_objective": "branch-dpo",
                                   "branch_loss_weight": 0.1, "per_device_train_batch_size": 2,
                                   "gradient_accumulation_steps": 1, "save_steps": 1, "logging_steps": 1}})
            calls = []

            def runner(command):
                calls.append(command)
                if "--checkpoint-path" in command:
                    miner.mine(miner.build_parser().parse_args(command[2:]))
                elif "--decay" in command:
                    reference.update_reference(**vars(reference.build_parser().parse_args(command[2:])))
                else:
                    with patch.object(sys, "argv", command[1:]):
                        training.train_round(training.parse_args())

            cfg = load_config(path)
            state = run_pipeline(cfg, runner=runner)
            self.assertTrue(state["complete"])
            self.assertEqual(len(state["checkpoint_pruning"]), 2)
            for i in range(5):
                directory = root / f"run/round-{i:03d}"
                self.assertEqual((directory / "training/latest/model.safetensors").exists(), i >= 2)
                self.assertEqual((directory / "reference/model.safetensors").exists(), i >= 2)
                self.assertTrue((directory / "branch_preferences.jsonl").is_file())
                self.assertTrue((directory / "branch_audit.json").is_file())
            self.assertFalse(state["final_inference"]["beam_early_stopping"])
            self.assertEqual(state["final_inference"]["checkpoint"], state["latest"])
            calls.clear()
            self.assertTrue(run_pipeline(cfg, runner=runner, resume=True)["complete"])
            self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
