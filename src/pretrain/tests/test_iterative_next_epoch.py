"""Continuing a full query pass carries forward both policy and EMA weights."""

from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from pretrain.iterative_dpo_utils import atomic_json, file_hash, read_json
from pretrain.start_next_iterative_epoch import next_epoch_config
from pretrain.train_iterative_ddro_vault import load_config, run_pipeline
from test_iterative_dpo import FixtureWorker, fixture


class NextEpochTest(unittest.TestCase):
    def make_source(self, root):
        path = fixture(root)
        raw = read_json(path)
        raw.update(rounds=None, epochs_per_round=1, steps_per_round=None,
                   reference_update="ema", reference_ema_decay=0.9)
        atomic_json(path, raw)
        return run_pipeline(load_config(path), runner=FixtureWorker())

    def test_second_pass_keeps_split_policy_and_ema_then_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root)
            saved = {p: file_hash(p) for p in (root / "run").rglob("*") if p.is_file()}
            cfg = next_epoch_config(root / "run", root / "epoch2")
            self.assertEqual(cfg["checkpoint_path"], source["latest"])
            self.assertEqual(cfg["initial_reference_checkpoint"], source["latest_reference"])
            self.assertNotEqual(cfg["checkpoint_path"], cfg["initial_reference_checkpoint"])
            path = root / "epoch2.json"
            atomic_json(path, cfg)
            cfg = load_config(path)
            worker = FixtureWorker()
            worker.fail_training = True
            with self.assertRaises(RuntimeError):
                run_pipeline(cfg, runner=worker)
            payload = read_json(root / "epoch2/round-000/round_inputs.json")
            self.assertEqual(payload["policy_checkpoint"], source["latest"])
            self.assertEqual(payload["reference_checkpoint"], source["latest_reference"])
            worker.fail_training = False
            result = run_pipeline(cfg, resume=True, runner=worker)
            self.assertTrue(result["complete"])
            for name in ("train_queries.jsonl", "validation_queries.jsonl", "document_metadata.jsonl"):
                self.assertEqual(file_hash(root / "run/metadata" / name),
                                 file_hash(root / "epoch2/metadata" / name))
            for i in range(source["planned_rounds"]):
                self.assertEqual(file_hash(root / f"run/round-{i:03d}/queries.jsonl"),
                                 file_hash(root / f"epoch2/round-{i:03d}/queries.jsonl"))
            self.assertEqual(saved, {p: file_hash(p) for p in saved})
            reference = Path(cfg["initial_reference_checkpoint"]) / "model.safetensors"
            reference.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline(cfg, resume=True, runner=worker)

    def test_refuses_unfinished_source_and_changed_final_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root)
            path = root / "run/run_manifest.json"
            atomic_json(path, {**source, "complete": False})
            with self.assertRaisesRegex(ValueError, "completed all"):
                next_epoch_config(root / "run", root / "epoch2")
            atomic_json(path, source)
            with self.assertRaisesRegex(ValueError, "separate output"):
                next_epoch_config(root / "run", root / "run/epoch2")
            (Path(source["latest_reference"]) / "model.safetensors").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference checkpoint"):
                next_epoch_config(root / "run", root / "epoch2")

    def test_previous_controller_without_explicit_reference_still_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_source(root)
            state["identity"]["config"].pop("initial_reference_checkpoint")
            state["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = (
                "0231e8a5f1ed667ff4346cfc5d066947bd0f2f2d8a411a564baba70cb25e38a0")
            atomic_json(root / "run/run_manifest.json", state)
            worker = FixtureWorker()
            result = run_pipeline(load_config(root / "config.json"), resume=True, runner=worker)
            self.assertTrue(result["complete"])
            self.assertFalse(worker.calls)

    def test_start_round_after_empty_partition_keeps_initial_ema_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root)
            cfg = next_epoch_config(root / "run", root / "epoch2")
            worker = FixtureWorker()
            worker.empty_mining_rounds.add(0)

            def interrupt(command):
                if "--round-id" in command and command[command.index("--round-id") + 1] == "1":
                    raise RuntimeError("stop before round one")
                worker(command)

            with self.assertRaisesRegex(RuntimeError, "stop before"):
                run_pipeline(cfg, runner=interrupt)
            run_pipeline(cfg, resume=True, start_round=1, runner=worker)
            payload = read_json(root / "epoch2/round-001/round_inputs.json")
            self.assertEqual(payload["reference_checkpoint"], source["latest_reference"])


if __name__ == "__main__":
    unittest.main()
