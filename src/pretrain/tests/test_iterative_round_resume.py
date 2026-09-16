"""Resume at a retained round after historical artifacts have been deleted."""
from pathlib import Path
import shutil
import tempfile
import unittest

from test_iterative_dpo import FixtureWorker, fixture
from pretrain.iterative_dpo_utils import atomic_json, file_hash, read_json
from pretrain.train_iterative_ddro_vault import SRC, load_config, run_pipeline


class RoundResumeTest(unittest.TestCase):
    def test_skip_deleted_history_preserves_policy_reference_and_epoch_count(self):
        for mode, decay in (("replace", 0.9), ("ema", 0.9), ("ema", 1.0)):
            with self.subTest(mode=mode, decay=decay), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = fixture(root)
                raw = read_json(path)
                raw.pop("steps_per_round")
                raw.update(rounds=3, epochs_per_round=1, reference_update=mode, reference_ema_decay=decay)
                atomic_json(path, raw)
                cfg = load_config(path)
                worker = FixtureWorker()
                def interrupt(command):
                    if "--round_manifest" in command and "round-002" in command[command.index("--round_manifest") + 1]:
                        raise RuntimeError("interrupt round two")
                    worker(command)
                with self.assertRaisesRegex(RuntimeError, "interrupt round two"):
                    run_pipeline(cfg, runner=interrupt)
                inputs = root / "run/round-002/round_inputs.json"
                before = file_hash(inputs)
                lineage = read_json(inputs)
                shutil.rmtree(root / "run/round-000")
                # Also exercise migration from the immediately preceding controller.
                manifest = root / "run/run_manifest.json"
                state = read_json(manifest)
                state["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = (
                    "a64a0b9ddd8bc57498cf729ccca88eadfb5e807a96ce0c1232a182f37cd33f84")
                atomic_json(manifest, state)
                worker.calls.clear()
                result = run_pipeline(cfg, resume=True, runner=worker, start_round=2)
                self.assertTrue(result["complete"])
                self.assertEqual(file_hash(inputs), before)
                self.assertEqual(result["rounds"][2]["completed_training_epochs"], 3)
                train = next(c for c in worker.calls if "--round_manifest" in c)
                self.assertEqual(train[train.index("--checkpoint_path") + 1], lineage["policy_checkpoint"])
                self.assertEqual(train[train.index("--reference_checkpoint_path") + 1], lineage["reference_checkpoint"])
                self.assertEqual(len([c for c in worker.calls if "--round_manifest" in c]), 1)
                worker.calls.clear()
                run_pipeline(cfg, resume=True, runner=worker, start_round=2)
                self.assertFalse(worker.calls)

    def test_missing_boundary_checkpoint_and_changed_current_data_still_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            worker = FixtureWorker()
            run_pipeline(cfg, runner=worker)
            weights = root / "run/round-000/training/latest/model.safetensors"
            saved = weights.read_bytes()
            weights.unlink()
            with self.assertRaisesRegex(ValueError, "required policy checkpoint"):
                run_pipeline(cfg, resume=True, runner=worker, start_round=1)
            weights.write_bytes(saved)
            (root / "run/round-001/queries.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run_pipeline(cfg, resume=True, runner=worker, start_round=1)

    def test_requires_resume_valid_range_and_completed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(fixture(Path(tmp)))
            worker = FixtureWorker()
            with self.assertRaisesRegex(ValueError, "requires --resume"):
                run_pipeline(cfg, start_round=1, runner=worker)
            run_pipeline(cfg, prepare_only=True, runner=worker)
            with self.assertRaisesRegex(ValueError, "less than 2"):
                run_pipeline(cfg, resume=True, start_round=2, runner=worker)
            with self.assertRaisesRegex(ValueError, "unfinished rounds"):
                run_pipeline(cfg, resume=True, start_round=1, runner=worker)


if __name__ == "__main__":
    unittest.main()
