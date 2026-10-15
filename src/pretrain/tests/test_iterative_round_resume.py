"""Resume at a retained round after historical artifacts have been deleted."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from test_iterative_dpo import FixtureWorker, fixture
from pretrain.iterative_dpo_utils import atomic_json, file_hash, read_json
from common import iter_json_records
from pretrain.train_iterative_ddro_vault import SRC, load_config, run_pipeline


class CompletedRunExtensionTest(unittest.TestCase):
    def pilot(self, root, mode="ema", decay=1.0):
        path = fixture(root)
        atomic_json(path, {**read_json(path), "rounds": 1,
                           "reference_update": mode, "reference_ema_decay": decay})
        cfg = load_config(path)
        worker = FixtureWorker()

        def run(command):
            worker(command)
            if "--evaluation-only" in command and command[command.index("--round-id") + 1] == "0":
                output = Path(command[command.index("--output") + 1])
                atomic_json(output.with_suffix(".stats.json"), {"metrics": {"mrr@8": 0.9}})

        state = run_pipeline(cfg, runner=run)
        worker.calls.clear()
        return path, cfg, worker, state

    def test_full_query_pass_derives_total_and_keeps_smaller_final_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, worker, pilot = self.pilot(root)
            old_pairs = file_hash(root / "run/round-000/preferences.jsonl")
            result = run_pipeline(cfg, resume=True, runner=worker, start_round=1, extend_rounds="all")
            self.assertTrue(result["complete"])
            self.assertEqual(result["identity"]["config"]["rounds"], 3)
            self.assertEqual(result["full_query_pass"]["partition_sizes"], [5, 5, 4])
            prepared = list(iter_json_records(root / "run/metadata/train_queries.jsonl"))
            chosen = [q for index in range(3) for q in iter_json_records(root / f"run/round-{index:03d}/queries.jsonl")]
            self.assertEqual(len(chosen), len(prepared))
            self.assertEqual(len({q["query_key"] for q in chosen}), len(chosen))
            self.assertEqual({q["query_key"] for q in chosen}, {q["query_key"] for q in prepared})
            self.assertEqual(result["rounds"][0], pilot["rounds"][0])
            self.assertEqual(file_hash(root / "run/round-000/preferences.jsonl"), old_pairs)
            worker.calls.clear()
            run_pipeline(cfg, resume=True, runner=worker, start_round=1, extend_rounds="all")
            self.assertFalse(worker.calls)
            with self.assertRaisesRegex(ValueError, "full-query pass"):
                run_pipeline(cfg, resume=True, runner=worker, start_round=3, extend_rounds=4)

    def test_extend_completed_pilot_preserves_history_policy_reference_and_best(self):
        for mode, decay in (("replace", 0.9), ("ema", 0.9), ("ema", 1.0)):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(mode=mode, decay=decay):
                root = Path(tmp)
                config, cfg, worker, pilot = self.pilot(root, mode, decay)
                files = [p for p in (root / "run/round-000").rglob("*") if p.is_file()]
                files += [root / "run/round_plan.json", config]
                before = {p: file_hash(p) for p in files}
                manifest_path = root / "run/run_manifest.json"
                old = read_json(manifest_path)
                old["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = (
                    "54640a8be5e13db7899f9cc106740ab0488c7b44cc5d182a45309d357d13bbb3")
                atomic_json(manifest_path, old)
                result = run_pipeline(cfg, resume=True, runner=worker, start_round=1, extend_rounds=3)
                self.assertTrue(result["complete"])
                self.assertEqual(result["planned_rounds"], 3)
                self.assertEqual(len(result["rounds"]), 3)
                self.assertEqual(result["rounds"][0], pilot["rounds"][0])
                self.assertEqual(result["best"]["round_id"], 0)
                self.assertEqual(result["identity_updates"][-1]["reason"], "round_count_extension")
                self.assertEqual({p: file_hash(p) for p in files}, before)
                self.assertEqual(read_json(root / "run/round_plan_extended_003.json")["total_rounds"], 3)
                trains = [c for c in worker.calls if "--round_manifest" in c]
                self.assertEqual(len(trains), 2)
                boundary = read_json(root / "run/round-001/round_inputs.json")
                self.assertEqual(boundary["policy_checkpoint"], pilot["latest"])
                self.assertEqual(boundary["reference_checkpoint"], pilot["latest_reference"])
                self.assertEqual(boundary["training"], read_json(root / "run/round-002/round_inputs.json")["training"]
                                 | {"seed": boundary["training"]["seed"]})
                self.assertFalse(any("--round-id" in c and c[c.index("--round-id") + 1] in {"-1", "0"}
                                     for c in worker.calls))
                worker.calls.clear()
                run_pipeline(cfg, resume=True, runner=worker, start_round=1, extend_rounds=3)
                self.assertFalse(worker.calls)
                self.assertEqual(len(read_json(manifest_path)["round_extensions"]), 1)
                run_pipeline(cfg, resume=True, runner=worker, start_round=3, extend_rounds=4)
                self.assertEqual(len([c for c in worker.calls if "--round_manifest" in c]), 1)

    def test_extension_rejects_unrelated_changes_and_corrupted_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, worker, _ = self.pilot(root)
            manifest = root / "run/run_manifest.json"
            before = file_hash(manifest)
            changed = {**cfg, "training": {**cfg["training"], "learning_rate": 0.1}}
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline(changed, resume=True, runner=worker, start_round=1, extend_rounds=3)
            self.assertEqual(file_hash(manifest), before)
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline({**cfg, "rounds": 3}, resume=True, runner=worker, start_round=1)
            target = root / "run/round-000/preferences.jsonl"
            target.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run_pipeline(cfg, resume=True, runner=worker, start_round=1, extend_rounds=3)
            self.assertEqual(file_hash(manifest), before)

    def test_requires_completed_run_and_larger_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            worker = FixtureWorker()
            for total in (0, -1, True):
                with self.subTest(total=total), self.assertRaisesRegex(ValueError, "positive total"):
                    run_pipeline(cfg, resume=True, runner=worker, extend_rounds=total)
            with self.assertRaisesRegex(ValueError, "requires --resume"):
                run_pipeline(cfg, runner=worker, extend_rounds=3)
            run_pipeline(cfg, prepare_only=True, runner=worker)
            with self.assertRaisesRegex(ValueError, "fully completed"):
                run_pipeline(cfg, resume=True, runner=worker, extend_rounds=3)
            run_pipeline(cfg, resume=True, runner=worker)
            for total in (1, 2):
                with self.subTest(total=total), self.assertRaisesRegex(ValueError, "increase"):
                    run_pipeline(cfg, resume=True, runner=worker, extend_rounds=total)

    def test_extension_can_resume_if_interrupted_while_writing_new_plan(self):
        import pretrain.train_iterative_ddro_vault as controller
        for write_first in (False, True):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(write_first=write_first):
                root = Path(tmp)
                _, cfg, worker, _ = self.pilot(root)
                original_plan = file_hash(root / "run/round_plan.json")
                actual_write = controller.atomic_json

                def interrupt(path, value):
                    if Path(path).name == "round_plan_extended_003.json":
                        if write_first:
                            actual_write(path, value)
                        raise RuntimeError("extension plan interruption")
                    actual_write(path, value)

                with patch.object(controller, "atomic_json", interrupt), \
                     self.assertRaisesRegex(RuntimeError, "extension plan interruption"):
                    run_pipeline(cfg, resume=True, runner=worker, start_round=1, extend_rounds=3)
                self.assertEqual(file_hash(root / "run/round_plan.json"), original_plan)
                self.assertFalse(read_json(root / "run/run_manifest.json")["complete"])
                result = run_pipeline(cfg, resume=True, runner=worker, start_round=1, extend_rounds=3)
                self.assertTrue(result["complete"])
                self.assertEqual(len(result["round_extensions"]), 1)


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
