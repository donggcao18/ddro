"""Final-only retrieval evaluation, data isolation, and beam-search settings."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import iter_json_records
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash, read_json
from pretrain.prepare_inference_queries import prepare_inference_queries
from pretrain.train_iterative_ddro_vault import load_config, run_pipeline
from test_iterative_dpo import FixtureWorker, fixture


class InferenceMetadataTest(unittest.TestCase):
    def test_single_label_multilabel_and_duplicates_for_both_id_designs(self):
        for target_type in ("url_based_id", "structure_id_v6"):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(target_type=target_type):
                root = Path(tmp)
                ids = ["repo/a", "repo/b", "repo/c"]
                atomic_jsonl(root / "docs.jsonl", [{"text_id": value, "doc_id_type": target_type} for value in ids])
                atomic_jsonl(root / "infer.jsonl", [
                    {"text": "first query", target_type: ids[0]},
                    {"text": "second query", target_type: ids[0], f"positive_{target_type}": ids[1:]},
                    {"text": "first query", target_type: ids[0]},
                ])
                report = prepare_inference_queries(root / "infer.jsonl", root / "docs.jsonl",
                                                   root / "prepared.jsonl", target_type, "multilabel")
                rows = list(iter_json_records(root / "prepared.jsonl"))
                self.assertEqual(report["queries"], 3)
                self.assertEqual(report["single_label_fallback_rows"], 2)
                self.assertEqual(rows[1]["positive_text_ids"], ids[1:])
                self.assertEqual(len({row["query_key"] for row in rows}), 3)

    def test_invalid_labels_missing_corpus_and_empty_file_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_jsonl(root / "docs.jsonl", [{"text_id": "a", "doc_id_type": "url_based_id"}])
            cases = [[], [{"text": "q", "url_based_id": "absent"}],
                     [{"text": "q", "url_based_id": "a", "positive_url_based_id": []}],
                     [{"text": "q"}], [{"text": [], "url_based_id": "a"}]]
            for records in cases:
                with self.subTest(records=records):
                    atomic_jsonl(root / "infer.jsonl", records)
                    with self.assertRaises(ValueError):
                        prepare_inference_queries(root / "infer.jsonl", root / "docs.jsonl",
                                                  root / "prepared.jsonl", "url_based_id", "multilabel")


class FinalInferenceControllerTest(unittest.TestCase):
    def setup_run(self, root, infer=True):
        path = fixture(root)
        atomic_jsonl(root / "infer.jsonl", [
            {"prompt": "final-only first", "target_text_id": "doc-0"},
            {"prompt": "final-only second", "target_text_id": "doc-0", "positive_text_ids": ["doc-1", "doc-2"]},
        ])
        if infer:
            atomic_json(path, {**read_json(path), "infer_file": "infer.jsonl", "eval_at_end": False,
                               "eval_every_rounds": 8})
        return path, load_config(path), FixtureWorker()

    def test_only_final_policy_is_evaluated_and_infer_metrics_do_not_select_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, worker = self.setup_run(root)

            def run(command):
                worker(command)
                if "--no-beam-early-stopping" in command:
                    output = Path(command[command.index("--output") + 1])
                    atomic_json(output.with_suffix(".stats.json"), {"metrics": {"mrr@8": 0.99}})

            result = run_pipeline(cfg, runner=run)
            self.assertTrue(result["complete"])
            inference = [c for c in worker.calls if "--no-beam-early-stopping" in c]
            self.assertEqual(len(inference), 1)
            self.assertEqual(worker.calls[-1], inference[0])
            self.assertIn("--evaluation-only", inference[0])
            self.assertEqual(inference[0][inference[0].index("--checkpoint-path") + 1], result["latest"])
            self.assertEqual(result["final_inference"]["metrics"]["mrr@8"], 0.99)
            self.assertEqual(result["best"]["metrics"]["mrr@8"], 0.5)
            self.assertEqual(result["final_inference"]["queries"], 2)
            self.assertFalse(result["final_inference"]["beam_early_stopping"])
            self.assertNotIn("infer_file", result["identity"]["config"])
            self.assertTrue(all(not q["prompt"].startswith("final-only") for q in
                                iter_json_records(root / "run/metadata/train_queries.jsonl")))
            prepared = list(iter_json_records(root / "run/metadata/infer_queries.jsonl"))
            self.assertEqual(prepared[1]["positive_text_ids"], ["doc-1", "doc-2"])
            self.assertEqual(read_json(root / "run/final_inference.json"), result["final_inference"])
            worker.calls.clear()
            run_pipeline(cfg, resume=True, runner=run)
            self.assertFalse(worker.calls)
            atomic_jsonl(root / "infer.jsonl", [{"prompt": "changed", "target_text_id": "doc-0"}])
            with self.assertRaisesRegex(ValueError, "Inference file.*changed"):
                run_pipeline(cfg, resume=True, runner=run)

    def test_interrupted_final_inference_resumes_without_training_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, worker = self.setup_run(root)

            def fail(command):
                if "--no-beam-early-stopping" in command:
                    raise RuntimeError("inference interruption")
                worker(command)

            with self.assertRaisesRegex(RuntimeError, "inference interruption"):
                run_pipeline(cfg, runner=fail)
            self.assertEqual(len(read_json(root / "run/run_manifest.json")["rounds"]), 2)
            original = file_hash(root / "run/round-001/training/latest/model.safetensors")
            worker.calls.clear()
            result = run_pipeline(cfg, resume=True, runner=worker, start_round=1)
            self.assertTrue(result["complete"])
            self.assertEqual(len(worker.calls), 1)
            self.assertIn("--no-beam-early-stopping", worker.calls[0])
            self.assertEqual(file_hash(root / "run/round-001/training/latest/model.safetensors"), original)

    def test_inference_can_be_added_to_completed_pre_upgrade_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, worker = self.setup_run(root, infer=False)
            before = run_pipeline(cfg, runner=worker)
            preserved = {p: file_hash(p) for i in range(2)
                         for p in (root / f"run/round-{i:03d}").rglob("*") if p.is_file()}
            path = root / "run/run_manifest.json"
            state = read_json(path)
            old_hashes = {
                "pretrain/train_iterative_ddro_vault.py": "4d0aae9a82ce5544b5c3200fa7d793a78450096bc675c8bc0a3c5740ffe27d01",
                "pretrain/mine_on_gpus.py": "fe6cbce8e6348979ba2978ca6eba98a9a82a92fe9e3aebf1a3c2b5b8744abea6",
                "scripts/bm25/the_vault/mine_model_confusion_negatives.py": "9692800ba1b4b98e322dd6f699a5a464c353845fa28408c44f95e9bb894b77be",
            }
            for relative, value in old_hashes.items():
                state["identity"]["code"][str(SRC / relative)] = value
            atomic_json(path, state)
            worker.calls.clear()
            result = run_pipeline(cfg, resume=True, runner=worker, start_round=1,
                                  infer_file=root / "infer.jsonl")
            self.assertEqual(result["latest"], before["latest"])
            self.assertEqual(len(worker.calls), 1)
            self.assertEqual({p: file_hash(p) for p in preserved}, preserved)
            with self.assertRaisesRegex(ValueError, "provide the same --infer-file"):
                run_pipeline(cfg, resume=True, runner=worker, start_round=1)

    def test_invalid_inference_is_checked_before_gpu_work_and_can_be_corrected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, worker = self.setup_run(root)
            atomic_jsonl(root / "infer.jsonl", [{"text": "q", "target_text_id": "unknown"}])
            with self.assertRaisesRegex(ValueError, "absent from the retrieval corpus"):
                run_pipeline(cfg, runner=worker)
            self.assertFalse(worker.calls)
            atomic_jsonl(root / "infer.jsonl", [{"text": "fixed", "target_text_id": "doc-0"}])
            self.assertTrue(run_pipeline(cfg, resume=True, runner=worker)["complete"])


@unittest.skipUnless(os.environ.get("RUN_DPO_INTEGRATION") == "1", "Set RUN_DPO_INTEGRATION=1")
class RealFinalInferenceTest(unittest.TestCase):
    def test_real_branch_training_and_generation_disable_early_stopping_only_for_infer(self):
        import torch
        from transformers import T5ForConditionalGeneration
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
                                                   "positive_url_based_id": [target], "family_id": f"family-{i // 2}"}
                                                  for i, target in enumerate(ids)])
            atomic_jsonl(root / "infer.jsonl", [
                {"text": "find C0", "url_based_id": ids[0]},
                {"text": "find C1", "positive_url_based_id": [ids[1], ids[2]]},
            ])
            path = root / "config.json"
            atomic_json(path, {"checkpoint_path": "model", "query_file": "queries.jsonl",
                               "infer_file": "infer.jsonl", "output_dir": "run", "input_format": "multilabel",
                               "target_type": "url_based_id", "queries_per_round": 3, "rounds": None,
                               "steps_per_round": 1, "num_beams": 4, "negatives_per_query": 2,
                               "mining_batch_size": 2, "max_prompt_length": 8, "max_target_length": 8,
                               "precision": "fp32", "device": "cpu", "selection_metric": "mrr@4",
                               "reference_update": "ema", "reference_ema_decay": 1.0,
                               "eval_every_rounds": 8, "eval_at_end": False, "training": {
                                   "preference_objective": "branch-dpo", "branch_loss_weight": 0.1,
                                   "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                                   "dataset_num_proc": 1, "dataloader_num_workers": 0,
                                   "save_steps": 1, "logging_steps": 1}})
            active_infer, seen, calls = [False], [], []
            actual_generate = T5ForConditionalGeneration.generate

            def generate(model, *args, **kwargs):
                seen.append((active_infer[0], kwargs["early_stopping"]))
                return actual_generate(model, *args, **kwargs)

            def runner(command):
                calls.append(command)
                if "--checkpoint-path" in command:
                    active_infer[0] = "--no-beam-early-stopping" in command
                    miner.mine(miner.build_parser().parse_args(command[2:]))
                elif "--decay" in command:
                    reference.update_reference(**vars(reference.build_parser().parse_args(command[2:])))
                else:
                    with patch.object(sys, "argv", command[1:]):
                        training.train_round(training.parse_args())

            cfg = load_config(path)
            with patch.object(T5ForConditionalGeneration, "generate", generate):
                result = run_pipeline(cfg, runner=runner)
                self.assertTrue(result["complete"])
                self.assertTrue(any(infer for infer, _ in seen))
                self.assertTrue(any(not infer for infer, _ in seen))
                self.assertTrue(all(early is (not infer) for infer, early in seen))
                self.assertEqual(result["final_inference"]["checkpoint"], result["latest"])
                stats = read_json(root / "run/round-001/infer_predictions.stats.json")
                self.assertFalse(stats["beam_early_stopping"])
                self.assertEqual(stats["queries_processed"], 2)
                calls.clear()
                run_pipeline(cfg, resume=True, runner=runner, start_round=1)
                self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
