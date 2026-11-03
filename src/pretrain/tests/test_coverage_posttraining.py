"""Independent preference gradients, missed-positive recovery and suite regressions."""

import json
from math import exp, log, log2
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretrain.coverage_experiments import prepare, summarize, transitions
from pretrain.calibrate_grpo_batch_size import probe_once
from pretrain.infer_grpo_vault_url import run_inference
from pretrain.train_coverage_vault import main as train_main, selection_score
from pretrain.train_grpo_vault import build_trainer
from scripts.bm25.the_vault.common import write_jsonl
import test_grpo as fixtures
from utils.coverage_trainer import CoveragePreferenceTrainer, CoverageSettings, OBJECTIVES, preference_loss, select_candidates
from utils.retrieval_metrics import score_ranking


class CoverageMathTests(unittest.TestCase):
    def test_metrics_distinguish_first_hit_from_multi_positive_coverage(self):
        a = score_ranking(["a", "x", "x", "y"], ["a", "b", "c", "d"])
        b = score_ranking(["a", "b", "c", "d"], ["a", "b", "c", "d"])
        self.assertEqual(a["hit1"], b["hit1"])
        self.assertEqual(a["mrr10"], b["mrr10"])
        self.assertEqual(a["recall10"], .25)
        self.assertEqual(b["recall10"], 1.)
        self.assertEqual(b["relevant_count10"], 4.)
        self.assertEqual(b["precision10"], .4)
        self.assertAlmostEqual(a["ndcg10"], 1 / sum(1 / log2(i + 2) for i in range(4)))

    def test_preference_matches_independent_logistic_and_multi_negative_formulas(self):
        for name in OBJECTIVES:
            scores = torch.tensor([-3., -4., -2., -5.], requires_grad=True)
            reference = torch.tensor([-4., -4., -2., -5.], requires_grad=True)
            loss = preference_loss(scores, reference, 2, torch.tensor([1., 2.]), name, .1)
            if name == "multi_negative_dpo":
                expected = (log(1 + 2 * exp(-.1)) + 2 * log(3)) / 3
            else:
                expected = (log(1 + exp(-.1)) + 2 * log(2)) / 3
            self.assertAlmostEqual(loss.item(), expected, places=6)
            loss.backward()
            self.assertTrue(scores.grad[:2].lt(0).all())
            self.assertTrue(scores.grad[2:].gt(0).all())
            self.assertIsNone(reference.grad)

    def test_positives_do_not_compete_in_multi_negative_loss(self):
        gradients = []
        for first in (-10., 10.):
            scores = torch.tensor([first, -2., -1.], requires_grad=True)
            loss = preference_loss(scores, torch.zeros(3), 2, torch.ones(2), "multi_negative_dpo", .1)
            loss.backward()
            gradients.append(scores.grad[1].item())
        self.assertEqual(gradients[0], gradients[1])

    def test_missing_positive_selection_and_false_negative_protection(self):
        settings = CoverageSettings(do_sample=False, positives_per_query=2, negatives_per_query=2,
                                    objective="missed_positive_dpo")
        for seed in range(10):
            group = select_candidates(["a", "b", "c"], ["a", "x", "b"], ["a", "b", "c", "x", "y"],
                                      settings, random.Random(seed))
            self.assertIn("c", group["positives"])
            self.assertFalse(set(group["negatives"]) & {"a", "b", "c"})
            self.assertEqual(len(set(group["positives"])), 2)
            self.assertEqual(group["weights"][group["positives"].index("c")], 2.)
        # All-correct beams still get safe negatives from the inventory.
        group = select_candidates(["a", "b"], ["a", "b"], ["a", "b", "x"], settings, random.Random(1))
        self.assertEqual(group["negatives"], ["x"])
        group = select_candidates(["a", "b"], ["a", "b"], ["a", "b"], settings, random.Random(1))
        self.assertEqual(group["negatives"], [])

    def test_guard_rejects_coverage_gain_with_excessive_hit1_loss(self):
        self.assertEqual(selection_score({"hit@1": .39, "recall@10": .9}, .4, .005), -1.)
        self.assertEqual(selection_score({"hit@1": .397, "recall@10": .2}, .4, .005), .2)

    def test_no_negative_preference_term_is_connected_zero(self):
        scores = torch.tensor([-2., -3.], requires_grad=True)
        value = preference_loss(scores, torch.zeros(2), 2, torch.ones(2), "multi_negative_dpo", .1)
        value.backward()
        self.assertEqual(value.item(), 0.)
        self.assertTrue(scores.grad.eq(0).all())


class CoveragePipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TinyT5Tests()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def settings(self, objective):
        return CoverageSettings(objective=objective, do_sample=False, num_generations=3,
                                eval_num_beams=3, max_target_length=3, negatives_per_query=1)

    def test_all_wrong_rollouts_still_update_annotated_positives_and_freeze_reference(self):
        for name in OBJECTIVES:
            args = self.fixture.arguments(name, ["--do_sample", "false", "--num_generations", "3"])
            trainer = build_trainer(args, settings=self.settings(name), trainer_class=CoveragePreferenceTrainer,
                                    require_reference=True)
            inputs = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[0]]))
            generated = torch.tensor([[0, 6, 1]] * 3, device=trainer.args.device)  # gamma, all wrong
            before = {key: value.detach().clone() for key, value in trainer.reference_model.state_dict().items()}
            optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.01)
            trainer.model.train()
            with patch.object(trainer, "generate_sequences", return_value=generated):
                initial = trainer.compute_loss(trainer.model, inputs)
                self.assertTrue(torch.isfinite(initial))
                initial.backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for p in trainer.model.parameters() if p.grad is not None), 0)
                optimizer.step()
                optimizer.zero_grad()
                improved = trainer.compute_loss(trainer.model, inputs)
                self.assertLess(improved.item(), initial.item())
            for key, value in trainer.reference_model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in trainer.reference_model.parameters()))

    def test_real_training_saves_candidate_and_preserves_sft_when_recall_cannot_improve(self):
        baseline = AutoModelForSeq2SeqLM.from_pretrained(self.root / "sft").state_dict()
        for name in OBJECTIVES:
            argv = ["train", "--objective", name, "--checkpoint_path", str(self.root / "sft"),
                    "--train_file", str(self.root / "train.jsonl"), "--eval_file", str(self.root / "eval.jsonl"),
                    "--corpus_file", str(self.root / "corpus.jsonl"), "--output_dir", str(self.root / name),
                    "--max_prompt_length", "16", "--max_target_length", "3", "--num_generations", "3",
                    "--eval_num_beams", "3", "--max_steps", "2", "--eval_steps", "1", "--save_steps", "1",
                    "--logging_steps", "1", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "2",
                    "--negatives_per_query", "1", "--learning_rate", ".001", "--warmup_ratio", "0",
                    "--no-gradient_checkpointing"]
            with patch.object(sys, "argv", argv):
                train_main()
            report = json.loads((self.root / name / "selection.json").read_text())
            self.assertEqual(report["selected"], "sft")  # All corpus IDs fit into top ten already.
            selected = AutoModelForSeq2SeqLM.from_pretrained(self.root / name / "final").state_dict()
            candidate = AutoModelForSeq2SeqLM.from_pretrained(self.root / name / "candidate_final").state_dict()
            self.assertTrue(any(not torch.equal(candidate[key], baseline[key]) for key in baseline))
            for key in baseline:
                torch.testing.assert_close(selected[key], baseline[key], rtol=0, atol=0)
            self.assertTrue((self.root / name / "checkpoint-1/trainer_state.json").is_file())

    def test_fixed_inventory_preparation_does_not_import_test_positive_labels(self):
        native = self.root / "native.jsonl"
        test = self.root / "test.jsonl"
        write_jsonl(native, [{"text": f"find {i}", "url_based_id": "alpha", "positive_url_based_id": ["alpha", "beta"]}
                            for i in range(4)])
        write_jsonl(test, [{"text": "find test", "url_based_id": "gamma", "positive_url_based_id": ["label-only"]}])
        args = SimpleNamespace(output_dir=str(self.root / "suite"), train_input_file=str(native),
                               test_input_file=str(test), eval_input_file=None, corpus_file=None, validation_split=.25, seed=42)
        prepare(args)
        catalog = (self.root / "suite/data/corpus.jsonl").read_text()
        self.assertIn("gamma", catalog)
        self.assertNotIn("label-only", catalog)
        self.assertNotIn("find test", (self.root / "suite/data/train.jsonl").read_text())
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare(args)

    def test_fixed_inference_keeps_unreachable_labels_in_denominator(self):
        write_jsonl(self.root / "fixed.jsonl", [{"target": "alpha"}])
        write_jsonl(self.root / "test.jsonl", [{"text": "find test", "url_based_id": "beta",
                                              "positive_url_based_id": ["alpha", "beta"]}])
        args = SimpleNamespace(model_path=str(self.root / "sft"), test_file=str(self.root / "test.jsonl"),
                               corpus_file=str(self.root / "fixed.jsonl"), output_dir=str(self.root / "infer"),
                               batch_size=1, num_beams=10, max_prompt_length=16, max_target_length=3,
                               catalog_mode="fixed", score_normalization="trie", fp16=False, device="cpu")
        result = run_inference(args)
        self.assertEqual(result["recall10"], .5)
        self.assertEqual(result["relevant_count10"], 1.)
        self.assertEqual(result["test_catalog_targets_added"], 0)
        self.assertEqual(result["missing_positive_targets"], 1)

    def test_preference_calibration_includes_anchor_reference_and_resident_gradients(self):
        args = SimpleNamespace(checkpoint_path=str(self.root / "sft"), beta=0., reserve_gib=0.,
                               num_gpus=1, fp16=False, max_prompt_length=8, num_generations=3,
                               do_sample="false", temperature=1., score_normalization="trie",
                               ranking_loss_weight=0., gradient_accumulation_steps=2,
                               preference_objective="multi_negative_dpo", positives_per_query=2,
                               negatives_per_query=1, preference_beta=.1, sft_anchor_weight=.1)
        with patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.synchronize"), \
                patch("torch.cuda.max_memory_allocated", return_value=123):
            self.assertEqual(probe_once(args, self.fixture.index, torch.device("cpu"), 2), (True, 123))

    def test_summary_keeps_candidate_failures_visible_and_uses_validation_fallback(self):
        root = self.root / "summary"
        baseline_path = root / "sft_baseline"
        candidate_path = root / "multi_positive_dpo/candidate_inference"
        baseline_path.mkdir(parents=True)
        candidate_path.mkdir(parents=True)
        common = {"num_queries": 1, "corpus_targets": 3, "num_beams": 3,
                  "score_normalization": "trie", "catalog_mode": "fixed"}
        baseline = {**score_ranking(["a", "x", "b"], ["a", "b"]), **common}
        candidate = {**score_ranking(["x", "a", "b"], ["a", "b"]), **common}
        for path, metrics in ((baseline_path, baseline), (candidate_path, candidate)):
            (path / "metrics.json").write_text(json.dumps(metrics))
            write_jsonl(path / "predictions.jsonl", [{"query_index": 0, "query": "q",
                                                       "positive_url_based_id": ["a", "b"], **metrics}])
        (candidate_path.parent / "selection.json").write_text(json.dumps({"selected": "sft"}))
        summarize(SimpleNamespace(output_dir=str(root), experiments=["multi_positive_dpo"]))
        report = json.loads((root / "comparison.json").read_text())
        experiment = report["experiments"]["multi_positive_dpo"]
        self.assertEqual(experiment["candidate_delta"]["hit1"], -1.)
        self.assertEqual(experiment["selected_delta"]["hit1"], 0.)
        self.assertEqual(experiment["query_transitions"]["hit1_lost_queries"], 1)
        self.assertFalse(report["selection_uses_test_metrics"])


if __name__ == "__main__":
    unittest.main()
