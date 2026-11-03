"""Native URL data and post-training inference regression tests (offline CPU)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretrain.infer_grpo_vault_url import METRIC_NAMES, run_inference, score_ranking
from pretrain.tests.test_grpo import tiny_checkpoint
from pretrain.train_grpo_vault import build_trainer, parse_args
from scripts.bm25.the_vault.common import iter_json_records, write_jsonl
from scripts.grpo.prepare_vault_url_grpo_data import main as prepare_main
from utils.grpo_data import filter_query_targets, prepare_url_file
from utils.grpo_trainer import TargetIndex


class DirectURLDataTests(unittest.TestCase):
    def test_direct_labels_catalog_deduplication_and_code_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.jsonl"
            write_jsonl(path, [
                {"numeric_id": "unused", "semantic_id": "wrong", "text": "Query: Find A",
                 "url_based_id": "own-a", "positive_url_based_id": ["positive-a", "positive-b"]},
                {"text": " find   a ", "url_based_id": "own-b", "positive_url_based_id": "positive-c"},
                {"text": "Code: def a; end", "url_based_id": "code-only"},
            ])
            rows, catalog = prepare_url_file(path)
            self.assertEqual(rows, [{"prompt": "Find A", "positive_targets": ["positive-a", "positive-b", "positive-c"]}])
            self.assertEqual(set(catalog), {"own-a", "own-b", "positive-a", "positive-b", "positive-c", "code-only"})
            # Test labels must not populate the inference candidate catalog.
            _, catalog = prepare_url_file(path, include_positive_targets_in_corpus=False)
            self.assertEqual(set(catalog), {"own-a", "own-b", "code-only"})

    def test_invalid_labels_fail_instead_of_silently_using_own_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.jsonl"
            for labels in (None, [], [""], [{"url": "a"}], [42]):
                write_jsonl(path, [{"text": "find a", "url_based_id": "a", "positive_url_based_id": labels}])
                with self.assertRaisesRegex(ValueError, "positive_url_based_id"):
                    prepare_url_file(path)

    def test_hit_and_truncated_mrr_use_first_of_all_positives(self):
        legacy_names = ("hit1", "hit5", "hit10", "mrr1", "mrr5", "mrr10")
        scores = score_ranking(["negative", "positive-b", "positive-a"], ["positive-a", "positive-b"])
        self.assertEqual({key: scores[key] for key in legacy_names},
                         {"hit1": 0., "hit5": 1., "hit10": 1., "mrr1": 0., "mrr5": .5, "mrr10": .5})
        ranked = [f"negative-{i}" for i in range(6)] + ["positive"]
        scores = score_ranking(ranked, ["positive"])
        self.assertEqual({key: scores[key] for key in legacy_names},
                         {"hit1": 0., "hit5": 0., "hit10": 1., "mrr1": 0., "mrr5": 0., "mrr10": 1 / 7})
        self.assertTrue(all(value == 0 for value in score_ranking([], ["positive"]).values()))
        with self.assertRaises(ValueError):
            score_ranking(["a"], [])


class URLInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        tiny_checkpoint(self.root / "sft")

    def tearDown(self):
        self.temporary.cleanup()

    def inference_args(self, model="sft", corpus="corpus.jsonl", output="inference"):
        return SimpleNamespace(
            model_path=str(self.root / model), test_file=str(self.root / "test.json"),
            corpus_file=str(self.root / corpus), output_dir=str(self.root / output),
            batch_size=2, num_beams=10, max_prompt_length=16, max_target_length=3,
            device="cpu", fp16=False,
        )

    def test_native_json_inference_metrics_predictions_and_missing_positives(self):
        write_jsonl(self.root / "corpus.jsonl", [{"target": "alpha"}])
        test_rows = [
            {"text": "find first", "url_based_id": "alpha", "positive_url_based_id": ["alpha", "beta"]},
            {"text": "Query: find second", "url_based_id": "beta", "positive_url_based_id": ["gamma"]},
            {"text": "find third", "url_based_id": "gamma", "positive_url_based_id": ["not-in-corpus"]},
            {"text": "QUERY: find first", "url_based_id": "alpha", "positive_url_based_id": ["beta"]},
            {"text": "Code: irrelevant document text", "url_based_id": "gamma"},
        ]
        (self.root / "test.json").write_text(json.dumps(test_rows), encoding="utf-8")
        with patch.dict(os.environ, RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"):
            result = run_inference(self.inference_args())
        self.assertEqual(result["num_queries"], 3)
        self.assertEqual(result["test_catalog_targets_added"], 2)
        self.assertEqual(result["corpus_targets"], 3)
        self.assertEqual(result["missing_positive_targets"], 1)
        self.assertEqual(result["queries_with_missing_positive_targets"], 1)
        self.assertAlmostEqual(result["hit5"], 2 / 3)
        self.assertAlmostEqual(result["hit10"], 2 / 3)
        predictions = list(iter_json_records(self.root / "inference/predictions.jsonl"))
        self.assertEqual([row["query_index"] for row in predictions], [0, 1, 2])
        for name in METRIC_NAMES:
            self.assertAlmostEqual(result[name], sum(row[name] for row in predictions) / 3)
        self.assertEqual(json.loads((self.root / "inference/metrics.json").read_text()), result)
        self.assertFalse(list((self.root / "inference").glob(".predictions.rank*")))

    def test_native_preparation_training_and_final_model_inference(self):
        native = self.root / "native.jsonl"
        write_jsonl(native, [
            {"text": f"find {query}", "url_based_id": "alpha", "positive_url_based_id": ["alpha", "beta"]}
            for query in ["first", "second", "third", "fourth", "validation"]
        ] + [{"text": "Code: gamma body", "url_based_id": "gamma"}])
        with patch.object(sys, "argv", ["prepare", "--input-file", str(native), "--output-dir", str(self.root / "data"), "--validation-split", ".2"]):
            prepare_main()
        self.assertEqual(len(list(iter_json_records(self.root / "data/train.jsonl"))), 4)
        self.assertEqual(len(list(iter_json_records(self.root / "data/validation.jsonl"))), 1)
        arguments = [
            "train", "--checkpoint_path", str(self.root / "sft"),
            "--train_file", str(self.root / "data/train.jsonl"), "--eval_file", str(self.root / "data/validation.jsonl"),
            "--corpus_file", str(self.root / "data/corpus.jsonl"), "--output_dir", str(self.root / "trained"),
            "--max_prompt_length", "16", "--max_target_length", "3", "--num_generations", "4",
            "--max_steps", "1", "--save_steps", "1", "--eval_steps", "1", "--logging_steps", "1",
            "--gradient_accumulation_steps", "1",
        ]
        with patch.object(sys, "argv", arguments):
            trainer = build_trainer(parse_args())
        trainer.args.disable_tqdm = True
        trainer.train()
        trainer.save_model(str(self.root / "trained/final"))
        (self.root / "test.json").write_text(json.dumps([
            {"text": "find first", "url_based_id": "alpha", "positive_url_based_id": ["alpha", "beta"]},
            {"text": "find second", "url_based_id": "gamma", "positive_url_based_id": ["gamma"]},
        ]), encoding="utf-8")
        with patch.dict(os.environ, RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"):
            result = run_inference(self.inference_args(model="trained/final", corpus="data/corpus.jsonl"))
        self.assertEqual(result["hit10"], 1.)
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["missing_positive_targets"], 0)


class InvalidURLTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer, self.config = tiny_checkpoint(self.root / "sft")

    def tearDown(self):
        self.temporary.cleanup()

    def test_skip_excludes_all_collisions_bad_tokens_and_long_ids(self):
        targets = ["alpha", "beta", "alpha alpha", "alpha  alpha", "alpha beta gamma", "absent",
                   "alpha [PAD]", "alpha [EOS] beta", "alpha\nbeta", "alpha\ud800", "\ufffd"]
        index = TargetIndex(self.tokenizer, self.config, targets, 3, invalid_target_policy="skip")
        self.assertEqual(set(index.target_to_sequence), {"alpha", "beta"})
        self.assertEqual(index.filtering_summary["excluded_targets"], 9)
        self.assertEqual(index.filtering_summary["excluded_by_reason"]["tokenizer_collision"], 2)
        self.assertEqual(index.excluded_targets["alpha beta gamma"]["token_length"], 4)
        self.assertEqual(index.excluded_targets["alpha beta gamma"]["reason"], "overlong")
        self.assertEqual(index.excluded_targets["alpha\ud800"]["reason"], "invalid_characters")
        index.write_filter_report(self.root / "reports")
        exclusions = list(iter_json_records(self.root / "reports/excluded_targets.jsonl"))
        self.assertEqual({row["target"] for row in exclusions}, set(targets) - {"alpha", "beta"})
        with self.assertRaisesRegex(ValueError, "No usable"):
            TargetIndex(self.tokenizer, self.config, ["alpha alpha", "alpha  alpha"], 3, invalid_target_policy="skip")
        long_target = " ".join(["alpha"] * 194)
        long_index = TargetIndex(self.tokenizer, self.config, ["alpha", long_target], 128, invalid_target_policy="skip")
        self.assertEqual(long_index.excluded_targets[long_target]["token_length"], 195)
        self.assertEqual(set(long_index.target_to_sequence), {"alpha"})

    def test_filtering_preserves_remaining_positives_and_counts_dropped_queries(self):
        rows, stats = filter_query_targets([
            {"prompt": "first", "positive_targets": ["alpha", "overlong"]},
            {"prompt": "second", "positive_targets": ["overlong"]},
            {"prompt": "third", "positive_targets": ["beta"]},
        ], {"alpha", "beta"})
        self.assertEqual(rows, [{"prompt": "first", "positive_targets": ["alpha"]},
                                {"prompt": "third", "positive_targets": ["beta"]}])
        self.assertEqual(stats["dropped_queries"], 1)
        self.assertEqual(stats["removed_positive_labels"], 2)

    def test_skip_mode_native_prepare_train_and_unfiltered_test_metrics(self):
        bad = "alpha beta gamma"
        surrogate = "alpha\ud800"
        native = self.root / "native.jsonl"
        write_jsonl(native, [
            {"text": "find first", "url_based_id": "alpha", "positive_url_based_id": ["alpha", bad, "absent"]},
            {"text": "find second", "url_based_id": "beta", "positive_url_based_id": ["beta"]},
            {"text": "find third", "url_based_id": bad, "positive_url_based_id": [bad]},
            {"text": "find fourth", "url_based_id": "alpha alpha", "positive_url_based_id": ["alpha alpha", "alpha  alpha"]},
            {"text": "find validation", "url_based_id": surrogate, "positive_url_based_id": [surrogate]},
        ], ensure_ascii=True)
        with patch.object(sys, "argv", ["prepare", "--input-file", str(native), "--output-dir", str(self.root / "data"), "--validation-split", "0"]):
            prepare_main()
        write_jsonl(self.root / "validation.jsonl", [{"prompt": "find gamma", "positive_targets": ["beta", bad]}])
        arguments = [
            "train", "--checkpoint_path", str(self.root / "sft"),
            "--train_file", str(self.root / "data/train.jsonl"), "--eval_file", str(self.root / "validation.jsonl"),
            "--corpus_file", str(self.root / "data/corpus.jsonl"), "--output_dir", str(self.root / "trained"),
            "--max_prompt_length", "16", "--max_target_length", "3", "--num_generations", "4",
            "--max_steps", "1", "--save_steps", "1", "--eval_steps", "1", "--logging_steps", "1",
            "--gradient_accumulation_steps", "1", "--invalid_target_policy", "skip",
        ]
        with patch.object(sys, "argv", arguments):
            trainer = build_trainer(parse_args())
        self.assertEqual(len(trainer.train_dataset), 2)
        self.assertEqual(trainer.train_dataset[0]["positive_targets"], ["alpha"])
        self.assertEqual(trainer.eval_dataset[0]["positive_targets"], ["beta"])
        report = json.loads((self.root / "trained/query_filtering.json").read_text())
        self.assertEqual(report["train"]["dropped_queries"], 3)
        trainer.args.disable_tqdm = True
        trainer.train()
        trainer.save_model(str(self.root / "trained/final"))
        test_rows = [
            {"text": "find first", "url_based_id": "alpha", "positive_url_based_id": ["alpha", bad]},
            {"text": "find second", "url_based_id": bad, "positive_url_based_id": [bad]},
            {"text": "find third", "url_based_id": "alpha alpha", "positive_url_based_id": ["alpha alpha", "alpha  alpha"]},
            {"text": "find fourth", "url_based_id": surrogate, "positive_url_based_id": [surrogate]},
        ]
        (self.root / "test.json").write_text(json.dumps(test_rows), encoding="utf-8")
        args = SimpleNamespace(
            model_path=str(self.root / "trained/final"), test_file=str(self.root / "test.json"),
            corpus_file=str(self.root / "data/corpus.jsonl"), output_dir=str(self.root / "inference"),
            batch_size=2, num_beams=10, max_prompt_length=16, max_target_length=3,
            device="cpu", fp16=False, invalid_target_policy="skip",
        )
        with patch.dict(os.environ, RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"):
            metrics = run_inference(args)
        self.assertEqual(metrics["num_queries"], 4)
        self.assertEqual(metrics["corpus_targets"], 2)
        self.assertEqual(metrics["queries_with_no_retrievable_positive"], 3)
        self.assertEqual(metrics["hit10"], .25)
        predictions = list(iter_json_records(self.root / "inference/predictions.jsonl"))
        self.assertEqual(predictions[3]["positive_url_based_id"], [surrogate])
        self.assertTrue((self.root / "inference/excluded_targets.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
