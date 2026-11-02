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
from utils.grpo_data import prepare_url_file


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
        self.assertEqual(score_ranking(["negative", "positive-b", "positive-a"], ["positive-a", "positive-b"]),
                         {"hit1": 0., "hit5": 1., "hit10": 1., "mrr1": 0., "mrr5": .5, "mrr10": .5})
        ranked = [f"negative-{i}" for i in range(6)] + ["positive"]
        self.assertEqual(score_ranking(ranked, ["positive"]),
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


if __name__ == "__main__":
    unittest.main()
