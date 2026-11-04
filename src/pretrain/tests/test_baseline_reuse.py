"""Completed baseline reuse must preserve comparisons and reject incompatible inputs."""

from copy import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretrain.coverage_experiments import baseline_artifacts, fingerprint
from scripts.bm25.the_vault.common import write_jsonl
from utils.retrieval_metrics import score_ranking


class BaselineReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = self.root / "old"
        self.source = self.old / "sft_baseline"
        self.source.mkdir(parents=True)
        (self.old / "data").mkdir()
        (self.root / "sft").mkdir()
        write_jsonl(self.root / "test.jsonl", [{"text": "q", "url_based_id": "a", "positive_url_based_id": ["a"]}])
        write_jsonl(self.old / "data/corpus.jsonl", [{"target": target} for target in ("a", "b", "c")])
        write_jsonl(self.root / "new-corpus.jsonl", [{"target": target} for target in ("c", "a", "b")])
        scores = score_ranking(["a", "b"], ["a"])
        metrics = {**scores, "num_queries": 1, "corpus_targets": 3, "num_beams": 3,
                   "model_path": str(self.root / "sft"), "score_normalization": "trie", "catalog_mode": "fixed",
                   "target_filtering": {"policy": "skip", "max_target_length": 128}}
        (self.source / "metrics.json").write_text(json.dumps(metrics))
        write_jsonl(self.source / "predictions.jsonl", [{"query_index": 0, "query": "q",
                                                        "positive_url_based_id": ["a"], "top_10_predictions": ["a", "b"], **scores}])
        (self.old / "suite_data.json").write_text(json.dumps({"test": fingerprint(self.root / "test.jsonl")}))
        (self.old / "multi_positive_dpo").mkdir()
        (self.old / "multi_positive_dpo/grpo_run.json").write_text(json.dumps({"arguments": {
            "max_prompt_length": 256, "max_target_length": 128, "fp16": True}}))
        self.args = SimpleNamespace(source_dir=str(self.source), destination_dir=str(self.root / "new/sft_baseline"),
                                    checkpoint_path=str(self.root / "sft"), test_input_file=str(self.root / "test.jsonl"),
                                    corpus_file=str(self.root / "new-corpus.jsonl"), max_prompt_length=256,
                                    max_target_length=128, num_beams=10, precision="fp16", invalid_target_policy="skip")

    def tearDown(self):
        self.temp.cleanup()

    def record(self):
        args = copy(self.args)
        args.source_dir = None
        args.destination_dir = str(self.source)
        baseline_artifacts(args)

    def test_legacy_suite_reused_without_decoding_and_catalog_order_does_not_matter(self):
        original = (self.source / "predictions.jsonl").read_bytes()
        baseline_artifacts(self.args)
        destination = Path(self.args.destination_dir)
        self.assertEqual((destination / "predictions.jsonl").read_bytes(), original)
        self.assertEqual((self.source / "predictions.jsonl").read_bytes(), original)
        self.assertEqual(json.loads((destination / "metrics.json").read_text())["hit1"], 1.)
        self.assertEqual(json.loads((destination / "baseline_provenance.json").read_text())["reused_from"], str(self.source.resolve()))

    def test_modern_cache_rejects_decoding_and_precision_changes(self):
        self.record()
        for name, value in (("max_prompt_length", 128), ("max_target_length", 64), ("precision", "fp32"),
                            ("invalid_target_policy", "error"), ("num_beams", 2)):
            with self.subTest(name=name):
                args = copy(self.args)
                setattr(args, name, value)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    baseline_artifacts(args)
                self.assertFalse(Path(args.destination_dir).exists())

    def test_changed_test_data_or_inventory_is_rejected(self):
        self.record()
        for kind in ("test", "corpus"):
            args = copy(self.args)
            if kind == "test":
                path = self.root / "different-test.jsonl"
                write_jsonl(path, [{"text": "other", "url_based_id": "a", "positive_url_based_id": ["a"]}])
                args.test_input_file = str(path)
            else:
                path = self.root / "different-corpus.jsonl"
                write_jsonl(path, [{"target": target} for target in ("a", "b", "d")])
                args.corpus_file = str(path)
            with self.assertRaisesRegex(ValueError, "mismatched"):
                baseline_artifacts(args)

    def test_changed_checkpoint_and_modified_cached_metrics_are_rejected(self):
        self.record()
        args = copy(self.args)
        args.checkpoint_path = str(self.root / "another-sft")
        with self.assertRaisesRegex(ValueError, "model_path"):
            baseline_artifacts(args)
        metrics = json.loads((self.source / "metrics.json").read_text())
        metrics["hit1"] = .5
        (self.source / "metrics.json").write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "changed since"):
            baseline_artifacts(self.args)

    def test_partial_prediction_outputs_are_rejected(self):
        (self.source / "predictions.jsonl").unlink()
        write_jsonl(self.source / ".predictions.rank0.jsonl", [{"query": "q"}])
        with self.assertRaisesRegex(ValueError, "completed predictions"):
            baseline_artifacts(self.args)


if __name__ == "__main__":
    unittest.main()
