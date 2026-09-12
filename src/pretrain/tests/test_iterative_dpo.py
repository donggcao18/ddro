"""Small controller/data tests; no ML dependencies or model downloads needed."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import iter_json_records
from data.data_prep.prepare_vault_dpo_metadata import prepare
from mine_model_confusion_negatives import retrieval_metrics
from pretrain.iterative_dpo_utils import (
    atomic_json, atomic_jsonl, checkpoint_hash, file_hash, latest_resumable_checkpoint,
    read_json, round_queries,
)
from pretrain.train_iterative_ddro_vault import audit_pairs, load_config, run_pipeline


def fake_checkpoint(path: Path, weights="initial"):
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "config.json", {"model_type": "t5"})
    atomic_json(path / "tokenizer.json", {"vocab": ["a", "b"]})
    (path / "model.safetensors").write_text(weights, encoding="utf-8")


def fixture(root: Path):
    corpus = root / "corpus.jsonl"
    queries = root / "queries.jsonl"
    atomic_jsonl(corpus, [{"text_id": f"doc-{i}", "numeric_id": str(i),
                          "url_based_id": f"repo/{i}", "structure_id_v3": f"path-{i}",
                          "text": "Query: held-out label must not be read"} for i in range(8)])
    atomic_jsonl(queries, [{"prompt": f"find function {i} variant {j}",
                           "target_text_id": f"doc-{i}", "family_id": f"source-{i}"}
                          for i in range(8) for j in range(2)])
    fake_checkpoint(root / "sft")
    config = root / "config.json"
    atomic_json(config, {"checkpoint_path": "sft", "corpus_files": ["corpus.jsonl"],
                         "query_file": "queries.jsonl", "output_dir": "run",
                         "rounds": 2, "queries_per_round": 5, "steps_per_round": 2,
                         "precision": "fp32", "num_beams": 8, "negatives_per_query": 2,
                         "selection_metric": "mrr@8"})
    return config


class FixtureWorker:
    """Implements the subprocess artifact contract, without pretending to train."""
    def __init__(self):
        self.calls = []
        self.fail_training = False

    def __call__(self, command):
        self.calls.append(command)
        def option(name):
            return command[command.index(name) + 1]
        if "--checkpoint-path" in command:
            queries = list(iter_json_records(option("--query-metadata")))
            output = Path(option("--output"))
            if "--evaluation-only" in command:
                atomic_jsonl(output, [{"query_key": q["query_key"], "predicted_text_ids": [q["target_text_id"]]}
                                     for q in queries])
                atomic_json(output.with_suffix(".stats.json"), {"metrics": {"mrr@8": 0.5}})
            else:
                documents = list(iter_json_records(option("--document-metadata")))
                pairs = []
                for query in queries:
                    rejected = next(d["text_id"] for d in documents if d["text_id"] not in query["positive_text_ids"])
                    pairs.append({"query_key": query["query_key"], "prompt": query["prompt"],
                                  "chosen_text_id": query["target_text_id"], "chosen": query["target_text_id"],
                                  "rejected_text_id": rejected, "rejected": rejected,
                                  "positive_text_ids": query["positive_text_ids"],
                                  "negative_source": "model_confusion", "round_id": int(option("--round-id")),
                                  "policy_fingerprint": option("--checkpoint-fingerprint")})
                atomic_jsonl(output, pairs)
                atomic_json(output.with_suffix(".stats.json"), {"model_pairs_written": len(pairs)})
        else:
            if self.fail_training:
                raise RuntimeError("Simulated training interruption")
            manifest = Path(option("--round_manifest"))
            output = Path(option("--output_dir"))
            fake_checkpoint(output / "latest", weights=f"round {read_json(manifest)['round_id']}")
            atomic_json(output / "training_metrics.json", {"train_loss": 0.5})
            atomic_json(output / "latest/round_complete.json", {
                "identity": file_hash(manifest), "global_step": 2,
                "checkpoint_fingerprint": checkpoint_hash(output / "latest"),
            })


class MetadataTest(unittest.TestCase):
    def test_no_corpus_labels_or_transitive_relevance_and_family_disjointness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(fixture(root))
            # q1 links A/B and q2 links B/C. q1 must not gain C as a positive.
            atomic_jsonl(root / "queries.jsonl", [
                {"prompt": "q1", "target_text_id": "doc-0", "positive_text_ids": ["doc-0", "doc-1"]},
                {"prompt": "q2", "target_text_id": "doc-1", "positive_text_ids": ["doc-1", "doc-2"]},
                {"prompt": "independent", "target_text_id": "doc-3"},
                {"prompt": "another", "target_text_id": "doc-4"},
            ])
            prepare(config["corpus_files"], config["query_file"], str(root / "meta"), 0.25)
            train = list(iter_json_records(root / "meta/train_queries.jsonl"))
            val = list(iter_json_records(root / "meta/validation_queries.jsonl"))
            q1 = next(r for r in train + val if r["prompt"] == "q1")
            self.assertEqual(q1["positive_text_ids"], ["doc-0", "doc-1"])
            self.assertFalse({r["family_id"] for r in train} & {r["family_id"] for r in val})
            self.assertFalse({p for r in train for p in r["positive_text_ids"]} &
                             {p for r in val for p in r["positive_text_ids"]})

    def test_rotation_covers_queries_before_wrap_without_duplicates(self):
        rows = [{"query_key": str(i)} for i in range(10)]
        subsets = [round_queries(rows, 4, r, 7) for r in range(3)]
        flattened = [row["query_key"] for group in subsets for row in group]
        self.assertEqual(len(set(flattened[:10])), 10)
        self.assertTrue(all(len({r["query_key"] for r in group}) == 4 for group in subsets))
        self.assertEqual(subsets[1], round_queries(list(reversed(rows)), 4, 1, 7))

    def test_ranking_metrics_keep_invalid_positions_and_deduplicate_recall(self):
        metrics = retrieval_metrics([None, "yes", "yes", "no"], {"yes", "other"}, [1, 4])
        self.assertEqual(metrics["mrr@4"], 0.5)
        self.assertEqual(metrics["recall@4"], 0.5)
        self.assertEqual(metrics["valid_generation_rate"], 0.75)

    def test_raw_numeric_queries_resolve_without_importing_corpus_query_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            atomic_jsonl(root / "queries.jsonl", [{"text": f"query {i}", "text_id": i} for i in range(8)])
            prepare(cfg["corpus_files"], cfg["query_file"], str(root / "meta"))
            rows = list(iter_json_records(root / "meta/train_queries.jsonl"))
            self.assertTrue(all(r["positive_text_ids"] == [r["target_text_id"]] for r in rows))


class ControllerTest(unittest.TestCase):
    def test_two_rounds_snapshot_lineage_fixed_data_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            worker = FixtureWorker()
            state = run_pipeline(cfg, runner=worker)
            self.assertTrue(state["complete"])
            first = read_json(root / "run/round-000/round_inputs.json")
            second = read_json(root / "run/round-001/round_inputs.json")
            self.assertEqual(first["policy_checkpoint"], cfg["checkpoint_path"])
            self.assertEqual(second["policy_checkpoint"], str(root / "run/round-000/training/latest"))
            train_calls = [c for c in worker.calls if "--round_manifest" in c]
            self.assertEqual(len(train_calls), 2)
            self.assertTrue(all("--resume_from_checkpoint" not in c for c in train_calls))
            self.assertTrue(all("--no-load_best_model_at_end" in c for c in train_calls))
            self.assertNotEqual(first["policy_fingerprint"], second["policy_fingerprint"])
            calls = len(worker.calls)
            run_pipeline(cfg, resume=True, runner=worker)
            self.assertEqual(len(worker.calls), calls)

    def test_interruption_reuses_pairs_and_resumes_only_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            worker = FixtureWorker()
            worker.fail_training = True
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_pipeline(cfg, runner=worker)
            manifest = root / "run/round-000/round_inputs.json"
            pairs = root / "run/round-000/preferences.jsonl"
            pair_hash = file_hash(pairs)
            directory = root / "run/round-000/training"
            checkpoint = directory / "checkpoint-1"
            fake_checkpoint(checkpoint)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (checkpoint / name).write_bytes(b"fixture")
            atomic_json(checkpoint / "trainer_state.json", {"global_step": 1})
            atomic_json(checkpoint / "round_checkpoint.json", {"identity": file_hash(manifest), "global_step": 1})
            fake_checkpoint(directory / "checkpoint-2")  # No completion marker: ignore.
            self.assertEqual(latest_resumable_checkpoint(directory, file_hash(manifest)), checkpoint)
            worker.calls.clear()
            worker.fail_training = False
            run_pipeline(cfg, resume=True, runner=worker)
            first_train = next(c for c in worker.calls if "--round_manifest" in c)
            self.assertIn(str(checkpoint), first_train)
            self.assertEqual(file_hash(pairs), pair_hash)
            self.assertFalse(any("--output" in c and str(pairs) in c for c in worker.calls))

    def test_resume_rejects_modified_pairs_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            worker = FixtureWorker()
            run_pipeline(cfg, runner=worker)
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline({**cfg, "steps_per_round": 3}, resume=True, runner=worker)
            pairs = root / "run/round-000/preferences.jsonl"
            with pairs.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run_pipeline(cfg, resume=True, runner=worker)

    def test_pair_audit_rejects_empty_duplicate_positive_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            run_pipeline(cfg, runner=FixtureWorker())
            directory = root / "run/round-000"
            original = list(iter_json_records(directory / "preferences.jsonl"))
            queries = list(iter_json_records(directory / "queries.jsonl"))
            fingerprint = read_json(directory / "round_inputs.json")["policy_fingerprint"]
            invalid = [[], [original[0], original[0]],
                       [{**original[0], "rejected_text_id": original[0]["chosen_text_id"]}],
                       [{**original[0], "round_id": 100}]]
            for rows in invalid:
                atomic_jsonl(root / "invalid.jsonl", rows)
                with self.assertRaises(ValueError):
                    audit_pairs(root / "invalid.jsonl", queries, root / "run/metadata/document_metadata.jsonl",
                                "text_id", 0, fingerprint, 2, None)


if __name__ == "__main__":
    unittest.main()
