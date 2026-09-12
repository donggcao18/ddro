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
from common import document_targets, iter_json_records
from data.data_prep.prepare_vault_dpo_metadata import multilabel_rows, prepare
from mine_model_confusion_negatives import retrieval_metrics
from pretrain.iterative_dpo_utils import (
    atomic_json, atomic_jsonl, checkpoint_hash, file_hash, latest_resumable_checkpoint,
    partition_queries, read_json, round_queries,
)
from pretrain.train_iterative_ddro_vault import audit_pairs, load_config, run_pipeline
from pretrain.update_dpo_reference import reference_identity


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
        self.empty_mining_rounds = set()

    def __call__(self, command):
        self.calls.append(command)
        def option(name):
            return command[command.index(name) + 1]
        if "--decay" in command:
            output = Path(option("--output"))
            identity = reference_identity(option("--reference-checkpoint"), option("--policy-checkpoint"),
                                          option("--reference-fingerprint"), option("--policy-fingerprint"),
                                          float(option("--decay")))
            fake_checkpoint(output, weights=json.dumps(identity, sort_keys=True))
            atomic_json(output / "reference_complete.json", {
                "identity": identity, "checkpoint_fingerprint": checkpoint_hash(output),
            })
        elif "--checkpoint-path" in command:
            queries = list(iter_json_records(option("--query-metadata")))
            output = Path(option("--output"))
            atomic_json(output.with_suffix(".excluded_text_ids.json"), {"overlength_targets": []})
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
                if int(option("--round-id")) in self.empty_mining_rounds:
                    pairs = []
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
    def test_full_partition_keeps_remainder_and_never_wraps(self):
        rows = [{"query_key": str(i)} for i in range(10)]
        for size, lengths in ((4, [4, 4, 2]), (5, [5, 5]), (20, [10]), (None, [10])):
            with self.subTest(size=size):
                partitions = partition_queries(rows, size, 42)
                self.assertEqual([len(part) for part in partitions], lengths)
                keys = [r["query_key"] for part in partitions for r in part]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(set(keys), {r["query_key"] for r in rows})
                self.assertNotEqual(keys, [r["query_key"] for r in rows])
                self.assertEqual(partitions, partition_queries(list(reversed(rows)), size, 42))

    def test_multilabel_uses_exact_per_row_labels_and_groups_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c, d = [f"repo/lib/file.rb/function_{i}(arg)" for i in range(4)]
            rows = [
                {"text": "same query", "url_based_id": a, "positive_url_based_id": [a, b, b]},
                {"text": "same query", "url_based_id": a, "positive_url_based_id": [a, c]},
                {"text": "different query", "url_based_id": a, "positive_url_based_id": [b]},
                {"text": "independent", "url_based_id": d, "positive_url_based_id": [d]},
            ]
            # Unrelated namespaces and labels must have no influence on URL identities.
            for row in rows:
                row.update(numeric_id="1090", semantic_id="9261", positive_text_ids=["wrong"],
                           text_id="wrong", original={"code": "unused"})
            atomic_jsonl(root / "queries.jsonl", rows + [rows[0]])
            manifest = prepare([], str(root / "queries.jsonl"), str(root / "meta"),
                               input_format="multilabel", doc_id_type="url_based_id")
            train = list(iter_json_records(root / "meta/train_queries.jsonl"))
            val = list(iter_json_records(root / "meta/validation_queries.jsonl"))
            prepared = train + val
            self.assertEqual(len(prepared), 4)  # Only the exact labeled duplicate is removed.
            same = [r for r in prepared if r["prompt"] == "same query"]
            self.assertEqual({tuple(r["positive_text_ids"]) for r in same}, {(a, b), (a, c)})
            other = next(r for r in prepared if r["prompt"] == "different query")
            self.assertEqual(other["positive_text_ids"], [b])
            self.assertEqual(other["target_text_id"], b)  # Never add source a as an implicit positive.
            self.assertEqual(len({r["family_id"] for r in prepared if r["source_doc_id"] == a}), 1)
            self.assertFalse({r["family_id"] for r in train} & {r["family_id"] for r in val})
            docs = list(iter_json_records(root / "meta/document_metadata.jsonl"))
            self.assertEqual({r["text_id"] for r in docs}, {a, b, c, d})  # b/c have no source row.
            self.assertTrue(all(document_targets(r, "url_based_id") == [r["text_id"]] for r in docs))
            self.assertEqual(manifest["candidate_documents"], 4)

    def test_multilabel_column_selection_optional_corpus_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"text": "find", "semantic_id": "s-1", "positive_semantic_id": ["s-2"],
                   "url_based_id": "ignored", "positive_url_based_id": ["also-ignored"]}
            atomic_jsonl(root / "queries.jsonl", [row])
            atomic_jsonl(root / "corpus.jsonl", [{"semantic_id": "s-3", "positive_semantic_id": ["not-a-candidate"]}])
            docs, queries = multilabel_rows([str(root / "corpus.jsonl")], str(root / "queries.jsonl"), "semantic_id")
            self.assertEqual(set(docs), {"s-1", "s-2", "s-3"})
            self.assertEqual(queries[0]["positive_text_ids"], ["s-2"])
            self.assertEqual(document_targets(docs["s-2"], "semantic_id"), ["s-2"])
            for labels in (None, [], "s-1", [None], [""], [" s-1"]):
                with self.subTest(labels=labels):
                    atomic_jsonl(root / "queries.jsonl", [{**row, "positive_semantic_id": labels}])
                    with self.assertRaisesRegex(ValueError, "positive_semantic_id"):
                        multilabel_rows([], str(root / "queries.jsonl"), "semantic_id")

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
        self.assertNotEqual(flattened[:10], [r["query_key"] for r in rows])
        self.assertNotEqual(subsets[0], round_queries(rows, 4, 0, 8))
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
    def test_automatic_round_count_covers_all_training_queries_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = fixture(root)
            raw = read_json(path)
            raw.pop("rounds")
            raw.pop("steps_per_round")
            raw["epochs_per_round"] = 1
            atomic_json(path, raw)
            cfg = load_config(path)
            self.assertIsNone(cfg["rounds"])
            worker = FixtureWorker()
            prepared = run_pipeline(cfg, prepare_only=True, runner=worker)
            self.assertFalse(worker.calls)
            plan = read_json(root / "run/round_plan.json")
            self.assertEqual(plan["mode"], "partition_all")
            self.assertEqual(plan["partition_sizes"], [5, 5, 4])
            self.assertEqual(prepared["planned_rounds"], 3)
            state = run_pipeline(cfg, resume=True, runner=worker)
            self.assertTrue(state["complete"])
            train = list(iter_json_records(root / "run/metadata/train_queries.jsonl"))
            selected = [row for i in range(plan["total_rounds"])
                        for row in iter_json_records(root / f"run/round-{i:03d}/queries.jsonl")]
            self.assertEqual(len(selected), len(train))
            self.assertEqual({r["query_key"] for r in selected}, {r["query_key"] for r in train})
            for command in worker.calls:
                if "--round_manifest" in command:
                    self.assertEqual(command[command.index("--num_train_epochs") + 1], "1")
                    self.assertEqual(command[command.index("--max_steps") + 1], "-1")
            calls = len(worker.calls)
            run_pipeline(cfg, resume=True, runner=worker)
            self.assertEqual(len(worker.calls), calls)

    def test_ema_reference_lineage_skips_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = fixture(root)
            atomic_json(path, {**read_json(path), "rounds": 3, "reference_update": "ema"})
            cfg = load_config(path)
            worker = FixtureWorker()
            worker.empty_mining_rounds = {1}
            state = run_pipeline(cfg, runner=worker)
            first = read_json(root / "run/round-000/round_inputs.json")
            third = read_json(root / "run/round-002/round_inputs.json")
            self.assertEqual(first["reference_checkpoint"], cfg["checkpoint_path"])
            self.assertEqual(third["reference_checkpoint"], str(root / "run/round-000/reference"))
            self.assertEqual(third["policy_checkpoint"], str(root / "run/round-000/training/latest"))
            self.assertEqual(state["latest_reference"], str(root / "run/round-002/reference"))
            self.assertEqual(len([c for c in worker.calls if "--decay" in c]), 2)
            train_calls = [c for c in worker.calls if "--round_manifest" in c]
            self.assertEqual(train_calls[-1][train_calls[-1].index("--reference_checkpoint_path") + 1],
                             third["reference_checkpoint"])
            calls = len(worker.calls)
            run_pipeline(cfg, resume=True, runner=worker)
            self.assertEqual(len(worker.calls), calls)
            (root / "run/round-000/reference/model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "EMA reference snapshot"):
                run_pipeline(cfg, resume=True, runner=worker)

    def test_epoch_rounds_accept_shortfalls_and_skip_empty_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = fixture(root)
            raw = read_json(config_path)
            raw.pop("steps_per_round")
            raw.update(epochs_per_round=1, rounds=3, num_beams=2, negatives_per_query=4,
                       selection_metric="mrr@2")
            atomic_json(config_path, raw)
            cfg = load_config(config_path)
            self.assertIsNone(cfg["steps_per_round"])
            # This worker produces one pair/query instead of the requested four.
            worker = FixtureWorker()
            worker.empty_mining_rounds = {0, 2}
            def runner(command):
                worker(command)
                if "--evaluation-only" in command:
                    output = Path(command[command.index("--output") + 1])
                    atomic_json(output.with_suffix(".stats.json"), {"metrics": {"mrr@2": 0.5}})
            state = run_pipeline(cfg, runner=runner)
            self.assertTrue(state["complete"])
            self.assertEqual([r["status"] for r in state["rounds"]],
                             ["skipped_no_negatives", "trained", "skipped_no_negatives"])
            self.assertEqual(state["rounds"][0]["checkpoint"], cfg["checkpoint_path"])
            self.assertEqual(state["rounds"][0]["optimizer_steps"], 0)
            self.assertEqual(state["latest"], str(root / "run/round-001/training/latest"))
            self.assertEqual(state["rounds"][1]["pair_audit"]["nominal_dataset_passes"], 1)
            train_calls = [c for c in worker.calls if "--round_manifest" in c]
            self.assertEqual(len(train_calls), 1)
            self.assertEqual(train_calls[0][train_calls[0].index("--max_steps") + 1], "-1")
            self.assertEqual(train_calls[0][train_calls[0].index("--num_train_epochs") + 1], "1")
            calls = len(worker.calls)
            run_pipeline(cfg, resume=True, runner=runner)
            self.assertEqual(len(worker.calls), calls)

    def test_corpus_free_url_multilabel_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = fixture(root)
            raw = read_json(config_path)
            raw.pop("corpus_files")
            raw.update(input_format="multilabel", target_type="url_based_id", num_gpus=2)
            atomic_json(config_path, raw)
            atomic_jsonl(root / "queries.jsonl", [
                {"text": f"query {i}", "url_based_id": f"repo/f{i}.rb/f(arg)",
                 "positive_url_based_id": [f"repo/f{i}.rb/f(arg)", f"repo/f{i}.rb/g(arg)"]}
                for i in range(8)
            ])
            cfg = load_config(config_path)
            self.assertEqual(cfg["corpus_files"], [])
            worker = FixtureWorker()
            state = run_pipeline(cfg, runner=worker)
            self.assertTrue(state["complete"])
            for command in worker.calls:
                if "--target-type" in command:
                    self.assertEqual(command[command.index("--target-type") + 1], "url_based_id")
                else:
                    self.assertIn("--nproc_per_node=2", command)
            pairs = list(iter_json_records(root / "run/round-000/preferences.jsonl"))
            self.assertTrue(all(p["chosen"].startswith("repo/") and
                                p["rejected"] not in p["positive_text_ids"] for p in pairs))
            calls = len(worker.calls)
            run_pipeline(cfg, resume=True, runner=worker)
            self.assertEqual(len(worker.calls), calls)

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

    def test_pair_audit_accepts_empty_but_rejects_duplicate_positive_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            run_pipeline(cfg, runner=FixtureWorker())
            directory = root / "run/round-000"
            original = list(iter_json_records(directory / "preferences.jsonl"))
            queries = list(iter_json_records(directory / "queries.jsonl"))
            fingerprint = read_json(directory / "round_inputs.json")["policy_fingerprint"]
            atomic_jsonl(root / "empty.jsonl", [])
            report = audit_pairs(root / "empty.jsonl", queries, root / "run/metadata/document_metadata.jsonl",
                                 "text_id", 0, fingerprint, 2, None)
            self.assertEqual(report["pairs"], 0)
            self.assertEqual(report["queries_without_negatives"], len(queries))
            self.assertEqual(report["previous_overlap_fraction"], 0)
            invalid = [[original[0], original[0]],
                       [{**original[0], "rejected_text_id": original[0]["chosen_text_id"]}],
                       [{**original[0], "round_id": 100}]]
            for rows in invalid:
                atomic_jsonl(root / "invalid.jsonl", rows)
                with self.assertRaises(ValueError):
                    audit_pairs(root / "invalid.jsonl", queries, root / "run/metadata/document_metadata.jsonl",
                                "text_id", 0, fingerprint, 2, None)


if __name__ == "__main__":
    unittest.main()
