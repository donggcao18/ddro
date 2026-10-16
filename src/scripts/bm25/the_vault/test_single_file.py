"""Regression coverage for the direct JSONL -> BM25 -> labeled DPO workflow."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from common import iter_json_records
from mine_dpo_negatives import mine
from prepare_bm25 import prepare
import run_pipeline


def row(document_id="repo/a.rb/a()", text="find a", positives=None):
    return {
        "numeric_id": "0", "text_id": "irrelevant-old-id", "semantic_id": "unused",
        "url_based_id": document_id, "text": text,
        "positive_url_based_id": positives if positives is not None else [document_id],
        "original": {"repo": "repo", "path": "lib/a.rb", "identifier": "a", "code": "def a; end"},
    }


class SingleFileTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.jsonl"
        self.work = self.root / "work"

    def write_source(self, rows):
        self.source.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    def prepare(self, **kwargs):
        options = dict(input=str(self.source), output_dir=str(self.work), id_field="url_based_id",
                       positive_id_field=None, code_only=False)
        options.update(kwargs)
        return prepare(SimpleNamespace(**options))

    def mine(self, run_text, **kwargs):
        run = self.work / "bm25_run.txt"
        run.write_text(run_text, encoding="utf-8")
        options = dict(run=str(run), query_metadata=str(self.work / "query_metadata.jsonl"),
                       document_metadata=str(self.work / "document_metadata.jsonl"),
                       output=str(self.work / "dpo_pairs_url.jsonl"), triples_output=None,
                       negatives_per_query=1, rank_ranges=[(1, 10)], rank_quotas=[1], seed=42,
                       fill_shortfall=False, pair_all_positives=False)
        options.update(kwargs)
        stats = mine(SimpleNamespace(**options))
        return stats, list(iter_json_records(options["output"]))

    def test_deduplicates_code_but_retains_every_query_and_source_bytes(self):
        self.write_source([row(text="one"), row(text="two"), row(text="three")])
        original = self.source.read_bytes()
        stats = self.prepare()
        self.assertEqual(stats["bm25_documents"], 1)
        self.assertEqual(stats["queries"], 3)
        corpus = list(iter_json_records(self.work / "corpus/documents.jsonl"))
        self.assertEqual(corpus[0]["id"], "repo/a.rb/a()")
        self.assertIn("def a; end", corpus[0]["contents"])
        self.assertNotIn("three", corpus[0]["contents"])
        queries = list(iter_json_records(self.work / "query_metadata.jsonl"))
        self.assertEqual([query["prompt"] for query in queries], ["one", "two", "three"])
        self.assertTrue(all("text_id" not in query for query in queries))
        self.assertEqual(self.source.read_bytes(), original)
        fingerprint = stats["corpus_sha256"]
        self.assertEqual(self.prepare()["corpus_sha256"], fingerprint)

    def test_explicit_positives_are_row_local_without_inference_or_transitive_expansion(self):
        self.write_source([
            row("repo/a", "shared query", ["repo/b", "repo/b"]),
            row("repo/a", "other query", []),
            row("repo/b", "shared query", ["repo/b", "repo/c"]),
            row("repo/c", "different query"),
        ])
        self.prepare()
        queries = list(iter_json_records(self.work / "query_metadata.jsonl"))
        self.assertEqual(queries[0]["positive_ids"], ["repo/a", "repo/b"])
        self.assertEqual(queries[1]["positive_ids"], ["repo/a"])
        stats, pairs = self.mine(
            "vault-000000000 Q0 repo/a 1 10.0 bm25\n"
            "vault-000000000 Q0 repo/b 2 9.0 bm25\n"
            "vault-000000000 Q0 repo/c 3 8.0 bm25\n"
            "vault-000000001 Q0 repo/b 1 7.0 bm25\n"
        )
        self.assertEqual(stats["filtered_multi_label_or_target_hits"], 2)
        self.assertEqual([(p["chosen"], p["rejected"]) for p in pairs],
                         [("repo/a", "repo/c"), ("repo/a", "repo/b")])
        self.assertEqual(pairs[0]["bm25_score"], 8.0)
        self.assertFalse(any("text_id" in key for pair in pairs for key in pair))

    def test_custom_fields_and_pair_all_positives_include_external_labels(self):
        a = row()
        a.update(doc_key=12, labels=[12, "outside-corpus"])
        b = row("repo/b")
        b.update(doc_key=13, labels=[13])
        self.write_source([a, b])
        stats = self.prepare(id_field="doc_key", positive_id_field="labels", code_only=True)
        self.assertEqual(stats["positive_ids_missing_from_corpus"], 1)
        self.assertEqual(next(iter_json_records(self.work / "corpus/documents.jsonl"))["contents"], "def a; end")
        _, pairs = self.mine("vault-000000000 Q0 13 1 1.0 bm25\n", pair_all_positives=True)
        self.assertEqual({p["chosen"] for p in pairs}, {"12", "outside-corpus"})
        self.assertEqual({p["rejected"] for p in pairs}, {"13"})

    def test_missing_or_malformed_required_fields_fail_before_writing(self):
        for field, invalid_value in [("positive_url_based_id", None), ("positive_url_based_id", "repo/a"),
                                     ("positive_url_based_id", [{}]), ("url_based_id", "has spaces"),
                                     ("text", ""), ("original", {"code": ""})]:
            with self.subTest(field=field, value=invalid_value):
                invalid = row()
                invalid[field] = invalid_value
                self.write_source([row(), invalid])
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.work.exists())

    def test_conflicting_document_content_is_rejected(self):
        changed = row(text="second query")
        changed["original"]["code"] = "def different; end"
        self.write_source([row(), changed])
        with self.assertRaisesRegex(ValueError, "conflicting document contents"):
            self.prepare()

    def test_no_negative_when_all_retrieved_ids_are_positive_or_unknown(self):
        self.write_source([row(positives=["repo/b"]), row("repo/b")])
        self.prepare()
        stats, pairs = self.mine(
            "vault-000000000 Q0 repo/b 1 3.0 bm25\n"
            "vault-000000000 Q0 unknown 2 2.0 bm25\n"
        )
        self.assertEqual(pairs, [])
        self.assertEqual(stats["queries_without_usable_negatives"], 1)
        self.assertEqual(stats["filtered_unknown_document_hits"], 1)

    def test_source_cannot_be_overwritten_and_stale_corpus_shards_are_rejected(self):
        self.work.mkdir()
        protected = self.work / "query_metadata.jsonl"
        protected.write_text(json.dumps(row()) + "\n", encoding="utf-8")
        original = protected.read_bytes()
        with self.assertRaisesRegex(ValueError, "generated BM25 artifact"):
            self.prepare(input=str(protected))
        self.assertEqual(protected.read_bytes(), original)
        self.write_source([row()])
        (self.work / "corpus").mkdir()
        (self.work / "corpus/stale.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "other files"):
            self.prepare()

    def test_cli_preparation_and_mining(self):
        self.write_source([row("repo/a", positives=["repo/b"]), row("repo/b"), row("repo/c")])
        script_dir = Path(__file__).parent
        subprocess.run([sys.executable, str(script_dir / "prepare_bm25.py"), "--input", str(self.source),
                        "--positive-id-field", "positive_url_based_id", "--output-dir", str(self.work)],
                       check=True, capture_output=True, text=True)
        run = self.work / "run.txt"
        run.write_text("vault-000000000 Q0 repo/b 1 2.0 bm25\nvault-000000000 Q0 repo/c 2 1.0 bm25\n", encoding="utf-8")
        output = self.work / "dpo.jsonl"
        subprocess.run([sys.executable, str(script_dir / "mine_dpo_negatives.py"), "--run", str(run),
                        "--query-metadata", str(self.work / "query_metadata.jsonl"),
                        "--document-metadata", str(self.work / "document_metadata.jsonl"),
                        "--output", str(output), "--negatives-per-query", "1", "--rank-ranges", "1:10",
                        "--rank-quotas", "1"], check=True, capture_output=True, text=True)
        self.assertEqual(next(iter_json_records(output))["rejected"], "repo/c")

    def run_pipeline(self, extra_args=()):
        def fake_run(command):
            if "prepare_bm25.py" in command[1]:
                self.prepare()
            elif "pyserini.index.lucene" in command:
                index = self.work / "index"
                index.mkdir(exist_ok=True)
                (index / "segments_test").touch()
        argv = ["run_pipeline.py", "--input", str(self.source), "--work-dir", str(self.work), *extra_args]
        with patch.object(sys, "argv", argv), patch.object(run_pipeline, "run", side_effect=fake_run) as calls, \
             contextlib.redirect_stdout(io.StringIO()):
            run_pipeline.main()
        return [call.args[0] for call in calls.call_args_list]

    def test_runner_wires_direct_ids_and_checks_index_reuse_against_actual_corpus(self):
        self.write_source([row()])
        commands = self.run_pipeline()
        self.assertIn("--input", commands[0])
        self.assertNotIn("--train-original", commands[0])
        self.assertNotIn("--storeRaw", commands[1])
        self.assertNotIn("--storeDocvectors", commands[1])
        self.assertIn("trec", commands[2])
        self.assertNotIn("--target-type", commands[3])
        self.assertTrue(commands[3][commands[3].index("--output") + 1].endswith("dpo_pairs_url.jsonl"))
        commands = self.run_pipeline(["--reuse-index"])
        self.assertFalse(any("pyserini.index.lucene" in command for command in commands))
        self.write_source([row(text="a different query", positives=["outside-corpus"])])
        commands = self.run_pipeline(["--reuse-index"])
        self.assertFalse(any("pyserini.index.lucene" in command for command in commands))
        changed = row()
        changed["original"]["code"] = "def changed; end"
        self.write_source([changed])
        with self.assertRaisesRegex(ValueError, "Cannot reuse"):
            self.run_pipeline(["--reuse-index"])
        # Retrying after preparation still cannot validate the stale index.
        with self.assertRaisesRegex(ValueError, "Cannot reuse"):
            self.run_pipeline(["--reuse-index"])


if __name__ == "__main__":
    unittest.main()
