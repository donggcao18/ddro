"""Offline BM25 provenance, separate quotas, and round-level hybrid training."""

from collections import Counter
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import iter_json_records
from pretrain.hybrid_preferences import BM25Cache, fuse_preferences
from pretrain.offline_iterative_bm25 import export, retrieve, pack
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash, read_json
from pretrain.train_iterative_ddro_vault import load_config, run_pipeline
from test_iterative_dpo import FixtureWorker, fake_checkpoint, fixture


def setup_config(root):
    fake_checkpoint(root / "sft")
    atomic_jsonl(root / "queries.jsonl", [
        {"url_based_id": f"repo/file{i:02d}/fn(arg)", "text": f"private pseudo query {i}",
         "positive_url_based_id": [f"repo/file{i:02d}/fn(arg)"],
         "original": {"code": f"def fn_{i}(arg): return arg + {i}"}} for i in range(20)])
    path = root / "config.json"
    atomic_json(path, {"checkpoint_path": "sft", "query_file": "queries.jsonl", "output_dir": "run",
                      "input_format": "multilabel", "target_type": "url_based_id",
                      "bm25_cache": "cache.sqlite", "bm25_negatives_per_query": 4,
                      "negatives_per_query": 4, "num_beams": 10, "selection_metric": "mrr@10",
                      "epochs_per_round": 1, "queries_per_round": 10,
                      "reference_update": "ema", "reference_ema_decay": 0.9,
                      "device": "cpu", "precision": "fp32", "max_target_length": 32})
    return load_config(path)


def build_cache(cfg, root, empty_query=None):
    work = root / "offline"
    export(cfg, work)
    def fake_retrieval(command):
        if "--topics" not in command:
            return
        docs = list(iter_json_records(work / "document_map.jsonl"))
        queries = list(iter_json_records(work / "metadata/train_queries.jsonl"))
        with (work / "bm25_run.trec").open("w", encoding="utf-8") as handle:
            # Reverse query order: pack must not rely on sequential query groups.
            for i in reversed(range(len(queries))):
                if i == empty_query:
                    continue
                for rank, doc in enumerate(docs, 1):
                    handle.write(f"{i} Q0 {doc['id']} {rank} {100-rank}.0 bm25\n")
    retrieve(work, hits=200, runner=fake_retrieval)
    pack(work, root / "cache.sqlite")
    return work


def model_rows(query, docs, round_id=0, fingerprint="policy"):
    return [{"query_key": query["query_key"], "prompt": query["prompt"],
             "chosen_text_id": query["target_text_id"], "chosen": query["target_text_id"],
             "rejected_text_id": doc, "rejected": doc, "positive_text_ids": query["positive_text_ids"],
             "negative_source": "model_confusion", "model_rank": rank, "model_score": -float(rank),
             "round_id": round_id, "policy_fingerprint": fingerprint}
            for rank, doc in enumerate(docs, 1)]


class HybridTest(unittest.TestCase):
    def open_cache(self, root):
        return BM25Cache(root / "cache.sqlite", root / "offline/metadata/train_queries.jsonl",
                         root / "offline/metadata/document_metadata.jsonl", "url_based_id")

    def test_export_uses_code_only_and_training_queries_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = setup_config(root)
            work = build_cache(cfg, root)
            contents = (work / "corpus/documents.jsonl").read_text()
            self.assertNotIn("private pseudo", contents)
            self.assertIn("def fn_", contents)
            train = list(iter_json_records(work / "metadata/train_queries.jsonl"))
            heldout = list(iter_json_records(work / "metadata/validation_queries.jsonl"))
            topics = (work / "queries.tsv").read_text().splitlines()
            self.assertEqual(len(topics), len(train))
            self.assertFalse({r["prompt"] for r in heldout} & {line.split("\t", 1)[1] for line in topics})
            cache = self.open_cache(root)
            try:
                self.assertEqual(len(cache.candidates(train[0])), 20)
                with self.assertRaisesRegex(ValueError, "Missing or changed"):
                    cache.candidates(heldout[0])
                with self.assertRaisesRegex(ValueError, "Missing or changed"):
                    cache.candidates({**train[0], "prompt": "modified"})
            finally:
                cache.close()

    def test_four_plus_four_deduplicates_overlap_and_token_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = build_cache(setup_config(root), root)
            query = list(iter_json_records(work / "metadata/train_queries.jsonl"))[0]
            cache = self.open_cache(root)
            try:
                others = [doc for doc, _, _ in cache.candidates(query) if doc not in query["positive_text_ids"]]
                atomic_jsonl(root / "model.jsonl", model_rows(query, others[:4]))
                atomic_json(root / "excluded.json", {"model_mining_excluded_text_ids": [], "collision_groups": [
                    {"text_ids": [query["target_text_id"], others[4]]}, {"text_ids": [others[0], others[5]]}]})
                stats = fuse_preferences([query], root / "model.jsonl", root / "excluded.json", cache,
                                         root / "pairs.jsonl", 0, "policy", file_hash(root / "cache.sqlite"))
                pairs = list(iter_json_records(root / "pairs.jsonl"))
                self.assertEqual(Counter(r["negative_source"] for r in pairs), {"model_confusion": 4, "bm25": 4})
                self.assertEqual(len({r["rejected"] for r in pairs}), 8)
                self.assertEqual([r["rejected"] for r in pairs[4:]], others[6:10])
                self.assertEqual(stats["source_mix_distribution"], {"4+4": 1})
                self.assertTrue(all(r["bm25_cache_fingerprint"] == file_hash(root / "cache.sqlite") for r in pairs[4:]))
            finally:
                cache.close()

    def test_multilabel_and_empty_or_bm25_only_shortfalls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = setup_config(root)
            rows = list(iter_json_records(root / "queries.jsonl"))
            rows[0]["positive_url_based_id"].append(rows[1]["url_based_id"])
            atomic_jsonl(root / "queries.jsonl", rows)
            work = build_cache(cfg, root, empty_query=0)
            queries = list(iter_json_records(work / "metadata/train_queries.jsonl"))
            cache = self.open_cache(root)
            try:
                atomic_jsonl(root / "model.jsonl", [])
                atomic_json(root / "excluded.json", {"model_mining_excluded_text_ids": [], "collision_groups": []})
                stats = fuse_preferences(queries, root / "model.jsonl", root / "excluded.json", cache,
                                         root / "pairs.jsonl", 0, "policy", "cachehash")
                pairs = list(iter_json_records(root / "pairs.jsonl"))
                self.assertEqual(stats["source_mix_distribution"]["0+0"], 1)
                self.assertEqual(stats["model_confusion_pairs"], 0)
                self.assertTrue(all(r["negative_source"] == "bm25" for r in pairs))
                self.assertTrue(all(r["rejected"] not in r["positive_text_ids"] for r in pairs))
                self.assertEqual(stats["bm25_pairs"], 4 * (len(queries) - 1))
            finally:
                cache.close()

    def test_cache_provenance_and_incomplete_retrieval_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = setup_config(root)
            export(cfg, root / "offline")
            with self.assertRaises(FileNotFoundError):
                pack(root / "offline", root / "cache.sqlite")
            self.assertFalse((root / "cache.sqlite").exists())
            # A completed search with an invalid document ID must not publish.
            work = root / "offline"
            run = work / "bm25_run.trec"
            run.write_text("0 Q0 99999 1 1.0 bm25\n")
            atomic_json(work / "retrieval_complete.json", {
                "export_fingerprint": file_hash(work / "export_manifest.json"),
                "artifacts": {str(run): file_hash(run)}, "hits": 200})
            with self.assertRaises(sqlite3.IntegrityError):
                pack(work, root / "cache.sqlite")
            self.assertFalse((root / "cache.sqlite").exists())
            run.write_text("0 Q0 0 1 1.0 bm25\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                pack(work, root / "cache.sqlite")

    def test_controller_resume_keeps_model_mining_and_uses_fused_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = setup_config(root)
            build_cache(cfg, root)
            worker = FixtureWorker()
            worker.fail_training = True

            def runner(command):
                worker(command)
                if "--evaluation-only" in command:
                    path = Path(command[command.index("--output") + 1]).with_suffix(".stats.json")
                    atomic_json(path, {"metrics": {"mrr@10": 0.5}})
                elif "--checkpoint-path" in command:
                    def option(flag):
                        return command[command.index(flag) + 1]
                    queries = list(iter_json_records(option("--query-metadata")))
                    docs = [r["text_id"] for r in iter_json_records(option("--document-metadata"))]
                    atomic_jsonl(option("--output"), [pair for q in queries for pair in model_rows(
                        q, [d for d in docs if d not in q["positive_text_ids"]][:4],
                        int(option("--round-id")), option("--checkpoint-fingerprint"))])

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_pipeline(cfg, runner=runner)
            model = root / "run/round-000/model_preferences.jsonl"
            original_hash = file_hash(model)
            worker.calls.clear()
            worker.fail_training = False
            state = run_pipeline(cfg, resume=True, runner=runner)
            self.assertTrue(state["complete"])
            self.assertEqual(file_hash(model), original_hash)
            self.assertFalse(any("--round-id" in c and "--evaluation-only" not in c
                                 and c[c.index("--round-id") + 1] == "0" for c in worker.calls))
            for row in state["rounds"]:
                report = row["pair_audit"]
                self.assertEqual(report["pairs"], report["selected_queries"] * 8)
                self.assertEqual(report["hybrid"]["source_mix_distribution"], {"4+4": report["selected_queries"]})
            with (root / "cache.sqlite").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline(cfg, resume=True, runner=runner)

    def test_model_only_old_run_can_resume_but_cannot_turn_hybrid_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            state = run_pipeline(cfg, runner=FixtureWorker())
            for key in ("bm25_cache", "bm25_negatives_per_query"):
                state["identity"]["config"].pop(key)
            state["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = (
                "8c56f70909846fe8174c1c357c7e858c44b035d032d64ce348efdb1119f4cd7b")
            atomic_json(root / "run/run_manifest.json", state)
            worker = FixtureWorker()
            run_pipeline(load_config(root / "config.json"), resume=True, runner=worker)
            self.assertFalse(worker.calls)
            cache = root / "fake-cache"
            cache.write_text("fake")
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline({**cfg, "bm25_cache": str(cache)}, resume=True, runner=worker)

    @unittest.skipUnless(os.environ.get("RUN_DPO_INTEGRATION") == "1", "Set RUN_DPO_INTEGRATION=1")
    def test_real_t5_mining_and_training_with_mixed_source_json(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
        import mine_model_confusion_negatives as miner
        from pretrain import train_ddro_vault as trainer
        from pretrain import update_dpo_reference as ema

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = setup_config(root)
            ids = [r["url_based_id"] for r in iter_json_records(root / "queries.jsonl")]
            vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{v: i + 3 for i, v in enumerate(ids)}}
            backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = WhitespaceSplit()
            backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
            model = T5ForConditionalGeneration(T5Config(vocab_size=len(vocab), d_model=16, d_kv=4,
                d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2,
                decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
            model.save_pretrained(root / "sft")
            tokenizer.save_pretrained(root / "sft")
            cfg.update(queries_per_round=100, max_target_length=8)
            cfg["training"].update(per_device_train_batch_size=16, gradient_accumulation_steps=1)
            build_cache(cfg, root)

            def runner(command):
                if "--decay" in command:
                    ema.update_reference(**vars(ema.build_parser().parse_args(command[2:])))
                elif "--checkpoint-path" in command:
                    miner.mine(miner.build_parser().parse_args(command[2:]))
                else:
                    with patch.object(sys, "argv", command[1:]):
                        trainer.train_round(trainer.parse_args())

            result = run_pipeline(cfg, runner=runner)
            self.assertTrue(result["complete"])
            report = result["rounds"][0]["pair_audit"]
            self.assertEqual(report["pairs"], 8 * report["selected_queries"])
            self.assertGreater(result["rounds"][0]["optimizer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
