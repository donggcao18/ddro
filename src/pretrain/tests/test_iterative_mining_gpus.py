"""GPU dispatch contracts, recovery, and optional real-model CPU worker tests."""

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import iter_json_records
from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, checkpoint_hash, file_hash, read_json
from pretrain.mine_on_gpus import merge_stats, run_sharded, run_workers, split_batches, validate_cuda
from pretrain.train_iterative_ddro_vault import (
    generation_command, load_config, run_pipeline,
)
from test_iterative_dpo import FixtureWorker, fake_checkpoint, fixture


class MiningGPUTest(unittest.TestCase):
    def setup_mining(self, root):
        fake_checkpoint(root / "sft")
        atomic_jsonl(root / "queries.jsonl", [{"query_key": str(i)} for i in range(7)])
        atomic_jsonl(root / "documents.jsonl", [{"text_id": "doc"}])
        return ["--checkpoint-path", str(root / "sft"), "--checkpoint-fingerprint", checkpoint_hash(root / "sft"),
                "--query-metadata", str(root / "queries.jsonl"), "--document-metadata", str(root / "documents.jsonl"),
                "--output", str(root / "preferences.jsonl"), "--batch-size", "2", "--num-beams", "10"]

    def complete_fixture(self, job):
        cmd = job["command"]
        rows = list(iter_json_records(cmd[cmd.index("--query-metadata") + 1]))
        output = job["output"]
        # Some queries have no negative; others produce one or two pairs.
        pairs = [{"query_key": row["query_key"], "rejected": str(j)}
                 for row in rows for j in range(int(row["query_key"]) % 3)]
        atomic_jsonl(output, pairs)
        atomic_json(output.with_suffix(".excluded_text_ids.json"), {"excluded_text_ids": []})
        atomic_json(output.with_suffix(".stats.json"), {
            "queries_seen": len(rows), "queries_processed": len(rows), "model_pairs_written": len(pairs),
            "documents": 123, "num_beams": 10, "device": f"cuda:{job['rank']}",
            "candidate_count_distribution": {"0": sum(int(row["query_key"]) % 3 == 0 for row in rows)},
        })

    def test_batch_partition_coverage_and_small_datasets(self):
        rows = [{"query_key": str(i)} for i in range(10000)]
        shards = split_batches(rows, 64, 2)
        self.assertEqual([len(s) for s in shards], [5056, 4944])
        self.assertEqual([r for s in shards for r in s], rows)
        for workers in (1, 3, 8, 200):
            shards = split_batches(rows, 64, workers)
            self.assertEqual([r for s in shards for r in s], rows)
            self.assertTrue(all(len(s) % 64 == 0 for s in shards[:-1]))
        self.assertEqual(split_batches(rows[:1], 64, 8), [rows[:1]])

    def test_config_dispatch_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(Path(tmp))
            raw = {**read_json(path), "num_gpus": 3}
            atomic_json(path, raw)
            cfg = load_config(path)
            def command(config):
                return generation_command(config, "policy", Path("queries"), Path("docs"), Path("out"), "hash", 0)
            cmd = command(cfg)
            self.assertIn("mine_on_gpus.py", cmd[1])
            self.assertEqual(cmd[cmd.index("--num-gpus") + 1], "3")
            self.assertIn("mine_model_confusion_negatives.py", command({**cfg, "mining_num_gpus": 1})[1])
            for value in (0, -1, True, 1.5, "2"):
                atomic_json(path, {**raw, "mining_num_gpus": value})
                with self.assertRaisesRegex(ValueError, "mining_num_gpus"):
                    load_config(path)
            atomic_json(path, {**raw, "device": "cpu"})
            with self.assertRaisesRegex(ValueError, "Multi-GPU"):
                load_config(path)

    def test_merge_statistics_are_weighted_and_corpus_not_summed(self):
        result = merge_stats([
            {"documents": 8, "queries_processed": 4, "metrics": {"mrr@10": 1.0}, "positive_beams_filtered": 3},
            {"documents": 8, "queries_processed": 1, "metrics": {"mrr@10": 0.0}},
        ], Path("out.jsonl"), 2)
        self.assertEqual(result["documents"], 8)
        self.assertEqual(result["queries_processed"], 5)
        self.assertEqual(result["positive_beams_filtered"], 3)
        self.assertEqual(result["metrics"]["mrr@10"], 0.8)
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            merge_stats([{"documents": 1}, {"documents": 2}], Path("out"), 2)

    def test_insufficient_visible_devices_fail_before_launch(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))}):
            with self.assertRaisesRegex(ValueError, "only 1 CUDA"):
                validate_cuda(2, "auto")
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                validate_cuda(2, "cuda:1")

    def test_failure_resume_merge_order_gpu_assignment_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_mining(root)
            def fail(jobs, commit):
                self.assertEqual(len(jobs), 2)
                for rank, job in enumerate(jobs):
                    cmd = job["command"]
                    self.assertEqual(cmd[cmd.index("--device") + 1], f"cuda:{rank}")
                self.complete_fixture(jobs[0])
                commit(jobs[0])
                raise RuntimeError("worker failed")
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                run_sharded(argv, 2, fail, lambda *_: None)
            self.assertFalse((root / "preferences.jsonl").exists())
            def resume(jobs, commit):
                self.assertEqual([job["rank"] for job in jobs], [1])
                for job in jobs:
                    self.complete_fixture(job)
                    commit(job)
            result = run_sharded(argv, 2, resume, lambda *_: None)
            self.assertEqual(result["queries_seen"], 7)
            self.assertEqual(result["documents"], 123)
            keys = [row["query_key"] for row in iter_json_records(root / "preferences.jsonl")]
            self.assertEqual(keys, [str(i) for i in range(7) for _ in range(i % 3)])
            before = file_hash(root / "preferences.jsonl")
            run_sharded(argv, 2, lambda *_: self.fail("Unexpected recomputation"), lambda *_: None)
            self.assertEqual(before, file_hash(root / "preferences.jsonl"))
            shard = root / "preferences.shards/gpu-001/preferences.jsonl"
            shard.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run_sharded(argv, 2, resume, lambda *_: None)

    def test_global_limit_and_empty_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_mining(root) + ["--limit-queries", "1"]
            def launch(jobs, commit):
                self.assertEqual(len(jobs), 1)
                self.assertNotIn("--limit-queries", jobs[0]["command"])
                self.complete_fixture(jobs[0])
                commit(jobs[0])
            result = run_sharded(argv, 4, launch, lambda *_: None)
            self.assertEqual(result["model_pairs_written"], 0)
            self.assertEqual(result["queries_seen"], 1)
            self.assertEqual((root / "preferences.jsonl").read_text(), "")

    def test_old_controller_resume_keeps_trained_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(fixture(root))
            worker = FixtureWorker()
            def interrupted(cmd):
                if "--round-id" in cmd and cmd[cmd.index("--round-id") + 1] == "1":
                    raise RuntimeError("stop after round zero")
                worker(cmd)
            with self.assertRaisesRegex(RuntimeError, "stop after"):
                run_pipeline(cfg, runner=interrupted)
            path = root / "run/run_manifest.json"
            state = read_json(path)
            state["identity"]["config"].pop("mining_num_gpus")
            state["identity"]["code"].pop(str(SRC / "pretrain/mine_on_gpus.py"))
            state["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = (
                "13c07e6d53f273b021498b572469322782b6af52333bbe6173a7466f50d88e1c")
            atomic_json(path, state)
            original = {p: file_hash(p) for p in (root / "run/round-000").rglob("*") if p.is_file()}
            worker.calls.clear()
            cfg = {**cfg, "mining_num_gpus": 2}
            run_pipeline(cfg, resume=True, runner=worker)
            self.assertEqual({p: file_hash(p) for p in original}, original)
            self.assertTrue(any("mine_on_gpus.py" in c[1] for c in worker.calls))
            self.assertEqual(len([c for c in worker.calls if "--round_manifest" in c]), 1)
            # Later edits to the launcher are not silently accepted.
            state = read_json(path)
            state["identity"]["code"][str(SRC / "pretrain/mine_on_gpus.py")] = "modified"
            atomic_json(path, state)
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline(cfg, resume=True, runner=worker)

    def test_processes_run_concurrently_and_worker_errors_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Both workers must start before either can finish.
            code = ("import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.touch(); "
                    "other=pathlib.Path(sys.argv[2]); deadline=time.monotonic()+10\n"
                    "while not other.exists() and time.monotonic()<deadline: time.sleep(.02)\n"
                    "assert other.exists(), 'peer was not started'\n")
            jobs = [{"rank": i, "log": str(root / f"{i}.log"), "command": [
                sys.executable, "-c", code, str(root / f"{i}.ready"), str(root / f"{1-i}.ready")]} for i in range(2)]
            done = []
            run_workers(jobs, lambda job: done.append(job["rank"]))
            self.assertEqual(sorted(done), [0, 1])
            jobs[0]["command"] = [sys.executable, "-c", "raise SystemExit(2)"]
            with self.assertRaisesRegex(RuntimeError, "failed.*exit 2"):
                run_workers(jobs[:1], lambda _: None)

    @unittest.skipUnless(os.environ.get("RUN_DPO_INTEGRATION") == "1", "Set RUN_DPO_INTEGRATION=1")
    def test_real_t5_sharded_mining_and_evaluation_match_single_worker(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, set_seed
        from mine_model_confusion_negatives import build_parser, mine

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"repo/lib/file_{i}.rb/function_{i}(arg)" for i in range(12)]
            pre = Whitespace()
            vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "find": 3}
            for target in ids:
                for token, _ in pre.pre_tokenize_str(target):
                    if token not in vocab:
                        vocab[token] = len(vocab)
            backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = pre
            backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
            set_seed(19)
            model = T5ForConditionalGeneration(T5Config(
                vocab_size=len(vocab), d_model=16, d_kv=4, d_ff=32, num_layers=1,
                num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
            model.save_pretrained(root / "sft")
            tokenizer.save_pretrained(root / "sft")
            atomic_jsonl(root / "documents.jsonl", [{"text_id": target, "url_based_ids": [target]} for target in ids])
            atomic_jsonl(root / "queries.jsonl", [
                {"query_key": str(i), "prompt": f"find {ids[i]}", "target_text_id": ids[i],
                 "positive_text_ids": [ids[i]] if i < 6 else ids} for i in range(7)])
            base = ["--checkpoint-path", str(root / "sft"), "--checkpoint-fingerprint", checkpoint_hash(root / "sft"),
                    "--query-metadata", str(root / "queries.jsonl"), "--document-metadata", str(root / "documents.jsonl"),
                    "--batch-size", "2", "--num-beams", "10", "--max-target-length", "20", "--strict-corpus"]

            def cpu_workers(jobs, commit):
                # Exercise actual concurrent subprocesses on CPU in CI. Production
                # receives the original cuda:<rank> commands without this adapter.
                adapted = []
                for job in jobs:
                    command = list(job["command"])
                    command[command.index("--device") + 1] = "cpu"
                    adapted.append({**job, "command": command})
                with patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}):
                    run_workers(adapted, commit)

            for evaluate in (False, True):
                with self.subTest(evaluate=evaluate):
                    options = base + (["--evaluation-only"] if evaluate else [])
                    single = root / f"single-{evaluate}.jsonl"
                    parallel = root / f"parallel-{evaluate}.jsonl"
                    expected = mine(build_parser().parse_args(options + ["--device", "cpu", "--output", str(single)]))
                    actual = run_sharded(options + ["--output", str(parallel)], 2, cpu_workers, lambda *_: None)
                    self.assertEqual(list(iter_json_records(single)), list(iter_json_records(parallel)))
                    for key in ("documents", "queries_seen", "queries_processed", "candidate_count_distribution"):
                        self.assertEqual(actual[key], expected[key])
                    if evaluate:
                        for key, value in expected["metrics"].items():
                            self.assertAlmostEqual(actual["metrics"][key], value)
                    else:
                        self.assertEqual(actual["model_pairs_written"], expected["model_pairs_written"])
                        self.assertEqual(actual["queries_with_no_model_negatives"], 1)
                    run_sharded(options + ["--output", str(parallel)], 2,
                                lambda *_: self.fail("Completed real shards should be reused"), lambda *_: None)


if __name__ == "__main__":
    unittest.main()
