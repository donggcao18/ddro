"""BM25 URL remapping, per-query source quotas, round isolation and resume."""

from collections import Counter
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
from pretrain.launch_branch_preference import build_parser, experiment_config
from pretrain.mix_bm25_preferences import prepare_bm25_pool, fuse_round_preferences
from pretrain.train_iterative_ddro_vault import generation_command, load_config, run_pipeline, skipped_round_stage
from test_iterative_dpo import fake_checkpoint, FixtureWorker


class TinyTokenizer:
    eos_token_id, pad_token_id, unk_token_id = 1, 0, 2

    def __init__(self, ids):
        self.tokens = {target: [i + 3, 1] for i, target in enumerate(ids)}

    def __call__(self, target, add_special_tokens=True, truncation=False, max_length=None):
        tokens = self.tokens[target]
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [1]
        return {"input_ids": tokens}


class MixedPreferencesTest(unittest.TestCase):
    def fixture(self, root, target_type="url_based_id"):
        ids = [f"repo/lib/function{i}()" for i in range(10)]
        tokenizer = TinyTokenizer(ids)
        atomic_jsonl(root / "documents.jsonl", [
            {"text_id": target, "doc_id_type": target_type,
             **({"url_based_ids": [target]} if target_type == "url_based_id" else {})} for target in ids])
        query = {"query_key": "current-hash", "prompt": "Find account creation", "target_text_id": ids[0],
                 "positive_text_ids": [ids[0], ids[1]], "family_id": "train-family"}
        atomic_jsonl(root / "queries.jsonl", [query])

        def bm25(rejected, rank=1, **extra):
            return {"prompt": "Query: FIND  account creation", "chosen": "ignored-chosen",
                    "rejected": "ignored-rejected", "query_key": "vault-000000001",
                    "chosen_text_id": "1002010611140201000", "rejected_text_id": "1401061011140201",
                    "positive_text_ids": ["1002010611140201000"],
                    f"chosen_{target_type}": ids[0], f"rejected_{target_type}": rejected,
                    "bm25_rank": rank, "bm25_score": None, **extra}

        def policy(rejected):
            return {"prompt": query["prompt"], "query_key": query["query_key"], "chosen": ids[0],
                    "chosen_text_id": ids[0], "rejected": rejected, "rejected_text_id": rejected,
                    "positive_text_ids": query["positive_text_ids"], "negative_source": "model_confusion",
                    "round_id": 3, "policy_fingerprint": "snapshot-3"}
        return ids, tokenizer, query, bm25, policy

    def prepare(self, root, tokenizer, target_type="url_based_id", **policies):
        return prepare_bm25_pool(root / "bm25.jsonl", root / "queries.jsonl", root / "documents.jsonl",
                                 root / "model", root / "pool.sqlite", target_type, 8,
                                 policies.get("length", "error"), policies.get("collision", "allow"),
                                 "error", tokenizer=tokenizer)

    def fuse(self, root, total=4, lexical=2):
        return fuse_round_preferences(root / "pool.sqlite", root / "policy.jsonl", root / "queries.jsonl",
                                      root / "mixed.jsonl", total, lexical, 3, "snapshot-3")

    def test_authoritative_url_fields_normalized_prompt_ranks_and_two_plus_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, query, bm, policy = self.fixture(root)
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[3], 15), bm(ids[2], 5), bm(ids[2], 2)])
            atomic_jsonl(root / "policy.jsonl", [policy(ids[i]) for i in (4, 5, 6, 7)])
            self.prepare(root, tokenizer)
            self.fuse(root)
            rows = list(iter_json_records(root / "mixed.jsonl"))
            self.assertEqual(Counter(r["negative_source"] for r in rows), {"bm25": 2, "model_confusion": 2})
            bm_rows = [r for r in rows if r["negative_source"] == "bm25"]
            self.assertEqual([r["bm25_rank"] for r in bm_rows], [2, 15])
            self.assertTrue(all(r["query_key"] == query["query_key"] for r in rows))
            self.assertTrue(all(r["chosen"] == ids[0] and r["chosen_text_id"] == ids[0] for r in rows))
            self.assertTrue(all(r["positive_text_ids"] == query["positive_text_ids"] for r in rows))
            self.assertEqual(bm_rows[0]["bm25_source_query_key"], "vault-000000001")
            self.assertIsNone(bm_rows[0]["bm25_score"])
            original = file_hash(root / "mixed.jsonl")
            self.fuse(root)
            self.assertEqual(file_hash(root / "mixed.jsonl"), original)

    def test_positive_aliases_unknown_targets_and_other_query_rows_are_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, query, bm, policy = self.fixture(root)
            tokenizer.tokens[ids[8]] = tokenizer.tokens[ids[0]]
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[1]), bm(ids[8]), bm("missing/doc"),
                                              bm(ids[2]), bm(ids[3], prompt="heldout query"),
                                              bm(ids[4], chosen_url_based_id=ids[1])])
            audit = self.prepare(root, tokenizer)
            self.assertEqual(audit["positive_or_alias_pairs_filtered"], 2)
            self.assertEqual(audit["unknown_corpus_target_rows"], 1)
            self.assertEqual(audit["unmatched_rows"], 2)
            atomic_jsonl(root / "policy.jsonl", [policy(ids[i]) for i in (4, 5, 6, 7)])
            mixed = self.fuse(root)
            self.assertEqual(mixed["counts"]["bm25"], 1)
            self.assertEqual(mixed["counts"]["model_confusion"], 3)
            self.assertEqual(mixed["pairs"], 4)

    def test_cross_source_token_alias_dedup_and_shortfalls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, _, bm, policy = self.fixture(root)
            tokenizer.tokens[ids[8]] = tokenizer.tokens[ids[2]]
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[8]), bm(ids[3], 2)])
            self.prepare(root, tokenizer)
            atomic_jsonl(root / "policy.jsonl", [policy(ids[2]), policy(ids[4])])
            result = self.fuse(root)
            rows = list(iter_json_records(root / "mixed.jsonl"))
            self.assertEqual(result["pairs"], 3)
            self.assertEqual(result["counts"]["queries_below_total_quota"], 1)
            self.assertEqual(len({tuple(tokenizer.tokens[r["rejected"]]) for r in rows}), 3)
            atomic_jsonl(root / "policy.jsonl", [])
            result = self.fuse(root)
            self.assertEqual(result["counts"]["bm25"], 2)
            self.assertEqual(result["pairs"], 2)

    def test_overlap_preserves_two_plus_two_when_unique_alternatives_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, _, bm, policy = self.fixture(root)
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[2]), bm(ids[3], 2)])
            atomic_jsonl(root / "policy.jsonl", [policy(ids[i]) for i in (2, 3, 4, 5)])
            self.prepare(root, tokenizer)
            result = self.fuse(root)
            self.assertEqual(result["source_mix_per_query"], {"2 bm25 + 2 policy": 1})
            self.assertEqual(result["pairs"], 4)
            rows = list(iter_json_records(root / "mixed.jsonl"))
            self.assertEqual({r["rejected"] for r in rows}, set(ids[2:6]))

    def test_same_prompt_with_different_chosen_labels_does_not_cross_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, query, bm, _ = self.fixture(root)
            other = {**query, "query_key": "other-query", "target_text_id": ids[1], "positive_text_ids": [ids[1]]}
            atomic_jsonl(root / "queries.jsonl", [query, other])
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[2])])
            self.prepare(root, tokenizer)
            atomic_jsonl(root / "policy.jsonl", [])
            self.fuse(root)
            self.assertEqual({r["query_key"] for r in iter_json_records(root / "mixed.jsonl")}, {query["query_key"]})

    def test_structure_ids_use_the_selected_primary_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, _, bm, _ = self.fixture(root, "structure_id_v6")
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[2])])
            self.prepare(root, tokenizer, "structure_id_v6")
            atomic_jsonl(root / "policy.jsonl", [])
            self.fuse(root)
            row = next(iter_json_records(root / "mixed.jsonl"))
            self.assertEqual(row["chosen_structure_id_v6"], ids[0])
            self.assertEqual(row["rejected_structure_id_v6"], ids[2])

    def test_no_matching_rows_and_bad_schema_fail_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, _, bm, _ = self.fixture(root)
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[2], prompt="different")])
            with self.assertRaisesRegex(ValueError, "No usable BM25 pairs"):
                self.prepare(root, tokenizer)
            atomic_jsonl(root / "bm25.jsonl", [{"prompt": "find", "chosen": ids[0], "rejected": ids[2]}])
            with self.assertRaisesRegex(ValueError, "chosen_url_based_id"):
                self.prepare(root, tokenizer)
            self.assertFalse((root / "pool.sqlite").exists())

    def test_stale_policy_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, tokenizer, _, bm, policy = self.fixture(root)
            atomic_jsonl(root / "bm25.jsonl", [bm(ids[2])])
            self.prepare(root, tokenizer)
            atomic_jsonl(root / "policy.jsonl", [{**policy(ids[3]), "policy_fingerprint": "stale"}])
            with self.assertRaisesRegex(ValueError, "current selected round"):
                self.fuse(root)


class MixedPipelineTest(unittest.TestCase):
    def test_clear_cli_names_and_legacy_aliases_resolve_identically(self):
        common = ["--checkpoint", "model", "--query-file", "q", "--output-dir", "run", "--bm25-file", "bm.jsonl"]
        old = experiment_config(build_parser().parse_args(common + [
            "--negatives-per-query", "10", "--bm25-negatives-per-query", "6"]))
        new = experiment_config(build_parser().parse_args(common + [
            "--total-training-negatives-per-query", "10", "--bm25-training-negatives-per-query", "6"]))
        self.assertEqual(old, new)
        explicit = experiment_config(build_parser().parse_args(common + [
            "--total-training-negatives-per-query", "10", "--bm25-training-negatives-per-query", "6",
            "--policy-mining-negatives-per-query", "4"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            atomic_json(path, explicit)
            cfg = load_config(path)
            command = generation_command(cfg, "model", Path("q"), Path("docs"), Path("out"), "hash", 0)
            self.assertEqual(command[command.index("--negatives-per-query") + 1], "4")
            self.assertEqual(cfg["negatives_per_query"], 10)
            self.assertEqual(cfg["bm25_negatives_per_query"], 6)
            for quota in (0, -1, True, 1.5):
                atomic_json(path, {**explicit, "policy_mining_negatives_per_query": quota})
                with self.assertRaisesRegex(ValueError, "policy_mining_negatives_per_query"):
                    load_config(path)
            atomic_json(path, {**explicit, "policy_mining_negatives_per_query": 10})
            self.assertIsNone(load_config(path)["policy_mining_negatives_per_query"])
            atomic_json(path, {**explicit, "bm25_file": None})
            with self.assertRaisesRegex(ValueError, "Without BM25 mixing"):
                load_config(path)

    def test_launcher_config_quotas_and_separate_experiment_identity(self):
        args = build_parser().parse_args(["--checkpoint", "model", "--query-file", "q", "--output-dir", "run",
                                          "--bm25-file", "bm25.jsonl"])
        config = experiment_config(args)
        self.assertEqual(config["bm25_negatives_per_query"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            atomic_json(path, config)
            self.assertEqual(load_config(path)["negatives_per_query"], 4)
            for quota in (0, -1, 4, True):
                atomic_json(path, {**config, "bm25_negatives_per_query": quota})
                with self.assertRaisesRegex(ValueError, "bm25_negatives_per_query"):
                    load_config(path)
        self.assertTrue(skipped_round_stage("mix-3", 4))

    def test_each_round_uses_only_selected_queries_and_resumes_committed_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"repo/function{i}()" for i in range(8)]
            tokenizer = TinyTokenizer(ids)
            fake_checkpoint(root / "model")
            atomic_jsonl(root / "queries.jsonl", [{"text": f"find {i}", "url_based_id": target,
                                                   "positive_url_based_id": [target]} for i, target in enumerate(ids)])
            atomic_jsonl(root / "bm25.jsonl", [
                {"prompt": f"find {i}", "chosen_url_based_id": target,
                 "rejected_url_based_id": negative, "query_key": f"vault-{i}", "bm25_rank": rank}
                for i, target in enumerate(ids) for rank, negative in enumerate(ids, 1) if negative != target])
            path = root / "config.json"
            atomic_json(path, {"checkpoint_path": "model", "query_file": "queries.jsonl", "bm25_file": "bm25.jsonl",
                               "input_format": "multilabel", "target_type": "url_based_id", "output_dir": "run",
                               "queries_per_round": 3, "steps_per_round": 2, "num_beams": 8,
                               "negatives_per_query": 4, "precision": "fp32", "selection_metric": "mrr@8"})
            cfg, worker = load_config(path), FixtureWorker()
            original = prepare_bm25_pool

            def prepare(*args, **kwargs):
                return original(*args, **kwargs, tokenizer=tokenizer)

            # Failure after the mixed dataset is committed must not remine it.
            worker.fail_training = True
            with patch("pretrain.mix_bm25_preferences.prepare_bm25_pool", prepare):
                with self.assertRaisesRegex(RuntimeError, "training interruption"):
                    run_pipeline(cfg, runner=worker)
            mixed_hash = file_hash(root / "run/round-000/mixed_preferences.jsonl")
            manifest = root / "run/run_manifest.json"
            state = read_json(manifest)
            state["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = (
                "5e7e23777e5895cf06ae2e61730a54f3659e4e6564d92542d3e338de2bd24569")
            atomic_json(manifest, state)
            worker.fail_training = False
            worker.calls.clear()
            result = run_pipeline(cfg, runner=worker, resume=True)
            self.assertTrue(result["complete"])
            self.assertEqual(file_hash(root / "run/round-000/mixed_preferences.jsonl"), mixed_hash)
            self.assertFalse(any("--round-id" in c and c[c.index("--round-id") + 1] == "0"
                                 and "--evaluation-only" not in c for c in worker.calls))
            selected_keys = set()
            for i in range(result["planned_rounds"]):
                directory = root / f"run/round-{i:03d}"
                selected = {r["query_key"] for r in iter_json_records(directory / "queries.jsonl")}
                mixed = list(iter_json_records(directory / "mixed_preferences.jsonl"))
                self.assertFalse(selected & selected_keys)
                selected_keys.update(selected)
                self.assertEqual({r["query_key"] for r in mixed}, selected)
                self.assertTrue(all(v <= 4 for v in Counter(r["query_key"] for r in mixed).values()))
                self.assertTrue(all(r["round_id"] == i for r in mixed))
                self.assertEqual(read_json(directory / "round_inputs.json")["pairs_file"], str(directory / "mixed_preferences.jsonl"))
            worker.calls.clear()
            run_pipeline(cfg, runner=worker, resume=True)
            self.assertFalse(worker.calls)
            # Neither source data nor the source ratio can change on resume.
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline({**cfg, "bm25_negatives_per_query": 1}, runner=worker, resume=True)
            with (root / "bm25.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Resume config"):
                run_pipeline(cfg, runner=worker, resume=True)


@unittest.skipUnless(os.environ.get("RUN_DPO_INTEGRATION") == "1", "Set RUN_DPO_INTEGRATION=1")
class RealMixedBranchTest(unittest.TestCase):
    def test_actual_mixed_branch_training_and_final_inference(self):
        import torch
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
                                                   "positive_url_based_id": [target]} for i, target in enumerate(ids)])
            atomic_jsonl(root / "bm25.jsonl", [
                {"prompt": f"find C{i}", "chosen_url_based_id": target, "rejected_url_based_id": negative,
                 "query_key": f"vault-{i}", "chosen_text_id": f"old-{i}", "positive_text_ids": [f"old-{i}"],
                 "bm25_rank": rank, "bm25_score": None}
                for i, target in enumerate(ids) for rank, negative in enumerate(ids, 1) if negative != target])
            atomic_jsonl(root / "infer.jsonl", [{"text": "find C0", "url_based_id": ids[0]}])
            path = root / "config.json"
            atomic_json(path, {"checkpoint_path": "model", "initial_reference_checkpoint": "model",
                               "query_file": "queries.jsonl", "bm25_file": "bm25.jsonl", "infer_file": "infer.jsonl",
                               "input_format": "multilabel", "target_type": "url_based_id", "output_dir": "run",
                               "queries_per_round": 4, "steps_per_round": 1, "num_beams": 8,
                               "negatives_per_query": 6, "policy_mining_negatives_per_query": 4, "mining_batch_size": 2,
                               "max_prompt_length": 8, "max_target_length": 8, "precision": "fp32", "device": "cpu",
                               "selection_metric": "mrr@8", "reference_update": "ema", "reference_ema_decay": 1.0,
                               "eval_every_rounds": 8, "training": {"preference_objective": "branch-dpo",
                                   "branch_loss_weight": 0.1, "per_device_train_batch_size": 2,
                                   "gradient_accumulation_steps": 1, "save_steps": 1, "logging_steps": 1}})
            calls = []

            def runner(command):
                calls.append(command)
                if "--checkpoint-path" in command:
                    miner.mine(miner.build_parser().parse_args(command[2:]))
                elif "--decay" in command:
                    reference.update_reference(**vars(reference.build_parser().parse_args(command[2:])))
                else:
                    with patch.object(sys, "argv", command[1:]):
                        training.train_round(training.parse_args())

            cfg = load_config(path)
            result = run_pipeline(cfg, runner=runner)
            self.assertTrue(result["complete"])
            for i in range(result["planned_rounds"]):
                directory = root / f"run/round-{i:03d}"
                mixed = list(iter_json_records(directory / "mixed_preferences.jsonl"))
                annotated = list(iter_json_records(directory / "branch_preferences.jsonl"))
                mined = list(iter_json_records(directory / "preferences.jsonl"))
                self.assertTrue(all(n <= 4 for n in Counter(r["query_key"] for r in mined).values()))
                self.assertTrue(all(n <= 6 for n in Counter(r["query_key"] for r in mixed).values()))
                self.assertEqual(len(mixed), len(annotated))
                self.assertEqual({r["negative_source"] for r in annotated}, {"bm25", "model_confusion"})
                self.assertTrue(any(r["branch_active"] for r in annotated))
                self.assertEqual(read_json(directory / "branch_audit.json")["sequence_rows"], len(mixed))
            final_stats = read_json(root / f"run/round-{result['planned_rounds'] - 1:03d}/infer_predictions.stats.json")
            self.assertFalse(final_stats["beam_early_stopping"])
            self.assertIn("hit@5", result["final_inference"]["metrics"])
            calls.clear()
            run_pipeline(cfg, resume=True, runner=runner)
            self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
