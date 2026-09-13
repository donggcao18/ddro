"""Opt-in real T5 mining/training/resume test; all weights are created locally."""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))


@unittest.skipUnless(os.environ.get("RUN_DPO_INTEGRATION") == "1", "Set RUN_DPO_INTEGRATION=1 with training dependencies installed")
class RealT5RoundTest(unittest.TestCase):
    def test_ema_tied_parameters_buffers_and_decay_endpoints(self):
        import torch
        from pretrain.update_dpo_reference import ema_parameters

        class Model(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
                self.tied_weight = self.weight
                self.register_buffer("average", torch.tensor([value], dtype=torch.float32))
                self.register_buffer("count", torch.tensor([int(value)]))

        for decay in (0.0, 0.9, 1.0):
            reference, policy = Model(2), Model(12)
            ema_parameters(reference, policy, decay)
            expected = torch.tensor([2 * decay + 12 * (1 - decay)])
            torch.testing.assert_close(reference.weight, expected)
            torch.testing.assert_close(reference.average, expected)
            self.assertIs(reference.weight, reference.tied_weight)
            self.assertEqual(reference.count.item(), 12)
            self.assertEqual(policy.weight.item(), 12)

    def test_two_round_training_frozen_reference_and_interrupted_resume(self):
        self.exercise_rounds(multilabel=False)

    def test_url_multilabel_training_and_resume_without_corpus(self):
        self.exercise_rounds(multilabel=True)

    def test_ema_reference_training_and_resume(self):
        self.exercise_rounds(multilabel=True, ema=True)

    def test_truncated_url_targets_mining_training_and_resume(self):
        self.exercise_rounds(multilabel=True, truncate=True)

    def test_allow_unknown_padding_and_collisions_in_mining_training_resume(self):
        self.exercise_rounds(multilabel=True, allow_tokens=True)

    def test_change_evaluation_schedule_after_round_zero_training(self):
        self.exercise_rounds(multilabel=True, schedule_change=True)

    def exercise_rounds(self, multilabel, ema=False, truncate=False, allow_tokens=False, schedule_change=False):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, TrainerCallback, set_seed
        import mine_model_confusion_negatives as miner
        from pretrain import train_ddro_vault as trainer_module
        from pretrain import update_dpo_reference as ema_module
        from pretrain.iterative_dpo_utils import atomic_json, atomic_jsonl, file_hash, read_json
        from common import iter_json_records
        from pretrain.train_iterative_ddro_vault import EVALUATION_COMPATIBLE_CONTROLLERS, load_config, run_pipeline

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"repo/lib/file_{i}.rb/function_{i}(arg)" if multilabel else f"doc-{i}" for i in range(8)]
            if allow_tokens:
                ids[:2] = [f"repo/<pad>/file_{i}.rb/function_{i}(arg)" for i in range(2)]
            pre_tokenizer = Whitespace() if multilabel else WhitespaceSplit()
            vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "find": 3}
            for target in ids:
                for token, _ in pre_tokenizer.pre_tokenize_str(target):
                    if allow_tokens and token in {"file_0", "file_1", "function_0", "function_1"}:
                        continue
                    if token not in vocab:
                        vocab[token] = len(vocab)
            backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = pre_tokenizer
            backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                               eos_token="</s>", unk_token="<unk>")
            set_seed(19)
            model = T5ForConditionalGeneration(T5Config(
                vocab_size=len(vocab), d_model=16, d_kv=4, d_ff=32,
                num_layers=1, num_decoder_layers=1, num_heads=2,
                decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
            ))
            model.save_pretrained(root / "sft")
            tokenizer.save_pretrained(root / "sft")
            atomic_jsonl(root / "corpus.jsonl", [{"text_id": f"doc-{i}"} for i in range(8)])
            atomic_jsonl(root / "queries.jsonl", [{"prompt": f"find doc-{i}", "target_text_id": f"doc-{i}"}
                                                  for i in range(8)])
            if multilabel:
                atomic_jsonl(root / "queries.jsonl", [
                    {"text": f"find {ids[i]}", "url_based_id": ids[i],
                     "family_id": f"family-{i // 2}",
                     "positive_url_based_id": [ids[i]] if allow_tokens and i < 2 else [ids[i], ids[i ^ 1]]}
                    for i in range(8)
                ])
                (root / "corpus.jsonl").unlink()
            config_path = root / "config.json"
            atomic_json(config_path, {
                "checkpoint_path": "sft", "corpus_files": ["corpus.jsonl"], "query_file": "queries.jsonl",
                "output_dir": "straight", "rounds": 2, "queries_per_round": 4, "steps_per_round": 2,
                "precision": "fp32", "device": "cpu", "num_beams": 8, "negatives_per_query": 2,
                "mining_batch_size": 2, "max_prompt_length": 8, "max_target_length": 4,
                "selection_metric": "mrr@8", "training": {
                    "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-3, "warmup_ratio": 0.0, "save_steps": 1, "logging_steps": 1,
                    "dataset_num_proc": 1, "dataloader_num_workers": 0,
                },
            })
            if multilabel:
                config = read_json(config_path)
                config.pop("corpus_files")
                config.pop("steps_per_round")
                config.update(input_format="multilabel", target_type="url_based_id",
                              max_prompt_length=32, max_target_length=32,
                              epochs_per_round=1, num_beams=10, negatives_per_query=4,
                              selection_metric="mrr@10")
                atomic_json(config_path, config)
            if ema:
                config = read_json(config_path)
                config.pop("rounds")  # Exercise full-data scheduling with a smaller final partition.
                config.update(reference_update="ema", reference_ema_decay=0.9)
                atomic_json(config_path, config)
            if truncate:
                atomic_json(config_path, {**read_json(config_path), "target_length_policy": "truncate",
                                          "max_target_length": 8})
            if schedule_change:
                atomic_json(config_path, {**read_json(config_path), "eval_every_epochs": 8})
            if allow_tokens:
                atomic_json(config_path, {**read_json(config_path), "target_token_policy": "allow",
                                          "target_collision_policy": "allow"})
                # Isolate the trainer preflight from the random validation split.
                from datasets import Dataset, DatasetDict
                shared_targets = DatasetDict(train=Dataset.from_list([
                    {"chosen": ids[0], "rejected": ids[2]}, {"chosen": ids[1], "rejected": ids[3]}]))
                trainer_module.validate_target_sequences(shared_targets, tokenizer, 32,
                                                          collision_policy="allow", token_policy="allow")
            loaded = []
            original_loader = trainer_module.load_policy_model
            def tracked_loader(*args):
                result = original_loader(*args)
                loaded.append((result, {k: v.detach().clone() for k, v in result.state_dict().items()}))
                return result
            interrupted = [False]
            interrupt_enabled = [False]
            ema_interrupted = [False]
            evaluation_interrupted = [False]
            real_trainer = trainer_module.DPOTrainer
            class InterruptOnce(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    if interrupt_enabled[0] and not interrupted[0] and not schedule_change:
                        interrupted[0] = True
                        raise RuntimeError("intentional interruption after checkpoint save")
            def tracked_trainer(*args, **kwargs):
                kwargs["callbacks"].append(InterruptOnce())
                result = real_trainer(*args, **kwargs)
                if truncate:
                    for i in range(len(result.train_dataset)):
                        for column in ("chosen", "rejected"):
                            original = kwargs["train_dataset"][i][column]
                            expected = tokenizer(original, truncation=True, max_length=8)["input_ids"]
                            self.assertGreater(len(tokenizer(original)["input_ids"]), 8)
                            self.assertEqual(result.train_dataset[i][column + "_labels"], expected)
                            self.assertEqual(expected[-1], tokenizer.eos_token_id)
                batch = result._prepare_inputs(result.data_collator([result.train_dataset[i] for i in range(2)]))
                result.model.train()
                initial_loss, _ = result.get_batch_loss_metrics(result.model, batch)
                if all(torch.equal(v, loaded[1][1][k]) for k, v in loaded[0][1].items()):
                    self.assertAlmostEqual(initial_loss.item(), float(torch.log(torch.tensor(2.0))), places=6)
                else:
                    self.assertTrue(ema)
                    self.assertTrue(torch.isfinite(initial_loss).item())
                return result

            def runner(command):
                if "--decay" in command:
                    options = vars(ema_module.build_parser().parse_args(command[2:]))
                    ema_module.update_reference(**options)
                    old = T5ForConditionalGeneration.from_pretrained(options["reference_checkpoint"]).state_dict()
                    policy = T5ForConditionalGeneration.from_pretrained(options["policy_checkpoint"]).state_dict()
                    mixed = T5ForConditionalGeneration.from_pretrained(options["output"]).state_dict()
                    for key in old:
                        torch.testing.assert_close(mixed[key], old[key] * 0.9 + policy[key] * 0.1,
                                                   rtol=1e-6, atol=1e-7)
                    if interrupt_enabled[0] and not ema_interrupted[0]:
                        ema_interrupted[0] = True
                        raise RuntimeError("intentional interruption after EMA export")
                elif "--checkpoint-path" in command:
                    if (schedule_change and interrupt_enabled[0] and not evaluation_interrupted[0]
                            and "--evaluation-only" in command
                            and command[command.index("--round-id") + 1] == "0"):
                        evaluation_interrupted[0] = True
                        raise RuntimeError("intentional interruption during evaluate-0")
                    miner.mine(miner.build_parser().parse_args(command[2:]))
                else:
                    loaded.clear()
                    with patch.object(sys, "argv", command[1:]), \
                         patch.object(trainer_module, "load_policy_model", tracked_loader), \
                         patch.object(trainer_module, "DPOTrainer", tracked_trainer):
                        try:
                            trainer_module.train_round(trainer_module.parse_args())
                        finally:
                            self.assertEqual(len(loaded), 2)
                            policy, before = loaded[0]
                            reference, frozen = loaded[1]
                            self.assertTrue(all(not p.requires_grad and p.grad is None for p in reference.parameters()))
                            self.assertTrue(all(torch.equal(v, reference.state_dict()[k]) for k, v in frozen.items()))
                            self.assertTrue(any(not torch.equal(v, policy.state_dict()[k]) for k, v in before.items()))
                            self.assertNotEqual(next(policy.parameters()).data_ptr(), next(reference.parameters()).data_ptr())
                    loaded.clear()
                    gc.collect()

            cfg = load_config(config_path)
            straight = run_pipeline(cfg, runner=runner)
            if allow_tokens:
                stats = read_json(root / "straight/baseline/validation_predictions.stats.json")
                self.assertEqual(stats["allowed_collision_groups"], 1)
                self.assertEqual(stats["excluded_collision_targets"], 0)
                self.assertEqual(stats["target_sequences_with_unknown_tokens"], 1)
                self.assertEqual(stats["target_sequences_with_padding_tokens"], 1)
                self.assertEqual(stats["usable_decoder_targets"], 7)
                self.assertEqual(tokenizer(ids[0])["input_ids"], tokenizer(ids[1])["input_ids"])
            if truncate:
                baseline_stats = read_json(root / "straight/baseline/validation_predictions.stats.json")
                self.assertEqual(baseline_stats["truncated_targets"], 8)
                self.assertEqual(baseline_stats["excluded_overlength_targets"], 0)
                self.assertEqual(baseline_stats["usable_decoder_targets"], 8)
                for result in straight["rounds"]:
                    self.assertEqual(result["pair_audit"]["query_coverage"], 1)
            if ema:
                plan = read_json(root / "straight/round_plan.json")
                self.assertEqual(plan["partition_sizes"], [4, 2])
            if multilabel:
                for result in straight["rounds"]:
                    self.assertEqual(result["optimizer_steps"], math.ceil(result["pair_audit"]["pairs"] / 2))
                    self.assertEqual(result["pair_audit"]["nominal_dataset_passes"], 1)
            else:
                self.assertEqual([r["optimizer_steps"] for r in straight["rounds"]], [2, 2])
            if multilabel:
                for round_id in range(2):
                    pairs = list(iter_json_records(root / f"straight/round-{round_id:03d}/preferences.jsonl"))
                    self.assertTrue(pairs)
                    for pair in pairs:
                        self.assertEqual(pair["target_type"], "url_based_id")
                        self.assertEqual(len(pair["positive_text_ids"]),
                                         1 if allow_tokens and pair["chosen"] in ids[:2] else 2)
                        self.assertIn(pair["chosen"], pair["positive_text_ids"])
                        self.assertNotIn(pair["rejected"], pair["positive_text_ids"])
                        self.assertIn(pair["rejected"], ids)
                        if allow_tokens:
                            rejected_tokens = tokenizer(pair["rejected"])["input_ids"]
                            self.assertTrue(all(rejected_tokens != tokenizer(p)["input_ids"]
                                                for p in pair["positive_text_ids"]))
            resumed_cfg = {**cfg, "output_dir": str(root / "resumed")}
            if schedule_change:
                resumed_cfg["eval_every_epochs"] = None
            interrupt_enabled[0] = True
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                run_pipeline(resumed_cfg, runner=runner)
            if schedule_change:
                manifest_path = root / "resumed/run_manifest.json"
                legacy = read_json(manifest_path)
                self.assertIn("train-0", legacy["stages"])
                self.assertNotIn("evaluate-0", legacy["stages"])
                legacy["identity"]["config"].pop("eval_every_epochs")
                legacy["identity"]["config"].pop("eval_at_end")
                legacy["identity"]["code"][str(SRC / "pretrain/train_iterative_ddro_vault.py")] = sorted(EVALUATION_COMPATIBLE_CONTROLLERS)[0]
                atomic_json(manifest_path, legacy)
                completion = root / "resumed/round-000/training/latest/round_complete.json"
                original_completion_hash = file_hash(completion)
                resumed_cfg["eval_every_epochs"] = 8
            if ema:
                with self.assertRaisesRegex(RuntimeError, "after EMA export"):
                    run_pipeline(resumed_cfg, resume=True, runner=runner)
            resumed = run_pipeline(resumed_cfg, resume=True, runner=runner)
            if schedule_change:
                self.assertEqual(file_hash(completion), original_completion_hash)
                self.assertIsNone(resumed["rounds"][0]["metrics"])
                self.assertIsNotNone(resumed["rounds"][1]["metrics"])
                self.assertEqual(len(resumed["identity_updates"]), 1)
            expected = T5ForConditionalGeneration.from_pretrained(straight["latest"]).state_dict()
            actual = T5ForConditionalGeneration.from_pretrained(resumed["latest"]).state_dict()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-7)
            round1 = read_json(root / "resumed/round-001/round_inputs.json")
            self.assertEqual(round1["policy_checkpoint"], str(root / "resumed/round-000/training/latest"))
            if ema:
                self.assertEqual(round1["reference_checkpoint"], str(root / "resumed/round-000/reference"))
                expected_ref = T5ForConditionalGeneration.from_pretrained(straight["latest_reference"]).state_dict()
                actual_ref = T5ForConditionalGeneration.from_pretrained(resumed["latest_reference"]).state_dict()
                for key in expected_ref:
                    torch.testing.assert_close(actual_ref[key], expected_ref[key], rtol=1e-5, atol=1e-7)
            self.assertTrue(resumed["complete"])


if __name__ == "__main__":
    unittest.main()
