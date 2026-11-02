"""Offline tests of multi-positive seq2seq GRPO, using a tiny random T5."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretrain.train_grpo_vault import build_trainer, main, parse_args
from pretrain.calibrate_grpo_batch_size import largest_fitting_batch, probe_once
from scripts.bm25.the_vault.common import write_jsonl
from utils.grpo_data import consolidate_queries, prepare_metadata, split_queries, validate_query_sets
from utils.grpo_trainer import (
    GRPOSettings, TargetIndex, completion_mask, constrained_logps,
    group_advantages, grpo_loss, retrieval_metrics,
)


torch.set_num_threads(1)


def tiny_checkpoint(directory: Path) -> tuple[PreTrainedTokenizerFast, T5Config]:
    vocabulary = {
        "[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
        "find": 3, "alpha": 4, "beta": 5, "gamma": 6,
        "first": 7, "second": 8, "third": 9, "fourth": 10, "validation": 11,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
    )
    config = T5Config(
        vocab_size=len(vocabulary), d_model=16, d_kv=8, d_ff=32,
        num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.3,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    torch.manual_seed(11)
    T5ForConditionalGeneration(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return tokenizer, config


class DataTests(unittest.TestCase):
    def test_union_split_and_missing_positive_validation(self):
        rows = consolidate_queries([
            {"prompt": "Query: Find A", "positive_targets": ["A"]},
            {"prompt": " find   a ", "positive_targets": ["B", "A"]},
            {"prompt": "find b", "positive_targets": ["B"]},
            {"prompt": "find c", "positive_targets": ["C"]},
        ])
        self.assertEqual(rows[0]["positive_targets"], ["A", "B"])
        first = split_queries(rows, 0.3, 42)
        self.assertEqual(first, split_queries(rows, 0.3, 42))
        validate_query_sets(*first, {"A", "B", "C"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_query_sets(rows, rows[:1], {"A", "B", "C"})
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_query_sets(rows, [], {"A", "B"})

    def test_metadata_all_target_types_and_explicit_v6_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [
                {"text_id": "old-a", "url_based_ids": ["repo/a"], "structure_id_v3s": ["v3-a"], "structure_id_v6s": ["v6-a"]},
                {"text_id": "old-b", "url_based_ids": ["repo/b"], "structure_id_v3s": ["v3-b"], "structure_id_v6s": ["v6-b"]},
                {"text_id": "old-c", "url_based_ids": ["repo/c"], "structure_id_v3s": ["v3-c"], "structure_id_v6s": ["v6-c"]},
            ]
            query = {"prompt": "find a", "target_text_id": "old-a", "positive_text_ids": ["old-a", "old-b"], "positive_structure_id_v6s": ["v6-c"]}
            write_jsonl(root / "documents.jsonl", docs)
            write_jsonl(root / "queries.jsonl", [query])
            for kind, expected in [
                ("text_id", ["old-a", "old-b"]), ("url", ["repo/a", "repo/b"]),
                ("structure_id_v3", ["v3-a", "v3-b"]), ("structure_id_v6", ["v6-a", "v6-b", "v6-c"]),
            ]:
                rows, corpus = prepare_metadata(root / "queries.jsonl", root / "documents.jsonl", kind)
                self.assertEqual(rows[0]["positive_targets"], expected)
                self.assertEqual(len(corpus), 3)
            query["positive_text_ids"].append("unknown")
            write_jsonl(root / "queries.jsonl", [query])
            with self.assertRaisesRegex(ValueError, "unmapped"):
                prepare_metadata(root / "queries.jsonl", root / "documents.jsonl", "structure_id_v6")


class ObjectiveTests(unittest.TestCase):
    def test_memory_calibration_finds_largest_integer_batch(self):
        for capacity in range(1, 34):
            trials = []

            def probe(batch_size):
                trials.append(batch_size)
                return batch_size <= capacity

            self.assertEqual(largest_fitting_batch(16, 32, probe), min(capacity, 32))
            self.assertTrue(all(1 <= value <= 32 for value in trials))
        with self.assertRaisesRegex(RuntimeError, "one query"):
            largest_fitting_batch(16, 32, lambda batch_size: False)
        with self.assertRaises(ValueError):
            largest_fitting_batch(32, 16, lambda batch_size: True)

    def test_advantages_and_eos_mask(self):
        rewards = torch.tensor([1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1.])
        advantages = group_advantages(rewards, 4)
        torch.testing.assert_close(advantages[:4], torch.tensor([1., -1., 1., -1.]), atol=3e-4, rtol=0)
        self.assertTrue(advantages[4:].eq(0).all())
        tokens = torch.tensor([[4, 1, 0, 0], [5, 6, 1, 0], [4, 6, 5, 6]])
        self.assertEqual(completion_mask(tokens, 1, 0).tolist(), [[True, True, False, False], [True, True, True, False], [True] * 4])

    def test_policy_gradient_clipping_and_kl(self):
        logps = torch.zeros(4, 2, requires_grad=True)
        advantages = torch.tensor([1., -1., 1., -1.])
        mask = torch.tensor([[True, True], [True, False], [True, True], [True, True]])
        loss, _ = grpo_loss(logps, logps.detach(), advantages, mask, 0.2)
        loss.backward()
        self.assertLess(logps.grad[0, 0].item(), 0)
        self.assertGreater(logps.grad[1, 0].item(), 0)
        self.assertEqual(logps.grad[1, 1].item(), 0)
        clipped = torch.tensor([[np.log(2)], [np.log(.5)]], requires_grad=True, dtype=torch.float32)
        loss, _ = grpo_loss(clipped, torch.zeros_like(clipped), torch.tensor([1., -1.]), torch.ones_like(clipped), .2)
        loss.backward()
        self.assertTrue(clipped.grad.eq(0).all())
        zero = torch.zeros(2, 2, requires_grad=True)
        loss, kl = grpo_loss(zero, zero.detach(), torch.zeros(2), torch.ones_like(zero), .2, .1, torch.full_like(zero, -.5))
        self.assertGreater(kl.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_zero_reward_loss_stays_connected(self):
        scores = torch.randn(8, 3, requires_grad=True)
        advantages = group_advantages(torch.zeros(8), 4)
        loss, _ = grpo_loss(scores, scores.detach(), advantages, torch.ones_like(scores), .2)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(scores.grad.eq(0).all())

    def test_zero_surrogate_value_has_policy_gradient_for_variable_lengths(self):
        scores = torch.tensor([[-.4, -.9, -.2], [-1.2, -.3, -.8], [-.7, -.2, -.5], [-.5, -.6, -.4]],
                              requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True], [True, False, False], [True, True, True]])
        advantages = group_advantages(torch.tensor([1., 0., 1., 0.]), 4)
        loss, _ = grpo_loss(scores, scores.detach(), advantages, mask, .2)
        self.assertAlmostEqual(loss.item(), 0., places=6)
        gradient, = torch.autograd.grad(loss, scores)
        diagnostic = -(advantages * ((scores * mask).sum(1) / mask.sum(1))).mean()
        diagnostic_gradient, = torch.autograd.grad(diagnostic, scores)
        torch.testing.assert_close(gradient, diagnostic_gradient)
        self.assertTrue(gradient[0, mask[0]].lt(0).all())
        self.assertTrue(gradient[1, mask[1]].gt(0).all())
        self.assertTrue(gradient[~mask].eq(0).all())

    def test_metric_definition(self):
        result = retrieval_metrics(type("Prediction", (), {"predictions": np.array([[1., .5], [.5, 1.]])})())
        self.assertEqual(result, {"mrr@10": .75, "recall@k": .75})


class TinyT5Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer, self.config = tiny_checkpoint(self.root / "sft")
        self.index = TargetIndex(self.tokenizer, self.config, ["alpha", "beta", "gamma"], 3)
        write_jsonl(self.root / "corpus.jsonl", [{"target": target} for target in ["alpha", "beta", "gamma"]])
        write_jsonl(self.root / "train.jsonl", [
            {"prompt": f"find {query}", "positive_targets": ["alpha", "beta"]}
            for query in ["first", "second", "third", "fourth"]
        ])
        write_jsonl(self.root / "eval.jsonl", [{"prompt": "find validation", "positive_targets": ["alpha", "beta"]}])

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, output="run", extra=None):
        arguments = [
            "train_grpo_vault.py", "--checkpoint_path", str(self.root / "sft"),
            "--train_file", str(self.root / "train.jsonl"),
            "--eval_file", str(self.root / "eval.jsonl"),
            "--corpus_file", str(self.root / "corpus.jsonl"),
            "--output_dir", str(self.root / output), "--max_target_length", "3",
            "--num_generations", "4", "--eval_num_beams", "3",
            "--max_steps", "2", "--save_steps", "1", "--eval_steps", "1",
            "--logging_steps", "1", "--gradient_accumulation_steps", "2",
            "--learning_rate", "0.001", "--warmup_ratio", "0", "--temperature", "0.7",
        ]
        with patch.object(sys, "argv", arguments + (extra or [])):
            return parse_args()

    def trainer(self, output="run", extra=None):
        trainer = build_trainer(self.arguments(output, extra))
        trainer.args.disable_tqdm = True
        return trainer

    def test_rewards_and_target_validation(self):
        sequences = torch.tensor([[0, 4, 1, 0], [0, 5, 1, 0], [0, 6, 1, 0], [0, 4, 0, 0]])
        self.assertEqual(self.index.rewards(sequences, [["alpha", "beta"]], 4).tolist(), [1., 1., 0., 0.])
        with self.assertRaisesRegex(ValueError, "increase"):
            TargetIndex(self.tokenizer, self.config, ["alpha beta"], 2)
        with self.assertRaisesRegex(ValueError, "collision"):
            TargetIndex(self.tokenizer, self.config, ["alpha", "alpha "], 3)
        with self.assertRaisesRegex(ValueError, "unknown"):
            TargetIndex(self.tokenizer, self.config, ["absent"], 3)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            GRPOSettings(num_generations=1)

    def test_capacity_probe_steps_and_oom_cleanup_on_cpu(self):
        # Exercise actual generation, backward and AdamW. Only GPU accounting is
        # replaced; this does not claim to validate CUDA capacity or FP16.
        args = SimpleNamespace(
            checkpoint_path=str(self.root / "sft"), beta=0.0, reserve_gib=0,
            num_gpus=1, fp16=False, max_prompt_length=8, num_generations=4,
            temperature=.7, do_sample="true",
        )
        original = T5ForConditionalGeneration.from_pretrained(self.root / "sft").shared.weight.detach().clone()
        with patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.synchronize"), \
                patch("torch.cuda.max_memory_allocated", return_value=123):
            for mode in ("true", "false"):
                args.do_sample = mode
                args.num_generations = 4 if mode == "true" else 3
                for beta in (0.0, .1):
                    args.beta = beta
                    fits, peak = probe_once(args, self.index, torch.device("cpu"), 2)
                    self.assertTrue(fits)
                    self.assertEqual(peak, 123)
            with patch("pretrain.calibrate_grpo_batch_size.AutoModelForSeq2SeqLM.from_pretrained",
                       side_effect=torch.cuda.OutOfMemoryError("synthetic OOM")):
                self.assertEqual(probe_once(args, self.index, torch.device("cpu"), 2), (False, 123))
        restored = T5ForConditionalGeneration.from_pretrained(self.root / "sft")
        torch.testing.assert_close(restored.shared.weight, original)

    def test_sampling_and_teacher_forcing_probabilities_match(self):
        trainer = self.trainer()
        # Shared prefixes and variable lengths exercise EOS and finished padding.
        index = TargetIndex(self.tokenizer, self.config, ["alpha", "alpha beta", "gamma"], 3)
        trainer.target_index = index
        model = trainer.model
        model.eval()
        inputs = trainer.data_collator([trainer.train_dataset[0]])
        torch.manual_seed(4)
        generated = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            generation_config=trainer.generation_config(True),
            prefix_allowed_tokens_fn=index.allowed,
            return_dict_in_generate=True, output_scores=True,
        )
        sequences = generated.sequences
        self.assertEqual(sequences.shape[0], 4)
        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"].repeat_interleave(4, 0),
                attention_mask=inputs["attention_mask"].repeat_interleave(4, 0),
                decoder_input_ids=sequences[:, :-1], use_cache=False,
            ).logits
        scored = constrained_logps(logits, index.score_layout(sequences), .7)
        sampled = torch.stack([
            scores.float().log_softmax(-1).gather(1, sequences[:, t + 1, None]).squeeze(1)
            for t, scores in enumerate(generated.scores)
        ], dim=1)
        mask = completion_mask(sequences[:, 1:], 1, 0)
        torch.testing.assert_close(scored[mask], sampled[mask], atol=1e-5, rtol=1e-5)
        # Ensure the T5 functional attention dropout is disabled too.
        self.assertTrue(all(module.dropout == 0 for module in model.modules() if isinstance(getattr(module, "dropout", None), float)))
        model.train()
        first = model(input_ids=inputs["input_ids"], decoder_input_ids=torch.tensor([[0, 4]])).logits
        second = model(input_ids=inputs["input_ids"], decoder_input_ids=torch.tensor([[0, 4]])).logits
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_deterministic_beam_rollouts_and_group_alignment(self):
        trainer = self.trainer(extra=["--do_sample", "False", "--num_generations", "3"])
        # Shared prefixes and variable lengths must produce valid, distinct IDs.
        index = TargetIndex(self.tokenizer, self.config, ["alpha", "alpha beta", "gamma"], 3)
        trainer.target_index = index
        inputs = trainer._prepare_inputs(trainer.data_collator([
            {"prompt": "find first", "positive_targets": ["alpha", "alpha beta"]},
            {"prompt": "find second", "positive_targets": ["gamma"]},
        ]))
        config = trainer.generation_config(True)
        self.assertFalse(config.do_sample)
        self.assertEqual((config.num_beams, config.num_return_sequences), (3, 3))
        self.assertEqual(trainer.grpo.scoring_temperature, 1.0)
        torch.manual_seed(5)
        first = trainer.generate_sequences(trainer.model, inputs, training=True)
        torch.manual_seed(77)
        second = trainer.generate_sequences(trainer.model, inputs, training=True)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(len(first), 6)
        for group in first.reshape(2, 3, -1):
            targets = [index.sequence_to_target[index.canonical(row)] for row in group.tolist()]
            self.assertEqual(set(targets), {"alpha", "alpha beta", "gamma"})
        rewards = index.rewards(first, inputs["positive_targets"], 3).reshape(2, 3)
        self.assertEqual(rewards.sum(1).tolist(), [2., 1.])
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        self.assertTrue(any(grad.abs().sum() > 0 for grad in gradients))
        manifest = json.loads((self.root / "run/grpo_run.json").read_text())
        self.assertEqual(manifest["rollout_mode"], "beam_selected_variant")
        self.assertFalse(manifest["grpo"]["do_sample"])
        self.assertEqual(manifest["scoring_temperature"], 1.)
        with self.assertRaisesRegex(ValueError, "usable corpus targets"):
            self.trainer("too-many-beams", ["--do_sample", "false"])

    def test_unprepared_dry_run_generation_enters_autocast(self):
        trainer = self.trainer(extra=["--do_sample", "false", "--num_generations", "3"])
        # No train()/prepare_model() call: reproduce the standalone dry-run path.
        trainer.model.train()
        inputs = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[0]]))
        dtypes = []
        handle = trainer.model.register_forward_hook(
            lambda model, arguments, output: dtypes.append(output.logits.dtype))
        try:
            trainer.generate_sequences(trainer.model, inputs, training=True)
            self.assertTrue(dtypes)
            self.assertEqual(set(dtypes), {torch.float32})
            dtypes.clear()
            # CPU BF16 exercises real lower-precision generation without CUDA.
            # On a V100 the configured Accelerator context instead uses FP16.
            with patch.object(trainer.accelerator, "autocast",
                              side_effect=lambda: torch.autocast("cpu", dtype=torch.bfloat16)) as autocast:
                sequences = trainer.generate_sequences(trainer.model, inputs, training=True)
            autocast.assert_called_once_with()
            self.assertTrue(dtypes)
            self.assertEqual(set(dtypes), {torch.bfloat16})
            self.assertEqual(len(sequences), 3)
            self.assertTrue(trainer.model.training)
            self.assertFalse(sequences.requires_grad)
            with patch.object(trainer.model, "generate", side_effect=RuntimeError("generation failure")):
                with self.assertRaisesRegex(RuntimeError, "generation failure"):
                    trainer.generate_sequences(trainer.model, inputs, training=True)
            self.assertTrue(trainer.model.training)
        finally:
            handle.remove()

    def test_beam_training_evaluation_checkpoint_and_resume(self):
        options = ["--do_sample", "false", "--num_generations", "3"]
        trainer = self.trainer("beam", options)
        original = trainer.model.shared.weight.detach().clone()
        trainer.train()
        self.assertFalse(torch.equal(original, trainer.model.shared.weight))
        self.assertEqual(trainer.evaluate()["eval_recall@k"], 1.)
        self.assertTrue((self.root / "beam/checkpoint-2/optimizer.pt").is_file())
        resumed = self.trainer("beam-resumed", options + ["--max_steps", "3"])
        resumed.train(resume_from_checkpoint=str(self.root / "beam/checkpoint-2"))
        self.assertEqual(resumed.state.global_step, 3)

    def test_zero_loss_update_increases_positive_target_probability(self):
        trainer = self.trainer("gradient-audit", ["--do_sample", "false", "--num_generations", "3"])
        inputs = trainer._prepare_inputs(trainer.data_collator([
            {"prompt": "find first", "positive_targets": ["alpha"]},
        ]))
        sequences = torch.tensor([[0, 4, 1], [0, 5, 1], [0, 6, 1]], device=trainer.args.device)

        def target_logps():
            with torch.no_grad():
                outputs = trainer.model(
                    input_ids=inputs["input_ids"].repeat_interleave(3, 0),
                    attention_mask=inputs["attention_mask"].repeat_interleave(3, 0),
                    decoder_input_ids=sequences[:, :-1], use_cache=False,
                )
                return constrained_logps(outputs.logits, trainer.target_index.score_layout(sequences), 1.)[:, 0]

        before = target_logps()
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.01)
        with patch.object(trainer, "generate_sequences", return_value=sequences):
            loss = trainer.compute_loss(trainer.model, inputs)
        self.assertAlmostEqual(loss.item(), 0., places=6)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.model.parameters()))
        optimizer.step()
        after = target_logps()
        self.assertGreater(after[0].item(), before[0].item())
        self.assertLess(after[1:].exp().sum().item(), before[1:].exp().sum().item())
        trainer.log({"loss": loss.item()})
        record = trainer.state.log_history[-1]
        self.assertGreater(record["grpo/advantage_abs_mean"], 0)
        advantages = group_advantages(torch.tensor([1., 0., 0.]), 3)
        expected_pg_loss = -(advantages * before.cpu() / 2).mean().item()  # EOS has logp zero.
        self.assertAlmostEqual(record["grpo/pg_loss"], expected_pg_loss, places=6)
        self.assertFalse(trainer._rollout_metrics["pg_loss"])

    def test_training_evaluation_save_and_resume(self):
        trainer = self.trainer()
        original = trainer.model.shared.weight.detach().clone()
        trainer.train()
        self.assertFalse(torch.equal(original, trainer.model.shared.weight))
        metrics = trainer.evaluate()
        self.assertAlmostEqual(metrics["eval_recall@k"], 1)
        self.assertGreater(metrics["eval_mrr@10"], 0)
        self.assertTrue((self.root / "run/checkpoint-2/optimizer.pt").is_file())
        self.assertIsNotNone(trainer.state.best_model_checkpoint)
        logs = [record for record in trainer.state.log_history if "grpo/reward" in record]
        self.assertTrue(logs)
        self.assertTrue(all(0 <= record["grpo/reward"] <= 1 for record in logs))
        destination = self.root / "run/final"
        trainer.save_model(str(destination))
        self.assertTrue((destination / "tokenizer.json").is_file())
        restored = T5ForConditionalGeneration.from_pretrained(destination, local_files_only=True)
        torch.testing.assert_close(restored.shared.weight, trainer.model.shared.weight)
        resumed = self.trainer("resumed", ["--max_steps", "3"])
        resumed.train(resume_from_checkpoint=str(self.root / "run/checkpoint-2"))
        self.assertEqual(resumed.state.global_step, 3)

    def test_optional_kl_and_zero_reward_groups_backward(self):
        trainer = self.trainer(extra=["--beta", "0.1"])
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.reference_model.parameters()))
        inputs = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[0]]))
        # Force an all-zero reward without changing the sampled trajectories.
        inputs["positive_targets"] = [[]]
        loss = trainer.compute_loss(trainer.model, inputs)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))

    def test_resume_matches_uninterrupted_training(self):
        class StopAfterOneStep(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_training_stop = True
                return control

        def training_run(name):
            args = self.arguments(name, ["--max_steps", "3"])
            args.eval_file = None
            args.validation_split = 0
            trainer = build_trainer(args)
            trainer.args.disable_tqdm = True
            return trainer

        uninterrupted = training_run("uninterrupted")
        uninterrupted.train()
        interrupted = training_run("interrupted")
        interrupted.add_callback(StopAfterOneStep())
        interrupted.train()
        self.assertEqual(interrupted.state.global_step, 1)
        resumed = training_run("continued")
        resumed.train(resume_from_checkpoint=str(self.root / "interrupted/checkpoint-1"))
        self.assertEqual(resumed.state.global_step, 3)
        for name, expected in uninterrupted.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[name], expected, atol=1e-7, rtol=1e-6)

    def test_dry_run_main_and_query_leakage(self):
        arguments = self.arguments("dry")
        arguments.dry_run = True
        with patch("pretrain.train_grpo_vault.parse_args", return_value=arguments):
            main()
        self.assertFalse(list((self.root / "dry").glob("checkpoint-*")))
        write_jsonl(self.root / "eval.jsonl", [{"prompt": "Query: FIND first", "positive_targets": ["alpha"]}])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.trainer("overlap")


if __name__ == "__main__":
    unittest.main()
