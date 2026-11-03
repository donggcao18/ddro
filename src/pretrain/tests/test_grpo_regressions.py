"""Independent generation-score and optimizer-update regressions for GRPO."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretrain.calibrate_grpo_batch_size import probe_once
from pretrain.train_grpo_vault import build_trainer
from pretrain.infer_grpo_vault_url import run_inference
from scripts.bm25.the_vault.common import write_jsonl, iter_json_records
import test_grpo as fixtures
from utils.grpo_trainer import (
    GRPOSettings, TargetIndex, completion_mask, listwise_ranking_loss,
    resolve_score_normalization, token_logps, grpo_loss,
)


class RankingTermTests(unittest.TestCase):
    def test_ranking_gradient_and_duplicate_invariance(self):
        scores = torch.tensor([-3., -2., -1., -4.], requires_grad=True)
        loss, eligible = listwise_ranking_loss(scores, torch.tensor([1., 1., 0., 0.]), 4,
                                              [(1,), (2,), (3,), (4,)])
        loss.backward()
        self.assertEqual(eligible.item(), 1.)
        self.assertTrue(scores.grad[:2].lt(0).all())
        self.assertTrue(scores.grad[2:].gt(0).all())
        base = torch.tensor([-3., -1., -4.], requires_grad=True)
        repeated = torch.tensor([-3., -1., -3., -4.], requires_grad=True)
        a, _ = listwise_ranking_loss(base, torch.tensor([1., 0., 0.]), 3, [(1,), (2,), (3,)])
        b, _ = listwise_ranking_loss(repeated, torch.tensor([1., 0., 1., 0.]), 4,
                                    [(1,), (2,), (1,), (3,)])
        torch.testing.assert_close(a, b)
        a.backward()
        b.backward()
        torch.testing.assert_close(base.grad, repeated.grad[[0, 1, 3]])
        self.assertEqual(repeated.grad[2].item(), 0.)

    def test_constant_groups_keep_connected_zero_ranking_loss(self):
        scores = torch.tensor([-2., -1., -3., -4.], requires_grad=True)
        loss, eligible = listwise_ranking_loss(scores, torch.tensor([1., 1., 0., 0.]), 2,
                                              [(1,), (2,), (3,), (4,)])
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(eligible.item(), 0.)
        loss.backward()
        self.assertTrue(scores.grad.eq(0).all())

    def test_invalid_settings_do_not_silently_change_sample_distribution(self):
        with self.assertRaisesRegex(ValueError, "constrained sampling"):
            GRPOSettings(do_sample=True, score_normalization="full_vocab")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            GRPOSettings(ranking_loss_weight=-1.)

    def test_masked_padding_cannot_overflow_kl(self):
        scores = torch.zeros(2, 2, requires_grad=True)
        reference = torch.tensor([[.01, 1000.], [.02, 1000.]])
        mask = torch.tensor([[True, False], [True, False]])
        loss, kl = grpo_loss(scores, scores.detach(), torch.zeros(2), mask, .2, .04, reference)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(kl))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(scores.grad[:, 1].eq(0).all())


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TinyT5Tests()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def test_beam_scores_match_teacher_forcing_in_both_normalizations(self):
        for normalization in ("trie", "full_vocab"):
            trainer = self.fixture.trainer(normalization, ["--do_sample", "false", "--num_generations", "3",
                                                           "--score_normalization", normalization])
            index = TargetIndex(self.fixture.tokenizer, self.fixture.config, ["alpha", "alpha beta", "gamma"], 3)
            trainer.target_index = index
            trainer.model.eval()
            inputs = trainer.data_collator([trainer.train_dataset[0]])
            for training in (True, False):
                config = trainer.generation_config(training)
                self.assertEqual(config.renormalize_logits, normalization == "trie")
                with torch.no_grad():
                    generated = trainer.model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        generation_config=config, prefix_allowed_tokens_fn=index.allowed,
                        return_dict_in_generate=True, output_scores=True,
                    )
                    sequences = generated.sequences
                    mask = completion_mask(sequences[:, 1:], index.eos_id, index.pad_id)
                    logits = trainer.model(
                        input_ids=inputs["input_ids"].repeat_interleave(3, 0),
                        attention_mask=inputs["attention_mask"].repeat_interleave(3, 0),
                        decoder_input_ids=sequences[:, :-1],
                        decoder_attention_mask=torch.cat((torch.ones_like(mask[:, :1]), mask[:, :-1]), 1),
                        use_cache=False,
                    ).logits
                    scored = token_logps(logits, index.score_layout(sequences), 1., normalization)
                    sequence_scores = (scored * mask).sum(1)
                # HF's beam bookkeeping is independent of our teacher-forcing scorer.
                torch.testing.assert_close(sequence_scores, generated.sequences_scores, atol=2e-6, rtol=1e-5)
                self.assertEqual(sequence_scores.argsort(descending=True).tolist(), [0, 1, 2])

    def test_checkpoint_metadata_overrides_legacy_parent_manifest(self):
        trainer = self.fixture.trainer("metadata", ["--do_sample", "false", "--num_generations", "3"])
        destination = self.root / "metadata/checkpoint-1"
        trainer.save_model(str(destination))
        self.assertEqual(resolve_score_normalization(destination), "trie")
        (destination.parent / "grpo_run.json").write_text(json.dumps({"grpo": {}}), encoding="utf-8")
        self.assertEqual(resolve_score_normalization(destination), "trie")
        self.assertEqual(resolve_score_normalization(destination, "full_vocab"), "full_vocab")
        legacy = destination.parent / "legacy-checkpoint"
        self.assertEqual(resolve_score_normalization(legacy), "full_vocab")
        self.assertEqual(resolve_score_normalization(self.root / "sft"), "trie")

    def test_saved_policy_inference_matches_trainer_decoding(self):
        for normalization in ("trie", "full_vocab"):
            trainer = self.fixture.trainer("inference-" + normalization, [
                "--do_sample", "false", "--num_generations", "3", "--score_normalization", normalization])
            index = TargetIndex(self.fixture.tokenizer, self.fixture.config, ["alpha", "alpha beta", "gamma"], 3)
            trainer.target_index = index
            inputs = trainer.data_collator([trainer.train_dataset[0]])
            sequences = trainer.generate_sequences(trainer.model, inputs, training=False)
            expected = [index.sequence_to_target[index.canonical(row)] for row in sequences.tolist()]
            destination = self.root / ("inference-" + normalization) / "final"
            trainer.save_model(str(destination))
            corpus = self.root / "inference-corpus.jsonl"
            write_jsonl(corpus, [{"target": target} for target in index.target_to_sequence])
            test = self.root / "inference-test.jsonl"
            write_jsonl(test, [{"text": "find first", "url_based_id": "alpha", "positive_url_based_id": ["alpha"]}])
            args = SimpleNamespace(model_path=str(destination), test_file=str(test), corpus_file=str(corpus),
                                   output_dir=str(destination.parent / "predictions"), batch_size=1, num_beams=10,
                                   max_prompt_length=16, max_target_length=3, device="cpu", fp16=False,
                                   score_normalization="auto")
            metrics = run_inference(args)
            self.assertEqual(metrics["score_normalization"], normalization)
            record = next(iter(iter_json_records(Path(args.output_dir) / "predictions.jsonl")))
            self.assertEqual(record["top_10_predictions"], expected)

    def training_run(self, name, lr, batch=1, accumulation=1, sgd=False):
        args = self.fixture.arguments(name, [
            "--do_sample", "false", "--num_generations", "3", "--max_steps", "1",
            "--learning_rate", str(lr), "--warmup_ratio", "0",
            "--per_device_train_batch_size", str(batch), "--gradient_accumulation_steps", str(accumulation),
            "--no-gradient_checkpointing",
        ])
        args.eval_file = None
        args.validation_split = 0
        trainer = build_trainer(args)
        trainer.args.disable_tqdm = True
        trainer.args.max_grad_norm = 0.
        trainer.args.lr_scheduler_type = "constant"
        if sgd:
            trainer.args.optim = "sgd"
        trainer.train_dataset = Dataset.from_list([
            {"prompt": "find first", "positive_targets": ["alpha"]},
            {"prompt": "find second", "positive_targets": ["alpha"]},
        ])
        sequence = torch.tensor([[0, 4, 1], [0, 5, 1], [0, 6, 1]], device=trainer.args.device)

        def fixed_rollouts(model, inputs, training):
            return sequence.repeat(len(inputs["positive_targets"]), 1)

        initial = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
        with patch.object(trainer, "generate_sequences", side_effect=fixed_rollouts):
            trainer.train()
        delta = sum((value.detach().double() - initial[name].double()).square().sum()
                    for name, value in trainer.model.state_dict().items()).sqrt().item()
        return trainer, delta

    def test_learning_rate_changes_actual_adamw_parameter_updates(self):
        low, low_delta = self.training_run("lr-low", 1e-6)
        high, high_delta = self.training_run("lr-high", 1e-4)
        self.assertGreater(low_delta, 0.)
        self.assertGreater(high_delta, 50 * low_delta)
        for trainer in (low, high):
            records = [row for row in trainer.state.log_history if "grpo/sampled_update_rms" in row]
            self.assertTrue(records)
            self.assertGreater(records[0]["grpo/sampled_update_rms"], 0.)
            self.assertEqual(records[0]["grpo/optimizer_step_skipped"], 0.)

    def test_accumulated_microbatches_match_full_batch_update(self):
        full, _ = self.training_run("full-batch", .001, batch=2, sgd=True)
        accumulated, _ = self.training_run("accumulated", .001, accumulation=2, sgd=True)
        for name, expected in full.model.state_dict().items():
            torch.testing.assert_close(accumulated.model.state_dict()[name], expected, atol=2e-7, rtol=1e-5)

    def test_optional_ranking_term_is_separate_from_exact_rewards(self):
        trainer = self.fixture.trainer("ranking", ["--do_sample", "false", "--num_generations", "3",
                                                  "--ranking_loss_weight", "0.1"])
        inputs = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[0]]))
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        trainer.log({"loss": loss.item()})
        record = trainer.state.log_history[-1]
        self.assertAlmostEqual(record["grpo/reward"], 2 / 3, places=6)
        self.assertEqual(record["grpo/ranking_groups"], 1.)
        self.assertGreater(record["grpo/ranking_loss"], 0.)
        self.assertAlmostEqual(record["grpo/total_loss"], record["grpo/policy_loss"] + record["grpo/kl_loss"]
                               + record["grpo/weighted_ranking_loss"], places=6)

    def test_calibration_exercises_full_vocab_ranking_and_resident_gradients(self):
        args = SimpleNamespace(
            checkpoint_path=str(self.root / "sft"), beta=.04, reserve_gib=0,
            num_gpus=1, fp16=False, max_prompt_length=8, num_generations=3,
            temperature=1., do_sample="false", score_normalization="full_vocab",
            ranking_loss_weight=.1, gradient_accumulation_steps=2,
        )
        with patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.synchronize"), \
                patch("torch.cuda.max_memory_allocated", return_value=123):
            self.assertEqual(probe_once(args, self.fixture.index, torch.device("cpu"), 2), (True, 123))


if __name__ == "__main__":
    unittest.main()
