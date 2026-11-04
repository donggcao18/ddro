"""Token reward arithmetic, T5 alignment, frozen references and end-to-end updates."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import test_grpo as fixtures
from pretrain.calibrate_tca_grpo_batch_size import probe_once
from pretrain.train_grpo_vault import build_trainer
from pretrain.train_tca_grpo_vault import main as train_main
from utils.grpo_trainer import TargetIndex, grpo_loss
from utils.tca_grpo_trainer import (TCASettings, TCAGrpoTrainer, GoldTrajectoryCache,
                                     gold_trajectories, trajectory_rewards, token_group_advantages)


class TokenMathTests(unittest.TestCase):
    def test_masked_depth_normalization_matches_independent_numpy(self):
        rewards = torch.tensor([[1., 4., 8., 999.], [2., 6., 999., 999.], [3., 999., 999., 999.]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        actual, informative = token_group_advantages(rewards, mask, 3, .001)
        expected = np.zeros((3, 4), dtype=np.float32)
        for t in range(4):
            keep = mask[:, t].numpy()
            values = rewards[:, t].numpy()[keep]
            if len(values) > 1 and values.std() > .001:
                expected[keep, t] = (values - values.mean()) / values.std()
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
        self.assertEqual(informative.tolist(), [[True, True, False, False]])
        flat, _ = token_group_advantages(torch.ones(3, 2), torch.ones(3, 2, dtype=torch.bool), 3)
        self.assertTrue(flat.eq(0).all())

    def test_variance_floor_does_not_amplify_tiny_reward_noise(self):
        rewards = torch.tensor([[1., 1.], [1.00001, 1.000001]])
        advantages, active = token_group_advantages(rewards, torch.ones(2, 2, dtype=torch.bool), 2, .001)
        self.assertFalse(active.any())
        self.assertTrue(advantages.eq(0).all())

    def test_token_advantages_drive_correct_gradient_and_padding_is_ignored(self):
        logps = torch.zeros(2, 3, requires_grad=True)
        advantage = torch.tensor([[0., 1., 500.], [0., -1., -500.]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        loss, _ = grpo_loss(logps, logps.detach(), advantage, mask, .2)
        loss.backward()
        self.assertLess(logps.grad[0, 1], 0)
        self.assertGreater(logps.grad[1, 1], 0)
        self.assertTrue(logps.grad[:, [0, 2]].eq(0).all())

    def test_cache_is_detached_query_specific_and_bounded(self):
        cache = GoldTrajectoryCache(16 / 1024**2)
        cache.put(("q1", "a"), torch.ones(4, requires_grad=True))
        self.assertFalse(cache.get(("q1", "a")).requires_grad)
        self.assertIsNone(cache.get(("q2", "a")))
        cache.put(("q2", "a"), torch.zeros(4))
        self.assertIsNone(cache.get(("q1", "a")))
        self.assertLessEqual(cache.bytes, 16)
        cache.put("too large", torch.zeros(20))
        self.assertNotIn("too large", cache.entries)


class TCAIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TinyT5Tests()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def trainer(self, name="tca", **overrides):
        args = self.fixture.arguments(name, ["--num_generations", "3", "--temperature", "1"])
        settings = TCASettings(num_generations=3, eval_num_beams=3, max_target_length=3,
                               beta=.04, gold_cache_mb=1.)
        settings = replace(settings, **overrides)
        trainer = build_trainer(args, settings=settings, trainer_class=TCAGrpoTrainer, require_reference=True)
        trainer.args.disable_tqdm = True
        return trainer

    def test_gold_features_match_full_t5_forward_without_off_by_one(self):
        trainer = self.trainer()
        inputs = trainer.data_collator([trainer.train_dataset[0]])
        reference = trainer.reference_model
        for alignment, shift in (("pre_token", 0), ("post_token", 1)):
            settings = replace(trainer.grpo, trajectory_alignment=alignment)
            cache = GoldTrajectoryCache(1.)
            features = gold_trajectories(reference, inputs, [["alpha", "beta"]], trainer.target_index, settings, cache)
            with torch.no_grad():
                expected = reference(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                     decoder_input_ids=torch.tensor([[0, 4]]), output_hidden_states=True,
                                     use_cache=False).decoder_hidden_states[-1][0, shift:shift + 1]
            torch.testing.assert_close(features[(0, "alpha")], expected)
            with patch.object(reference, "get_encoder", side_effect=AssertionError("cache should satisfy this call")):
                cached = gold_trajectories(reference, inputs, [["alpha", "beta"]], trainer.target_index, settings, cache)
            torch.testing.assert_close(cached[(0, "alpha")], expected)

    def test_any_positive_gets_exact_bonus_and_overflow_ignores_eos_and_padding(self):
        trainer = self.trainer()
        index = TargetIndex(self.fixture.tokenizer, self.fixture.config, ["alpha", "beta", "alpha beta"], 3)
        sequences = torch.tensor([[0, 4, 5, 1], [0, 5, 1, 0], [0, 4, 1, 0]])
        hidden = torch.tensor([[[1., 0.]] * 3, [[0., 1.]] * 3, [[1., 0.]] * 3])
        features = {(0, "alpha"): torch.tensor([[1., 0.]]), (0, "beta"): torch.tensor([[0., 1.]])}
        rewards, exact = trajectory_rewards(hidden, sequences, [["alpha", "beta"]], [["alpha"]], features,
                                            index, trainer.grpo)
        torch.testing.assert_close(rewards, torch.tensor([[1., -.5, 0.], [3., 2., 0.], [3., 2., 0.]]))
        self.assertEqual(exact.tolist(), [0., 1., 1.])
        self.assertFalse(rewards.requires_grad)

    def test_best_gold_path_is_selected_as_a_whole(self):
        trainer = self.trainer(num_generations=2)
        index = TargetIndex(self.fixture.tokenizer, self.fixture.config, ["alpha beta", "beta alpha", "gamma gamma"], 3)
        sequences = torch.tensor([[0, 6, 6, 1]] * 2)
        hidden = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]] * 2)
        features = {(0, "alpha beta"): torch.tensor([[1., 0.], [1., 0.]]),
                    (0, "beta alpha"): torch.tensor([[0., 1.], [0., 1.]])}
        reward, exact = trajectory_rewards(hidden, sequences, [["alpha beta", "beta alpha"]],
                                           [["alpha beta", "beta alpha"]], features, index, trainer.grpo)
        torch.testing.assert_close(reward, torch.tensor([[1., 0., 0.]] * 2))
        self.assertEqual(exact.sum(), 0)

    def test_pre_token_cannot_distinguish_last_action_but_post_token_can(self):
        generated = torch.tensor([[0, 5, 1], [0, 6, 1], [0, 5, 1]])
        for alignment in ("pre_token", "post_token"):
            trainer = self.trainer(alignment, trajectory_alignment=alignment, beta=0.,
                                   all_zero_ranking_weight=0., sft_anchor_weight=0.)
            inputs = trainer.data_collator([{"prompt": "find first", "positive_targets": ["alpha"]}])
            with patch.object(trainer, "generate_sequences", return_value=generated):
                loss = trainer.compute_loss(trainer.model, inputs)
            magnitude = trainer._rollout_metrics["advantage_abs_mean"][-1]
            self.assertTrue(torch.isfinite(loss))
            if alignment == "pre_token":
                self.assertEqual(magnitude, 0.)
            else:
                self.assertGreater(magnitude, .1)
                loss.backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for p in trainer.model.parameters() if p.grad is not None), 0.)

    def test_all_wrong_fallback_updates_policy_and_reference_stays_frozen(self):
        trainer = self.trainer()
        inputs = trainer.data_collator([trainer.train_dataset[0]])
        generated = torch.tensor([[0, 6, 1]] * 3)
        reference_before = {name: value.clone() for name, value in trainer.reference_model.state_dict().items()}
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.001)
        with patch.object(trainer, "generate_sequences", return_value=generated):
            loss = trainer.compute_loss(trainer.model, inputs)
            self.assertEqual(trainer._rollout_metrics["all_zero_groups"][-1], 1.)
            self.assertGreater(trainer._rollout_metrics["all_zero_ranking_loss"][-1], 0.)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            after = trainer.compute_loss(trainer.model, inputs)
            self.assertLess(after.item(), loss.item())
        for name, value in trainer.reference_model.state_dict().items():
            torch.testing.assert_close(value, reference_before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in trainer.reference_model.parameters()))
        self.assertLessEqual(trainer.gold_cache.bytes, 1024**2)

    def test_real_training_and_checkpoint_resume_keep_sft_fallback_explicit(self):
        argv = ["train", "--checkpoint_path", str(self.root / "sft"), "--output_dir", str(self.root / "trained"),
                "--train_file", str(self.root / "train.jsonl"), "--eval_file", str(self.root / "eval.jsonl"),
                "--corpus_file", str(self.root / "corpus.jsonl"), "--max_prompt_length", "16", "--max_target_length", "3",
                "--num_generations", "3", "--eval_num_beams", "3", "--max_steps", "2", "--save_steps", "1",
                "--eval_steps", "1", "--logging_steps", "1", "--gradient_accumulation_steps", "2",
                "--learning_rate", ".001", "--warmup_ratio", "0", "--trajectory_alignment", "post_token"]
        with patch.object(sys, "argv", argv):
            train_main()
        report = json.loads((self.root / "trained/selection.json").read_text())
        self.assertEqual(report["selected"], "sft")
        manifest = json.loads((self.root / "trained/grpo_run.json").read_text())
        self.assertEqual(manifest["objective"], "tca_grpo")
        self.assertEqual(manifest["rollout_mode"], "sampled_grpo")
        state = json.loads((self.root / "trained/trainer_state.json").read_text())
        self.assertTrue(any(row.get("grpo/sampled_update_rms", 0.) > 0 for row in state["log_history"]))
        initial = AutoModelForSeq2SeqLM.from_pretrained(self.root / "sft").state_dict()
        final = AutoModelForSeq2SeqLM.from_pretrained(self.root / "trained/final").state_dict()
        for key in initial:
            torch.testing.assert_close(initial[key], final[key], rtol=0, atol=0)
        # Resume from an existing optimizer checkpoint into a new destination.
        resumed = [*argv, "--output_dir", str(self.root / "resumed"), "--resume_from_checkpoint",
                   str(self.root / "trained/checkpoint-1")]
        with patch.object(sys, "argv", resumed):
            train_main()
        resumed_state = json.loads((self.root / "resumed/trainer_state.json").read_text())
        self.assertEqual(resumed_state["global_step"], 2)

    def test_capacity_probe_uses_actual_tca_graph(self):
        args = SimpleNamespace(checkpoint_path=str(self.root / "sft"), num_gpus=1, reserve_gib=0., fp16=False,
                               max_prompt_length=8, max_target_length=3, num_generations=3, do_sample="true",
                               gradient_accumulation_steps=2, beta=.04, gold_candidates_per_query=2,
                               trajectory_alignment="post_token")
        with patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.synchronize"), \
                patch("torch.cuda.max_memory_allocated", return_value=123):
            self.assertEqual(probe_once(args, self.fixture.index, torch.device("cpu"), 2), (True, 123))


if __name__ == "__main__":
    unittest.main()
