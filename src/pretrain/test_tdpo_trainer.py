"""CPU tests: python -m unittest discover -s src/pretrain -p test_tdpo_trainer.py."""

import ast
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from typing import Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
from trl import DPOTrainer

from tdpo_trainer import PreferenceConfig, TokenDPOTrainer, sequence_statistics, tdpo_loss


class TDPOTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_reference_implementation_parity(self):
        source = Path(__file__).resolve().parents[2] / "Token-level-Direct-Preference-Optimization/trainers.py"
        if not source.is_file():
            self.skipTest("Optional upstream checkout not present")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"tdpo_loss", "_tdpo_get_batch_logps"}]
        namespace = {"torch": torch, "F": F, "Tuple": Tuple}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        labels = torch.tensor([[2, 3, 1, -100], [4, 1, -100, -100]])
        ref = torch.randn(2, 4, 8)
        for objective in ["tdpo1", "tdpo2"]:
            policy = torch.randn(2, 4, 8, requires_grad=True)
            actual = sequence_statistics(policy, ref, labels)
            # Adapt the upstream causal shift to these already-aligned decoder logits.
            upstream = namespace["_tdpo_get_batch_logps"](
                torch.cat([policy, policy[:, :1]], dim=1),
                torch.cat([ref, ref[:, :1]], dim=1),
                torch.cat([torch.zeros(2, 1, dtype=torch.long), labels], dim=1),
            )
            for a, b in zip(actual, upstream):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            margin, kl, _ = actual
            expected = namespace["tdpo_loss"](
                upstream[0][:1], upstream[0][1:], upstream[1][:1], upstream[1][1:],
                beta=0.4, alpha=0.5, if_tdpo2=objective == "tdpo2",
            )
            result = tdpo_loss(margin[:1], margin[1:], kl[:1], kl[1:], 0.4, 0.5, objective)
            for a, b in zip(result, expected):
                torch.testing.assert_close(a, b)
            expected_grad = torch.autograd.grad(expected[0].sum(), policy, retain_graph=True)[0]
            actual_grad = torch.autograd.grad(result[0].sum(), policy)[0]
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_masking_alignment_and_reference_gradients(self):
        policy = torch.randn(2, 3, 8, requires_grad=True)
        ref = torch.randn(2, 3, 8, requires_grad=True)
        labels = torch.tensor([[2, 3, 1], [4, 1, -100]])
        before = labels.clone()
        margin, kl, logps = sequence_statistics(policy, ref, labels)
        expected = policy.log_softmax(-1)[0, torch.arange(3), labels[0]].sum()
        torch.testing.assert_close(logps[0], expected)
        padded = sequence_statistics(
            torch.cat([policy, torch.randn(2, 2, 8) * 1000], dim=1),
            torch.cat([ref, torch.randn(2, 2, 8) * 1000], dim=1),
            torch.cat([labels, torch.full((2, 2), -100)], dim=1),
        )
        for original, extended in zip((margin, kl, logps), padded):
            torch.testing.assert_close(original, extended)
        (margin + kl).sum().backward()
        self.assertIsNone(ref.grad)
        self.assertEqual(policy.grad[1, 2].abs().sum().item(), 0)
        self.assertGreater(policy.grad[0, 0].abs().sum().item(), 0)
        torch.testing.assert_close(labels, before)

    def test_tdpo2_detaches_only_chosen_kl(self):
        for objective in ["tdpo1", "tdpo2"]:
            tensors = [torch.tensor([v], requires_grad=True) for v in [0.8, -0.2, 0.1, 0.4]]
            tdpo_loss(*tensors, 0.4, 0.5, objective)[0].sum().backward()
            for index in [0, 1, 3]:
                self.assertIsNotNone(tensors[index].grad)
                self.assertNotEqual(tensors[index].grad.item(), 0)
            if objective == "tdpo2":
                self.assertIsNone(tensors[2].grad)
            else:
                self.assertNotEqual(tensors[2].grad.item(), 0)

    def test_alpha_zero_matches_dpo_values_and_gradients(self):
        policy = torch.randn(2, 3, 8, requires_grad=True)
        ref = torch.randn_like(policy)
        labels = torch.tensor([[2, 3, 1], [4, 1, -100]])
        margin, kl, _ = sequence_statistics(policy, ref, labels)
        loss = tdpo_loss(margin[:1], margin[1:], kl[:1], kl[1:], 0.4, 0, "tdpo2")[0].sum()
        mask = labels != -100
        ids = labels.masked_fill(~mask, 0).unsqueeze(-1)
        logratio = (policy.log_softmax(-1).gather(-1, ids).squeeze(-1)
                    - ref.log_softmax(-1).gather(-1, ids).squeeze(-1))
        sums = (logratio * mask).sum(-1)
        dpo = -F.logsigmoid(0.4 * (sums[0] - sums[1]))
        torch.testing.assert_close(loss, dpo)
        actual_grad = torch.autograd.grad(loss, policy, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(dpo, policy)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_extreme_logits_are_finite(self):
        policy = torch.tensor([[[1000., -1000., 0.]]], dtype=torch.bfloat16, requires_grad=True)
        ref = -policy.detach()
        values = sequence_statistics(policy, ref, torch.tensor([[1]]))
        self.assertTrue(all(torch.isfinite(value).all() for value in values))
        sum(value.sum() for value in values).backward()
        self.assertTrue(torch.isfinite(policy.grad).all())

    def test_tiny_t5_train_eval_save_and_resume(self):
        tokenizer_impl = Tokenizer(models.WordLevel(
            {"<pad>": 0, "</s>": 1, "<unk>": 2, "query": 3, "chosen": 4, "rejected": 5, "long": 6},
            unk_token="<unk>",
        ))
        tokenizer_impl.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer_impl.post_processor = processors.TemplateProcessing(
            single="$A </s>", special_tokens=[("</s>", 1)]
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_impl, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
        )
        dataset = Dataset.from_dict({
            "prompt": ["query"] * 4,
            "chosen": ["chosen long", "chosen"] * 2,
            "rejected": ["rejected", "rejected long"] * 2,
        })
        for objective in ["dpo", "tdpo1", "tdpo2"]:
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                model = T5ForConditionalGeneration(T5Config(
                    vocab_size=8, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
                    num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
                    dropout_rate=0.0,
                ))
                reference = deepcopy(model).requires_grad_(False).eval()
                config = PreferenceConfig(
                    output_dir=directory, preference_objective=objective,
                    max_steps=1, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                    learning_rate=1e-3, max_prompt_length=8, max_target_length=8,
                    max_length=16, report_to=[], use_cpu=True, save_steps=1,
                    logging_steps=1, gradient_checkpointing=True, remove_unused_columns=False,
                    dataset_num_proc=1, disable_tqdm=True,
                )
                trainer_type = DPOTrainer if objective == "dpo" else TokenDPOTrainer
                trainer = trainer_type(
                    model=model, ref_model=reference, args=config, tokenizer=tokenizer,
                    train_dataset=dataset, eval_dataset=dataset, is_encoder_decoder=True,
                )
                before = model.shared.weight.detach().clone()
                trainer.train()
                self.assertFalse(torch.equal(before, model.shared.weight))
                self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
                evaluation = trainer.evaluate()
                self.assertTrue(torch.isfinite(torch.tensor(evaluation["eval_loss"])))
                if objective != "dpo":
                    self.assertIn("eval_kl/chosen", evaluation)
                saved = torch.load(Path(directory) / "checkpoint-1/training_args.bin", weights_only=False)
                self.assertEqual(saved.preference_objective, objective)
                self.assertEqual(saved.tdpo_alpha, 0.5)
                trainer.save_model(str(Path(directory) / "final"))
                restored = T5ForConditionalGeneration.from_pretrained(Path(directory) / "final")
                torch.testing.assert_close(restored.shared.weight, model.shared.weight)
                trainer.args.max_steps = 2
                trainer.train(resume_from_checkpoint=str(Path(directory) / "checkpoint-1"))
                self.assertEqual(trainer.state.global_step, 2)


if __name__ == "__main__":
    unittest.main()
