"""Numerical, gradient, padding, and actual TRL parity checks on a tiny local T5."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
import torch
import torch.nn.functional as F
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
from trl import DPOConfig, DPOTrainer
from pretrain.branch_preference_trainer import (
    BranchCollator, BranchDPOConfig, BranchDPOTrainer, local_branch_statistics,
)
from pretrain.branch_preference_utils import assign_branch_weights, describe_branch, tokenizer_fingerprint
from pretrain.train_ddro_vault import disable_t5_attention_dropout

torch.set_num_threads(1)


def tiny_checkpoint(root):
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "find": 3, "A": 4, "B": 5}
    for i in range(8):
        vocab[f"C{i}"] = len(vocab)
        vocab[f"D{i}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                       eos_token="</s>", unk_token="<unk>")
    torch.manual_seed(19)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(vocab), d_model=16, d_kv=4, d_ff=32, num_layers=1,
        num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    ))
    model.save_pretrained(root)
    tokenizer.save_pretrained(root)
    return model, tokenizer


def annotated_rows(tokenizer, model):
    rows = []
    for i, (chosen, rejected) in enumerate((("A B C0 D0", "A B C2 D2"), ("A C1", "A B C3 D3"))):
        row = {"prompt": "find", "chosen": chosen, "rejected": rejected, "query_key": f"q{i}",
               "branch_positive_targets": [chosen],
               "branch_tokenizer_fingerprint": tokenizer_fingerprint(tokenizer, model.config)}
        row.update(describe_branch(row["query_key"], tokenizer(chosen)["input_ids"],
                                   tokenizer(rejected)["input_ids"], [tokenizer(chosen)["input_ids"]]))
        rows.append(row)
    assign_branch_weights(rows)
    return rows


class BranchLossTest(unittest.TestCase):
    def test_formula_and_gradients_use_one_history(self):
        policy = torch.randn(2, 4, 7, requires_grad=True)
        reference = torch.randn(2, 4, 7, requires_grad=True)
        batch = {"branch_position": torch.tensor([2, 0]), "branch_preferred_token": torch.tensor([3, 4]),
                 "branch_rejected_token": torch.tensor([5, 6]), "branch_active": torch.tensor([True, False])}
        local, _, _ = local_branch_statistics(policy, reference, batch, 0.4)
        p, r = policy.float().log_softmax(-1), reference.float().log_softmax(-1)
        expected = F.softplus(-0.4 * (p[0, 2, 3] - p[0, 2, 5] - r[0, 2, 3] + r[0, 2, 5]))
        torch.testing.assert_close(local[0], expected)
        self.assertEqual(local[1].item(), 0)
        local.sum().backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(torch.count_nonzero(policy.grad).item(), 2)
        self.assertLess(policy.grad[0, 2, 3].item(), 0)
        self.assertGreater(policy.grad[0, 2, 5].item(), 0)

    def test_half_precision_extreme_logits_remain_finite(self):
        policy = torch.tensor([[[65000.0, -65000.0]]], dtype=torch.float16, requires_grad=True)
        reference = -policy.detach()
        batch = {"branch_position": torch.tensor([0]), "branch_preferred_token": torch.tensor([1]),
                 "branch_rejected_token": torch.tensor([0]), "branch_active": torch.tensor([True])}
        local, _, _ = local_branch_statistics(policy, reference, batch, 0.4)
        self.assertTrue(torch.isfinite(local).all())
        local.sum().backward()
        self.assertTrue(torch.isfinite(policy.grad).all())

    def test_invalid_coefficients_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            for key, value in (("branch_loss_weight", -1), ("branch_beta", 0), ("branch_beta", float("nan"))):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    BranchDPOConfig(output_dir=temp, **{key: value}, report_to="none")


class ActualTrainerTest(unittest.TestCase):
    def make_trainer(self, root, weight):
        model, tokenizer = tiny_checkpoint(root / "model")
        reference = deepcopy(model)
        disable_t5_attention_dropout(model)
        disable_t5_attention_dropout(reference)
        rows = annotated_rows(tokenizer, model)
        args = BranchDPOConfig(output_dir=str(root / "training"), branch_loss_weight=weight,
                               branch_beta=0.4, beta=0.4, max_prompt_length=8,
                               max_target_length=8, max_length=16, dataset_num_proc=1,
                               remove_unused_columns=False, report_to="none", use_cpu=True)
        trainer = BranchDPOTrainer(model=model, ref_model=reference, args=args, tokenizer=tokenizer,
                                   train_dataset=Dataset.from_list(rows), is_encoder_decoder=True)
        return trainer, tokenizer, rows

    def test_zero_coefficient_exact_loss_and_gradient_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            branch, tokenizer, rows = self.make_trainer(root, 0)
            ordinary_model = deepcopy(branch.model)
            ordinary_ref = deepcopy(branch.ref_model)
            args = DPOConfig(output_dir=str(root / "ordinary"), beta=0.4, max_prompt_length=8,
                             max_target_length=8, max_length=16, dataset_num_proc=1,
                             remove_unused_columns=False, report_to="none", use_cpu=True)
            ordinary = DPOTrainer(model=ordinary_model, ref_model=ordinary_ref, args=args, tokenizer=tokenizer,
                                  train_dataset=Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")}
                                                                  for r in rows]), is_encoder_decoder=True)
            branch_batch = branch._prepare_inputs(branch.data_collator(list(branch.train_dataset)))
            ordinary_batch = ordinary._prepare_inputs(ordinary.data_collator(list(ordinary.train_dataset)))
            for key in ("chosen_labels", "rejected_labels", "prompt_input_ids"):
                torch.testing.assert_close(branch_batch[key], ordinary_batch[key], rtol=0, atol=0)
            left, _ = branch.get_batch_loss_metrics(branch.model, branch_batch)
            right, _ = ordinary.get_batch_loss_metrics(ordinary.model, ordinary_batch)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            left.backward()
            right.backward()
            for p, q in zip(branch.model.parameters(), ordinary.model.parameters()):
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_auxiliary_loss_padding_reference_freeze_and_saved_config(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer, _, _ = self.make_trainer(Path(temp), 0.1)
            batch = trainer._prepare_inputs(trainer.data_collator(list(trainer.train_dataset)))
            self.assertEqual(batch["chosen_labels"][1, -1].item(), -100)
            loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch)
            self.assertAlmostEqual(metrics["loss/sequence"].item(), 0.69314718, places=6)
            self.assertAlmostEqual(metrics["loss/branch"].item(), 0.69314718, places=6)
            self.assertAlmostEqual(loss.item(), 1.1 * 0.69314718, places=6)
            loss.backward()
            self.assertTrue(all(p.grad is None for p in trainer.ref_model.parameters()))
            saved = trainer.args.to_dict()
            self.assertEqual(saved["branch_loss_weight"], 0.1)
            self.assertEqual(saved["preference_objective"], "branch-dpo")
            batch["branch_active"].fill_(False)
            batch["branch_loss_weight"].zero_()
            loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(metrics["loss/branch"].item(), 0)


if __name__ == "__main__":
    unittest.main()
