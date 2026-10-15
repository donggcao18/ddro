"""Pure extraction/normalization/configuration checks without model downloads."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from pretrain.branch_preference_utils import (
    assign_branch_weights, describe_branch, first_divergence, validate_branch_rows,
)
from pretrain.iterative_dpo_utils import atomic_json
from pretrain.launch_branch_preference import build_parser, experiment_config
from pretrain.train_iterative_ddro_vault import load_config, skipped_round_stage


class BranchExtractionTest(unittest.TestCase):
    def test_first_divergence_and_same_prefix(self):
        self.assertEqual(first_divergence([3, 4, 5, 1], [3, 4, 6, 1]), 2)
        branch = describe_branch("q", [3, 4, 5, 1], [3, 4, 6, 1], [[3, 4, 5, 1]])
        self.assertTrue(branch["branch_active"])
        self.assertEqual(branch["branch_prefix_tokens"], [3, 4])
        self.assertEqual(branch["branch_preferred_token"], 5)
        self.assertEqual(branch["branch_rejected_token"], 6)

    def test_empty_prefix_and_real_eos(self):
        self.assertEqual(describe_branch("q", [3, 1], [4, 1], [[3, 1]])["branch_position"], 0)
        branch = describe_branch("q", [3, 1], [3, 4, 1], [[3, 1]])
        self.assertEqual(branch["branch_preferred_token"], 1)
        self.assertEqual(branch["branch_prefix_tokens"], [3])

    def test_other_positive_protects_branch(self):
        branch = describe_branch("q", [3, 4, 1], [3, 5, 6, 1], [[3, 4, 1], [3, 5, 7, 1]])
        self.assertFalse(branch["branch_active"])
        self.assertEqual(branch["branch_skip_reason"], "negative_branch_contains_positive")
        self.assertEqual(branch["branch_positive_children"], [4, 5])

    def test_invalid_and_identical_sequences(self):
        self.assertFalse(describe_branch("q", [3, 1], [3, 1], [[3, 1]])["branch_active"])
        with self.assertRaisesRegex(ValueError, "EOS"):
            describe_branch("q", [3], [3, 4], [[3]])
        with self.assertRaisesRegex(ValueError, "known positives"):
            describe_branch("q", [3, 1], [4, 1], [[5, 1]])
        with self.assertRaisesRegex(ValueError, "corpus trie"):
            describe_branch("q", [3, 1], [4, 1], [[3, 1]], lambda h: [3])

    def rows(self):
        tokens = {"p": [3, 4, 1], "n": [3, 5, 1], "n2": [3, 5, 6, 1]}
        rows = []
        for query, rejected in (("q1", "n"), ("q1", "n2"), ("q2", "n")):
            row = {"query_key": query, "chosen": "p", "rejected": rejected,
                   "branch_positive_targets": ["p"], "branch_tokenizer_fingerprint": "fp"}
            row.update(describe_branch(query, tokens["p"], tokens[rejected], [tokens["p"]]))
            rows.append(row)
        assign_branch_weights(rows)
        return rows, tokens

    def test_duplicate_decisions_are_query_balanced(self):
        rows, _ = self.rows()
        self.assertEqual([r["branch_loss_weight"] for r in rows], [0.75, 0.75, 1.5])
        self.assertEqual(rows[0]["branch_key"], rows[1]["branch_key"])
        self.assertNotEqual(rows[0]["branch_key"], rows[2]["branch_key"])

    def test_schema_tokenizer_and_weights_are_verified(self):
        rows, tokens = self.rows()
        validate_branch_rows(rows, tokens.__getitem__, "fp")
        for key, value in (("branch_position", 0), ("branch_loss_weight", 7.0),
                           ("branch_positive_children", [4, 5]),
                           ("branch_tokenizer_fingerprint", "other")):
            bad = deepcopy(rows)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_branch_rows(bad, tokens.__getitem__, "fp")

    def test_empty_and_inactive_datasets(self):
        self.assertEqual(assign_branch_weights([])["sequence_rows"], 0)
        row = {"query_key": "q", **describe_branch("q", [3, 1], [4, 1], [[3, 1], [4, 1]])}
        audit = assign_branch_weights([row])
        self.assertEqual(audit["eligible_queries"], 0)
        self.assertEqual(row["branch_loss_weight"], 0)


class BranchConfigTest(unittest.TestCase):
    def config(self, arm):
        args = build_parser().parse_args(["--checkpoint", "dpo", "--query-file", "q.jsonl",
                                          "--output-dir", "run", "--arm", arm])
        return experiment_config(args)

    def test_launcher_arms_and_frozen_reference(self):
        for arm in ("dpo", "zero", "branch"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                config = self.config(arm)
                path = Path(temp) / "config.json"
                atomic_json(path, config)
                loaded = load_config(path)
                self.assertEqual(loaded["reference_ema_decay"], 1)
                self.assertEqual(loaded["rounds"], 3)
                self.assertEqual(loaded["target_length_policy"], "truncate")
                self.assertEqual(loaded["target_collision_policy"], "allow")
                self.assertEqual(loaded["target_token_policy"], "allow")
                if arm != "dpo":
                    self.assertEqual(loaded["training"]["branch_loss_weight"], 0 if arm == "zero" else 0.1)
                else:
                    self.assertNotIn("preference_objective", loaded["training"])
        self.assertTrue(skipped_round_stage("branches-0", 1))

    def test_invalid_branch_options_fail(self):
        config = self.config("branch")
        for field, value in (("branch_loss_weight", -1), ("branch_beta", 0),
                             ("preference_objective", "tdpo")):
            with tempfile.TemporaryDirectory() as temp, self.subTest(field=field):
                bad = deepcopy(config)
                bad["training"][field] = value
                path = Path(temp) / "config.json"
                atomic_json(path, bad)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_explicit_target_policies_are_preserved(self):
        for length, collision in (("error", "error"), ("skip", "skip"), ("truncate", "allow")):
            with tempfile.TemporaryDirectory() as temp, self.subTest(length=length):
                args = build_parser().parse_args([
                    "--checkpoint", "dpo", "--query-file", "q.jsonl", "--output-dir", "run",
                    "--target-length-policy", length, "--target-collision-policy", collision,
                    "--target-token-policy", "error"])
                path = Path(temp) / "config.json"
                atomic_json(path, experiment_config(args))
                loaded = load_config(path)
                self.assertEqual(loaded["target_length_policy"], length)
                self.assertEqual(loaded["target_collision_policy"], collision)
                self.assertEqual(loaded["target_token_policy"], "error")


if __name__ == "__main__":
    unittest.main()
