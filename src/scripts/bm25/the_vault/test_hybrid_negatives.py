"""Regression tests for additive Vault model-confusion and hybrid mining."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from combine_hybrid_dpo import combine
from common import iter_json_records
from mine_model_confusion_negatives import (
    build_document_targets,
    build_target_index,
    canonical_generated_tokens,
    select_model_candidates,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def base_pair(rejected: str, rank: int, source: str) -> dict:
    row = {
        "prompt": "find the target",
        "chosen": "target",
        "rejected": rejected,
        "query_key": "vault-000000000",
        "chosen_text_id": "target",
        "rejected_text_id": rejected,
        "positive_text_ids": ["target", "other-positive"],
    }
    if source == "bm25":
        row.update({"bm25_rank": rank, "bm25_score": 100.0 - rank})
    else:
        row.update(
            {
                "negative_source": "model_confusion",
                "model_rank": rank,
                "model_score": -float(rank),
            }
        )
    return row


class FakeTokenizer:
    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        ids = self.encodings[text]
        if kwargs.get("truncation") and len(ids) > kwargs["max_length"]:
            ids = ids[:kwargs["max_length"] - 1] + [ids[-1]]
        return {"input_ids": ids}


class ModelConfusionHelperTest(unittest.TestCase):
    def test_truncated_targets_keep_full_ids_and_reject_prefix_collisions(self):
        tokenizer = FakeTokenizer({"long-a": [10, 11, 12, 1], "long-b": [14, 15, 16, 1],
                                   "same-prefix": [10, 11, 99, 1]})
        encoded, mapping, excluded, groups, overlong = build_target_index(
            tokenizer, ["long-a", "long-b"], 3, length_policy="truncate")
        self.assertEqual(mapping, {(10, 11, 1): "long-a", (14, 15, 1): "long-b"})
        self.assertEqual(excluded, set())
        self.assertEqual(len(overlong), 2)
        self.assertTrue(all(len(tokens) == 3 and tokens[-1] == 1 for tokens in encoded))
        with self.assertRaisesRegex(ValueError, "same tokenizer target sequence"):
            build_target_index(tokenizer, ["long-a", "same-prefix"], 3, length_policy="truncate")

    def test_canonicalizes_and_filters_model_beams(self) -> None:
        sequence_to_id = {
            (11, 1): "target",
            (12, 1): "other-positive",
            (13, 1): "negative-a",
            (14, 1): "negative-b",
        }
        sequences = [
            [0, 11, 1, 0],
            [0, 12, 1, 0],
            [0, 13, 1, 0],
            [0, 13, 1, 0],
            [0, 14, 1, 0],
        ]
        selected, stats = select_model_candidates(
            sequences,
            [-0.1, -0.2, -0.3, -0.4, -0.5],
            sequence_to_id,
            {"target", "other-positive"},
            2,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
        self.assertEqual(
            canonical_generated_tokens([0, 13, 1, 0], 0, 0, 1),
            (13, 1),
        )
        self.assertEqual([row["target"] for row in selected], ["negative-a", "negative-b"])
        self.assertEqual([row["rank"] for row in selected], [3, 5])
        self.assertEqual(stats["positive_beams_filtered"], 2)
        self.assertEqual(stats["duplicate_beams_filtered"], 1)
        self.assertEqual(stats["queries_with_top1_positive"], 1)

    def test_rejects_tokenizer_collisions_and_long_docids(self) -> None:
        tokenizer = FakeTokenizer({"a": [10, 1], "b": [10, 1], "c": [11, 1]})
        with self.assertRaisesRegex(ValueError, "same tokenizer target sequence"):
            build_target_index(tokenizer, ["a", "b"], max_target_length=4)

        encoded, mapping, excluded, groups, overlong = build_target_index(
            tokenizer,
            ["a", "b", "c"],
            max_target_length=4,
            collision_policy="skip",
        )
        self.assertEqual(encoded, [[11, 1]])
        self.assertEqual(mapping, {(11, 1): "c"})
        self.assertEqual(excluded, {"a", "b"})
        self.assertEqual(groups, [["a", "b"]])
        self.assertEqual(overlong, [])

        tokenizer = FakeTokenizer({"a": [10, 11, 12, 1]})
        with self.assertRaisesRegex(ValueError, "must not be truncated"):
            build_target_index(tokenizer, ["a"], max_target_length=3)

        encoded, mapping, excluded, groups, overlong = build_target_index(
            FakeTokenizer({"short": [10, 1], "long": [10, 11, 12, 1]}),
            ["short", "long"],
            max_target_length=3,
            length_policy="skip",
        )
        self.assertEqual(encoded, [[10, 1]])
        self.assertEqual(mapping, {(10, 1): "short"})
        self.assertEqual(excluded, {"long"})
        self.assertEqual(groups, [])
        self.assertEqual(overlong, [(4, "long")])

    def test_builds_unique_url_targets_and_tracks_invalid_mappings(self) -> None:
        text_id_to_target, target_to_text_id, invalid = build_document_targets(
            {
                "text-a": {"url_based_ids": ["repo/a.rb/a()"]},
                "text-b": {"url_based_ids": ["repo/b.rb/b()"]},
                "text-c": {"url_based_ids": []},
            },
            "url",
        )
        self.assertEqual(
            text_id_to_target,
            {
                "text-a": "repo/a.rb/a()",
                "text-b": "repo/b.rb/b()",
            },
        )
        self.assertEqual(target_to_text_id["repo/a.rb/a()"], "text-a")
        self.assertEqual(invalid, 1)

        with self.assertRaisesRegex(ValueError, "maps to multiple text IDs"):
            build_document_targets(
                {
                    "text-a": {"url_based_ids": ["same-url"]},
                    "text-b": {"url_based_ids": ["same-url"]},
                },
                "url",
            )


class HybridCombinerTest(unittest.TestCase):
    def make_inputs(
        self,
        root: Path,
        model_ids: list[str],
    ) -> tuple[Path, Path, Path]:
        bm25 = root / "bm25.jsonl"
        model = root / "model.jsonl"
        metadata = root / "document_metadata.jsonl"
        bm25_ids = [f"bm25-{index}" for index in range(1, 9)]
        ranks = [1, 2, 3, 21, 22, 101, 102, 103]
        write_rows(
            bm25,
            [
                base_pair(text_id, rank, "bm25")
                for text_id, rank in zip(bm25_ids, ranks)
            ],
        )
        write_rows(
            model,
            [
                base_pair(text_id, rank, "model")
                for rank, text_id in enumerate(model_ids, start=1)
            ],
        )
        all_ids = {"target", "other-positive", *bm25_ids, *model_ids}
        write_rows(metadata, [{"text_id": text_id} for text_id in sorted(all_ids)])
        return bm25, model, metadata

    def run_combine(
        self,
        root: Path,
        model_ids: list[str],
        require_exact_mix: bool = False,
    ) -> tuple[list[dict], dict]:
        bm25, model, metadata = self.make_inputs(root, model_ids)
        output = root / "hybrid.jsonl"
        stats = combine(
            SimpleNamespace(
                bm25_input=str(bm25),
                model_input=str(model),
                document_metadata=str(metadata),
                output=str(output),
                model_per_query=4,
                total_per_query=8,
                seed=42,
                require_exact_mix=require_exact_mix,
            )
        )
        return list(iter_json_records(output)), stats

    def test_combines_four_model_and_four_diverse_bm25(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            rows, stats = self.run_combine(
                Path(temp_directory),
                ["model-1", "model-2", "model-3", "model-4"],
            )
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["rejected"] for row in rows}), 8)
            self.assertEqual(
                sum(row["negative_source"].startswith("model_confusion") for row in rows),
                4,
            )
            selected_bm25_ranks = [
                row["bm25_rank"]
                for row in rows
                if row["negative_source"] == "bm25"
            ]
            self.assertEqual(sum(rank <= 20 for rank in selected_bm25_ranks), 2)
            self.assertEqual(sum(21 <= rank <= 100 for rank in selected_bm25_ranks), 1)
            self.assertEqual(sum(rank >= 101 for rank in selected_bm25_ranks), 1)
            self.assertEqual(stats["queries_with_exact_mix"], 1)

    def test_fills_model_shortfall_and_handles_bm25_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            rows, stats = self.run_combine(
                Path(temp_directory),
                ["bm25-1", "model-2"],
            )
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["rejected"] for row in rows}), 8)
            self.assertEqual(
                sum(row["negative_source"].startswith("model_confusion") for row in rows),
                2,
            )
            overlapping = next(row for row in rows if row["rejected"] == "bm25-1")
            self.assertEqual(overlapping["negative_source"], "model_confusion+bm25")
            self.assertEqual(overlapping["also_bm25_rank"], 1)
            self.assertEqual(stats["queries_using_bm25_fallback"], 1)
            self.assertEqual(stats["model_bm25_overlap"], 1)

    def test_strict_mix_skips_model_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            rows, stats = self.run_combine(
                Path(temp_directory),
                ["model-1", "model-2"],
                require_exact_mix=True,
            )
            self.assertEqual(rows, [])
            self.assertEqual(stats["groups_skipped"], 1)
            self.assertEqual(stats["skipped_model_shortfall"], 1)

    def test_combiner_preserves_url_decoder_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            bm25, model, metadata = self.make_inputs(
                root,
                ["model-1", "model-2", "model-3", "model-4"],
            )
            url_by_text_id = {
                "target": "repo/target.rb/target()",
                **{
                    f"bm25-{index}": f"repo/bm25-{index}.rb/call()"
                    for index in range(1, 9)
                },
                **{
                    f"model-{index}": f"repo/model-{index}.rb/call()"
                    for index in range(1, 5)
                },
            }
            for path in (bm25, model):
                converted: list[dict] = []
                for row in iter_json_records(path):
                    copied = dict(row)
                    copied["chosen"] = url_by_text_id["target"]
                    copied["rejected"] = url_by_text_id[copied["rejected_text_id"]]
                    converted.append(copied)
                write_rows(path, converted)

            output = root / "hybrid_url.jsonl"
            combine(
                SimpleNamespace(
                    bm25_input=str(bm25),
                    model_input=str(model),
                    document_metadata=str(metadata),
                    output=str(output),
                    model_per_query=4,
                    total_per_query=8,
                    seed=42,
                    require_exact_mix=False,
                )
            )
            rows = list(iter_json_records(output))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["chosen"].startswith("repo/") for row in rows))
            self.assertTrue(all(row["rejected"].startswith("repo/") for row in rows))
            self.assertTrue(all(row["rejected_text_id"] for row in rows))

    def test_combiner_excludes_tokenizer_collision_docids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            bm25, model, metadata = self.make_inputs(
                root,
                ["model-1", "model-2", "model-3", "model-4"],
            )
            exclusions = root / "excluded.json"
            exclusions.write_text(
                json.dumps({"excluded_text_ids": ["bm25-1", "model-1"]}),
                encoding="utf-8",
            )
            output = root / "hybrid.jsonl"
            stats = combine(
                SimpleNamespace(
                    bm25_input=str(bm25),
                    model_input=str(model),
                    document_metadata=str(metadata),
                    output=str(output),
                    excluded_text_ids=str(exclusions),
                    model_per_query=4,
                    total_per_query=8,
                    seed=42,
                    require_exact_mix=False,
                )
            )
            rows = list(iter_json_records(output))
            self.assertEqual(len(rows), 8)
            self.assertNotIn("bm25-1", {row["rejected_text_id"] for row in rows})
            self.assertNotIn("model-1", {row["rejected_text_id"] for row in rows})
            self.assertEqual(stats["excluded_text_ids"], 2)


if __name__ == "__main__":
    unittest.main()
