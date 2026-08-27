"""Small regression tests for Vault ID mapping and multi-label filtering."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import iter_json_records
from mine_dpo_negatives import mine, stratified_sample
from mine_model_confusion_negatives import build_document_targets
from postprocess_dpo_urls import convert
from prepare_bm25 import prepare


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class VaultPipelineTest(unittest.TestCase):
    def test_model_miner_accepts_structure_id_v3_targets(self) -> None:
        forward, reverse, invalid = build_document_targets(
            {
                "target-a": {
                    "text_id": "target-a",
                    "structure_id_v3s": ["compare|arg|path one"],
                },
                "target-b": {
                    "text_id": "target-b",
                    "structure_id_v3s": ["other|arg|path two"],
                },
            },
            "structure_id_v3",
        )
        self.assertEqual(invalid, 0)
        self.assertEqual(forward["target-a"], "compare|arg|path one")
        self.assertEqual(reverse["other|arg|path two"], "target-b")

    def test_postprocesses_mined_pairs_to_url_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            metadata = root / "document_metadata.jsonl"
            mined = root / "dpo_text_ids.jsonl"
            output = root / "dpo_urls.jsonl"

            write_rows(
                metadata,
                [
                    {"text_id": "target-a", "url_based_ids": ["repo/a.rb/a()"]},
                    {"text_id": "target-c", "url_based_ids": ["repo/c.rb/c()"]},
                ],
            )
            write_rows(
                mined,
                [
                    {
                        "prompt": "find a",
                        "chosen": "target-a",
                        "rejected": "target-c",
                        "chosen_text_id": "target-a",
                        "rejected_text_id": "target-c",
                        "bm25_rank": 3,
                    }
                ],
            )

            stats = convert(
                SimpleNamespace(
                    input=str(mined),
                    document_metadata=str(metadata),
                    output=str(output),
                    stats_output=None,
                    on_mapping_error="error",
                )
            )

            self.assertEqual(stats["input_rows"], 1)
            self.assertEqual(stats["output_rows"], 1)
            pair = next(iter_json_records(output))
            self.assertEqual(pair["chosen"], "repo/a.rb/a()")
            self.assertEqual(pair["rejected"], "repo/c.rb/c()")
            self.assertEqual(pair["chosen_text_id"], "target-a")
            self.assertEqual(pair["rejected_text_id"], "target-c")
            self.assertEqual(pair["bm25_rank"], 3)

    def test_postprocessor_can_skip_an_invalid_referenced_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            metadata = root / "document_metadata.jsonl"
            mined = root / "dpo_text_ids.jsonl"
            output = root / "dpo_urls.jsonl"

            write_rows(
                metadata,
                [
                    {"text_id": "target-a", "url_based_ids": ["repo/a.rb/a()"]},
                    {"text_id": "target-c", "url_based_ids": []},
                ],
            )
            write_rows(
                mined,
                [{"prompt": "find a", "chosen": "target-a", "rejected": "target-c"}],
            )

            stats = convert(
                SimpleNamespace(
                    input=str(mined),
                    document_metadata=str(metadata),
                    output=str(output),
                    stats_output=None,
                    on_mapping_error="skip",
                )
            )

            self.assertEqual(stats["output_rows"], 0)
            self.assertEqual(stats["skipped_mapping_errors"], 1)
            self.assertEqual(stats["invalid_document_url_mappings"], 1)
            self.assertEqual(list(iter_json_records(output)), [])

    def test_vault_rank_bucket_allocation(self) -> None:
        candidates = [
            {"text_id": f"doc-{rank}", "rank": rank, "score": None}
            for rank in range(1, 201)
        ]
        selected = stratified_sample(
            candidates,
            8,
            [(1, 20), (21, 100), (101, 200)],
            random.Random(42),
            rank_quotas=[3, 2, 3],
        )
        self.assertEqual(sum(item["rank"] <= 20 for item in selected), 3)
        self.assertEqual(sum(21 <= item["rank"] <= 100 for item in selected), 2)
        self.assertEqual(sum(item["rank"] >= 101 for item in selected), 3)

        undersupplied = [item for item in candidates if item["rank"] <= 102]
        self.assertEqual(
            stratified_sample(
                undersupplied[:111],
                8,
                [(1, 20), (21, 100), (101, 200)],
                random.Random(42),
                rank_quotas=[3, 2, 3],
            ),
            [],
        )

    def test_numeric_mapping_and_multilabel_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            original = root / "original.jsonl"
            augmentation = root / "augmentation.jsonl"
            work = root / "work"

            write_rows(
                original,
                [
                    {"numeric_id": "1", "text_id": "target-a", "text": "Query: shared query"},
                    {"numeric_id": "2", "text_id": "target-b", "text": "Query: shared query"},
                    {"numeric_id": "101", "text_id": "target-a", "text": "Code: def a; end"},
                    {"numeric_id": "102", "text_id": "target-b", "text": "Code: def b; end"},
                    {"numeric_id": "103", "text_id": "target-c", "text": "Code: def c; end"},
                ],
            )
            # Raw q10 format: text_id is the original numeric_id.
            write_rows(
                augmentation,
                [{"text_id": 1, "text": "a generated shared query"}],
            )

            prepare_stats = prepare(
                SimpleNamespace(
                    train_original=str(original),
                    test_original=[],
                    augmentation=str(augmentation),
                    output_dir=str(work),
                    augmentation_id_mode="auto",
                    code_only=False,
                    strict=True,
                )
            )
            self.assertEqual(prepare_stats["multi_label_query_groups"], 1)

            query = next(iter_json_records(work / "query_metadata.jsonl"))
            self.assertEqual(query["target_text_id"], "target-a")
            self.assertEqual(query["positive_text_ids"], ["target-a", "target-b"])

            run = work / "run.txt"
            run.write_text(
                "\n".join(
                    [
                        "vault-000000000 Q0 target-a 1 10.0 test",
                        "vault-000000000 Q0 target-b 2 9.0 test",
                        "vault-000000000 Q0 target-c 3 8.0 test",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            dpo_output = work / "dpo.jsonl"
            mine_stats = mine(
                SimpleNamespace(
                    run=str(run),
                    query_metadata=str(work / "query_metadata.jsonl"),
                    document_metadata=str(work / "document_metadata.jsonl"),
                    output=str(dpo_output),
                    triples_output=None,
                    negatives_per_query=1,
                    rank_ranges=[(1, 100)],
                    seed=42,
                    pair_all_positives=False,
                    fill_shortfall=False,
                    rank_quotas=None,
                )
            )
            self.assertEqual(mine_stats["filtered_multi_label_or_target_hits"], 2)
            self.assertEqual(mine_stats["dpo_pairs"], 1)

            pair = next(iter_json_records(dpo_output))
            self.assertEqual(pair["chosen"], "target-a")
            self.assertEqual(pair["rejected"], "target-c")

    def test_structure_id_v3_join_and_bm25_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            original = root / "original.jsonl"
            augmentation = root / "augmentation.jsonl"
            structures = root / "structures.jsonl"
            work = root / "work"

            write_rows(
                original,
                [
                    {"numeric_id": "1", "text_id": "target-a", "text": "Query: shared query"},
                    {"numeric_id": "2", "text_id": "target-b", "text": "Query: shared query"},
                    {"numeric_id": "3", "text_id": "target-c", "text": "Query: different query"},
                    {"numeric_id": "101", "text_id": "target-a", "text": "Code: def a; end"},
                    {"numeric_id": "102", "text_id": "target-b", "text": "Code: def b; end"},
                    {"numeric_id": "103", "text_id": "target-c", "text": "Code: def c; end"},
                ],
            )
            write_rows(
                structures,
                [
                    {"numeric_id": "1", "structure_id_v3": "shared_a|arg|path one"},
                    {"numeric_id": "2", "structure_id_v3": "shared_b|arg|path two"},
                    {"numeric_id": "3", "structure_id_v3": "different|arg|path three"},
                ],
            )
            write_rows(augmentation, [{"text_id": 1, "text": "generated query"}])

            prepare_stats = prepare(
                SimpleNamespace(
                    train_original=str(original),
                    test_original=[],
                    augmentation=str(augmentation),
                    output_dir=str(work),
                    augmentation_id_mode="auto",
                    code_only=False,
                    strict=True,
                    structure_id_source=[str(structures)],
                    structure_id_field="structure_id_v3",
                )
            )
            self.assertEqual(prepare_stats["documents_with_structure_id_v3"], 3)
            query = next(iter_json_records(work / "query_metadata.jsonl"))
            self.assertEqual(query["target_structure_id_v3"], "shared_a|arg|path one")
            self.assertEqual(
                query["positive_structure_id_v3s"],
                ["shared_a|arg|path one", "shared_b|arg|path two"],
            )

            run = work / "run.txt"
            run.write_text(
                "\n".join(
                    [
                        "vault-000000000 Q0 target-a 1 10.0 test",
                        "vault-000000000 Q0 target-b 2 9.0 test",
                        "vault-000000000 Q0 target-c 3 8.0 test",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = work / "dpo_structure.jsonl"
            stats = mine(
                SimpleNamespace(
                    run=str(run),
                    query_metadata=str(work / "query_metadata.jsonl"),
                    document_metadata=str(work / "document_metadata.jsonl"),
                    output=str(output),
                    triples_output=None,
                    negatives_per_query=1,
                    rank_ranges=[(1, 100)],
                    seed=42,
                    pair_all_positives=False,
                    fill_shortfall=False,
                    rank_quotas=None,
                    target_type="structure_id_v3",
                )
            )
            self.assertEqual(stats["filtered_multi_label_or_target_hits"], 2)
            pair = next(iter_json_records(output))
            self.assertEqual(pair["chosen"], "shared_a|arg|path one")
            self.assertEqual(pair["rejected"], "different|arg|path three")
            self.assertEqual(pair["chosen_text_id"], "target-a")
            self.assertEqual(pair["rejected_text_id"], "target-c")


if __name__ == "__main__":
    unittest.main()
