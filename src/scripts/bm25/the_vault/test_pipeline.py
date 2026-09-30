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
from relabel_structure_v6_metadata import relabel


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class VaultPipelineTest(unittest.TestCase):
    def test_v6_merged_queries_join_by_url_and_filter_explicit_positives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.jsonl"
            merged = root / "merged.jsonl"
            work = root / "work"
            write_rows(original, [
                {"numeric_id": "1", "text_id": "old-a", "url_based_id": "repo/a", "text": "Query: find a"},
                {"numeric_id": "2", "text_id": "old-b", "url_based_id": "repo/b", "text": "Query: find b"},
                {"numeric_id": "3", "text_id": "old-c", "url_based_id": "repo/c", "text": "Query: find c"},
                {"numeric_id": "11", "text_id": "old-a", "url_based_id": "repo/a", "text": "Code: a"},
                {"numeric_id": "12", "text_id": "old-b", "url_based_id": "repo/b", "text": "Code: b"},
                {"numeric_id": "13", "text_id": "old-c", "url_based_id": "repo/c", "text": "Code: c"},
            ])
            write_rows(merged, [
                {"numeric_id": "2", "semantic_id": "unrelated", "url_based_id": "repo/a",
                 "structure_id_v6": "v6|a", "text": "new merged query",
                 "positive_structure_id_v6": ["v6|a", "v6|b"]},
                {"numeric_id": "20", "url_based_id": "repo/b", "structure_id_v6": "v6|b"},
                {"numeric_id": "30", "url_based_id": "repo/c", "structure_id_v6": "v6|c"},
            ])
            prepare(SimpleNamespace(
                train_original=str(original), test_original=[],
                augmentation=str(merged), output_dir=str(work),
                augmentation_id_mode="url_based_id", code_only=False, strict=True,
                structure_id_source=[str(merged)], structure_id_field="structure_id_v6",
                structure_id_join_key="url_based_id",
            ))
            query = next(iter_json_records(work / "query_metadata.jsonl"))
            self.assertEqual(query["target_text_id"], "old-a")
            self.assertEqual(query["target_structure_id_v6"], "v6|a")
            self.assertEqual(query["positive_structure_id_v6s"], ["v6|a", "v6|b"])
            run = work / "bm25_run.txt"
            run.write_text(
                "vault-000000000 Q0 old-b 1 2.0 test\n"
                "vault-000000000 Q0 old-c 2 1.0 test\n",
                encoding="utf-8",
            )
            output = work / "dpo_pairs_structure_id_v6.jsonl"
            stats = mine(SimpleNamespace(
                run=str(run), query_metadata=str(work / "query_metadata.jsonl"),
                document_metadata=str(work / "document_metadata.jsonl"),
                output=str(output), triples_output=None, negatives_per_query=1,
                rank_ranges=[(1, 2)], rank_quotas=None, seed=42,
                pair_all_positives=False, fill_shortfall=False,
                target_type="structure_id_v6",
            ))
            self.assertEqual(stats["filtered_multi_label_or_target_hits"], 1)
            self.assertEqual(next(iter_json_records(output))["rejected"], "v6|c")

    def test_relabels_existing_bm25_run_without_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "existing"
            output_dir = root / "v6"
            source.mkdir()
            structures = root / "structures.jsonl"
            write_rows(structures, [
                {"url_based_id": "repo/a", "structure_id_v6": "v6|a"},
                {"url_based_id": "repo/b", "structure_id_v6": "v6|b"},
                {"url_based_id": "repo/c", "structure_id_v6": "v6|a"},
                {"url_based_id": "repo/d", "structure_id_v6": "v6|d"},
            ])
            write_rows(source / "document_metadata.jsonl", [
                {"text_id": "a", "url_based_ids": ["repo/a"]},
                {"text_id": "b", "url_based_ids": ["repo/b"]},
                {"text_id": "c", "url_based_ids": ["repo/c"]},
                {"text_id": "d", "url_based_ids": ["repo/d"]},
            ])
            write_rows(source / "query_metadata.jsonl", [
                {"query_key": "vault-000000000", "prompt": "find a",
                 "target_text_id": "a", "positive_text_ids": ["a"],
                 "url_based_id": "repo/a"},
            ])
            run = source / "bm25_run.txt"
            run.write_text(
                "vault-000000000 Q0 c 1 3.0 test\n"
                "vault-000000000 Q0 b 2 2.0 test\n"
                "vault-000000000 Q0 d 3 1.0 test\n",
                encoding="utf-8",
            )
            stats = relabel(SimpleNamespace(
                source_query_metadata=str(source / "query_metadata.jsonl"),
                source_document_metadata=str(source / "document_metadata.jsonl"),
                structure_id_source=[str(structures)],
                output_query_metadata=str(output_dir / "query_metadata.jsonl"),
                output_document_metadata=str(output_dir / "document_metadata.jsonl"),
            ))
            self.assertEqual(stats["documents_with_unique_structure_id_v6"], 4)
            query = next(iter_json_records(output_dir / "query_metadata.jsonl"))
            self.assertEqual(query["target_structure_id_v6"], "v6|a")
            self.assertEqual(query["positive_structure_id_v6s"], ["v6|a"])
            output = output_dir / "dpo_pairs_structure_id_v6.jsonl"
            mined = mine(SimpleNamespace(
                run=str(run),
                query_metadata=str(output_dir / "query_metadata.jsonl"),
                document_metadata=str(output_dir / "document_metadata.jsonl"),
                output=str(output), triples_output=None, negatives_per_query=1,
                rank_ranges=[(1, 3)], rank_quotas=None, seed=42,
                pair_all_positives=False, fill_shortfall=False,
                target_type="structure_id_v6",
            ))
            self.assertEqual(mined["dpo_pairs"], 1)
            self.assertEqual(mined["filtered_multi_label_or_target_hits"], 1)
            pair = next(iter_json_records(output))
            self.assertEqual(pair["chosen"], "v6|a")
            self.assertNotEqual(pair["rejected"], "v6|a")
            self.assertEqual(run.read_text(encoding="utf-8").count("\n"), 3)

    def test_structure_id_v6_is_used_for_preparation_and_bm25_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.jsonl"
            augmentation = root / "augmentation.jsonl"
            structures = root / "structures.jsonl"
            work = root / "work"
            write_rows(original, [
                {"numeric_id": "1", "text_id": "a", "url_based_id": "repo/a", "text": "Query: find a"},
                {"numeric_id": "2", "text_id": "b", "url_based_id": "repo/b", "text": "Query: find b"},
                {"numeric_id": "3", "text_id": "c", "url_based_id": "repo/c", "text": "Query: find c"},
                {"numeric_id": "11", "text_id": "a", "url_based_id": "repo/a", "text": "Code: a"},
                {"numeric_id": "12", "text_id": "b", "url_based_id": "repo/b", "text": "Code: b"},
                {"numeric_id": "13", "text_id": "c", "url_based_id": "repo/c", "text": "Code: c"},
            ])
            write_rows(structures, [
                {"url_based_id": "repo/a", "structure_id_v6": "v6|a"},
                {"url_based_id": "repo/b", "structure_id_v6": "v6|b"},
                {"url_based_id": "repo/c", "structure_id_v6": "v6|c"},
            ])
            write_rows(augmentation, [
                {"text_id": 1, "url_based_id": "repo/a", "text": "find a"},
            ])
            stats = prepare(SimpleNamespace(
                train_original=str(original), test_original=[],
                augmentation=str(augmentation), output_dir=str(work),
                augmentation_id_mode="auto", code_only=False, strict=True,
                structure_id_source=[str(structures)],
                structure_id_field="structure_id_v6",
                structure_id_join_key="url_based_id",
            ))
            self.assertEqual(stats["documents_with_structure_id_v6"], 3)
            query = next(iter_json_records(work / "query_metadata.jsonl"))
            self.assertEqual(query["target_structure_id_v6"], "v6|a")
            run = work / "bm25_run.txt"
            run.write_text(
                "vault-000000000 Q0 b 1 2.0 test\n"
                "vault-000000000 Q0 c 2 1.0 test\n", encoding="utf-8"
            )
            output = work / "dpo_pairs_structure_id_v6.jsonl"
            mined = mine(SimpleNamespace(
                run=str(run), query_metadata=str(work / "query_metadata.jsonl"),
                document_metadata=str(work / "document_metadata.jsonl"),
                output=str(output), triples_output=None, negatives_per_query=1,
                rank_ranges=[(1, 2)], rank_quotas=None, seed=42,
                pair_all_positives=False, fill_shortfall=False,
                target_type="structure_id_v6",
            ))
            self.assertEqual(mined["dpo_pairs"], 1)
            pair = next(iter_json_records(output))
            self.assertEqual(pair["chosen"], "v6|a")
            self.assertIn(pair["rejected"], {"v6|b", "v6|c"})
            self.assertEqual(pair["chosen_structure_id_v6"], "v6|a")

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
                    {"numeric_id": "1", "text_id": "target-a", "url_based_id": "repo/a.rb/a()", "text": "Query: shared query"},
                    {"numeric_id": "2", "text_id": "target-b", "url_based_id": "repo/b.rb/b()", "text": "Query: shared query"},
                    {"numeric_id": "3", "text_id": "target-c", "url_based_id": "repo/c.rb/c()", "text": "Query: different query"},
                    {"numeric_id": "101", "text_id": "target-a", "url_based_id": "repo/a.rb/a()", "text": "Code: def a; end"},
                    {"numeric_id": "102", "text_id": "target-b", "url_based_id": "repo/b.rb/b()", "text": "Code: def b; end"},
                    {"numeric_id": "103", "text_id": "target-c", "url_based_id": "repo/c.rb/c()", "text": "Code: def c; end"},
                ],
            )
            write_rows(
                structures,
                [
                    {"numeric_id": "9001", "url_based_id": "repo/a.rb/a()", "structure_id_v3": "shared_a|arg|path one"},
                    {"numeric_id": "9002", "url_based_id": "repo/b.rb/b()", "structure_id_v3": "shared_b|arg|path two"},
                    {"numeric_id": "9003", "url_based_id": "repo/c.rb/c()", "structure_id_v3": "different|arg|path three"},
                ],
            )
            write_rows(
                augmentation,
                [{"text_id": 1, "url_based_id": "repo/a.rb/a()", "text": "generated query"}],
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
                    structure_id_source=[str(structures)],
                    structure_id_field="structure_id_v3",
                    structure_id_join_key="url_based_id",
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

    def test_duplicate_structure_targets_are_counted_and_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            query_metadata = root / "query_metadata.jsonl"
            document_metadata = root / "document_metadata.jsonl"
            run = root / "run.txt"
            output = root / "dpo_structure.jsonl"

            write_rows(
                query_metadata,
                [
                    {
                        "query_key": "vault-000000000",
                        "prompt": "create the artist table",
                        "target_text_id": "target-a",
                        "positive_text_ids": ["target-a"],
                        "positive_structure_id_v3s": ["shared|args|path"],
                    }
                ],
            )
            write_rows(
                document_metadata,
                [
                    {"text_id": "target-a", "structure_id_v3s": ["shared|args|path"]},
                    {"text_id": "target-b", "structure_id_v3s": ["shared|args|path"]},
                    {"text_id": "target-c", "structure_id_v3s": ["different|args|path"]},
                ],
            )
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

            stats = mine(
                SimpleNamespace(
                    run=str(run),
                    query_metadata=str(query_metadata),
                    document_metadata=str(document_metadata),
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

            self.assertEqual(stats["duplicate_decoder_targets"], 1)
            self.assertEqual(stats["text_ids_in_duplicate_decoder_targets"], 2)
            self.assertEqual(stats["extra_text_ids_sharing_decoder_targets"], 1)
            self.assertEqual(stats["filtered_multi_label_or_target_hits"], 2)
            pair = next(iter_json_records(output))
            self.assertEqual(pair["chosen"], "shared|args|path")
            self.assertEqual(pair["rejected"], "different|args|path")


if __name__ == "__main__":
    unittest.main()
