"""Small regression tests for Vault ID mapping and multi-label filtering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import iter_json_records
from mine_dpo_negatives import mine
from prepare_bm25 import prepare


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class VaultPipelineTest(unittest.TestCase):
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
                    negatives_per_query=3,
                    rank_ranges=[(1, 100), (101, 500), (501, 1000)],
                    seed=42,
                    pair_all_positives=False,
                )
            )
            self.assertEqual(mine_stats["filtered_multi_label_or_target_hits"], 2)
            self.assertEqual(mine_stats["dpo_pairs"], 1)

            pair = next(iter_json_records(dpo_output))
            self.assertEqual(pair["chosen"], "target-a")
            self.assertEqual(pair["rejected"], "target-c")


if __name__ == "__main__":
    unittest.main()
