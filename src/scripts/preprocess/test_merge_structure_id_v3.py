from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from merge_structure_id_v3 import iter_json_records, merge


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class MergeStructureIdV3Test(unittest.TestCase):
    def test_joins_by_url_when_numeric_ids_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            original = root / "original.jsonl"
            structures = root / "structures.jsonl"
            current = root / "current.jsonl"
            output = root / "output.jsonl"
            write_rows(
                original,
                [
                    {"numeric_id": "1", "text_id": "legacy-a", "url_based_id": "repo/a.rb/a()"},
                    {"numeric_id": "2", "text_id": "legacy-b", "url_based_id": "repo/b.rb/b()"},
                ],
            )
            write_rows(
                structures,
                [
                    {"numeric_id": "9001", "url_based_id": "repo/a.rb/a()", "structure_id_v3": "structure a"},
                    {"numeric_id": "9002", "url_based_id": "repo/b.rb/b()", "structure_id_v3": "structure b"},
                ],
            )
            write_rows(
                current,
                [
                    {
                        "numeric_id": "5001",
                        "url_based_id": "repo/a.rb/a()",
                        "text_id": ["legacy-a", "legacy-b"],
                        "text": "shared query",
                    }
                ],
            )

            stats = merge(
                SimpleNamespace(
                    input=str(current),
                    structure_source=str(structures),
                    original=[str(original)],
                    output=str(output),
                    structure_id_field="structure_id_v3",
                    join_key="url_based_id",
                    output_target_field="text_id",
                    expand_multilabel=True,
                    on_missing="error",
                )
            )
            rows = list(iter_json_records(output))
            self.assertEqual(stats["output_rows"], 2)
            self.assertEqual([row["text_id"] for row in rows], ["structure a", "structure b"])
            self.assertEqual(
                rows[0]["positive_structure_id_v3s"],
                ["structure a", "structure b"],
            )
            self.assertEqual(rows[0]["legacy_text_ids"], ["legacy-a", "legacy-b"])

    def test_skip_mode_counts_and_logs_unmapped_positives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            original = root / "original.jsonl"
            structures = root / "structures.jsonl"
            current = root / "current.jsonl"
            output = root / "output.jsonl"
            write_rows(
                original,
                [
                    {"numeric_id": "1", "text_id": "legacy-a", "url_based_id": "repo/a.rb/a()"},
                    {"numeric_id": "2", "text_id": "legacy-missing", "url_based_id": "repo/b.rb/b()"},
                ],
            )
            write_rows(
                structures,
                [{"numeric_id": "9001", "url_based_id": "repo/a.rb/a()", "structure_id_v3": "structure a"}],
            )
            write_rows(
                current,
                [
                    {
                        "numeric_id": "1",
                        "url_based_id": "repo/a.rb/a()",
                        "text_id": ["legacy-a", "legacy-missing"],
                        "text": "shared query",
                    }
                ],
            )

            stats = merge(
                SimpleNamespace(
                    input=str(current),
                    structure_source=str(structures),
                    original=[str(original)],
                    output=str(output),
                    structure_id_field="structure_id_v3",
                    join_key="url_based_id",
                    output_target_field="text_id",
                    expand_multilabel=True,
                    on_missing="skip",
                )
            )
            self.assertEqual(stats["output_rows"], 1)
            self.assertEqual(stats["rows_with_unmapped_positive_text_ids"], 1)
            self.assertEqual(stats["unmapped_positive_text_ids"], 1)
            self.assertEqual(stats["unique_unmapped_positive_text_ids"], 1)
            missing_rows = list(iter_json_records(stats["missing_log"]))
            self.assertEqual(missing_rows[0]["text_ids"], ["legacy-missing"])


if __name__ == "__main__":
    unittest.main()
