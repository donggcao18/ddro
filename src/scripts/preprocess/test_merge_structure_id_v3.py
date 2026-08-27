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
    def test_joins_by_numeric_id_and_expands_multilabel_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            original = root / "original.jsonl"
            structures = root / "structures.jsonl"
            current = root / "current.jsonl"
            output = root / "output.jsonl"
            write_rows(
                original,
                [
                    {"numeric_id": "1", "text_id": "legacy-a"},
                    {"numeric_id": "2", "text_id": "legacy-b"},
                ],
            )
            write_rows(
                structures,
                [
                    {"numeric_id": "1", "structure_id_v3": "structure a"},
                    {"numeric_id": "2", "structure_id_v3": "structure b"},
                ],
            )
            write_rows(
                current,
                [
                    {
                        "numeric_id": "1",
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


if __name__ == "__main__":
    unittest.main()
