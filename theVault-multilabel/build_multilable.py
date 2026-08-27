import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path



QUERY_PREFIX_RE = re.compile(r"^\s*Query\s*:\s*", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def iter_json_records(path):
    """Yield objects from either a JSONL file or a JSON list file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        first = ""
        while True:
            ch = f.read(1)
            if not ch:
                return
            if not ch.isspace():
                first = ch
                break
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is JSON but does not contain a top-level list")
            for row in data:
                if row:
                    yield row
        else:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc


def normalize_query(text):
    text = QUERY_PREFIX_RE.sub("", str(text))
    text = SPACE_RE.sub(" ", text.strip().lower())
    return text


def add_unique(mapping, key, value):
    if key not in mapping:
        mapping[key] = OrderedDict()
    mapping[key][value] = None


def build_positive_maps(original_paths, text_field, id_field, query_prefix_only):
    query_to_ids = OrderedDict()
    query_to_row_count = OrderedDict()
    counts = []
    used_counts = []

    for path in original_paths:
        count = 0
        used_count = 0
        for row in iter_json_records(path):
            count += 1
            if text_field not in row or id_field not in row:
                continue
            raw_text = str(row[text_field])
            if query_prefix_only and not QUERY_PREFIX_RE.match(raw_text):
                continue
            query = normalize_query(raw_text)
            text_id = str(row[id_field])
            if query:
                add_unique(query_to_ids, query, text_id)
                query_to_row_count[query] = query_to_row_count.get(query, 0) + 1
                used_count += 1
        counts.append(count)
        used_counts.append(used_count)

    query_to_positive_ids = {
        query: list(ids.keys())
        for query, ids in query_to_ids.items()
    }

    text_id_to_positive_ids = {}
    for positive_ids in query_to_positive_ids.values():
        for text_id in positive_ids:
            text_id_to_positive_ids[text_id] = positive_ids

    return counts, used_counts, query_to_positive_ids, query_to_row_count, text_id_to_positive_ids


def convert_augmentation_file(
    augmentation_path,
    output_path,
    text_id_to_positive_ids,
    id_field,
    deduplicate_positive_groups=False,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    multi_positive_rows = 0
    missing_mappings = 0
    skipped_duplicate_groups = 0
    seen_positive_ids = set()

    with output_path.open("w", encoding="utf-8") as out:
        for row in iter_json_records(augmentation_path):
            processed += 1
            raw_text_id = row.get(id_field)
            text_id = str(raw_text_id)
            positive_ids = text_id_to_positive_ids.get(text_id)

            if positive_ids is None:
                missing_mappings += 1
                positive_ids = [text_id]

            if len(positive_ids) > 1:
                multi_positive_rows += 1

            if deduplicate_positive_groups:
                if any(positive_id in seen_positive_ids for positive_id in positive_ids):
                    skipped_duplicate_groups += 1
                    continue
                seen_positive_ids.update(positive_ids)

            row[id_field] = positive_ids
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return processed, written, multi_positive_rows, missing_mappings, skipped_duplicate_groups


def main():
    parser = argparse.ArgumentParser(
        description="Convert ready-to-feed DSI-QG augmentation data to multi-positive text_id lists."
    )
    parser.add_argument("--train_original", required=True, help="Original indexed train JSON/JSONL file")
    parser.add_argument("--test_original", required=True, help="Original indexed test JSON/JSONL file")
    parser.add_argument("--augmentation", required=True, help="Ready-to-feed augmentation JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--text_field", default="text", help="Original query text field")
    parser.add_argument("--id_field", default="text_id", help="Document ID field")
    parser.add_argument(
        "--include_non_query_texts",
        action="store_true",
        help="Also group non-Query: rows from original data. By default, only Query: rows are used.",
    )
    parser.add_argument(
        "--deduplicate_positive_groups",
        action="store_true",
        help=(
            "Write only the first row for each multi-positive text_id group. "
            "Use this when creating evaluation/test files so the same query group is evaluated once."
        ),
    )
    args = parser.parse_args()

    (
        counts,
        used_counts,
        query_to_positive_ids,
        query_to_row_count,
        text_id_to_positive_ids,
    ) = build_positive_maps(
        [args.train_original, args.test_original],
        text_field=args.text_field,
        id_field=args.id_field,
        query_prefix_only=not args.include_non_query_texts,
    )

    (
        processed,
        written,
        multi_positive_rows,
        missing_mappings,
        skipped_duplicate_groups,
    ) = convert_augmentation_file(
        args.augmentation,
        args.output,
        text_id_to_positive_ids,
        id_field=args.id_field,
        deduplicate_positive_groups=args.deduplicate_positive_groups,
    )

    group_sizes = [len(ids) for ids in query_to_positive_ids.values()]
    multi_positive_groups = sum(size > 1 for size in group_sizes)
    max_group_size = max(group_sizes, default=0)
    affected_original_query_rows = sum(
        query_to_row_count[query]
        for query, ids in query_to_positive_ids.items()
        if len(ids) > 1
    )
    affected_original_text_ids = sum(
        len(ids)
        for ids in query_to_positive_ids.values()
        if len(ids) > 1
    )

    print(f"Original train examples: {counts[0] if len(counts) > 0 else 0}")
    print(f"Original test examples: {counts[1] if len(counts) > 1 else 0}")
    print(f"Total original examples: {sum(counts)}")
    print(f"Original train query rows used: {used_counts[0] if len(used_counts) > 0 else 0}")
    print(f"Original test query rows used: {used_counts[1] if len(used_counts) > 1 else 0}")
    print(f"Total original query rows used: {sum(used_counts)}")
    print(f"Unique normalized queries: {len(query_to_positive_ids)}")
    print(f"Query groups with multiple positives: {multi_positive_groups}")
    print(f"Affected original query rows: {affected_original_query_rows}")
    print(f"Affected original unique text_ids: {affected_original_text_ids}")
    print(f"Max positives in one query group: {max_group_size}")
    print(f"Augmented rows processed: {processed}")
    print(f"Output rows written: {written}")
    print(f"Augmented rows with multiple positives: {multi_positive_rows}")
    print(f"Missing text_id mappings: {missing_mappings}")
    print(f"Skipped duplicate positive groups: {skipped_duplicate_groups}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()


# python data/augmentation/build_multilabel_ready_to_feed.py \
#   --train_original /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_train_r32.0.json \
#   --test_original /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_test_r32.0.json \
#   --augmentation /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_ready_to_feed_numeric.jsonl \
#   --output /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_ready_to_feed_multilabel.jsonl

# python data/augmentation/build_multilabel_ready_to_feed.py \
#   --train_original /data/scratch/projects/punim1928/east/CodeGR/data/original_indexed_data/Ruby_train_r32.0.json \
#   --test_original /data/scratch/projects/punim1928/east/CodeGR/data/original_indexed_data/Ruby_test_r32.0.json \
#   --augmentation /data/scratch/projects/punim1928/east/CodeGR/data/original_indexed_data/Ruby_test_r32.0.json \
#   --output /data/scratch/projects/punim1928/east/CodeGR/data/original_indexed_data/Ruby_test_r32.0_multilabel.jsonl \
#   --deduplicate_positive_groups
