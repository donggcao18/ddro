import json
import argparse
from collections import defaultdict

def merge_queries(file_a, file_b, output_file):
    # Read file A metadata and query texts
    file_a_records = {}
    original_queries = {}
    with open(file_a, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            text_id = str(item["text_id"])
            numeric_id = str(item.get("numeric_id", ""))
            url_based_id = item.get("url_based_id", item.get("url_id", ""))
            if not numeric_id:
                continue

            # Keep metadata for all rows (both Query and Code rows).
            file_a_records[numeric_id] = {
                "text_id": text_id,
                "numeric_id": numeric_id,
                "url_based_id": str(url_based_id) if url_based_id is not None else ""
            }

            if item["text"].startswith("Query:"):
                query = item["text"]
                original_queries[numeric_id] = query[6:].strip()  # Remove "Query: " prefix
    
    print(f"Loaded {len(file_a_records)} metadata records from {file_a}")
    print(f"Loaded {len(original_queries)} original queries from {file_a}")
    
    # Read augmented queries (file B)
    augmented_queries = defaultdict(list)
    with open(file_b, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            text_id = str(item["text_id"])
            query = item["text"]
            augmented_queries[text_id].append(query)
    
    print(f"Loaded augmented queries for {len(augmented_queries)} documents from {file_b}")
    
    # Merge queries
    merged_results = []
    total_original_added = 0
    
    for numeric_id_key, aug_queries in augmented_queries.items():
        if numeric_id_key in file_a_records:
            is_unique = True
            metadata = file_a_records[numeric_id_key]
            original_query = original_queries.get(numeric_id_key)
            
            for aug_query in aug_queries:
                if original_query is not None and original_query.lower().strip() == aug_query.lower().strip():
                    is_unique = False
                merged_results.append({
                    "text_id": metadata["text_id"],
                    "numeric_id": metadata["numeric_id"],
                    "url_based_id": metadata["url_based_id"],
                    "text": aug_query,
                    "is_original": False
                })
            
            # Add original query if unique
            if original_query is not None and is_unique:
                merged_results.append({
                    "text_id": metadata["text_id"],
                    "numeric_id": metadata["numeric_id"],
                    "url_based_id": metadata["url_based_id"],
                    "text": original_query,
                    "is_original": True
                })
                total_original_added += 1
        # else:
            # for aug_query in aug_queries:
            #     merged_results.append({
            #         "text_id": "",
            #         "numeric_id": numeric_id_key,
            #         "url_based_id": "",
            #         "text": aug_query,
            #         "is_original": False
            #     })
            
    
    # Write merged results
    with open(output_file, 'w') as f:
        for item in merged_results:
            # Remove the is_original field before writing
            output_item = {
                "text_id": item["text_id"],
                "numeric_id": item["numeric_id"],
                "url_based_id": item["url_based_id"],
                "text": item["text"],
                "is_original": item["is_original"]
            }
            f.write(json.dumps(output_item) + '\n')
    
    print(f"Merged results written to {output_file}")
    print(f"Added {total_original_added} original queries that were unique")
    print(f"Total queries in output: {len(merged_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge original queries with augmented queries")
    parser.add_argument("--file_a", type=str,  default="/home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_train_r32.0.json", help="Path to file with original queries")
    parser.add_argument("--file_b", type=str,  default="/home/users/congthanh_le/scratch/east/CodeGR/data/augmentation/Ruby_q10.jsonl", help="Path to file with augmented queries")
    parser.add_argument("--output", type=str,  default="/home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_ready_to_feed_numeric.jsonl", help="Path to output file")

    args = parser.parse_args()
    
    merge_queries(args.file_a, args.file_b, args.output)    