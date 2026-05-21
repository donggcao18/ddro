from huggingface_hub import HfApi
import os

repo_id = "kiyam/ddro-nq-dataset"
base = "resources/datasets/processed/nq-data"

files = [
    ("hf_datasets/ddro-nq-dataset/README.md", "README.md"),
    (f"{base}/nq_merged.json",   "nq_merged.json"),
    (f"{base}/nq_train.gz",      "nq_train.gz"),
    (f"{base}/nq_val.gz",        "nq_val.gz"),
]

api = HfApi()

# Create repo if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

for local_path, repo_filename in files:
    size_gb = os.path.getsize(local_path) / 1e9
    print(f"[upload] {repo_filename}  ({size_gb:.1f} GB) ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {repo_filename}",
    )
    print(f"[done]   {repo_filename}")

print("All files uploaded to", repo_id)
