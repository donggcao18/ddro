---
license: cc-by-sa-3.0
language:
- en
tags:
  - generative-retrieval
  - information-retrieval
  - natural-questions
  - ddro
  - document-retrieval
task_categories:
  - text-retrieval
pretty_name: DDRO NQ320K Dataset
size_categories:
  - 100K<n<1M
---

# DDRO — NQ320K Processed Dataset

This dataset contains the preprocessed **Natural Questions (NQ320K)** corpus used to train and evaluate the DDRO generative retrieval models from:

**[Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval (SIGIR 2025)](https://arxiv.org/abs/2504.05181)**

The raw NQ data (from Google) is processed into a unified format with document text, queries, and relevance annotations, ready for use in generative retrieval training pipelines.

---

## Files

| File | Description | Size |
|---|---|---|
| `nq_merged.json` | Full NQ320K document corpus (109,739 unique docs, JSONL format) | 5.5 GB |
| `nq_train.gz` | Training split — queries + document fields (TSV, gzipped) | 6.5 GB |
| `nq_val.gz` | Validation split — queries + document fields (TSV, gzipped) | 159 MB |

---

## Format

### `nq_merged.json` (JSONL)
One document per line:
```json
{
  "id": "5655493461695504401",
  "query": "what is the most common use of opt-in email marketing",
  "long_answer": "...",
  "short_answer": "a newsletter sent to an advertising firm's customers",
  "title": "email marketing",
  "abstract": "...",
  "content": "...",
  "document_url": "https://en.wikipedia.org/...",
  "doc_tac": "<title> + <abstract> + <content>",
  "language": "en"
}
```

### `nq_train.gz` / `nq_val.gz` (TSV, gzipped)
Tab-separated, no header. Columns: `query`, `id`, `long_answer`, `short_answer`, `title`, `abstract`, `content`, `document_url`, `doc_tac`, `language`

---

## Dataset Statistics

| Split | Examples |
|---|---|
| Train | ~307,373 (raw) → 87,791 (after dedup by title) |
| Dev | ~7,830 (raw) → 21,948 (after dedup) |
| Unique documents | 109,739 |

---

## Usage

### Load the corpus
```python
import json

docs = {}
with open("nq_merged.json") as f:
    for line in f:
        doc = json.loads(line)
        docs[doc["id"]] = doc
```

### Load train/val splits
```python
import gzip
import pandas as pd

columns = ["query", "id", "long_answer", "short_answer", "title",
           "abstract", "content", "document_url", "doc_tac", "language"]

with gzip.open("nq_train.gz", "rt") as f:
    train_df = pd.read_csv(f, sep="\t", names=columns)
```

---

## Corresponding Models

Trained on this dataset:

| Model | Docid Type | MRR@10 | R@10 |
|---|---|---|---|
| [`kiyam/ddro-nq-pq`](https://huggingface.co/kiyam/ddro-nq-pq) | PQ (Product Quantization) | 55.51 | 67.31 |
| [`kiyam/ddro-nq-tu`](https://huggingface.co/kiyam/ddro-nq-tu) | TU (Title + URL) | 45.99 | 55.98 |

SFT reference policies: [`kiyam/ddro-nq-pq-sft`](https://huggingface.co/kiyam/ddro-nq-pq-sft), [`kiyam/ddro-nq-tu-sft`](https://huggingface.co/kiyam/ddro-nq-tu-sft)

---

## Source Data

This dataset is derived from the **Google Natural Questions (NQ)** dataset:

> Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. **Natural Questions: a Benchmark for Question Answering Research.** *Transactions of the Association of Computational Linguistics.*

- Homepage: https://ai.google.com/research/NaturalQuestions
- License: [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- Raw files: `gs://natural_questions/v1.0-simplified/`

The original data is made available by Google under CC BY-SA 3.0. This processed version inherits the same license terms.

See [`process_nq_dataset.py`](https://github.com/kidist-amde/ddro/blob/main/src/data/data_prep/nq/process_nq_dataset.py) for the full preprocessing pipeline.

---

## Citation

```bibtex
@inproceedings{amdie2025ddro,
  title={Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval},
  author={Amdie, Kidist},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025},
}
```
