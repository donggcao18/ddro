---
license: ms-pl
language:
- en
tags:
  - generative-retrieval
  - information-retrieval
  - msmarco
  - ddro
  - document-retrieval
task_categories:
  - text-retrieval
pretty_name: DDRO MS MARCO Top-300K Dataset
size_categories:
  - 100K<n<1M
---

# DDRO — MS MARCO Top-300K Processed Dataset

This dataset contains the preprocessed **MS MARCO Top-300K** document corpus used to train and evaluate the DDRO generative retrieval models from:

**[Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval (SIGIR 2025)](https://arxiv.org/abs/2504.05181)**

---

## Files

| File | Description | Size |
|---|---|---|
| `msmarco-docs-sents.top.300k.json` | Top-300K documents selected by click frequency, with sentence tokenization (JSONL format) | ~2 GB |

---

## Format

### `msmarco-docs-sents.top.300k.json` (JSONL)
One document per line:
```json
{
  "docid": "D1650436",
  "url": "https://...",
  "title": "...",
  "body": "...",
  "sents": ["sentence 1", "sentence 2", "..."]
}
```

---

## Dataset Statistics

| | |
|---|---|
| Documents | 300,000 (top by click frequency) |
| Selection | Ranked by click counts from MS MARCO training qrels |
| Preprocessing | Full body text tokenized into sentences |

**Note:** Only the Top-300K split is used. Random sampling is not used in any experiments.

---

## Corresponding Models

Trained on this dataset:

| Model | Docid Type | MRR@10 | R@10 |
|---|---|---|---|
| [`kiyam/ddro-msmarco-pq`](https://huggingface.co/kiyam/ddro-msmarco-pq) | PQ (Product Quantization) | 45.76 | 73.02 |
| [`kiyam/ddro-msmarco-tu`](https://huggingface.co/kiyam/ddro-msmarco-tu) | TU (Title + URL) | 50.07 | 74.01 |

SFT reference policies: [`kiyam/ddro-msmarco-pq-sft`](https://huggingface.co/kiyam/ddro-msmarco-pq-sft), [`kiyam/ddro-msmarco-tu-sft`](https://huggingface.co/kiyam/ddro-msmarco-tu-sft)

---

## Source Data

This dataset is derived from the **Microsoft MS MARCO Document Ranking** dataset:

> Tri Nguyen, Mir Rosenberg, Xia Song, Jauhar Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. **MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.** *NIPS 2016 Workshop on Cognitive Computation.*

- Homepage: https://microsoft.github.io/msmarco/Datasets.html#document-ranking-dataset
- License: [Microsoft Research License Terms (ms-pl)](https://microsoft.github.io/msmarco/)

The original data is provided by Microsoft under the MS-PL license. This processed version inherits the same license terms.

See [`sample_top_docs.sh`](https://github.com/kidist-amde/ddro/blob/main/src/scripts/preprocess/sample_top_docs.sh) for the full preprocessing pipeline.

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{mekonnen2025lightweight,
  title={Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval},
  author={Mekonnen, Kidist Amde and Tang, Yubao and de Rijke, Maarten},
  journal={arXiv preprint arXiv:2504.05181},
  year={2025}
}
```