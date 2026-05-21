# DDRO: Direct Document Relevance Optimization for Generative Information Retrieval

[![SIGIR 2025](https://img.shields.io/badge/SIGIR-2025-blue)](https://sigir2025.dei.unipd.it/call-full-papers.html)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2504.05181-b31b1b.svg)](https://arxiv.org/abs/2504.05181)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HF-Datasets-blueviolet)](https://huggingface.co/collections/kiyam/ddro-generative-retrieval-datasets)
[![HuggingFace](https://img.shields.io/badge/🤗%20SFT_Models-PURPLE)](https://huggingface.co/collections/kiyam/ddro-reference-policies-sft)
[![HuggingFace](https://img.shields.io/badge/🤗%20DDRO_Models-yellow)](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval)

Official implementation of our SIGIR 2025 paper:
**[Lightweight and Direct Document Relevance Optimization for Generative IR](https://arxiv.org/abs/2504.05181)**

---

## Table of Contents

- [Motivation](#motivation)
- [Method](#method)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Datasets and Checkpoints](#datasets-and-checkpoints)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Motivation

Generative IR models are typically trained via next-token prediction (cross-entropy loss) over docid tokens. While effective for language modeling, this objective optimizes **token-level generation** ,  not **document-level ranking**, which is the core requirement in IR systems.

DDRO addresses this misalignment by directly optimizing the model for document-level ranking using pairwise preference learning, without reinforcement learning or reward modeling.

---

## Method

DDRO trains in two phases:

<img src="src/arc_images/DDRO.drawio.png" alt="DDRO training pipeline overview" width="800"/>

### Phase 1 — Supervised Fine-Tuning (SFT)

The model learns to generate the correct docid sequence for a given query via autoregressive next-token prediction across three stages:

1. **Pretraining** — document content to docid (`doc → docid`)
2. **Search pretraining** — pseudo queries to docid (`pseudoquery → docid`)
3. **Fine-tuning** — real queries to docid using qrels supervision (`query → docid`)

<img src="src/arc_images/loss_ntp.png" alt="SFT loss" width="300"/>
<img src="src/arc_images/objective_ntp.png" alt="SFT objective" width="300"/>

### Phase 2 — Pairwise Ranking Optimization (DDRO Loss)

The model is fine-tuned with a pairwise learning-to-rank objective inspired by [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290), adapted for structured docid generation under beam decoding constraints.

<img src="src/arc_images/dpo_loss.png" alt="DDRO loss" width="600"/>
<img src="src/arc_images/dpo_objective.png" alt="DDRO objective" width="500"/>

The DDRO loss trains the model to prefer relevant documents (`docid+`) over non-relevant ones (`docid-`) relative to a frozen SFT reference policy:

| Symbol | Description |
|---|---|
| `docid+` | Relevant document for query `q` |
| `docid-` | Non-relevant document |
| `π_θ` | Current model being optimized |
| `π_ref` | Frozen SFT reference model |
| `β` | Temperature controlling preference sensitivity |

### Why DDRO differs from standard DPO

| | DPO | DDRO |
|---|---|---|
| Architecture | Decoder-only | Encoder-decoder |
| Output | Free-form text | Structured docid sequences |
| Decoding | Greedy/sampling | Constrained beam search |
| Objective | Open-ended preference | Document-level ranking |

---

## Project Structure

```
src/
├── data/          # Downloading, preprocessing, and docid instance generation
├── pretrain/      # Model training and evaluation
├── scripts/       # Shell scripts for SFT, DDRO, BM25, and preprocessing
└── utils/         # Tokenization, trie, metrics, trainers

ddro_env.yml       # Conda environment for training
pyserini.yml       # Conda environment for BM25 retrieval
requirements.txt   # Python dependencies
```

> Each subdirectory contains a detailed `README.md` with further instructions.

---

## Setup

### 1. Install environment

```bash
git clone https://github.com/kidist-amde/ddro.git
cd ddro
conda env create -f ddro_env.yml
conda activate ddro_env
```

### 2. Download datasets and pretrained model

We use MS MARCO (top-300K) and Natural Questions (NQ-320K), plus a pretrained T5-base model.

```bash
bash ./src/data/download/download_msmarco_datasets.sh
bash ./src/data/download/download_nq_datasets.sh
python ./src/data/download/download_t5_model.py
```

See [src/data/download/README.md](https://github.com/kidist-amde/ddro/tree/main/src/data/download#readme) for details.

---

## Data Preparation

### MS MARCO — sample top-300K subset

```bash
bash scripts/preprocess/sample_top_docs.sh
```

Output: `resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz`

### Expected directory structure

```
resources/
├── datasets/
│   ├── raw/
│   │   ├── msmarco-data/
│   │   └── nq-data/
│   └── processed/
└── transformer_models/
    └── t5-base/
```

For full preprocessing instructions (docid generation, training/eval instance creation):
[src/data/data_prep/README.md](https://github.com/kidist-amde/ddro/tree/main/src/data/data_prep#readme)

---

## Training

### Phase 1 — SFT

Run all three SFT stages with a single command:

```bash
bash src/scripts/sft/launch_SFT_training.sh
```

The `--encoding` flag controls the docid format (`pq` or `url_title`).

### Phase 2 — DDRO

After SFT, run pairwise ranking optimization:

```bash
bash scripts/ddro/slurm_submit_ddro_training.sh
```

DDRO is implemented using a custom version of HuggingFace's [DPOTrainer](https://github.com/huggingface/trl).

---

## Evaluation

### Option A — Quick evaluation via launcher (recommended)

```bash
# SLURM
sbatch src/pretrain/hf_eval/slurm_submit_hf_eval.sh

# Direct
python src/pretrain/hf_eval/launch_hf_eval_from_config.py \
  --dataset msmarco \
  --encoding pq \
  --scale top_300k \
  --hf_docids_repo kiyam/ddro-docids \
  --hf_tests_repo  kiyam/ddro-testsets
```

### Option B — Manual evaluation with HF URIs

**NQ + Title+URL**
```bash
python src/pretrain/hf_eval/eval_hf_docid_ranking.py \
  --pretrain_model_path kiyam/ddro-nq-tu \
  --docid_path "hf:dataset:kiyam/ddro-docids:tu_nq_docids.txt" \
  --test_file_path "hf:dataset:kiyam/ddro-testsets:nq/test_data/query_dev.t5_128_1.url_title_nq.json" \
  --dataset_script_dir src/data/data_scripts \
  --dataset_cache_dir ./cache \
  --num_beams 50 --add_doc_num 6144 \
  --max_seq_length 64 --max_docid_length 100 \
  --use_docid_rank True --docid_format nq \
  --lookup_fallback True --device cuda:0
```

**MS MARCO + PQ**
```bash
python src/pretrain/hf_eval/eval_hf_docid_ranking.py \
  --pretrain_model_path kiyam/ddro-msmarco-pq \
  --docid_path "hf:dataset:kiyam/ddro-docids:pq_msmarco_docids.txt" \
  --test_file_path "hf:dataset:kiyam/ddro-testsets:msmarco/test_data_top_300k/query_dev.t5_128_1.pq.top_300k.json" \
  --dataset_script_dir src/data/data_scripts \
  --dataset_cache_dir ./cache \
  --num_beams 80 --add_doc_num 6144 \
  --max_seq_length 64 --max_docid_length 24 \
  --use_docid_rank True --docid_format msmarco \
  --lookup_fallback True --device cuda:0
```

### Results

| Dataset | Docid | Model | MRR@10 | R@10 |
|---|---|---|---|---|
| MS MARCO | PQ | `kiyam/ddro-msmarco-pq` | 45.76 | 73.02 |
| MS MARCO | TU | `kiyam/ddro-msmarco-tu` | 50.07 | 74.01 |
| NQ | PQ | `kiyam/ddro-nq-pq` | 55.51 | 67.31 |
| NQ | TU | `kiyam/ddro-nq-tu` | 45.99 | 55.98 |

### Notes

- Recommended: `transformers==4.37.2`, `tokenizers==0.15.2`
- NQ–PQ uses canonical integer docids; NQ–TU uses lowercased url_title strings — do not mix assets across sources
- Default beam counts: NQ-PQ (100), NQ-TU (50), MS MARCO-PQ (80)
- Logs saved to `logs/<dataset>/dpo_*.log` and `logs/<dataset>/dpo_*.csv`

---

## Datasets and Checkpoints

All preprocessed datasets, docid encodings, and model checkpoints:
[DDRO Generative IR Collection on Hugging Face](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5)

| Resource | Link |
|---|---|
| MS MARCO Top-300K dataset | [kiyam/ddro-ms-dataset](https://huggingface.co/datasets/kiyam/ddro-ms-dataset) |
| NQ-320K dataset | [kiyam/ddro-nq-dataset](https://huggingface.co/datasets/kiyam/ddro-nq-dataset) |
| DocID tables | [kiyam/ddro-docids](https://huggingface.co/datasets/kiyam/ddro-docids) |
| Eval test sets | [kiyam/ddro-testsets](https://huggingface.co/datasets/kiyam/ddro-testsets) |

---

## Acknowledgments

- [ULTRON](https://github.com/smallporridge/WebUltron)
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [NCI (Neural Corpus Indexer)](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)
- [docTTTTTquery](https://github.com/castorini/docTTTTTquery)

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Citation

```bibtex
@inproceedings{mekonnen2025lightweight,
  title={Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval},
  author={Mekonnen, Kidist Amde and Tang, Yubao and de Rijke, Maarten},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1327--1338},
  year={2025}
}
```

---

For questions, please open an [issue](https://github.com/kidist-amde/DDRO-Direct-Document-Relevance-Optimization/issues).

© 2025 Kidist Amde Mekonnen — [IRLab](https://irlab.science.uva.nl/), University of Amsterdam
