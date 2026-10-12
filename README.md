# Memtale Prototype — Emotion-Aware Summarisation and Retrieval

This repository contains a **fully runnable research prototype** of the *Memtale* system for **emotion-aware summarisation and retrieval of autobiographical-style narratives**.

The prototype demonstrates how **emotional alignment** can be incorporated as an explicit ranking signal on top of standard information retrieval pipelines, using a transparent two-stage architecture.

---

## 1. System Overview

The prototype implements and compares the following retrieval approaches:

- **BM25** — lexical retrieval baseline  
- **SBERT** — dense semantic retrieval baseline  
- **RRF** — hybrid retrieval using Reciprocal Rank Fusion (BM25 + SBERT)  
- **CE-Rerank** — two-stage retrieval using **BM25 candidate generation (topK=100)** followed by a **cross-encoder neural reranker**
  (**cross-encoder/ms-marco-MiniLM-L-6-v2**)  
- **OURS** — proposed two-stage retrieval:

  **BM25 candidate generation (topK=100) → semantic + emotion-aware reranking (SETL-2)**

The system supports **emotion-intent queries**, reflecting how users recall memories through affective cues rather than purely topical descriptions.

All experiments, metrics, and figures can be reproduced using a **single command**.

---

## 2. Project Structure

'''text
Memtale-Co-Archivist/
│
├── src/
│   ├── summarisation.py
│   ├── setl2.py
│   ├── retrieval.py
│   ├── data_build_empathetic.py
│   ├── precompute_summaries.py
│   ├── build_eval_queries_v3.py
│   ├── build_judging_pool_v3.py
│   └── convert_judged_to_qrels_v3.py
│
├── evaluation/
│   ├── run_two_stage_retrieval.py
│   ├── compute_metrics.py
│   ├── compute_emotional_fidelity.py
│   ├── compute_significance_test.py
│   ├── make_main_results_figure.py
│   ├── make_qualitative_rerank_example.py
│   ├── make_subset_for_iaa.py          # (Optional) creates 20% subset for 2nd annotator
│   └── compute_iaa_kappa.py            # (Optional) Cohen’s kappa agreement + adjudication
│
├── dataset/
│   ├── raw/empathetic_stories.csv
│   ├── cache/summaries.csv
│   └── eval_v3/
│       ├── queries.csv
│       ├── qrels.csv
│       ├── judging_pool_blind.csv
│       ├── judging_pool_blind_subset20_a1.csv
│       ├── judging_pool_blind_subset20_a2.csv
│       └── judging_pool_blind_subset20_adjudication.csv
│
├── main.py
|__ app.py
├── requirements.txt
└── README.md
'''

---

## 3. Environment Setup

### Requirements
- Python **3.10**
- Tested on **Windows (PowerShell)**

### Create and activate a virtual environment

```powershell
python -m venv venv_new
.\venv_new\Scripts\Activate.ps1
```

---

## 4. Install dependencies

```PowerShell
pip install -r requirements.txt
```

Note: The virtual environment (venv/) is not included in the repository and should be created locally.

---

## 5. Running the Prototype

After activating the virtual environment, run:

```powershell
python main.py
```

This single command:

- Runs retrieval methods: BM25, SBERT, RRF, CE-Rerank, OURS
- Generates additional weight sensitivity runs for OURS by varying wemo:
0.00, 0.10, 0.20, 0.30, 0.40, 0.50
- Computes ranking metrics (nDCG@5, MRR)
- Computes emotional alignment (EmoFidelity@5 and EmoFidelity@10)
- Performs statistical significance testing (Wilcoxon signed-rank)
- Generates the main results plot and a qualitative rerank example

No manual annotation is required in run mode.

---

## 6. Outputs

*Metrics + tests*
- evaluation/results/evaluation_summary.csv
  → nDCG@5 and MRR for all methods (including CE-Rerank and wemo variants)
- evaluation/results/emotional_fidelity_summary.csv
  → Emotional alignment summary (overall and emotion_intent)
- evaluation/results/significance_summary.csv
  → Paired Wilcoxon signed-rank test results

*Figures + qualitative analysis*
- evaluation/results/main_results_bar_chart.png
  → Combined grouped bar chart (ranking + emotional alignment)
- evaluation/results/qualitative_rerank_example.csv
  → Automatically selected reranking illustration

*Run files (ranked outputs)*
- evaluation/results/bm25_run.csv
- evaluation/results/sbert_run.csv
- evaluation/results/rrf_run.csv
- evaluation/results/ce_run.csv
- evaluation/results/wemo_0.00_run.csv
- evaluation/results/wemo_0.10_run.csv
- evaluation/results/wemo_0.20_run.csv ← **final chosen OURS configuration**
- evaluation/results/wemo_0.30_run.csv
- evaluation/results/wemo_0.40_run.csv
- evaluation/results/wemo_0.50_run.csv

---

## 7. Optional: Inter-Annotator Agreement (IAA) — 2nd Annotator + Cohen’s κ

This step is not required to reproduce the retrieval results, but supports by reporting annotation reliability.

### 7.1 Create a 20% subset (432 pairs) for double annotation

```PowerShell
python -m evaluation.make_subset_for_iaa
```

This produces:
- dataset/eval_v3/judging_pool_blind_subset20_a1.csv (annotator 1 labels)
- dataset/eval_v3/judging_pool_blind_subset20_a2.csv (blank relevance column for annotator 2)

Send judging_pool_blind_subset20_a2.csv to the second annotator and ask them to fill only the relevance column (0/1/2).

### 7.2 Compute Cohen’s kappa + create adjudication file

After the second annotator returns the completed file:

```PowerShell
python -m evaluation.compute_iaa_kappa
```
This produces:
- evaluation/results/iaa_subset20_report.txt
- dataset/eval_v3/judging_pool_blind_subset20_adjudication.csv

---

## 8. Reproducibility Notes

- All datasets, summaries, queries, and relevance judgments are pre-generated and frozen
- Manual judging is not required to reproduce results in run mode
- The --rebuild mode regenerates datasets and requires manual relevance annotation:

```PowerShell
python main.py --mode rebuild
```

---