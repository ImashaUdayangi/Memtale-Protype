# evaluation/compute_metrics.py

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

def dcg(rels):
    return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k_from_qrels(qid, ranked_docs, qrels_dict, k=5):
    # gains for retrieved docs (missing judged docs -> 0)
    retrieved_rels = [qrels_dict[qid].get(doc_id, 0) for doc_id in ranked_docs[:k]]

    # ideal gains from ALL judged docs for that query
    all_judged_rels = list(qrels_dict[qid].values())
    ideal_rels = sorted(all_judged_rels, reverse=True)[:k]

    if sum(ideal_rels) == 0:
        return 0.0

    return dcg(retrieved_rels) / dcg(ideal_rels)

def mrr_from_qrels(qid, ranked_docs, qrels_dict):
    for i, doc_id in enumerate(ranked_docs):
        if qrels_dict[qid].get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def pretty_method_name(stem: str) -> str:
    """
    Convert run filename stem -> nice method name.
    Examples:
      bm25 -> BM25
      sbert -> SBERT
      rrf -> RRF
      ours -> OURS
      ce -> CE-Rerank
      wemo_0.20 -> OURS-wemo=0.20
    """
    s = stem.lower()
    if s == "bm25":
        return "BM25"
    if s == "sbert":
        return "SBERT"
    if s == "rrf":
        return "RRF"
    if s == "ours":
        return "OURS"
    if s == "ce":
        return "CE-Rerank"
    if s.startswith("wemo_"):
        val = s.split("_", 1)[1]
        return f"OURS-wemo={val}"
    return stem  # fallback

QRELS_PATH = Path("dataset/eval_v3/qrels.csv")
RUNS_DIR = Path("evaluation/results")

qrels = pd.read_csv(QRELS_PATH)

# build qrels dict
qrels_dict = defaultdict(dict)
for _, row in qrels.iterrows():
    qrels_dict[str(row["query_id"])][int(row["story_id"])] = int(row["relevance"])

# Auto-detect run files (anything that ends with _run.csv)
run_paths = sorted(RUNS_DIR.glob("*_run.csv"))
if not run_paths:
    raise FileNotFoundError(f"No *_run.csv files found in {RUNS_DIR.resolve()}")

results = []

for run_path in run_paths:
    stem = run_path.name.replace("_run.csv", "")
    method = pretty_method_name(stem)

    run = pd.read_csv(run_path)
    required_cols = {"query_id", "story_id", "rank"}
    missing = required_cols - set(run.columns)
    if missing:
        raise ValueError(f"{run_path.name} missing columns: {missing}")

    run["query_id"] = run["query_id"].astype(str)
    run["story_id"] = run["story_id"].astype(int)
    run["rank"] = run["rank"].astype(int)

    ndcgs = []
    mrrs = []

    for qid, group in run.groupby("query_id"):
        if qid not in qrels_dict:
            continue

        ranked_docs = group.sort_values("rank")["story_id"].tolist()
        ranked_docs = list(dict.fromkeys(ranked_docs))  # safety: drop duplicates, keep order

        ndcgs.append(ndcg_at_k_from_qrels(qid, ranked_docs, qrels_dict, k=5))
        mrrs.append(mrr_from_qrels(qid, ranked_docs, qrels_dict))

    results.append({
        "Method": method,
        "nDCG@5": round(float(np.mean(ndcgs)), 4),
        "MRR": round(float(np.mean(mrrs)), 4),
        "RunFile": run_path.name,
    })

results_df = pd.DataFrame(results)

# Sort: show main baselines first, then CE, then OURS, then wemo grid
def sort_key(m):
    order = {
        "BM25": 0,
        "SBERT": 1,
        "RRF": 2,
        "CE-Rerank": 3,
        "OURS": 4,
    }
    if m in order:
        return (order[m], 0.0)
    if m.startswith("OURS-wemo="):
        try:
            val = float(m.split("=", 1)[1])
        except:
            val = 99.0
        return (5, val)
    return (9, 0.0)

results_df = results_df.sort_values(by="Method", key=lambda s: s.map(sort_key))

OUT_PATH = RUNS_DIR / "evaluation_summary.csv"
results_df.to_csv(OUT_PATH, index=False)

print("\n=== Evaluation Results (nDCG@5, MRR) ===")
print(results_df[["Method", "nDCG@5", "MRR", "RunFile"]].to_string(index=False))
print(f"\nSaved to {OUT_PATH}")