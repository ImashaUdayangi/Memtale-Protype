import os
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset/eval_v3"
QRELS_PATH = os.path.join(DATASET_DIR, "qrels.csv")
QUERIES_PATH = os.path.join(DATASET_DIR, "queries.csv")

RUNS_DIR = "evaluation/results"
RUN_PATHS = {
    "BM25": os.path.join(RUNS_DIR, "bm25_run.csv"),
    "SBERT": os.path.join(RUNS_DIR, "sbert_run.csv"),
    "RRF": os.path.join(RUNS_DIR, "rrf_run.csv"),
    "CE-Rerank": os.path.join(RUNS_DIR, "ce_run.csv"),
    "OURS": os.path.join(RUNS_DIR, "wemo_0.20_run.csv"),   # ✅ final OURS
}

CORPUS_PATH = "dataset/raw/empathetic_stories.csv"  # story_id, emotion_label, Story, ...
RESULTS_DIR = "evaluation/results"

K_NDCG = 5
K_EMO = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Helpers: IR metrics ----------------
def dcg_at_k(rels, k):
    rels = rels[:k]
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))

def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return 0.0 if ideal == 0 else dcg / ideal

def mrr(rels):
    for i, r in enumerate(rels, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0

# ---------------- Helpers: Wilcoxon + effect size ----------------
def wilcoxon_with_effect(x, y):
    """
    Wilcoxon signed-rank test on paired per-query scores.
    Returns: n_used, Wplus, Wminus, statistic(min), pvalue, rank-biserial correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    d = y - x

    # remove zeros (wilcoxon ignores 0 diffs depending on zero_method; we remove to be explicit)
    d_nz = d[np.abs(d) > 1e-12]
    n = len(d_nz)
    if n == 0:
        return {
            "n_used": 0,
            "Wplus": np.nan,
            "Wminus": np.nan,
            "statistic": np.nan,
            "pvalue": np.nan,
            "r_rank_biserial": np.nan,
        }

    # Wilcoxon test
    res = wilcoxon(d_nz, zero_method="wilcox", alternative="two-sided", mode="auto")

    # Compute W+ and W- ourselves for rank-biserial
    abs_d = np.abs(d_nz)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()

    Wplus = float(np.sum(ranks[d_nz > 0]))
    Wminus = float(np.sum(ranks[d_nz < 0]))
    # Rank-biserial correlation (range [-1, 1], positive means y > x)
    rbc = (Wplus - Wminus) / (Wplus + Wminus)

    return {
        "n_used": n,
        "Wplus": Wplus,
        "Wminus": Wminus,
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "r_rank_biserial": float(rbc),
    }

# ---------------- Load data ----------------
# qrels: query_id, story_id, relevance
qrels = pd.read_csv(QRELS_PATH)
qrels["query_id"] = qrels["query_id"].astype(str)
qrels["story_id"] = qrels["story_id"].astype(int)
qrels["relevance"] = qrels["relevance"].astype(int)

# queries: query_id, query_text, query_type, target_emotion
queries = pd.read_csv(QUERIES_PATH)
queries["query_id"] = queries["query_id"].astype(str)
queries["query_type"] = queries["query_type"].astype(str)
queries["target_emotion"] = queries["target_emotion"].astype(str)

# Build qrels lookup: (qid -> dict(story_id -> relevance))
qrels_lookup = {}
for row in qrels.itertuples(index=False):
    qrels_lookup.setdefault(row.query_id, {})[row.story_id] = row.relevance

# Load corpus emotions: story_id -> emotion_label
stories = pd.read_csv(CORPUS_PATH)
if "story_id" not in stories.columns or "emotion_label" not in stories.columns:
    raise ValueError(
        f"Corpus must have story_id and emotion_label. Found columns: {list(stories.columns)}"
    )
story_emotion = dict(zip(stories["story_id"].astype(int), stories["emotion_label"].astype(str)))

# Load run files
runs = {}
for method, path in RUN_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing run file for {method}: {path}")
    df = pd.read_csv(path)
    df["query_id"] = df["query_id"].astype(str)
    df["story_id"] = df["story_id"].astype(int)
    df["rank"] = df["rank"].astype(int)

    # sort and group
    df = df.sort_values(["query_id", "rank"])
    runs[method] = df

    # sanity check: each query should have ranks starting at 1
    # (not mandatory, but helps catch incomplete files)

# ---------------- Compute per-query metrics ----------------
# We'll compute for ALL queries in queries.csv
per_query_rows = []

for q in queries.itertuples(index=False):
    qid = str(q.query_id)
    qtype = str(q.query_type)
    target_em = str(q.target_emotion)

    relmap = qrels_lookup.get(qid, {})

    row_out = {
        "query_id": qid,
        "query_type": qtype,
        "target_emotion": target_em,
    }

    for method in ["BM25", "SBERT", "RRF", "CE-Rerank", "OURS"]:
        df_run = runs[method]
        docs = df_run[df_run["query_id"] == qid]["story_id"].tolist()

        # relevance list for ranking positions
        rels = [int(relmap.get(doc_id, 0)) for doc_id in docs]

        row_out[f"{method}_nDCG@{K_NDCG}"] = ndcg_at_k(rels, K_NDCG)
        row_out[f"{method}_MRR"] = mrr(rels)  # MRR usually over full list

        # Emotional fidelity@5: fraction of top-5 whose emotion equals target
        topk = docs[:K_EMO]
        if len(topk) == 0:
            emo_fid = np.nan
        else:
            matches = 0
            for sid in topk:
                em = story_emotion.get(int(sid), "")
                if em == target_em:
                    matches += 1
            emo_fid = matches / len(topk)
        row_out[f"{method}_EmoFidelity@{K_EMO}"] = emo_fid

    per_query_rows.append(row_out)

per_query = pd.DataFrame(per_query_rows)
per_query_out = os.path.join(RESULTS_DIR, "per_query_metrics_for_significance.csv")
per_query.to_csv(per_query_out, index=False)
print(f"✅ Saved per-query metrics: {per_query_out}")

# ---------------- Wilcoxon tests ----------------
tests = []

def add_test(metric_col, base_method, ours_method="OURS", label=None):
    x = per_query[f"{base_method}_{metric_col}"].to_numpy()
    y = per_query[f"{ours_method}_{metric_col}"].to_numpy()
    stats = wilcoxon_with_effect(x, y)
    tests.append({
        "comparison": label or f"{ours_method} vs {base_method}",
        "metric": metric_col,
        **stats
    })

# nDCG@5
add_test(f"nDCG@{K_NDCG}", "BM25", label="OURS vs BM25")
add_test(f"nDCG@{K_NDCG}", "SBERT", label="OURS vs SBERT")
add_test(f"nDCG@{K_NDCG}", "RRF",  label="OURS vs RRF")
add_test(f"nDCG@{K_NDCG}", "CE-Rerank", label="OURS vs CE-Rerank")

# MRR
add_test("MRR", "BM25", label="OURS vs BM25")
add_test("MRR", "SBERT", label="OURS vs SBERT")
add_test("MRR", "RRF",  label="OURS vs RRF")
add_test("MRR", "CE-Rerank", label="OURS vs CE-Rerank")

# Emotional fidelity@5 (do against all key baselines now)
add_test(f"EmoFidelity@{K_EMO}", "BM25", label="OURS vs BM25")
add_test(f"EmoFidelity@{K_EMO}", "CE-Rerank", label="OURS vs CE-Rerank")
add_test(f"EmoFidelity@{K_EMO}", "RRF", label="OURS vs RRF")
add_test(f"EmoFidelity@{K_EMO}", "SBERT", label="OURS vs SBERT")

summary = pd.DataFrame(tests)
summary_out = os.path.join(RESULTS_DIR, "significance_summary.csv")
summary.to_csv(summary_out, index=False)

print("\n=== Wilcoxon Signed-Rank Results ===")
print(summary.to_string(index=False))
print(f"\n Saved significance summary: {summary_out}")
