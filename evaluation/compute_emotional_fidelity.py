import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset/eval_v3"
CORPUS_PATH = "dataset/raw/empathetic_stories.csv"
RESULTS_DIR = "evaluation/results"
K_LIST = [5, 10]
# ----------------------------------------

def pretty_method_name(stem: str) -> str:
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
    return stem

def load_corpus(corpus_path: str) -> pd.DataFrame:
    df = pd.read_csv(corpus_path)

    required_cols = {"story_id", "emotion_label", "Story"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Corpus missing columns: {missing}. Found columns: {list(df.columns)}"
        )

    df = df.rename(columns={"emotion_label": "emotion", "Story": "text"})
    df["emotion"] = df["emotion"].fillna("").astype(str)
    df["story_id"] = df["story_id"].astype(int)

    return df[["story_id", "emotion"]]

def load_queries(dataset_dir: str) -> pd.DataFrame:
    qpath = os.path.join(dataset_dir, "queries.csv")
    q = pd.read_csv(qpath)

    required_cols = {"query_id", "query_type", "target_emotion"}
    missing = required_cols - set(q.columns)
    if missing:
        raise ValueError(
            f"Queries missing columns: {missing}. Found columns: {list(q.columns)}"
        )

    q["query_id"] = q["query_id"].astype(str)
    q["query_type"] = q["query_type"].fillna("").astype(str)
    q["target_emotion"] = q["target_emotion"].fillna("").astype(str)
    return q

def load_run(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"query_id", "story_id", "rank"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Run file {path} missing columns: {missing}")

    df["query_id"] = df["query_id"].astype(str)
    df["story_id"] = df["story_id"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df

def emotional_fidelity_at_k(run_df, queries_df, storyid_to_emotion, k: int) -> pd.DataFrame:
    """
    per-query emotional fidelity@k:
    proportion of top-k retrieved docs with emotion == target_emotion.
    """
    topk = run_df[run_df["rank"] <= k].copy()

    topk = topk.merge(
        queries_df[["query_id", "target_emotion", "query_type"]],
        on="query_id",
        how="left",
    )

    topk["retrieved_emotion"] = topk["story_id"].map(storyid_to_emotion).fillna("")
    topk["em_match"] = (topk["retrieved_emotion"] == topk["target_emotion"]).astype(int)

    per_q = (
        topk.groupby(["query_id", "query_type", "target_emotion"], as_index=False)["em_match"]
        .mean()
        .rename(columns={"em_match": f"emo_fidelity@{k}"})
    )
    return per_q

def summarize(per_query_df: pd.DataFrame, k: int, method_name: str) -> pd.DataFrame:
    col = f"emo_fidelity@{k}"

    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else np.nan

    overall = safe_mean(per_query_df[col].values)

    emo_mask = per_query_df["query_type"] == "emotion_intent"
    emo_intent = safe_mean(per_query_df.loc[emo_mask, col].values)

    non_mask = ~emo_mask
    non_emo = safe_mean(per_query_df.loc[non_mask, col].values)

    return pd.DataFrame(
        [
            {"Method": method_name, "Subset": "overall", f"EmoFidelity@{k}": overall},
            {"Method": method_name, "Subset": "emotion_intent", f"EmoFidelity@{k}": emo_intent},
            {"Method": method_name, "Subset": "non_emotion_intent", f"EmoFidelity@{k}": non_emo},
        ]
    )

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs_dir = Path(RESULTS_DIR)

    run_paths = sorted(runs_dir.glob("*_run.csv"))
    if not run_paths:
        raise FileNotFoundError(f"No *_run.csv files found in {runs_dir.resolve()}")

    queries = load_queries(DATASET_DIR)
    corpus = load_corpus(CORPUS_PATH)
    storyid_to_emotion = dict(zip(corpus["story_id"], corpus["emotion"]))

    # Start per-query table with base query info
    per_query_all = queries[["query_id", "query_type", "target_emotion"]].copy()
    summary_rows = []

    for run_path in run_paths:
        stem = run_path.name.replace("_run.csv", "")
        method = pretty_method_name(stem)

        run_df = load_run(str(run_path))

        for k in K_LIST:
            per_q = emotional_fidelity_at_k(run_df, queries, storyid_to_emotion, k)

            # Merge per-query scores into wide table
            per_query_all = per_query_all.merge(
                per_q[["query_id", f"emo_fidelity@{k}"]].rename(
                    columns={f"emo_fidelity@{k}": f"{method}_emo_fidelity@{k}"}
                ),
                on="query_id",
                how="left",
            )

            summary_rows.append(summarize(per_q, k, method))

    summary_df = pd.concat(summary_rows, ignore_index=True)

    out_summary = os.path.join(RESULTS_DIR, "emotional_fidelity_summary.csv")
    out_per_query = os.path.join(RESULTS_DIR, "emotional_fidelity_per_query.csv")

    summary_df.to_csv(out_summary, index=False)
    per_query_all.to_csv(out_per_query, index=False)

    print("\n=== Emotional Fidelity Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out_summary}")
    print(f"Saved: {out_per_query}")

if __name__ == "__main__":
    main()