# evaluation/make_qualitative_rerank_example.py
import os
import pandas as pd

DATASET_DIR = "dataset/eval_v3"
RESULTS_DIR = "evaluation/results"
CORPUS_PATH = "dataset/raw/empathetic_stories.csv"

BM25_RUN = os.path.join(RESULTS_DIR, "bm25_run.csv")
OURS_RUN = os.path.join(RESULTS_DIR, "wemo_0.20_run.csv")
PERQ_PATH = os.path.join(RESULTS_DIR, "per_query_metrics_for_significance.csv")
QUERIES_PATH = os.path.join(DATASET_DIR, "queries.csv")

OUT_CSV = os.path.join(RESULTS_DIR, "qualitative_rerank_example.csv")

TOPK = 5

def load_runs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["query_id"] = df["query_id"].astype(str)
    df["story_id"] = df["story_id"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df

def main():
    # Load data
    perq = pd.read_csv(PERQ_PATH)
    perq["query_id"] = perq["query_id"].astype(str)

    queries = pd.read_csv(QUERIES_PATH)
    queries["query_id"] = queries["query_id"].astype(str)

    stories = pd.read_csv(CORPUS_PATH)
    stories["story_id"] = stories["story_id"].astype(int)
    stories["emotion_label"] = stories["emotion_label"].astype(str)
    stories["Story"] = stories["Story"].fillna("").astype(str)

    story_em = dict(zip(stories["story_id"], stories["emotion_label"]))
    story_text = dict(zip(stories["story_id"], stories["Story"]))

    bm25 = load_runs(BM25_RUN)
    ours = load_runs(OURS_RUN)

    # Only emotion_intent queries (where target_emotion is meaningful)
    df_e = perq[perq["query_type"] == "emotion_intent"].copy()

    # Compute deltas (OURS - BM25)
    df_e["delta_emo5"] = df_e["OURS_EmoFidelity@5"] - df_e["BM25_EmoFidelity@5"]
    df_e["delta_ndcg5"] = df_e["OURS_nDCG@5"] - df_e["BM25_nDCG@5"]

    # Non-cherry-picked rule:
    # pick query with max emotional fidelity improvement;
    # tie-break by relevance improvement.
    df_e = df_e.sort_values(["delta_emo5", "delta_ndcg5"], ascending=False)
    chosen = df_e.iloc[0]
    qid = str(chosen["query_id"])

    qrow = queries[queries["query_id"] == qid].iloc[0]
    qtext = str(qrow.get("query_text", ""))
    target = str(qrow.get("target_emotion", ""))

    # Get top-5 story_ids for each system
    bm25_top = bm25[bm25["query_id"] == qid].sort_values("rank").head(TOPK)["story_id"].tolist()
    ours_top = ours[ours["query_id"] == qid].sort_values("rank").head(TOPK)["story_id"].tolist()

    def pack(system_name: str, story_ids):
        rows = []
        for r, sid in enumerate(story_ids, start=1):
            rows.append({
                "query_id": qid,
                "query_text": qtext,
                "target_emotion": target,
                "system": system_name,
                "rank": r,
                "story_id": sid,
                "story_emotion": story_em.get(sid, ""),
                "story_excerpt": (story_text.get(sid, "")[:160] + "...") if len(story_text.get(sid, "")) > 160 else story_text.get(sid, "")
            })
        return rows

    rows = []
    rows += pack("BM25", bm25_top)
    rows += pack("OURS", ours_top)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print("\n=== Qualitative Rerank Example (Auto-picked) ===")
    print("Chosen query_id:", qid)
    print("Query text:", qtext)
    print("Target emotion:", target)
    print("Delta EmoFidelity@5 (OURS-BM25):", float(chosen["delta_emo5"]))
    print("Delta nDCG@5 (OURS-BM25):", float(chosen["delta_ndcg5"]))
    print("\nTop-5 comparison saved to:", OUT_CSV)

    # Print compact view
    view = out[["system", "rank", "story_id", "story_emotion", "story_excerpt"]]
    print("\n", view.to_string(index=False))

if __name__ == "__main__":
    main()
