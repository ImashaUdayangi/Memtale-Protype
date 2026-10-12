import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "evaluation/results"
EVAL_SUMMARY_PATH = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
EMO_SUMMARY_PATH = os.path.join(RESULTS_DIR, "emotional_fidelity_summary.csv")

OUT_PNG = os.path.join(RESULTS_DIR, "main_results_bar_chart.png")


def normalize_method_name(m: str) -> str:
    m = str(m).strip()
    if m.lower() == "ours":
        return "OURS"
    if m.lower() == "bm25":
        return "BM25"
    if m.lower() == "sbert":
        return "SBERT"
    if m.lower() == "rrf":
        return "RRF"
    return m


def main():
    if not os.path.exists(EVAL_SUMMARY_PATH):
        raise FileNotFoundError(f"Missing file: {EVAL_SUMMARY_PATH}")

    if not os.path.exists(EMO_SUMMARY_PATH):
        raise FileNotFoundError(f"Missing file: {EMO_SUMMARY_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load evaluation summary (nDCG@5, MRR) ----
    eval_df = pd.read_csv(EVAL_SUMMARY_PATH)
    # --- Choose final OURS (wemo=0.20) and drop other OURS variants ---
    eval_df = eval_df[eval_df["Method"].isin(["BM25", "SBERT", "RRF", "CE-Rerank", "OURS-wemo=0.20"])].copy()
    eval_df.loc[eval_df["Method"] == "OURS-wemo=0.20", "Method"] = "OURS"

    if not {"Method", "nDCG@5", "MRR"}.issubset(eval_df.columns):
        raise ValueError(
            f"{EVAL_SUMMARY_PATH} must contain columns: Method, nDCG@5, MRR. "
            f"Found: {list(eval_df.columns)}"
        )

    eval_df["Method"] = eval_df["Method"].apply(normalize_method_name)

    # ---- Load emotional fidelity summary (EmoFidelity@5 overall) ----
    emo_df = pd.read_csv(EMO_SUMMARY_PATH)
    # --- Choose final OURS (wemo=0.20) and drop other OURS variants ---
    emo_df = emo_df[emo_df["Method"].isin(["BM25", "SBERT", "RRF", "CE-Rerank", "OURS-wemo=0.20"])].copy()
    emo_df.loc[emo_df["Method"] == "OURS-wemo=0.20", "Method"] = "OURS"
    # Expect columns: Method, Subset, EmoFidelity@5, EmoFidelity@10
    needed = {"Method", "Subset", "EmoFidelity@5"}
    if not needed.issubset(emo_df.columns):
        raise ValueError(
            f"{EMO_SUMMARY_PATH} must contain columns: {sorted(list(needed))}. "
            f"Found: {list(emo_df.columns)}"
        )

    emo_df["Method"] = emo_df["Method"].apply(normalize_method_name)

    # Pick overall @5 (your file has overall rows + some NaNs in separate lines)
    emo_overall_5 = (
        emo_df[(emo_df["Subset"] == "overall") & (emo_df["EmoFidelity@5"].notna())]
        .copy()
    )

    # ---- Merge into one table ----
    merged = pd.merge(
        eval_df[["Method", "nDCG@5", "MRR"]],
        emo_overall_5[["Method", "EmoFidelity@5"]],
        on="Method",
        how="left",
    )

    # Desired order for the plot
    method_order = ["BM25", "SBERT", "RRF", "CE-Rerank", "OURS"]
    merged["Method"] = pd.Categorical(merged["Method"], categories=method_order, ordered=True)
    merged = merged.sort_values("Method")

    # Fill missing emo fidelity with 0 if any method doesn't have it
    merged["EmoFidelity@5"] = merged["EmoFidelity@5"].fillna(0.0)

    # ---- Plot: grouped bars (3 metrics per method) ----
    methods = merged["Method"].astype(str).tolist()
    x = np.arange(len(methods))
    width = 0.25

    ndcg_vals = merged["nDCG@5"].astype(float).to_numpy()
    mrr_vals = merged["MRR"].astype(float).to_numpy()
    emo_vals = merged["EmoFidelity@5"].astype(float).to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, ndcg_vals, width, label="nDCG@5")
    plt.bar(x,         mrr_vals,  width, label="MRR")
    plt.bar(x + width, emo_vals,  width, label="EmoFidelity@5")

    plt.xticks(x, methods)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Main Results: Relevance vs Emotional Alignment")
    plt.legend()

    # Add value labels (optional but helpful)
    def add_labels(xs, ys):
        for xi, yi in zip(xs, ys):
            plt.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

    add_labels(x - width, ndcg_vals)
    add_labels(x, mrr_vals)
    add_labels(x + width, emo_vals)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    print(" Saved combined grouped bar chart:")
    print(f" - {OUT_PNG}")
    print("\nData used:")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
