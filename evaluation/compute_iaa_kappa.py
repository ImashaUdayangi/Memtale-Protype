import pandas as pd
import numpy as np

A1_FILE = "dataset/eval_v3/judging_pool_blind_subset20_a1.csv"
A2_FILE = "dataset/eval_v3/judging_pool_blind_subset20_a2.csv"
OUT_REPORT = "evaluation/results/iaa_subset20_report.txt"
OUT_ADJ = "dataset/eval_v3/judging_pool_blind_subset20_adjudication.csv"

KEY_COLS = ["query_id", "story_id", "pool_pos"]
LABELS = [0, 1, 2]

def weighted_kappa(y1, y2, labels=(0,1,2), weight_type="quadratic"):
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(labels)
    idx = {lab:i for i,lab in enumerate(labels)}

    O = np.zeros((n, n), dtype=float)
    for a, b in zip(y1, y2):
        O[idx[a], idx[b]] += 1

    N = O.sum()
    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col) / N

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if weight_type == "linear":
                W[i, j] = abs(i - j) / (n - 1)
            elif weight_type == "quadratic":
                W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
            else:  # unweighted
                W[i, j] = 0.0 if i == j else 1.0

    num = (W * O).sum() / N
    den = (W * E).sum() / N
    return 1.0 - (num / den)

def main():
    a1 = pd.read_csv(A1_FILE)
    a2 = pd.read_csv(A2_FILE)

    # Safety: ensure rows match (same items in same order)
    if not a1[KEY_COLS].equals(a2[KEY_COLS]):
        # If order differs, merge to align
        a1 = a1.sort_values(KEY_COLS).reset_index(drop=True)
        a2 = a2.sort_values(KEY_COLS).reset_index(drop=True)
        assert a1[KEY_COLS].equals(a2[KEY_COLS]), "A1/A2 subsets do not match on keys."

    y1 = pd.to_numeric(a1["relevance"], errors="raise").astype(int)
    y2 = pd.to_numeric(a2["relevance"], errors="raise").astype(int)

    k_unw = weighted_kappa(y1, y2, labels=LABELS, weight_type="unweighted")
    k_lin = weighted_kappa(y1, y2, labels=LABELS, weight_type="linear")
    k_quad = weighted_kappa(y1, y2, labels=LABELS, weight_type="quadratic")

    # Per-emotion kappas (nice to report as extra evidence)
    per_emo = {}
    if "target_emotion" in a1.columns:
        for emo, g in a1.groupby("target_emotion"):
            idxs = g.index
            per_emo[emo] = weighted_kappa(y1.loc[idxs], y2.loc[idxs], labels=LABELS, weight_type="quadratic")

    # Build adjudication sheet (does NOT change your qrels automatically)
    adj = a1.copy()
    adj = adj.rename(columns={"relevance": "relevance_a1"})
    adj["relevance_a2"] = y2
    adj["disagree"] = (adj["relevance_a1"].astype(int) != adj["relevance_a2"].astype(int))
    adj["final_relevance"] = ""  # you will fill only where disagree=True

    adj.to_csv(OUT_ADJ, index=False)

    lines = []
    lines.append(f"Pairs: {len(a1)}")
    lines.append(f"Cohen kappa (unweighted): {k_unw:.4f}")
    lines.append(f"Cohen kappa (linear):     {k_lin:.4f}")
    lines.append(f"Cohen kappa (quadratic):  {k_quad:.4f}")
    lines.append(f"Disagreements: {adj['disagree'].sum()} ({adj['disagree'].mean()*100:.2f}%)")

    if per_emo:
        lines.append("\nQuadratic kappa per emotion:")
        for emo, val in sorted(per_emo.items()):
            lines.append(f"  {emo:12s}: {val:.4f}")

    report = "\n".join(lines)
    print(report)

    # also save report to file
    import os
    os.makedirs("evaluation/results", exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\nSaved adjudication file:", OUT_ADJ)
    print("Saved report:", OUT_REPORT)

if __name__ == "__main__":
    main()