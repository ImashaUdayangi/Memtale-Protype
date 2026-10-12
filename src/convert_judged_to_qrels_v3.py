# src/convert_judged_to_qrels_v3.py
from pathlib import Path
import pandas as pd

IN_PATH  = Path("dataset/eval_v3/judging_pool_blind.csv")
OUT_PATH = Path("dataset/eval_v3/qrels.csv")
REPORT_PATH = Path("dataset/eval_v3/qrels_v3_report.csv")

VALID = {0, 1, 2}

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # Basic required columns check
    required = {"query_id", "story_id", "relevance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in judging pool: {missing}")

    # Ensure relevance is numeric/int
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")

    # Check missing relevance
    n_missing = df["relevance"].isna().sum()
    if n_missing > 0:
        bad_rows = df[df["relevance"].isna()][["query_id", "story_id"]].head(10)
        raise ValueError(
            f"{n_missing} rows have missing/non-numeric relevance.\n"
            f"Example rows:\n{bad_rows}"
        )

    # Convert to int
    df["relevance"] = df["relevance"].astype(int)

    # Validate relevance values
    bad = df[~df["relevance"].isin(VALID)]
    if not bad.empty:
        example = bad[["query_id", "story_id", "relevance"]].head(10)
        raise ValueError(
            f"Found invalid relevance values (must be 0/1/2). Example:\n{example}"
        )

    # Keep only needed columns
    qrels = df[["query_id", "story_id", "relevance"]].copy()

    # Remove duplicates safely (should not exist, but guard anyway)
    dup = qrels.duplicated(subset=["query_id", "story_id"], keep=False)
    if dup.any():
        dups = qrels[dup].sort_values(["query_id", "story_id"])
        raise ValueError(
            "Duplicate (query_id, story_id) pairs found. Fix before proceeding.\n"
            f"Example duplicates:\n{dups.head(20)}"
        )

    # Save qrels
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    qrels.to_csv(OUT_PATH, index=False)

    # Create a small report to sanity check label distribution
    report = (
        qrels.groupby(["query_id", "relevance"])
             .size()
             .unstack(fill_value=0)
             .rename(columns={0: "rel0", 1: "rel1", 2: "rel2"})
             .reset_index()
             .sort_values("query_id")
    )
    report.to_csv(REPORT_PATH, index=False)

    # Print summary
    print("✅ Saved qrels:", OUT_PATH)
    print("✅ Saved report:", REPORT_PATH)

    total = len(qrels)
    counts = qrels["relevance"].value_counts().to_dict()
    print("\n=== Overall label counts ===")
    print("Total rows:", total)
    print("rel=2:", counts.get(2, 0))
    print("rel=1:", counts.get(1, 0))
    print("rel=0:", counts.get(0, 0))

    print("\n=== Per-query check (first 10 queries) ===")
    print(report.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
