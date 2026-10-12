#src/precompute_summaries.py
import pandas as pd
from pathlib import Path
from src.summarisation import naive_summary, emotion_aware_summary

IN_PATH = Path("dataset/raw/empathetic_stories.csv")
OUT_PATH = Path("dataset/cache/summaries.csv")

def main():
    df = pd.read_csv(IN_PATH)

    # Use the correct column name (your dataset schema uses "Story")
    texts = df["Story"].fillna("").tolist()

    naive_list = []
    aware_list = []

    for t in texts:
        base = naive_summary(t)
        aware = emotion_aware_summary(t, base_summary=base)  # no second summarizer call
        naive_list.append(base)
        aware_list.append(aware)

    out = pd.DataFrame({
        "story_id": df["story_id"].astype(int),
        "naive_summary": naive_list,
        "aware_summary": aware_list
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("Saved summaries to", OUT_PATH)

if __name__ == "__main__":
    main()
