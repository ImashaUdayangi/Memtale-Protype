import pandas as pd

INFILE = "dataset/eval_v3/judging_pool_blind.csv"
OUT_A1 = "dataset/eval_v3/judging_pool_blind_subset20_a1.csv"
OUT_A2 = "dataset/eval_v3/judging_pool_blind_subset20_a2.csv"

RANDOM_SEED = 42
PER_EMOTION = 48   # 9 emotions * 48 = 432

df = pd.read_csv(INFILE)

# We need target_emotion per row. If not present in judging_pool_blind.csv, merge from queries.csv.
if "target_emotion" not in df.columns:
    q = pd.read_csv("dataset/eval_v3/queries.csv")[["query_id", "target_emotion"]]
    df = df.merge(q, on="query_id", how="left")

assert df["target_emotion"].notna().all(), "Missing target_emotion after merge."

# Stratified sampling inside each emotion by relevance label
subsets = []
for emo, g in df.groupby("target_emotion"):
    g = g.copy()
    g["relevance"] = pd.to_numeric(g["relevance"], errors="coerce")
    # If any relevance missing, that's a problem for annotator1 file
    assert g["relevance"].notna().all(), f"Missing relevance in emotion {emo}"

    # sample roughly evenly from labels 0/1/2
    per_label = PER_EMOTION // 3  # 16
    picked = []
    for lab in [0, 1, 2]:
        gl = g[g["relevance"] == lab]
        n = min(per_label, len(gl))
        picked.append(gl.sample(n=n, random_state=RANDOM_SEED))

    picked_df = pd.concat(picked, ignore_index=True)

    # If not enough to reach 48 (some label might be rare), fill from remaining randomly
    remaining_needed = PER_EMOTION - len(picked_df)
    if remaining_needed > 0:
        remaining_pool = g.drop(picked_df.index, errors="ignore")
        fill = remaining_pool.sample(n=remaining_needed, random_state=RANDOM_SEED)
        picked_df = pd.concat([picked_df, fill], ignore_index=True)

    subsets.append(picked_df)

subset_a1 = pd.concat(subsets, ignore_index=True)

# Save annotator 1 subset (your current labels)
subset_a1.to_csv(OUT_A1, index=False)

# Create annotator 2 file with blank relevance
subset_a2 = subset_a1.copy()
subset_a2["relevance"] = ""
subset_a2.to_csv(OUT_A2, index=False)

print("Saved:", OUT_A1, "rows:", len(subset_a1))
print("Saved:", OUT_A2, "rows:", len(subset_a2))
print(subset_a1["target_emotion"].value_counts())
print(subset_a1["relevance"].value_counts())