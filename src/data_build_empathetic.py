# src/data_build_empathetic.py
import pandas as pd
from pathlib import Path
import re

IN_PATH = Path("dataset/raw/emotion-emotion_69k.csv")              # Kaggle file
OUT_PATH = Path("dataset/raw/empathetic_stories.csv")  # output clean dataset

TARGET_TOTAL = 5000         # total stories target
MIN_WORDS = 100
MAX_WORDS = 600
SOURCE_NAME = "EmpatheticDialogues_Kaggle"
RANDOM_SEED = 42

#15th Jan Removed as per 5000 record creation
# TARGET_PER_EMOTION = 10     # ~10 per emotion
# MIN_WORDS = 200
# MAX_WORDS = 600
# SOURCE_NAME = "EmpatheticDialogues_Kaggle"


def _clean_text(x) -> str:
    """Safe text cleanup for NaNs and spacing."""
    if pd.isna(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()

#15th jan Added function to strip speaker tags
def _strip_speaker_tags(text: str) -> str:
    """
    Remove repeated speaker tags like 'Customer :' and 'Agent :' from dialogue text.
    Also normalizes whitespace after removal.
    """
    if not text:
        return ""
    # remove variations like "Customer:" "Customer :" "Agent:" "Agent :"
    text = re.sub(r"\b(Customer|Agent)\s*:\s*", "", text, flags=re.IGNORECASE)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    # 1) Load
    df = pd.read_csv(IN_PATH)

    # 2) Standardize column names (Kaggle sometimes has casing or extra unnamed columns)
    df.columns = [c.strip() for c in df.columns]

    # Drop extra empty columns created by the trailing ",,"
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Expected columns: id, Situation, emotion, empathetic_dialogues, labels
    # Some Kaggle exports include unnamed columns at the end; ignore them.
    required = {"Situation", "emotion", "empathetic_dialogues", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    # 3) Clean key fields
    df["Situation"] = df["Situation"].apply(_clean_text)
    df["emotion"] = df["emotion"].apply(_clean_text).str.lower()
    df["empathetic_dialogues"] = df["empathetic_dialogues"].apply(_clean_text)
    df["labels"] = df["labels"].apply(_clean_text)

    # Remove blank situations or emotions
    df = df[(df["Situation"] != "") & (df["emotion"] != "")].copy()

    # 4) Build per-row "turn" text = dialogue + agent reply
    # This matches requirement: merge empathetic_dialogues + labels into Story
    def make_turn(row):
        d = row["empathetic_dialogues"]
        a = row["labels"]
        if d and a:
            return f"{d} {a}"
        return d or a

    df["turn_text"] = df.apply(make_turn, axis=1)
    df = df[df["turn_text"] != ""].copy()

    # 5) Group rows with the same Situation (+ emotion) into ONE story row
    # Note: Using (Situation, emotion) avoids mixing different emotions for same situation if any exist.
    grouped = (
        df.groupby(["Situation", "emotion"])["turn_text"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index(name="Story")
    )

    # Rename to target schema
    grouped = grouped.rename(columns={"Situation": "situation", "emotion": "emotion_label"})
    grouped["source"] = SOURCE_NAME

    # Remove 'Customer:' / 'Agent:' speaker tags from the merged story text #15th Jan Added
    grouped["Story"] = grouped["Story"].apply(_strip_speaker_tags)

    # --- NEW: enforce Story length 100–600 words ---
    def _word_count(t: str) -> int:
        return len((t or "").split())

    def _truncate_words(t: str, max_words: int) -> str:
        words = (t or "").split()
        return " ".join(words[:max_words])

    # truncate long stories to MAX_WORDS
    grouped["Story"] = grouped["Story"].apply(lambda t: _truncate_words(t, MAX_WORDS))

    # drop short stories below MIN_WORDS
    grouped = grouped[grouped["Story"].apply(_word_count) >= MIN_WORDS].copy()

    # 6) Sample a total of TARGET_TOTAL stories (keep natural emotion distribution)
    if len(grouped) <= TARGET_TOTAL:
        sampled = grouped.copy()
    else:
        sampled = grouped.sample(n=TARGET_TOTAL, random_state=RANDOM_SEED).reset_index(drop=True)

    #15th Jan Removed as per 5000 record creation
    # # 6) Balanced sampling: ~10 per emotion
    # emotions = sorted(grouped["emotion_label"].unique().tolist())

    # parts = []
    # for emo in emotions:
    #     chunk = grouped[grouped["emotion_label"] == emo]
    #     if len(chunk) <= TARGET_PER_EMOTION:
    #         parts.append(chunk)
    #     else:
    #         parts.append(chunk.sample(n=TARGET_PER_EMOTION, random_state=42))

    # sampled = pd.concat(parts, ignore_index=True).reset_index(drop=True)

    # 7) Add story_id
    sampled.insert(0, "story_id", range(1, len(sampled) + 1))

    # 8) Final column order 
    sampled = sampled[["story_id", "emotion_label", "situation", "Story", "source"]]

    # 9) Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(OUT_PATH, index=False)

    print("✅ Saved:", OUT_PATH)
    print("Rows:", len(sampled)) #15th Jan Added 
    print("Unique emotions:", sampled["emotion_label"].nunique())
    print("Emotion distribution (top 10):")
    print(sampled["emotion_label"].value_counts().head(10))
    print("Emotion distribution (tail 10):")
    print(sampled["emotion_label"].value_counts().tail(10))

    #15th Jan Removed as per 5000 record creation
    # print("Rows:", len(sampled))
    # print("Unique emotions:", sampled["emotion_label"].nunique())
    # print("Example emotions distribution (top 10):")
    # print(sampled["emotion_label"].value_counts().head(10))


if __name__ == "__main__":
    main()