# src/build_judging_pool_v3.py
"""
Build a manual judging pool (multi-system pooling) for emotion-intent queries.

Outputs:
  1) dataset/eval/judging_pool_blind.csv   (for manual labeling: no emotion_label shown)
  2) dataset/eval/judging_pool_key.csv     (contains emotion_label; keep separate)
  3) dataset/eval/judging_pool_meta.csv    (pool stats for sanity checks)

Design goals (reviewer-safe):
- Pool candidates using multiple retrieval systems to avoid bias/circularity.
- Do NOT use any system to "define relevance"; only to select candidates for human judgment.
- Blind the judge from gold labels and system ranks to reduce bias.
"""

from pathlib import Path
import random
import re

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ------------------------
# Paths
# ------------------------
STORIES_PATH = Path("dataset/raw/empathetic_stories.csv")
QUERIES_PATH = Path("dataset/eval_v3/queries.csv")  
OUT_BLIND = Path("dataset/eval_v3/judging_pool_blind.csv")
OUT_KEY = Path("dataset/eval_v3/judging_pool_key.csv")
OUT_META = Path("dataset/eval_v3/judging_pool_meta.csv")

# ------------------------
# Settings (tune if needed)
# ------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Candidate pooling sizes
TOP_BM25 = 30
TOP_SBERT = 30
TOP_RRF = 30

# Final pool size per query (what you will manually judge)
POOL_SIZE = 30

# Text excerpt sizes (for human judging UI)
MAX_SITUATION_WORDS = 35
MAX_STORY_WORDS = 110

# Retrieval text mode (recommended = situation + Story)
INDEX_MODE = "full"  # "full" only in this pool builder (keep it consistent)


# ------------------------
# Helpers
# ------------------------
def tokenize(text: str):
    return re.findall(r"\b\w+\b", (text or "").lower())

def truncate_words(text: str, n: int):
    w = (text or "").split()
    return " ".join(w[:n])

def rrf_fusion(ranked_lists, k=60):
    """
    Reciprocal Rank Fusion. Returns scores aligned to doc indices [0..N-1].
    """
    if not ranked_lists:
        return None
    N = max(len(r) for r in ranked_lists)
    scores = np.zeros(N, dtype=float)
    for r in ranked_lists:
        for rank, doc_idx in enumerate(r, start=1):
            if doc_idx < N:
                scores[doc_idx] += 1.0 / (k + rank)
    return scores

def build_index_texts(stories: pd.DataFrame):
    # "full" = situation + Story (recommended for autobiographical retrieval)
    return (stories["situation"].fillna("") + " " + stories["Story"].fillna("")).tolist()


def main():
    # ------------------------
    # Load data
    # ------------------------
    stories = pd.read_csv(STORIES_PATH)
    queries = pd.read_csv(QUERIES_PATH)

    # Sanity
    required_story_cols = {"story_id", "situation", "Story", "emotion_label"}
    missing = required_story_cols - set(stories.columns)
    if missing:
        raise ValueError(f"Missing story cols: {missing}")

    required_query_cols = {"query_id", "query_text"}
    missing_q = required_query_cols - set(queries.columns)
    if missing_q:
        raise ValueError(f"Missing query cols: {missing_q}")

    stories["story_id"] = stories["story_id"].astype(int)

    story_ids = stories["story_id"].tolist()
    index_texts = build_index_texts(stories)

    # ------------------------
    # Build BM25 index (once)
    # ------------------------
    tokenized_docs = [tokenize(t) for t in index_texts]
    tokenized_docs = [toks if toks else ["_empty_"] for toks in tokenized_docs]
    bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

    # ------------------------
    # Build SBERT embeddings (once)
    # ------------------------
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embs = sbert.encode(index_texts, convert_to_numpy=True, normalize_embeddings=True)

    # ------------------------
    # Build pools
    # ------------------------
    blind_rows = []
    key_rows = []
    meta_rows = []

    for q in queries.itertuples(index=False):
        qid = str(q.query_id)
        qtext = str(q.query_text)

        # ---- BM25 ranking ----
        bm25_scores = np.asarray(bm25.get_scores(tokenize(qtext)))
        bm25_ranked = np.argsort(bm25_scores)[::-1].tolist()

        # ---- SBERT ranking ----
        q_emb = sbert.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)[0]
        sem_sims = doc_embs @ q_emb
        sbert_ranked = np.argsort(sem_sims)[::-1].tolist()

        # ---- RRF ranking (BM25 + SBERT) ----
        fused = rrf_fusion([bm25_ranked, sbert_ranked], k=60)
        rrf_ranked = np.argsort(fused)[::-1].tolist()

        # Candidate set = union of top lists
        cand = set(bm25_ranked[:TOP_BM25]) | set(sbert_ranked[:TOP_SBERT]) | set(rrf_ranked[:TOP_RRF])
        cand = list(cand)

        # If union is small, expand slightly using RRF
        if len(cand) < POOL_SIZE:
            extra = [x for x in rrf_ranked if x not in set(cand)]
            cand += extra[: (POOL_SIZE - len(cand))]

        # Now choose final pool = top POOL_SIZE by RRF score (stable + strong)
        cand_sorted = sorted(cand, key=lambda i: fused[i], reverse=True)[:POOL_SIZE]

        # Randomize order for judging (IMPORTANT: reduce bias)
        random.shuffle(cand_sorted)

        # Save rows
        for pos, doc_idx in enumerate(cand_sorted, start=1):
            sid = int(story_ids[doc_idx])

            situation = str(stories.loc[doc_idx, "situation"])
            story = str(stories.loc[doc_idx, "Story"])

            # Blind judge content (no emotion label)
            blind_rows.append({
                "query_id": qid,
                "query_text": qtext,
                "pool_pos": pos,
                "story_id": sid,
                "situation_excerpt": truncate_words(situation, MAX_SITUATION_WORDS),
                "story_excerpt": truncate_words(story, MAX_STORY_WORDS),

                # Manual label column (fill later):
                # 2 = relevant, 1 = somewhat relevant, 0 = not relevant
                "relevance": ""
            })

            # Key file (keep separate; do NOT use while judging)
            key_rows.append({
                "query_id": qid,
                "story_id": sid,
                "emotion_label": str(stories.loc[doc_idx, "emotion_label"]),
                "bm25_rank": int(bm25_ranked.index(doc_idx) + 1),
                "sbert_rank": int(sbert_ranked.index(doc_idx) + 1),
                "rrf_rank": int(rrf_ranked.index(doc_idx) + 1),
            })

        meta_rows.append({
            "query_id": qid,
            "cand_union_size": len(set(cand)),
            "final_pool_size": len(cand_sorted)
        })

    # ------------------------
    # Write outputs
    # ------------------------
    OUT_BLIND.parent.mkdir(parents=True, exist_ok=True)

    blind_df = pd.DataFrame(blind_rows)
    key_df = pd.DataFrame(key_rows)
    meta_df = pd.DataFrame(meta_rows)

    blind_df.to_csv(OUT_BLIND, index=False)
    key_df.to_csv(OUT_KEY, index=False)
    meta_df.to_csv(OUT_META, index=False)

    print("✅ Saved judging pool (blind):", OUT_BLIND, "rows:", len(blind_df))
    print("✅ Saved judging pool key:", OUT_KEY, "rows:", len(key_df))
    print("✅ Saved pool meta:", OUT_META, "rows:", len(meta_df))
    print("\nTip: Open judging_pool_blind.csv and fill relevance with 0/1/2 manually.")


if __name__ == "__main__":
    main()
