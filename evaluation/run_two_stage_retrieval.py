# evaluation/run_two_stage_retrieval.py
import os
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from src.setl2 import setl2_vector_np  # SETL-2 emotion vector (8 dims)

# ---------------- CONFIG ----------------
K = 100  # topK candidates from BM25
DATASET_DIR = "dataset/eval_v3"
SUMMARIES_PATH = "dataset/cache/summaries.csv"   #use aware summaries
RESULTS_DIR = "evaluation/results"

SBERT_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60

# ----- Weight Sensitivity Configs (M3) -----
WEIGHT_CONFIGS = {
    "wemo_0.00": (0.40, 0.60, 0.00),
    "wemo_0.10": (0.35, 0.55, 0.10),
    "wemo_0.20": (0.30, 0.50, 0.20),
    "wemo_0.30": (0.30, 0.40, 0.30),  # current OURS
    "wemo_0.40": (0.25, 0.35, 0.40),
    "wemo_0.50": (0.20, 0.30, 0.50),
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def tokenize(text: str):
    return re.findall(r"\b\w+\b", (text or "").lower())

def minmax_norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if (mx - mn) < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float((a / na) @ (b / nb))

def rrf(rank1, rank2, k=60):
    """
    Reciprocal Rank Fusion over index lists.
    rank1/rank2 are arrays of doc indices sorted best->worst
    """
    scores = {}
    for i, doc_idx in enumerate(rank1):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + i + 1)
    for i, doc_idx in enumerate(rank2):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------------- Load Queries ----------------
queries = pd.read_csv(os.path.join(DATASET_DIR, "queries.csv"))

required_qcols = {"query_id", "query_text", "query_type"}
missing_q = required_qcols - set(queries.columns)
if missing_q:
    raise ValueError(f"queries.csv missing columns: {missing_q}. Found: {list(queries.columns)}")

# target_emotion is required for emotion_intent queries
if "target_emotion" not in queries.columns:
    raise ValueError("queries.csv must contain 'target_emotion' column for emotion_intent reranking.")

# ---------------- Load Corpus (Summaries) ----------------
summ = pd.read_csv(SUMMARIES_PATH)

required_scols = {"story_id", "aware_summary"}
missing_s = required_scols - set(summ.columns)
if missing_s:
    raise ValueError(f"summaries.csv missing columns: {missing_s}. Found: {list(summ.columns)}")

summ["aware_summary"] = summ["aware_summary"].fillna("").astype(str)
summ["story_id"] = summ["story_id"].astype(int)

docs_text = summ["aware_summary"].tolist()
doc_ids = summ["story_id"].tolist()

# ---------------- Build BM25 index ONCE ----------------
tokenized_docs = [tokenize(t) if tokenize(t) else ["_empty_"] for t in docs_text]
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

# ---------------- Build SBERT embeddings ONCE ----------------
sbert = SentenceTransformer(SBERT_MODEL)
doc_embs = sbert.encode(
    docs_text,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ---------------- Build Cross-Encoder ONCE (CE-Rerank baseline) ----------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- Build SETL-2 emotion vectors ONCE ----------------
# each doc vector is 8-dim in fixed order [joy, sadness, anger, fear, trust, disgust, surprise, anticipation]
doc_emo_vecs = np.vstack([setl2_vector_np(t) for t in docs_text]).astype(float)

# ---------------- Run containers ----------------
all_runs = {"bm25": [], "sbert": [], "rrf": [], "ce": [], "ours": []}

for cfg_name in WEIGHT_CONFIGS.keys():
    all_runs[cfg_name] = []

for _, q in queries.iterrows():
    qid = q["query_id"]
    qtext = str(q["query_text"])
    qtype = str(q["query_type"]).strip().lower()
    target_emotion = str(q["target_emotion"]).strip().lower()

    # ----- Stage 1: BM25 full corpus -> topK -----
    bm25_scores = np.asarray(bm25.get_scores(tokenize(qtext)))
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:K]

    # ----- SBERT baseline full corpus -> topK -----
    q_emb = sbert.encode(qtext, convert_to_tensor=True, normalize_embeddings=True)
    sbert_scores = util.cos_sim(q_emb, doc_embs)[0].cpu().numpy()
    sbert_top_idx = np.argsort(sbert_scores)[::-1][:K]

    # ----- RRF baseline (BM25 + SBERT) -----
    rrf_rank = rrf(bm25_top_idx, sbert_top_idx, k=RRF_K)

    # ----- CE-Rerank baseline: rerank BM25 topK using CrossEncoder -----
    cand_texts = [docs_text[i] for i in bm25_top_idx]
    ce_pairs = [(qtext, t) for t in cand_texts]
    ce_scores = cross_encoder.predict(ce_pairs)  # higher = better
    ce_order = np.argsort(ce_scores)[::-1]
    ce_ranked_idx = bm25_top_idx[ce_order]

    # ----- OURS: rerank BM25 topK using (BM25_norm + SBERT_sim + SETL2_emotion_sim) -----
    # Adaptive weights (rule-based expert system, not tuned on test)
    if qtype == "emotion_intent":
        w_bm25, w_sem, w_emo = 0.30, 0.40, 0.30
    else:
        w_bm25, w_sem, w_emo = 0.50, 0.45, 0.05

    # Query emotion vector:
    # We incorporate the "emotion intent" using the target emotion label.
    # Simple + interpretable: represent query emotion as a one-hot over SETL emotions if it matches,
    # otherwise fallback to SETL from query text.
    setl_emotions = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"]
    if qtype == "emotion_intent" and target_emotion in setl_emotions:
        q_emo_vec = np.zeros(len(setl_emotions), dtype=float)
        q_emo_vec[setl_emotions.index(target_emotion)] = 1.0
    else:
        q_emo_vec = setl2_vector_np(qtext)

    # compute rerank score for each candidate
    bm25_cand = bm25_scores[bm25_top_idx]
    bm25_norm = minmax_norm(bm25_cand)

    rerank_scores = {}
    for pos, idx in enumerate(bm25_top_idx):
        sem = float(sbert_scores[idx])
        emo = cosine_sim(doc_emo_vecs[idx], q_emo_vec)
        rerank_scores[idx] = (w_bm25 * float(bm25_norm[pos])) + (w_sem * sem) + (w_emo * emo)

    reranked = sorted(rerank_scores.items(), key=lambda x: x[1], reverse=True)

    # ----- M3 Weight Sensitivity (emotion_intent only): produce extra run files -----
    if qtype == "emotion_intent":
        for cfg_name, (wb, ws, we) in WEIGHT_CONFIGS.items():
            cfg_scores = {}
            for pos, idx in enumerate(bm25_top_idx):
                sem = float(sbert_scores[idx])
                emo = cosine_sim(doc_emo_vecs[idx], q_emo_vec)
                cfg_scores[idx] = (wb * float(bm25_norm[pos])) + (ws * sem) + (we * emo)

            cfg_ranked = sorted(cfg_scores.items(), key=lambda x: x[1], reverse=True)

            for rank, (idx, _) in enumerate(cfg_ranked[:K], start=1):
                all_runs[cfg_name].append([qid, int(doc_ids[idx]), rank])

    # ----- Save runs in TREC-like format: query_id, story_id, rank -----
    for rank, idx in enumerate(bm25_top_idx, start=1):
        all_runs["bm25"].append([qid, int(doc_ids[idx]), rank])

    for rank, idx in enumerate(sbert_top_idx, start=1):
        all_runs["sbert"].append([qid, int(doc_ids[idx]), rank])

    for rank, (idx, _) in enumerate(rrf_rank[:K], start=1):
        all_runs["rrf"].append([qid, int(doc_ids[idx]), rank])

    for rank, idx in enumerate(ce_ranked_idx[:K], start=1):
        all_runs["ce"].append([qid, int(doc_ids[idx]), rank])

    for rank, (idx, _) in enumerate(reranked[:K], start=1):
        all_runs["ours"].append([qid, int(doc_ids[idx]), rank])

# ---------------- Write Outputs ----------------
for name, rows in all_runs.items():
    out_path = os.path.join(RESULTS_DIR, f"{name}_run.csv")
    pd.DataFrame(rows, columns=["query_id", "story_id", "rank"]).to_csv(out_path, index=False)

print("✅ Saved run files to evaluation/results/:")
for name in all_runs.keys():
    print(f" - {name}_run.csv")
