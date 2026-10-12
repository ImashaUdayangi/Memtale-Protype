# src/retrieval.py
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# SBERT model (good default)
_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# B1: Keyword retrieval (BM25)
# -------------------------
def _tokenize(text: str):
    # simple word tokenizer (fast + stable)
    return re.findall(r"\b\w+\b", (text or "").lower())

def bm25_retrieval(query: str, docs: list[str], top_k: int = 5):
    tokenized_docs = [_tokenize(d) for d in docs]
    tokenized_docs = [toks if toks else ["_empty_"] for toks in tokenized_docs] 
    bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

    scores = bm25.get_scores(_tokenize(query))  # numpy array
    ranked = np.argsort(scores)[::-1][:top_k]
    return ranked.tolist(), scores[ranked].tolist()

#16th Jan Add functions to build BM25 index and get scores for all docs
def build_bm25_index(docs: list[str], k1: float = 1.5, b: float = 0.75) -> BM25Okapi:
    tokenized_docs = [_tokenize(d) for d in docs]
    tokenized_docs = [toks if toks else ["_empty_"] for toks in tokenized_docs]
    return BM25Okapi(tokenized_docs, k1=k1, b=b)

def bm25_rank(bm25: BM25Okapi, query: str) -> np.ndarray:
    """Return BM25 scores for all docs."""
    return np.asarray(bm25.get_scores(_tokenize(query)))
#----

# -------------------------
# B2/B3: Semantic retrieval (SBERT)
# -------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    return _sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def semantic_retrieval(query: str, docs: list[str], top_k: int = 5):
    doc_embs = embed_texts(docs)
    q_emb = embed_texts([query])[0]
    sims = doc_embs @ q_emb  # cosine sim because normalized
    ranked = np.argsort(sims)[::-1][:top_k]
    return ranked.tolist(), sims[ranked].tolist()

# -------------------------
# Ours: Semantic + Emotion hybrid retrieval
# -------------------------
def emotion_aware_retrieval(
    query: str,
    docs: list[str],
    doc_emotions: list[dict],
    query_emotion: dict,
    alpha: float = 0.7,
    top_k: int = 5
):
    """
    Score = alpha * semantic_sim + (1-alpha) * emotion_sim
    emotion_sim computed as cosine similarity over emotion vectors.
    """
    # semantic sims
    doc_embs = embed_texts(docs)
    q_emb = embed_texts([query])[0]
    sem_sims = doc_embs @ q_emb

    # emotion sims
    keys = sorted(query_emotion.keys())
    q_e = np.array([query_emotion[k] for k in keys], dtype=float)
    q_norm = np.linalg.norm(q_e)

    # FALLBACK: if query has no meaningful emotion, use semantic only
    if q_norm < 1e-6:
        ranked = np.argsort(sem_sims)[::-1][:top_k]
        return ranked.tolist(), sem_sims[ranked].tolist(), sem_sims[ranked].tolist(), None

    # normalize query emotion
    q_e = q_e / q_norm


    emo_sims = []
    for e in doc_emotions:
        d_e = np.array([e[k] for k in keys], dtype=float)
        d_e = d_e / (np.linalg.norm(d_e) + 1e-12)
        emo_sims.append(float(d_e @ q_e))

    emo_sims = np.array(emo_sims)

    final = alpha * sem_sims + (1 - alpha) * emo_sims
    ranked = np.argsort(final)[::-1][:top_k]
    return ranked.tolist(), final[ranked].tolist(), sem_sims[ranked].tolist(), emo_sims[ranked].tolist()

# -------------------------
# Helpers for label-based emotion vectors (32 ED labels)
# -------------------------
def build_label_space(labels: list[str]):
    labels = sorted(set(str(x).lower().strip() for x in labels if str(x).strip()))
    idx = {lab: i for i, lab in enumerate(labels)}
    return labels, idx

def onehot(label: str, idx: dict):
    v = np.zeros(len(idx), dtype=float)
    lab = str(label).lower().strip()
    if lab in idx:
        v[idx[lab]] = 1.0
    return v

# -------------------------
# Ours (NEW): Semantic + Label-Emotion hybrid
# -------------------------
def emotion_aware_retrieval_vec(
    query: str,
    docs: list[str],
    doc_emo_vecs: np.ndarray,
    q_emo_vec: np.ndarray | None,
    alpha: float = 0.7,
    top_k: int = 5
):
    doc_embs = embed_texts(docs)
    q_emb = embed_texts([query])[0]
    sem_sims = doc_embs @ q_emb

    if q_emo_vec is None or np.linalg.norm(q_emo_vec) < 1e-6:
        ranked = np.argsort(sem_sims)[::-1][:top_k]
        return ranked.tolist(), sem_sims[ranked].tolist(), sem_sims[ranked].tolist(), None

    q = q_emo_vec / (np.linalg.norm(q_emo_vec) + 1e-12)
    D = doc_emo_vecs / (np.linalg.norm(doc_emo_vecs, axis=1, keepdims=True) + 1e-12)

    emo_sims = D @ q
    final = alpha * sem_sims + (1 - alpha) * emo_sims
    ranked = np.argsort(final)[::-1][:top_k]

    return ranked.tolist(), final[ranked].tolist(), sem_sims[ranked].tolist(), emo_sims[ranked].tolist()

#Add baseline B4: BM25+SBERT Fusion (RRF)
def rrf_fusion(
    ranked_lists: list[list[int]],
    k: int = 60
) -> list[float]:
    """
    Reciprocal Rank Fusion (RRF).
    Returns fused scores aligned to doc indices [0..N-1].
    ranked_lists: each is a list of doc indices sorted best->worst
    """
    if not ranked_lists:
        return []

    # infer N from longest list
    N = max(len(r) for r in ranked_lists)
    scores = np.zeros(N, dtype=float)

    for r in ranked_lists:
        for rank, doc_idx in enumerate(r, start=1):
            if doc_idx < N:
                scores[doc_idx] += 1.0 / (k + rank)

    return scores.tolist()


def bm25_sbert_rrf_retrieval(query: str, docs: list[str], top_k: int = 5, rrf_k: int = 60):
    """
    B4 baseline: BM25 + SBERT fusion using RRF.
    """
    # get full rankings
    bm25_ranked, _ = bm25_retrieval(query, docs, top_k=len(docs))
    sem_ranked, _ = semantic_retrieval(query, docs, top_k=len(docs))

    fused_scores = rrf_fusion([bm25_ranked, sem_ranked], k=rrf_k)
    fused_scores = np.array(fused_scores)

    ranked = np.argsort(fused_scores)[::-1][:top_k]
    return ranked.tolist(), fused_scores[ranked].tolist()


#16th Jan Add a cached semantic retrieval function (precompute doc embeddings once)
def semantic_retrieval_cached(q_emb: np.ndarray, doc_embs: np.ndarray, top_k: int = 5):
    """
    Fast semantic retrieval when doc embeddings are precomputed.
    q_emb: (d,)
    doc_embs: (N,d) normalized
    """
    sims = doc_embs @ q_emb
    ranked = np.argsort(sims)[::-1][:top_k]
    return ranked.tolist(), sims[ranked].tolist()

#16th Jan Add a cached emotion-aware retrieval function (precompute doc embeddings once)
def emotion_aware_retrieval_cached(
    q_emb: np.ndarray,
    doc_embs: np.ndarray,
    doc_emotions: list[dict],
    query_emotion: dict,
    alpha: float = 0.7,
    top_k: int = 5
):
    """
    Cached version:
    Score = alpha * (doc_embs @ q_emb) + (1-alpha) * emotion_sim
    Assumes doc_embs and q_emb are already normalized.
    """
    sem_sims = doc_embs @ q_emb

    keys = sorted(query_emotion.keys())
    q_e = np.array([query_emotion[k] for k in keys], dtype=float)
    q_norm = np.linalg.norm(q_e)

    # if no emotion in query -> semantic only
    if q_norm < 1e-6:
        ranked = np.argsort(sem_sims)[::-1][:top_k]
        return ranked.tolist(), sem_sims[ranked].tolist(), sem_sims[ranked].tolist(), None

    q_e = q_e / q_norm

    emo_sims = np.empty(len(doc_emotions), dtype=float)
    for i, e in enumerate(doc_emotions):
        d_e = np.array([e[k] for k in keys], dtype=float)
        d_e = d_e / (np.linalg.norm(d_e) + 1e-12)
        emo_sims[i] = float(d_e @ q_e)

    final = alpha * sem_sims + (1 - alpha) * emo_sims
    ranked = np.argsort(final)[::-1][:top_k]
    return ranked.tolist(), final[ranked].tolist(), sem_sims[ranked].tolist(), emo_sims[ranked].tolist()

# -------------------------
#1st Feb Added two-stage BM25 + SETL-2 reranking
# -------------------------
def minmax_norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if (mx - mn) < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

def cosine_sim_vec(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float((a / na) @ (b / nb))

def two_stage_bm25_rerank_setl2(
    query_text: str,
    docs_text: List[str],
    docs_emotion_vecs: np.ndarray,     # shape: (N, d)
    query_emotion_vec: np.ndarray,     # shape: (d,)
    bm25_top_k: int = 100,
    final_top_k: int = 100,
    query_type: str = "emotion_intent",
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    bm25_index: BM25Okapi | None = None,
):
    """
    Stage 1: BM25 over ALL docs -> topK candidates
    Stage 2: rerank ONLY topK using:
        - normalized BM25 score
        - SBERT semantic similarity
        - SETL-2 emotion cosine similarity

    Adaptive weights:
        emotion_intent -> more emotion weight
        non_emotion_intent -> less emotion weight
    """

    # ---- Stage 1: BM25 ----
    if bm25_index is None:
        bm25_index = build_bm25_index(docs_text, k1=bm25_k1, b=bm25_b)

    bm25_scores_all = bm25_rank(bm25_index, query_text)  # (N,)
    cand_idx = np.argsort(bm25_scores_all)[::-1][:bm25_top_k]

    # ---- Stage 2 features on candidates ----
    # semantic sims (SBERT)
    # (we compute embeddings only for candidate docs for speed)
    cand_texts = [docs_text[i] for i in cand_idx]
    cand_embs = embed_texts(cand_texts)          # (K, d)
    q_emb = embed_texts([query_text])[0]         # (d,)
    sem_sims = cand_embs @ q_emb                 # (K,)

    # emotion sims (SETL-2)
    # cosine between query_emotion_vec and each doc emotion vec
    q_e = query_emotion_vec.astype(float)
    emo_sims = np.array([cosine_sim_vec(docs_emotion_vecs[i].astype(float), q_e) for i in cand_idx])

    # normalize bm25 within candidates
    bm25_cand = bm25_scores_all[cand_idx]
    bm25_norm = minmax_norm(bm25_cand)

    # ---- Adaptive weights (rule-based expert system) ----
    # (not tuned on test results; fixed rule = acceptable)
    if str(query_type).lower() == "emotion_intent":
        w_bm25, w_sem, w_emo = 0.30, 0.50, 0.20
    else:
        w_bm25, w_sem, w_emo = 0.50, 0.45, 0.05

    # fused rerank score
    fused = (w_bm25 * bm25_norm) + (w_sem * sem_sims) + (w_emo * emo_sims)

    rerank_order = np.argsort(fused)[::-1]
    reranked_idx = cand_idx[rerank_order][:final_top_k]

    # also return scores for analysis if needed
    return reranked_idx.tolist(), fused[rerank_order][:final_top_k].tolist()
