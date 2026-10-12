# src/setl2.py
"""
SETL-2: Semantic–Emotional Tagging Layer (Interpretable)
Combines lexicon-based emotion signals with sentiment polarity.
"""
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex
from collections import Counter

# Initialize VADER once
_vader = SentimentIntensityAnalyzer()

# Fixed emotion set (small & interpretable)
EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "disgust",
    "surprise",
    "anticipation"
]

#1st Feb Added two-stage BM25 + SETL-2 reranking
def setl2_vector_np(text: str) -> np.ndarray:
    """
    Returns SETL-2 vector as numpy array in fixed EMOTIONS order.
    """
    v = setl2_emotion_vector(text)
    return np.array([v[e] for e in EMOTIONS], dtype=float)

def vader_compound(text: str) -> float:
    """Return VADER compound sentiment score."""
    return _vader.polarity_scores(text)["compound"]

def nrc_distribution(text: str) -> dict:
    """Return normalized NRC emotion distribution."""
    nrc = NRCLex(text)
    counts = Counter(nrc.raw_emotion_scores)

    total = sum(counts[e] for e in EMOTIONS) or 1
    return {e: counts[e] / total for e in EMOTIONS}

def setl2_emotion_vector(text: str) -> dict:
    """
    Hybrid SETL-2 emotion vector:
    - Base NRC distribution
    - Adjusted using VADER polarity for emotional emphasis
    """
    base = nrc_distribution(text)
    polarity = vader_compound(text)

    adjusted = {}
    for emotion, value in base.items():
        if polarity > 0.3 and emotion in ("joy", "trust", "anticipation"):
            adjusted[emotion] = value * (1 + polarity)
        elif polarity < -0.3 and emotion in ("sadness", "anger", "fear", "disgust"):
            adjusted[emotion] = value * (1 + abs(polarity))
        else:
            adjusted[emotion] = value

    # Re-normalize
    total = sum(adjusted.values()) or 1
    return {e: adjusted[e] / total for e in EMOTIONS}