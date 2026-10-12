# src/summarisation.py
from transformers import pipeline
from transformers.utils import logging as hf_logging

# Stop HuggingFace from spamming warnings every call
hf_logging.set_verbosity_error()

_summariser = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6", #15th Jan changed from facebook/bart-base to reduce model size
    tokenizer="sshleifer/distilbart-cnn-12-6",
)

# Force tokenizer max length for safe truncation behavior
# BART max input length (~1024 tokens). Truncate characters to be safe.
_summariser.tokenizer.model_max_length = 1024

def _truncate_chars(text: str, max_chars: int = 3500) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:max_chars]

def naive_summary(text: str, max_new_tokens: int = 80, min_length: int = 30) -> str:
    text = _truncate_chars(text, max_chars=3500)

    # Use max_new_tokens ONLY (no max_length) => removes warning
    out = _summariser(
        text,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        do_sample=False,
        truncation=True
    )
    return out[0]["summary_text"].strip()

def emotion_aware_summary(
    text: str,
    base_summary: str | None = None,
    max_new_tokens: int = 80,
    min_length: int = 30
) -> str:
    """
    Minimal 'emotion-aware' variant:
    - Use base summary (passed in) so we DON'T summarize twice
    - Append 1 emotional sentence if missing
    """
    text_clean = _truncate_chars(text, max_chars=3500)

    # If base summary already provided, don't call summarizer again
    base = base_summary if base_summary is not None else naive_summary(
        text_clean, max_new_tokens=max_new_tokens, min_length=min_length
    )

    # Simple heuristic: keep one sentence that contains emotion words
    emotion_cues = ["happy", "sad", "scared", "afraid", "proud", "pain", "joy", "angry", "excited", "cry", "love"]

    # Use the same truncated version here so sentence splitting doesn't scan huge text
    sentences = [s.strip() for s in text_clean.split(".") if s.strip()]

    chosen = ""
    for s in sentences:
        low = s.lower()
        if any(w in low for w in emotion_cues):
            chosen = s
            break

    # If base doesn't already contain that emotional sentence, append it.
    if chosen and chosen.lower() not in base.lower():
        return (base.rstrip(".") + ". " + chosen.rstrip(".") + ".").strip()

    return base
