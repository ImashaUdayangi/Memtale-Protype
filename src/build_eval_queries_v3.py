# src/build_eval_queries_v3.py
from pathlib import Path
import pandas as pd
import random

OUT_DIR = Path("dataset/eval_v3")
OUT_Q = OUT_DIR / "queries.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Your 9 emotions
EMOTIONS = [
    "angry", "joyful", "impressed", "proud", "confident",
    "surprised", "anticipating", "guilty", "lonely"
]

# We generate 4 direct + 4 paraphrase per emotion
N_DIRECT = 4
N_PARA = 4

# Direct templates: some explicit (contains the emotion word), some implicit (no exact label)
DIRECT_TEMPLATES = {
    "angry": [
        "a memory where I felt angry",
        "a time I was furious after being treated unfairly",
        "a time I felt disrespected and wanted to push back",
        "a moment when I felt wronged and couldn't let it go",
        "a time someone crossed a line and I couldn't stay calm",
    ],
    "joyful": [
        "a memory where I felt joyful",
        "a time I felt genuinely happy and uplifted",
        "a moment when everything felt bright and positive",
        "a time I couldn’t stop smiling because it felt so good",
        "a moment that left me feeling light and thankful",
    ],
    "impressed": [
        "a memory where I felt impressed",
        "a time I admired someone’s skills or achievement",
        "a moment when someone exceeded my expectations",
        "a time I saw something surprisingly excellent",
        "a moment that made me think “wow, that was amazing”",
    ],
    "proud": [
        "a memory where I felt proud",
        "a time I accomplished something meaningful",
        "a moment when my effort finally paid off",
        "a time I proved to myself I could do it",
        "a moment I felt I earned my success",
    ],
    "confident": [
        "a memory where I felt confident",
        "a time I felt capable and in control",
        "a situation where I felt sure of myself",
        "a time I handled something difficult without doubt",
        "a moment I knew I could succeed",
    ],
    "surprised": [
        "a memory where I felt surprised",
        "a time something unexpected happened",
        "a moment when news changed everything suddenly",
        "a time I didn’t see it coming at all",
        "a moment that shocked me in a good or bad way",
    ],
    "anticipating": [
        "a memory where I was anticipating something",
        "a time I was waiting for something important",
        "a moment before something big happened",
        "a time I couldn’t relax waiting for what was next",
        "a moment where I felt on edge, waiting",
    ],
    "guilty": [
        "a memory where I felt guilty",
        "a time I regretted something I did",
        "a time I kept replaying a mistake in my head",
        "a moment I felt responsible for harm",
        "a time I wished I could undo what happened",
    ],
    "lonely": [
        "a memory where I felt lonely",
        "a time I felt alone even around other people",
        "a time it felt like I was completely on my own",
        "a moment when no one understood me",
        "a time I felt unsupported and isolated",
    ],
}

# Paraphrases should avoid repeating the exact direct templates
PARAPHRASE_TEMPLATES = {
    "angry": [
        "a time it felt unfair and I wanted to react",
        "a moment I felt provoked and ready to argue",
        "a time I couldn’t accept how I was treated",
        "a moment I felt my patience snap",
        "a situation where I felt wronged by someone",
    ],
    "joyful": [
        "a moment when everything felt uplifting",
        "a time I felt warm and positive inside",
        "a memory that left me feeling cheerful for hours",
        "a moment that filled me with happiness",
        "a time I felt delighted by something simple",
    ],
    "impressed": [
        "a moment when someone did something surprisingly well",
        "a time I couldn’t believe their skill",
        "a moment I saw an achievement that stood out",
        "a time I felt genuine admiration",
        "a moment that raised my expectations",
    ],
    "proud": [
        "a time I felt valued and accomplished",
        "a moment I felt my hard work mattered",
        "a time I felt I became a better version of myself",
        "a moment I felt proud of what I achieved",
        "a time I felt I truly earned it",
    ],
    "confident": [
        "a time I felt ready and capable",
        "a moment I trusted myself completely",
        "a time I felt I could handle anything",
        "a situation where I felt steady and sure",
        "a moment I acted without hesitation",
    ],
    "surprised": [
        "a time I didn’t see it coming at all",
        "a moment when unexpected news changed things",
        "a time something happened out of nowhere",
        "a moment that completely caught me off guard",
        "a time I was shocked in an unexpected way",
    ],
    "anticipating": [
        "a moment before something important happened",
        "a time I felt restless waiting",
        "a moment when I couldn’t stop thinking about what was next",
        "a time I felt tense during a buildup",
        "a moment where the waiting felt intense",
    ],
    "guilty": [
        "a time I couldn’t forgive myself",
        "a moment I felt I let someone down",
        "a time I felt ashamed of my actions",
        "a moment regret kept bothering me",
        "a time I wished I could fix what I did",
    ],
    "lonely": [
        "a moment when no one understood me",
        "a time I felt disconnected from everyone",
        "a time I felt emotionally isolated",
        "a moment when I felt left out",
        "a time I felt like I didn’t belong",
    ],
}

def pick_unique(pool, n, used):
    """Pick up to n unique strings from pool, not in used."""
    choices = [x for x in pool if x not in used]
    random.shuffle(choices)
    picked = choices[:n]
    used.update(picked)
    return picked

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = []
    qid = 1

    for emo in EMOTIONS:
        used = set()

        direct = pick_unique(DIRECT_TEMPLATES[emo], N_DIRECT, used)
        para = pick_unique(PARAPHRASE_TEMPLATES[emo], N_PARA, used)

        # If still short (shouldn't happen unless pools are too small), fill by reshuffling without duplicates
        while len(direct) < N_DIRECT:
            for x in DIRECT_TEMPLATES[emo]:
                if x not in used:
                    direct.append(x); used.add(x)
                if len(direct) == N_DIRECT:
                    break

        while len(para) < N_PARA:
            for x in PARAPHRASE_TEMPLATES[emo]:
                if x not in used:
                    para.append(x); used.add(x)
                if len(para) == N_PARA:
                    break

        for t in direct:
            queries.append({
                "query_id": f"q{qid}",
                "query_type": "emotion_intent",
                "query_subtype": "direct",
                "target_emotion": emo,
                "query_text": t
            })
            qid += 1

        for t in para:
            queries.append({
                "query_id": f"q{qid}",
                "query_type": "emotion_intent",
                "query_subtype": "paraphrase",
                "target_emotion": emo,
                "query_text": t
            })
            qid += 1

    out = pd.DataFrame(queries)

    # Safety: assert uniqueness
    dup = out.duplicated(subset=["query_text"]).sum()
    if dup > 0:
        raise ValueError(f"Duplicate query_text still exists: {dup}")

    out.to_csv(OUT_Q, index=False)
    print("✅ Saved:", OUT_Q)
    print("Total queries:", len(out))
    print("Unique query_text:", out["query_text"].nunique())

if __name__ == "__main__":
    main()
