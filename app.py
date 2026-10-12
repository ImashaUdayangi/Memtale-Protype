# app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st

# -----------------------------
# Paths (relative to project root)
# -----------------------------
ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "evaluation" / "results"
EVAL_V3_DIR = ROOT / "dataset" / "eval_v3"
RAW_DIR = ROOT / "dataset" / "raw"
CACHE_DIR = ROOT / "dataset" / "cache"

EVAL_SUMMARY = RESULTS_DIR / "evaluation_summary.csv"
EMO_SUMMARY = RESULTS_DIR / "emotional_fidelity_summary.csv"
MAIN_FIG = RESULTS_DIR / "main_results_bar_chart.png"
QUAL_EXAMPLE = RESULTS_DIR / "qualitative_rerank_example.csv"

QUERIES = EVAL_V3_DIR / "queries.csv"
QRELS = EVAL_V3_DIR / "qrels.csv"
STORIES = RAW_DIR / "empathetic_stories.csv"
SUMMARIES = CACHE_DIR / "summaries.csv"

# Run files (all baselines + final OURS choice)
RUN_FILES = {
    "BM25": RESULTS_DIR / "bm25_run.csv",
    "SBERT": RESULTS_DIR / "sbert_run.csv",
    "RRF": RESULTS_DIR / "rrf_run.csv",
    "CE-Rerank": RESULTS_DIR / "ce_run.csv",
    "OURS (wemo=0.20)": RESULTS_DIR / "wemo_0.20_run.csv",  # final chosen OURS
}

# -----------------------------
# Helper loaders (cached)
# -----------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["query_id"] = df["query_id"].astype(str)
    df["story_id"] = df["story_id"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df.sort_values(["query_id", "rank"])

def safe_exists(path: Path) -> bool:
    return path.exists()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Memtale-Co-Archivist Demo", layout="wide")

st.title("Memtale-Co-Archivist — Demo UI")
st.caption("Interactive viewer for datasets, retrieval runs, evaluation metrics, and qualitative examples (Streamlit).")

# -----------------------------
# Sidebar: run status + datasets + run files
# -----------------------------
st.sidebar.header("Run status")

required_files = [
    EVAL_SUMMARY, EMO_SUMMARY, MAIN_FIG,
    QUERIES, QRELS, STORIES, SUMMARIES,
    RUN_FILES["BM25"], RUN_FILES["SBERT"], RUN_FILES["RRF"], RUN_FILES["OURS (wemo=0.20)"]
]
missing = [str(p) for p in required_files if not safe_exists(p)]

if missing:
    st.sidebar.error("Some required files are missing ❌")
    for m in missing:
        st.sidebar.write("•", m)
    st.sidebar.info("Fix: run `python main.py` first to generate results and run files.")
    st.stop()
else:
    st.sidebar.success("All required files found ✅")

st.sidebar.subheader("Datasets")

DATA_FILES = {
    "empathetic_stories.csv": STORIES,
    "summaries.csv": SUMMARIES,
    "queries.csv": QUERIES,
    "qrels.csv": QRELS,
}

if "selected_file_label" not in st.session_state:
    st.session_state["selected_file_label"] = "empathetic_stories.csv"
    st.session_state["selected_file_path"] = STORIES

for label, path in DATA_FILES.items():
    if st.sidebar.button(label, key=f"open_{label}", width="stretch"):
        st.session_state["selected_file_label"] = label
        st.session_state["selected_file_path"] = path

st.sidebar.subheader("Run files")
for name, p in RUN_FILES.items():
    tag = "✅" if safe_exists(p) else "—"
    st.sidebar.write(f"{tag} {name}: `{p.name}`")

st.sidebar.subheader("Final configuration")
st.sidebar.write("Final OURS run:", "`wemo_0.20_run.csv`")
st.sidebar.write("Cross-encoder run:", "`ce_run.csv`")
st.sidebar.write("Baselines:", "`bm25_run.csv`, `sbert_run.csv`, `rrf_run.csv`")

st.divider()

# -----------------------------
# Section 0: Dataset preview
# -----------------------------
st.subheader("0) Dataset Preview")

st.markdown(f"### Selected file: {st.session_state['selected_file_label']}")
df_sel = load_csv(st.session_state["selected_file_path"])
st.caption(f"Rows: {len(df_sel):,} | Columns: {list(df_sel.columns)}")
st.dataframe(df_sel.head(25), width="stretch", hide_index=True)
st.divider()

colA, colB = st.columns(2)

with colA:
    st.markdown("### empathetic_stories.csv")
    stories_df = load_csv(STORIES)
    st.caption(f"Rows: {len(stories_df):,} | Columns: {list(stories_df.columns)}")
    st.dataframe(stories_df.head(10), width="stretch", hide_index=True)

with colB:
    st.markdown("### summaries.csv")
    summ_df = load_csv(SUMMARIES)
    st.caption(f"Rows: {len(summ_df):,} | Columns: {list(summ_df.columns)}")
    st.dataframe(summ_df.head(10), width="stretch", hide_index=True)

st.divider()

# -----------------------------
# Section 1: Main results
# -----------------------------
st.subheader("1) Main Results Summary")

col1, col2 = st.columns(2)

with col1:
    eval_df = load_csv(EVAL_SUMMARY)
    st.markdown("### Evaluation (nDCG@5, MRR)")
    st.dataframe(eval_df, width="stretch", hide_index=True)

with col2:
    emo_df = load_csv(EMO_SUMMARY)
    st.markdown("### Emotional Fidelity Summary")
    st.dataframe(emo_df, width="stretch", hide_index=True)

st.markdown("### Main Results Figure")
if safe_exists(MAIN_FIG):
    st.image(str(MAIN_FIG), width="stretch")
else:
    st.warning("main_results_bar_chart.png not found. Run: `python -m evaluation.make_main_results_figure`")

# Short interpretation helper (safe + accurate)
st.info(
    "Note: In this current results, RRF has the highest nDCG@5, OURS(wemo=0.20) has the highest MRR, "
    "and CE-Rerank has the highest EmoFidelity@5."
)

st.divider()

# -----------------------------
# Section 2: Query explorer (BM25 vs SBERT vs RRF vs OURS vs CE)
# -----------------------------
st.subheader("2) Query Explorer (Compare Retrieval Methods)")

queries_df = load_csv(QUERIES)
queries_df["query_id"] = queries_df["query_id"].astype(str)

# story maps
stories_df = stories_df.copy()
stories_df["story_id"] = stories_df["story_id"].astype(int)
stories_df["emotion_label"] = stories_df["emotion_label"].fillna("").astype(str)
stories_df["Story"] = stories_df["Story"].fillna("").astype(str)
story_map = stories_df.set_index("story_id")[["emotion_label", "Story"]].to_dict(orient="index")

# Load runs for available systems
available_systems = [name for name, p in RUN_FILES.items() if safe_exists(p)]
runs = {name: load_run(RUN_FILES[name]) for name in available_systems}

# Pick query
query_ids = queries_df["query_id"].tolist()
default_qid = "q61" if "q61" in query_ids else query_ids[0]
qid = st.selectbox("Select a query_id", query_ids, index=query_ids.index(default_qid))

qrow = queries_df[queries_df["query_id"] == qid].iloc[0]
st.markdown(
    f"**Query text:** {qrow['query_text']}  \n"
    f"**Target emotion:** {qrow['target_emotion']}  \n"
    f"**Query type:** {qrow['query_type']}"
)

topk = st.slider("Top-K to display", min_value=3, max_value=20, value=5, step=1)

default_compare = ["BM25", "RRF", "CE-Rerank", "OURS (wemo=0.20)"]
default_compare = [x for x in default_compare if x in available_systems]

selected_systems = st.multiselect(
    "Select systems to compare",
    options=available_systems,
    default=default_compare
)

def build_topk_table(run_df: pd.DataFrame, system_name: str) -> pd.DataFrame:
    rows = run_df[run_df["query_id"] == qid].sort_values("rank").head(topk)
    out = []
    for _, r in rows.iterrows():
        sid = int(r["story_id"])
        meta = story_map.get(sid, {"emotion_label": "", "Story": ""})
        out.append({
            "system": system_name,
            "rank": int(r["rank"]),
            "story_id": sid,
            "story_emotion": meta["emotion_label"],
            "excerpt": (meta["Story"][:220] + "...") if len(meta["Story"]) > 220 else meta["Story"]
        })
    return pd.DataFrame(out)

if not selected_systems:
    st.warning("Select at least one system to compare.")
else:
    cols = st.columns(len(selected_systems))
    for i, sysname in enumerate(selected_systems):
        with cols[i]:
            st.markdown(f"### {sysname}")
            table = build_topk_table(runs[sysname], sysname)
            st.dataframe(table, width="stretch", hide_index=True)

st.divider()

# -----------------------------
# Section 3: Qualitative example
# -----------------------------
st.subheader("3) Qualitative Rerank Example (Auto-picked)")

if safe_exists(QUAL_EXAMPLE):
    qual_df = load_csv(QUAL_EXAMPLE)
    st.dataframe(qual_df, width="stretch", hide_index=True)
else:
    st.warning("qualitative_rerank_example.csv not found. Run: `python -m evaluation.make_qualitative_rerank_example`")