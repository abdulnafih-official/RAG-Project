"""Phase 8 — Streamlit UI. Run from project root: streamlit run src/app.py"""
import re
import time

import chromadb
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer

from build_bm25 import load_index as load_bm25
from query import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME, retrieve
from query import answer as generate_answer

st.set_page_config(page_title="Supercapacitor Research Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading indexes and models...")
def load_resources():
    bm25, chunks = load_bm25()
    lookup = {c["chunk_id"]: c for c in chunks}
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
    return bm25, chunks, lookup, embed_model, collection, genai.Client()


st.title("Supercapacitor Research Assistant")
st.caption("Hybrid retrieval (BM25 + dense, fused with RRF) over arXiv supercapacitor papers")

# Loaded once per server process, so query timing below excludes model loading.
bm25, chunks, lookup, embed_model, collection, gemini = load_resources()

with st.sidebar:
    k = st.slider("Chunks to retrieve (k)", min_value=3, max_value=10, value=5)

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What energy density values are reported for graphene-based supercapacitors?",
)

if question:
    start = time.perf_counter()
    with st.spinner("Retrieving and generating..."):
        retrieved = retrieve(bm25, chunks, lookup, embed_model, collection, question, top_k=k)
        answer_text = generate_answer(gemini, question, retrieved)
    elapsed = time.perf_counter() - start

    # Citations look like [chunk_id] or [id1, id2]; check each id against what was retrieved.
    retrieved_ids = {c["chunk_id"] for c in retrieved}
    tokens = [
        t.strip()
        for group in re.findall(r"\[([^\[\]]+)\]", answer_text)
        for t in re.split(r"[,;]", group)
    ]
    cited = {t for t in tokens if t in retrieved_ids}
    unknown = sorted({t for t in tokens if t not in retrieved_ids})

    st.subheader("Answer")
    st.markdown(answer_text)

    st.subheader("Sources cited")
    if not cited:
        st.caption("No citations in this answer.")
    for c in retrieved:
        if c["chunk_id"] in cited:
            section = f" — {c['section']}" if c.get("section") else ""
            st.markdown(f"**[{c['chunk_id']}]** {c['title']}{section}")
    if unknown:
        st.warning(f"Bracketed reference(s) not matching any retrieved chunk: {unknown}")

    st.subheader("Retrieved chunks")
    for c in retrieved:
        is_cited = c["chunk_id"] in cited
        label = f"[{c['chunk_id']}] {c['title']} ({'cited' if is_cited else 'not cited'})"
        with st.expander(label, expanded=is_cited):
            st.write(c["text"])

    (st.success if elapsed < 10 else st.warning)(
        f"Completed in {elapsed:.1f}s (checkpoint: under ~10s)"
    )