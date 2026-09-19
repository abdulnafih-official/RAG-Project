"""Take a user query, retrieve top chunks via hybrid search, and generate a
cited answer using the Gemini API."""
import os

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from build_bm25 import load_index as load_bm25, tokenize
from build_hybrid import bm25_ranking, dense_ranking, reciprocal_rank_fusion

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supercap_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-3.5-flash-lite"   

SYSTEM_INSTRUCTION = """You are a research assistant answering questions about
supercapacitor technology using only the provided excerpts from academic papers.

Rules:
- Answer only using the provided context. If the context doesn't contain the
  answer, say so explicitly rather than guessing.
- After each claim, cite the source using the format [chunk_id].
- Be concise and technical, matching the register of the source material.
"""


def retrieve(bm25, chunks, chunk_lookup, embed_model, collection, query_text, top_k=5):
    bm25_ranked = bm25_ranking(bm25, chunks, query_text)
    dense_ranked = dense_ranking(embed_model, collection, query_text)
    fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
    return [chunk_lookup[cid] for cid in fused[:top_k]]


def build_prompt(query_text, retrieved_chunks):
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(
            f"[{c['chunk_id']}] (from \"{c['title']}\")\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"


def answer(client, query_text, retrieved_chunks):
    prompt = build_prompt(query_text, retrieved_chunks)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"system_instruction": SYSTEM_INSTRUCTION},
    )
    return response.text


if __name__ == "__main__":
    bm25, chunks = load_bm25()
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    gemini_client = genai.Client()  # reads GEMINI_API_KEY from env

    test_queries = [
        "What materials show good cycling stability over 10000 cycles?",
        "How does electrolyte ionic conductivity affect supercapacitor performance?",
        "What energy density values are reported for graphene-based supercapacitors?",
    ]

    for q in test_queries:
        print(f"\n=== Query: {q!r} ===")
        retrieved = retrieve(bm25, chunks, chunk_lookup, embed_model, collection, q, top_k=5)
        print("Retrieved chunks:", [c["chunk_id"] for c in retrieved])
        result = answer(gemini_client, q, retrieved)
        print(f"\nAnswer:\n{result}")
