"""Build a BM25 index over chunks.jsonl and provide a simple query interface
to test exact-term retrieval quality."""
import json
import pickle
import re

from rank_bm25 import BM25Okapi

CHUNKS_PATH = "chunks.jsonl"
INDEX_PATH = "bm25_index.pkl"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_chunks():
    with open(CHUNKS_PATH) as f:
        return [json.loads(line) for line in f]


def build_index():
    chunks = load_chunks()
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Indexed {len(chunks)} chunks -> {INDEX_PATH}")
    return bm25, chunks


def load_index():
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def query(bm25, chunks, query_text, top_k=5):
    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(
        range(len(chunks)), key=lambda i: scores[i], reverse=True
    )[:top_k]
    return [(chunks[i], scores[i]) for i in ranked]


if __name__ == "__main__":
    bm25, chunks = build_index()

    test_queries = [
        "MnO2 electrode capacitance",
        "cycling stability 10000 cycles",
        "energy density Wh/kg",
        "activated carbon electrode",
        "electrolyte ionic conductivity",
    ]

    for q in test_queries:
        print(f"\n=== Query: {q!r} ===")
        results = query(bm25, chunks, q, top_k=3)
        for chunk, score in results:
            print(f"[{score:.2f}] {chunk['chunk_id']} - {chunk['title'][:60]}")
            print(f"    {chunk['text'][:150]}")
