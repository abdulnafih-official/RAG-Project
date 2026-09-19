"""Combine BM25 (sparse) and dense embedding rankings using Reciprocal Rank
Fusion (RRF), so a chunk ranking well under either method gets pulled up."""
import chromadb
from sentence_transformers import SentenceTransformer

from build_bm25 import load_index as load_bm25, tokenize

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supercap_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60  # standard RRF constant


def bm25_ranking(bm25, chunks, query_text, top_k=20):
    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(
        range(len(chunks)), key=lambda i: scores[i], reverse=True
    )[:top_k]
    return [chunks[i]["chunk_id"] for i in ranked_indices]


def dense_ranking(model, collection, query_text, top_k=20):
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["ids"][0]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """rankings: list of ranked chunk_id lists (one per retrieval method).
    Returns chunk_ids sorted by fused RRF score, descending."""
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)


def hybrid_query(bm25, chunks, model, collection, chunk_lookup, query_text, top_k=5):
    bm25_ranked = bm25_ranking(bm25, chunks, query_text)
    dense_ranked = dense_ranking(model, collection, query_text)
    fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
    return [chunk_lookup[cid] for cid in fused[:top_k]]


if __name__ == "__main__":
    bm25, chunks = load_bm25()
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    test_queries = [
        "MnO2 electrode capacitance",
        "cycling stability 10000 cycles",
        "energy density Wh/kg",
        "activated carbon electrode",
        "electrolyte ionic conductivity",
        "how well does the material store charge",
        "long-term durability after repeated charge-discharge",
        "how much energy is stored per unit weight",
        "carbon-based electrode materials",
        "how ions move through the electrolyte",
    ]

    for q in test_queries:
        print(f"\n=== Query: {q!r} ===")
        results = hybrid_query(bm25, chunks, model, collection, chunk_lookup, q, top_k=3)
        for chunk in results:
            print(f"{chunk['chunk_id']} - {chunk['title'][:60]}")
            print(f"    {chunk['text'][:150]}")
