"""Embed chunks with a small sentence-transformer model and store in ChromaDB
for dense similarity search."""
import json

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.jsonl"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supercap_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_chunks():
    with open(CHUNKS_PATH) as f:
        return [json.loads(line) for line in f]


def build_index():
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {"arxiv_id": c["arxiv_id"], "title": c["title"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    print(f"Embedding {len(chunks)} chunks with {MODEL_NAME}...")
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE].tolist(),
            documents=texts[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )

    print(f"Indexed {len(chunks)} chunks -> {CHROMA_DIR}/")
    return model, collection


def query(model, collection, query_text, top_k=5):
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return list(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))


if __name__ == "__main__":
    model, collection = build_index()

    # Paraphrased versions of the same 5 concepts tested in BM25 —
    # dense search should handle these even without exact term overlap
    test_queries = [
        "how well does the material store charge",
        "long-term durability after repeated charge-discharge",
        "how much energy is stored per unit weight",
        "carbon-based electrode materials",
        "how ions move through the electrolyte",
    ]

    for q in test_queries:
        print(f"\n=== Query: {q!r} ===")
        results = query(model, collection, q, top_k=3)
        for chunk_id, text, meta, distance in results:
            print(f"[dist={distance:.3f}] {chunk_id} - {meta['title'][:60]}")
            print(f"    {text[:150]}")
