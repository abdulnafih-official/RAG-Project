"""Run the same exact-term queries used for BM25 through dense-only search,
to confirm dense embeddings underperform on exact terms (the gap hybrid
fusion exists to close)."""
from build_dense import query
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supercap_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

exact_term_queries = [
    "MnO2 electrode capacitance",
    "cycling stability 10000 cycles",
    "energy density Wh/kg",
    "activated carbon electrode",
    "electrolyte ionic conductivity",
]

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

for q in exact_term_queries:
    print(f"\n=== Query: {q!r} ===")
    results = query(model, collection, q, top_k=3)
    for chunk_id, text, meta, distance in results:
        print(f"[dist={distance:.3f}] {chunk_id} - {meta['title'][:60]}")
        print(f"    {text[:150]}")
