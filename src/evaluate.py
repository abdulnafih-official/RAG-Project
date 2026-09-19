"""Compare dense-only vs hybrid retrieval@k against a hand-labeled query set.
Fill in eval/test_queries.json with real queries and the correct chunk_id(s)
for each before running this."""
import json

import chromadb
from sentence_transformers import SentenceTransformer

from build_bm25 import load_index as load_bm25
from build_hybrid import bm25_ranking, dense_ranking, reciprocal_rank_fusion

EVAL_PATH = "eval/test_queries.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supercap_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 3


def hit_at_k(ranked_ids, correct_ids, k):
    return int(any(cid in correct_ids for cid in ranked_ids[:k]))


def main():
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    bm25, chunks = load_bm25()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    dense_hits, hybrid_hits = 0, 0
    rows = []

    for entry in eval_set:
        q, correct_ids = entry["query"], set(entry["correct_chunk_ids"])

        # Fusion needs a deep candidate pool from both methods (top_k=20),
        # independent of the final TOP_K used for the hit-rate metric.
        dense_ranked_full = dense_ranking(embed_model, collection, q, top_k=20)
        bm25_ranked = bm25_ranking(bm25, chunks, q, top_k=20)
        fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked_full])

        d_hit = hit_at_k(dense_ranked_full, correct_ids, TOP_K)
        h_hit = hit_at_k(fused, correct_ids, TOP_K)

        dense_hits += d_hit
        hybrid_hits += h_hit
        rows.append((q, entry.get("type", "?"), d_hit, h_hit))

    print(f"{'Query':<55} {'Type':<10} {'Dense@'+str(TOP_K):<10} {'Hybrid@'+str(TOP_K)}")
    print("-" * 90)
    for q, qtype, d, h in rows:
        marker = "  <-- fixed by hybrid" if h and not d else ""
        print(f"{q[:53]:<55} {qtype:<10} {d:<10} {h}{marker}")

    n = len(eval_set)
    print(f"\nDense-only@{TOP_K}: {dense_hits}/{n} ({100*dense_hits/n:.0f}%)")
    print(f"Hybrid@{TOP_K}:     {hybrid_hits}/{n} ({100*hybrid_hits/n:.0f}%)")


if __name__ == "__main__":
    main()
