"""Advanced Retrieval Module: Hybrid Search, Reranking, and Context Compression."""

from __future__ import annotations

import numpy as np
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

def create_vector_db(chunks: list[str], metadata: list[dict], persist_directory: str = "/tmp/chroma_db"):
    """Create a Chroma vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata,
        persist_directory=persist_directory
    )

def create_bm25(chunks: list[str]):
    """Build a BM25 index over whitespace-tokenised chunks."""
    from rank_bm25 import BM25Okapi
    tokenized = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized)

def hybrid_search(
    query: str,
    vector_db,
    bm25,
    chunks: list[str],
    metadata_list: list[dict] = None,
    k: int = 10,
    alpha: float = 0.5
) -> list[dict]:
    """Perform hybrid search using Reciprocal Rank Fusion (RRF)."""
    
    results_map = {}
    
    # 1. Dense Search
    if vector_db:
        dense_results = vector_db.similarity_search_with_relevance_scores(query, k=k)
        for rank, (doc, score) in enumerate(dense_results):
            text = doc.page_content
            if text not in results_map:
                results_map[text] = {"text": text, "metadata": doc.metadata, "dense_rank": rank + 1, "sparse_rank": 60}
            else:
                results_map[text]["dense_rank"] = rank + 1

    # 2. Sparse Search
    if bm25:
        bm25_scores = bm25.get_scores(query.split())
        top_idx = np.argsort(bm25_scores)[-k:][::-1]
        for rank, idx in enumerate(top_idx):
            score = float(bm25_scores[idx])
            if score > 0:
                text = chunks[idx]
                if text not in results_map:
                    results_map[text] = {
                        "text": text, 
                        "metadata": metadata_list[idx] if metadata_list else {}, 
                        "dense_rank": 60, 
                        "sparse_rank": rank + 1
                    }
                else:
                    results_map[text]["sparse_rank"] = rank + 1

    # 3. Reciprocal Rank Fusion
    k_rrf = 60
    final_results = []
    for text, data in results_map.items():
        rrf_score = (alpha / (k_rrf + data["dense_rank"])) + ((1 - alpha) / (k_rrf + data["sparse_rank"]))
        final_results.append({
            "text": data["text"],
            "metadata": data["metadata"],
            "score": rrf_score
        })
        
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:k]

def filter_results_by_documents(results: list[dict], relevant_docs: list[str]) -> list[dict]:
    """Keep only results whose source is in relevant_docs."""
    if not relevant_docs:
        return results
    return [item for item in results if item.get("metadata", {}).get("source", "") in relevant_docs]

def rerank_results(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Apply a Cross-Encoder to re-score and compress context."""
    if not results:
        return []
        
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    pairs = [[query, r["text"]] for r in results]
    scores = model.predict(pairs)
    
    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])
        
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]

def format_sources(retrieved_chunks: list[dict]) -> str:
    """Render a de-duplicated bullet list of source (Page n) citations."""
    sources: list[str] = []
    seen: set[str] = set()
    for item in retrieved_chunks:
        metadata = item.get("metadata", {})
        source = metadata.get("source", "Unknown Document")
        page = metadata.get("page", "?")
        key = f"{source}-{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"- {source} (Page {page})")
    return "\n".join(sources)
