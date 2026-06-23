"""Hybrid retrieval using ChromaDB (Dense) and BM25 (Sparse)."""

from __future__ import annotations
import os
import numpy as np
from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

Result = dict[str, float | dict | str]

def init_vector_db():
    """Initialize ChromaDB with local sentence transformers."""
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import chromadb
    
    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Use ephemeral client for simple deployment, or persistent if needed
    chroma_client = chromadb.EphemeralClient()
    vector_db = Chroma(
        client=chroma_client,
        collection_name="docmind_collection",
        embedding_function=embeddings
    )
    return vector_db


def create_bm25(chunks: list[str]):
    """Build a BM25 index over whitespace-tokenised ``chunks``."""
    from rank_bm25 import BM25Okapi
    tokenized = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized)


def hybrid_search(
    query: str,
    vector_db,
    bm25,
    chunks: list[str],
    metadata: list[dict] = None,
    k: int | None = None,
    threshold: float | None = None,
    filters: dict = None
) -> list[Result]:
    """Perform True Hybrid Search (BM25 + ChromaDB Semantic) with Metadata Filtering."""
    settings = get_settings()
    k = k or settings.default_top_k

    results_dict = {}

    # 1. Sparse Search (BM25)
    if bm25 and chunks:
        bm25_scores = bm25.get_scores(query.split())
        top_idx = np.argsort(bm25_scores)[-k:]
        for idx in top_idx:
            score = float(bm25_scores[idx])
            meta = metadata[idx] if metadata and idx < len(metadata) else {}
            
            if filters:
                if any(meta.get(f_key) != f_val for f_key, f_val in filters.items()):
                    continue

            if score > 0:
                text = chunks[idx]
                results_dict[text] = {"text": text, "sparse_score": score, "dense_score": 0.0, "metadata": meta}

    # 2. Dense Search (ChromaDB)
    if vector_db:
        try:
            dense_docs = vector_db.similarity_search_with_relevance_scores(query, k=k, filter=filters)
            for doc, score in dense_docs:
                text = doc.page_content
                if text in results_dict:
                    results_dict[text]["dense_score"] = score
                else:
                    results_dict[text] = {"text": text, "sparse_score": 0.0, "dense_score": score, "metadata": doc.metadata}
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")

    # 3. Combine Scores (Reciprocal Rank Fusion or simple weighted sum)
    final_results = []
    for text, data in results_dict.items():
        hybrid_score = (data["sparse_score"] * 0.3) + (data["dense_score"] * 0.7)
        final_results.append({
            "text": text,
            "score": hybrid_score,
            "metadata": data["metadata"],
            "confidence_score": min(1.0, hybrid_score)
        })

    # Sort best first
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:k]


def filter_results_by_documents(results: list[Result], relevant_docs: list[str]) -> list[Result]:
    """Keep only results whose source is in ``relevant_docs`` (if provided)."""
    if not relevant_docs:
        return results
    return [item for item in results if item.get("metadata", {}).get("source", "") in relevant_docs]


def rerank_results(query: str, results: list[Result]) -> list[Result]:
    """Passthrough for now, can be expanded with CrossEncoder."""
    return results


def format_sources(retrieved_chunks: list[Result]) -> str:
    """Render a de-duplicated bullet list of ``source (Page n)`` citations."""
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
