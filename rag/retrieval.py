from __future__ import annotations
import os
import numpy as np
from typing import List, Dict, Any

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

Result = dict[str, float | dict | str]

def create_bm25(chunks: list[str]):
    """Build a BM25 index over whitespace-tokenised ``chunks``."""
    from rank_bm25 import BM25Okapi
    tokenized = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized)

def create_vector_db(chunks: list[str], metadata_list: list[dict]):
    """Build an in-memory vector store using Gemini Embeddings."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found. Falling back to BM25-only.")
        return None
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        db = InMemoryVectorStore(embeddings)
        docs = [Document(page_content=chunk, metadata=meta) for chunk, meta in zip(chunks, metadata_list)]
        db.add_documents(docs)
        return db
    except Exception as e:
        logger.error(f"Failed to create vector db: {e}")
        return None

def hybrid_search(
    query: str,
    vector_db: InMemoryVectorStore | None,
    bm25,
    chunks: list[str],
    metadata_list: list[dict] = None,
    k: int = 5,
    alpha: float = 0.5, # 0.0 = BM25 only, 1.0 = Vector only
) -> list[Result]:
    """Perform Hybrid Search using Reciprocal Rank Fusion (RRF)."""
    settings = get_settings()
    k = k or settings.default_top_k
    
    # 1. BM25 Search
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_idx = np.argsort(bm25_scores)[-k*2:][::-1] # Get top 2k
    
    bm25_results = []
    for rank, idx in enumerate(bm25_top_idx):
        score = float(bm25_scores[idx])
        if score > 0:
            bm25_results.append({
                "id": idx,
                "text": chunks[idx],
                "score": score,
                "rank": rank + 1,
                "metadata": metadata_list[idx] if metadata_list else {},
            })
            
    # 2. Vector Search
    vector_results = []
    if vector_db:
        docs_with_scores = vector_db.similarity_search_with_score(query, k=k*2)
        for rank, (doc, score) in enumerate(docs_with_scores):
            # Find original index by text matching (or id)
            try:
                idx = chunks.index(doc.page_content)
                vector_results.append({
                    "id": idx,
                    "text": doc.page_content,
                    "score": float(score),
                    "rank": rank + 1,
                    "metadata": doc.metadata,
                })
            except ValueError:
                pass
                
    # 3. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    fused_scores = {}
    
    for res in bm25_results:
        idx = res["id"]
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rrf_k + res["rank"])) * (1.0 - alpha)
        
    for res in vector_results:
        idx = res["id"]
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rrf_k + res["rank"])) * alpha
        
    # Combine results
    final_results = []
    seen_idx = set()
    for idx, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        if idx not in seen_idx:
            seen_idx.add(idx)
            final_results.append({
                "text": chunks[idx],
                "score": score,
                "metadata": metadata_list[idx] if metadata_list else {},
            })
            
    return final_results[:k]

def filter_results_by_documents(results: list[Result], relevant_docs: list[str]) -> list[Result]:
    """Keep only results whose source is in ``relevant_docs`` (if provided)."""
    if not relevant_docs:
        return results
    return [item for item in results if item.get("metadata", {}).get("source", "") in relevant_docs]

def rerank_results(query: str, results: list[Result], llm=None) -> list[Result]:
    """Rerank retrieved chunks using LLM or simple passthrough."""
    # LLM-based reranking is slow and consumes a lot of tokens, 
    # so we'll just passthrough for now since RRF is already very strong.
    # A true Cross-Encoder is not possible on Vercel limits without an API like Cohere.
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
