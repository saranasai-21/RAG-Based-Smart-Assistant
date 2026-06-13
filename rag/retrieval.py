"""BM25 sparse retrieval only for lightweight Vercel deployment.

This module strips out ChromaDB and Sentence-Transformers because they exceed
Vercel Serverless Function memory and disk footprint limitations.
"""

from __future__ import annotations

import numpy as np

from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

Result = dict[str, float | dict | str]


def create_bm25(chunks: list[str]):
    """Build a BM25 index over whitespace-tokenised ``chunks``."""
    from rank_bm25 import BM25Okapi

    tokenized = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized)


def hybrid_search(
    query: str,
    vector_db,  # unused now, kept for signature compatibility
    bm25,
    chunks: list[str],
    k: int | None = None,
    threshold: float | None = None,
) -> list[Result]:
    """Perform BM25-only search (renamed from hybrid for compatibility)."""
    settings = get_settings()
    k = k or settings.default_top_k

    results: list[Result] = []

    bm25_scores = bm25.get_scores(query.split())
    # Get top K indices
    top_idx = np.argsort(bm25_scores)[-k:]
    for idx in top_idx:
        score = float(bm25_scores[idx])
        if score > 0:
            results.append(
                {
                    "text": chunks[idx],
                    "score": score,
                    "metadata": {},
                }
            )

    # Sort best first
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def filter_results_by_documents(results: list[Result], relevant_docs: list[str]) -> list[Result]:
    """Keep only results whose source is in ``relevant_docs`` (if provided)."""
    if not relevant_docs:
        return results
    return [item for item in results if item.get("metadata", {}).get("source", "") in relevant_docs]


def rerank_results(query: str, results: list[Result]) -> list[Result]:
    """Passthrough for Vercel deployment (reranker removed for size)."""
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
