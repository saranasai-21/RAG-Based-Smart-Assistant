"""Embeddings, vector store, BM25 and cross-encoder reranking.

Heavy ML models (embeddings + reranker) are loaded lazily and cached so that
simply importing this module never triggers a model download. This keeps the
import side-effect free, which matters for tests and fast app start-up.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

import numpy as np

from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

Result = dict[str, Any]


@lru_cache(maxsize=1)
def get_device() -> str:
    """Return ``"cuda"`` when a GPU is available, otherwise ``"cpu"``."""

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    return device


@lru_cache(maxsize=1)
def load_embeddings():
    """Load and cache the sentence-embedding model."""

    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": get_device()},
    )


@lru_cache(maxsize=1)
def load_reranker():
    """Load and cache the cross-encoder reranker (lazy)."""

    from sentence_transformers import CrossEncoder

    settings = get_settings()
    logger.info("Loading reranker model: %s", settings.reranker_model)
    return CrossEncoder(settings.reranker_model, device=get_device())


def create_vector_db(chunks: list[str], metadata: list[dict]):
    """Create a persistent Chroma collection from ``chunks``."""

    from langchain_chroma import Chroma

    settings = get_settings()
    collection_name = "rag_" + str(uuid.uuid4())[:8]
    logger.info("Creating vector DB collection=%s chunks=%d", collection_name, len(chunks))
    return Chroma.from_texts(
        texts=chunks,
        embedding=load_embeddings(),
        metadatas=metadata,
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_dir,
    )


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
    k: int | None = None,
    threshold: float | None = None,
) -> list[Result]:
    """Combine dense (vector) and sparse (BM25) retrieval results."""

    settings = get_settings()
    k = k or settings.default_top_k
    threshold = threshold if threshold is not None else settings.default_threshold

    results: list[Result] = []

    semantic_results = vector_db.similarity_search_with_score(query, k=k * 3)
    for doc, score in semantic_results:
        similarity = 1 / (1 + score)
        if similarity >= threshold:
            results.append(
                {
                    "text": doc.page_content,
                    "score": similarity,
                    "metadata": doc.metadata,
                }
            )

    bm25_scores = bm25.get_scores(query.split())
    top_idx = np.argsort(bm25_scores)[-k:]
    for idx in top_idx:
        results.append(
            {
                "text": chunks[idx],
                "score": float(bm25_scores[idx]),
                "metadata": {},
            }
        )

    return results


def filter_results_by_documents(results: list[Result], relevant_docs: list[str]) -> list[Result]:
    """Keep only results whose source is in ``relevant_docs`` (if provided)."""

    if not relevant_docs:
        return results
    return [item for item in results if item.get("metadata", {}).get("source", "") in relevant_docs]


def rerank_results(query: str, results: list[Result]) -> list[Result]:
    """Re-score ``results`` with the cross-encoder, sorted best-first."""

    if not results:
        return []

    pairs = [[query, item["text"]] for item in results]
    scores = load_reranker().predict(pairs)

    ranked = sorted(zip(results, scores, strict=False), key=lambda x: x[1], reverse=True)
    reranked: list[Result] = []
    for item, score in ranked:
        item["score"] = float(score)
        reranked.append(item)
    return reranked


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
