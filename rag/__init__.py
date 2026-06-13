"""RAG-Based Smart Assistant package.

A modular Retrieval-Augmented Generation toolkit that powers the Streamlit
assistant. The package is split into small, individually testable modules:

- :mod:`rag.config` - environment-driven configuration.
- :mod:`rag.logging_config` - centralised logging setup.
- :mod:`rag.text_utils` - hashing, cleaning and chunking helpers.
- :mod:`rag.query_classification` - lightweight intent detection.
- :mod:`rag.prompts` - prompt templates.
- :mod:`rag.loaders` - multi-format document loaders.
- :mod:`rag.retrieval` - embeddings, vector store, BM25 and reranking.
- :mod:`rag.llm` - LLM helpers (multi-query, summaries, rewriting).
"""

from rag.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]

__version__ = "1.0.0"
