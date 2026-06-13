"""Central configuration for the RAG assistant.

All tunable values live here and can be overridden through environment
variables, which keeps secrets and deployment-specific values out of the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

try:  # Optional: load variables from a local .env file when present.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is an optional convenience dep
    pass


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get_env(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from the environment."""

    # --- Models -------------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _get_env("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    reranker_model: str = field(
        default_factory=lambda: _get_env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    groq_model: str = field(default_factory=lambda: _get_env("GROQ_MODEL", "llama-3.1-8b-instant"))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.0))

    # --- Credentials --------------------------------------------------------
    groq_api_key: str = field(default_factory=lambda: _get_env("GROQ_API_KEY", ""))

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 180))
    min_chunk_len: int = field(default_factory=lambda: _get_int("MIN_CHUNK_LEN", 100))
    min_page_len: int = field(default_factory=lambda: _get_int("MIN_PAGE_LEN", 20))

    # --- Ingestion limits ---------------------------------------------------
    max_file_size_mb: int = field(default_factory=lambda: _get_int("MAX_FILE_SIZE_MB", 25))
    max_total_chunks: int = field(default_factory=lambda: _get_int("MAX_TOTAL_CHUNKS", 300))
    max_chunks_per_page: int = field(default_factory=lambda: _get_int("MAX_CHUNKS_PER_PAGE", 40))
    summary_char_limit: int = field(default_factory=lambda: _get_int("SUMMARY_CHAR_LIMIT", 15000))

    # --- Retrieval ----------------------------------------------------------
    default_top_k: int = field(default_factory=lambda: _get_int("DEFAULT_TOP_K", 8))
    default_threshold: float = field(default_factory=lambda: _get_float("DEFAULT_THRESHOLD", 0.25))

    # --- Storage ------------------------------------------------------------
    chroma_persist_dir: str = field(
        default_factory=lambda: _get_env("CHROMA_PERSIST_DIR", "./chroma_db")
    )

    # --- Logging ------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
