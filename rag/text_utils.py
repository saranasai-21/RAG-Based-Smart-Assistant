"""Text hashing, cleaning and chunking utilities.

These helpers intentionally avoid heavy ML dependencies so they can be unit
tested quickly and imported in lightweight environments (e.g. CI).
"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import get_settings


def get_file_hash(data: bytes) -> str:
    """Return a stable MD5 hash for raw file ``bytes``.

    Accepts either ``bytes`` or any object exposing ``getvalue()`` (such as a
    Streamlit ``UploadedFile``) for convenience.
    """

    if hasattr(data, "getvalue"):
        data = data.getvalue()
    return hashlib.md5(data).hexdigest()


def clean_text(text: object) -> str:
    """Normalise whitespace and strip problematic characters from ``text``."""

    text = str(text)
    text = text.replace("\x00", " ")
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return " ".join(text.split())


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_len: int | None = None,
) -> list[str]:
    """Split ``text`` into overlapping, cleaned chunks.

    Chunks shorter than ``min_chunk_len`` are discarded so the retriever is not
    polluted with low-signal fragments.
    """

    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    min_chunk_len = min_chunk_len if min_chunk_len is not None else settings.min_chunk_len

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    cleaned: list[str] = []
    for chunk in splitter.split_text(text):
        chunk = clean_text(chunk)
        if len(chunk) > min_chunk_len:
            cleaned.append(chunk)
    return cleaned
