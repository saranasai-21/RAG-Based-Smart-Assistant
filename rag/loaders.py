"""Multi-format document loaders.

Each loader returns a list of ``(page_number, text)`` tuples so the rest of the
pipeline can treat every file type uniformly. PDFs keep their real page
numbers; single-stream formats (txt/markdown/docx) are returned as page ``1``.
"""

from __future__ import annotations

import os

from rag.config import get_settings
from rag.logging_config import get_logger
from rag.text_utils import clean_text

logger = get_logger(__name__)

Page = tuple[int, str]

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md", ".markdown", ".docx")


def _keep(text: str) -> bool:
    return len(text) > get_settings().min_page_len


def load_pdf_pages(path: str) -> list[Page]:
    """Load a PDF into cleaned per-page tuples."""

    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(path).load()
    pages: list[Page] = []
    for idx, doc in enumerate(docs):
        text = clean_text(doc.page_content)
        if _keep(text):
            pages.append((idx + 1, text))
    return pages


def load_text_file(path: str) -> list[Page]:
    """Load a plain-text or markdown file as a single page."""

    with open(path, encoding="utf-8", errors="ignore") as handle:
        text = clean_text(handle.read())
    return [(1, text)] if _keep(text) else []


def load_docx_file(path: str) -> list[Page]:
    """Load a Word ``.docx`` document as a single page."""

    import docx  # python-docx

    document = docx.Document(path)
    text = clean_text("\n".join(p.text for p in document.paragraphs if p.text.strip()))
    return [(1, text)] if _keep(text) else []


def load_document(path: str) -> list[Page]:
    """Dispatch ``path`` to the loader matching its file extension.

    Raises:
        ValueError: if the file extension is not supported.
    """

    ext = os.path.splitext(path)[1].lower()
    logger.info("Loading document %s (type=%s)", path, ext)

    if ext == ".pdf":
        return load_pdf_pages(path)
    if ext in (".txt",):
        return load_text_file(path)
    if ext in (".md", ".markdown"):
        return load_text_file(path)
    if ext == ".docx":
        return load_docx_file(path)

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
