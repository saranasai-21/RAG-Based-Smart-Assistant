"""LLM helpers: client construction and prompt-driven utilities."""

from __future__ import annotations

from collections.abc import Mapping
import os

from rag.config import get_settings
from rag.logging_config import get_logger
from rag.prompts import (
    build_followup_rewrite_prompt,
    build_multi_query_prompt,
    build_relevant_docs_prompt,
    build_summary_prompt,
)
from rag.query_classification import is_comparison_query

logger = get_logger(__name__)


class MissingAPIKeyError(RuntimeError):
    """Raised when no Groq API key is configured."""


def load_llm(api_key: str | None = None):
    """Construct a :class:`ChatGoogleGenerativeAI` client.

    Args:
        api_key: Optional explicit key; falls back to ``GOOGLE_API_KEY``.

    Raises:
        MissingAPIKeyError: when no key can be resolved.
    """

    settings = get_settings()
    key = api_key or os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY is not set. Add it to your environment or .env file."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=key,
        temperature=settings.llm_temperature,
    )


def _content(response) -> str:
    """Extract text from an LLM response that may be a message or string."""

    if hasattr(response, "content"):
        return response.content
    return str(response)


def generate_multi_queries(llm, query: str) -> list[str]:
    """Expand ``query`` into several semantic variants for retrieval."""

    if is_comparison_query(query):
        return [
            query,
            f"document comparison {query}",
            f"main differences {query}",
            f"document similarities {query}",
            f"compare overall themes {query}",
        ]

    try:
        response = _content(llm.invoke(build_multi_query_prompt(query)))
        queries = []
        for line in response.split("\n"):
            line = line.strip()
            if line:
                queries.append(line.replace("-", "").strip())
        queries.append(query)
        return list(set(queries))
    except Exception as exc:  # pragma: no cover - network/LLM failure
        logger.warning("Multi-query generation failed: %s", exc)
        return [query]


def summarize_document(llm, text: str, filename: str = "") -> str:
    """Produce a holistic, document-wide summary."""

    settings = get_settings()
    prompt = build_summary_prompt(text[: settings.summary_char_limit], filename)
    try:
        return _content(llm.invoke(prompt))
    except Exception as exc:  # pragma: no cover - network/LLM failure
        logger.warning("Summary failed for %s: %s", filename, exc)
        return f"Summary failed: {exc}"


def get_recent_chat_history(chat_history: list[dict], limit: int = 4) -> str:
    """Render the last ``limit`` turns of ``chat_history`` as text."""

    if not chat_history:
        return ""

    history_text = ""
    for item in chat_history[-limit:]:
        history_text += f"""
USER:
{item["user"]}

ASSISTANT:
{item["assistant"]}

"""
    return history_text


def rewrite_followup_query(llm, query: str, chat_history: list[dict]) -> str:
    """Rewrite a context-dependent follow-up into a standalone question."""

    history_text = get_recent_chat_history(chat_history)
    prompt = build_followup_rewrite_prompt(query, history_text)
    try:
        return _content(llm.invoke(prompt)).strip()
    except Exception as exc:  # pragma: no cover - network/LLM failure
        logger.warning("Follow-up rewrite failed: %s", exc)
        return query


def detect_relevant_documents(query: str, document_summaries: Mapping[str, str], llm) -> list[str]:
    """Return the filenames the LLM judges relevant to ``query``."""

    docs_text = ""
    for idx, (doc, summary) in enumerate(document_summaries.items(), start=1):
        docs_text += f"""

DOCUMENT {idx}:
{doc}

SUMMARY:
{summary}

"""

    prompt = build_relevant_docs_prompt(query, docs_text)
    try:
        response = _content(llm.invoke(prompt))
        return [name for name in document_summaries.keys() if name.lower() in response.lower()]
    except Exception as exc:  # pragma: no cover - network/LLM failure
        logger.warning("Relevant-doc detection failed: %s", exc)
        return list(document_summaries.keys())
