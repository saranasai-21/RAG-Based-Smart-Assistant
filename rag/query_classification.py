"""Lightweight, dependency-free query intent detection.

The assistant routes each user message to a specialised handler (comparison,
metadata, general chat, ...) based on simple keyword heuristics. Keeping these
pure functions makes the routing logic easy to test and reason about.
"""

from __future__ import annotations

_COMPARISON_TERMS = (
    "compare",
    "comparison",
    "difference",
    "differences",
    "similarities",
    "similar",
    "contrast",
    "distinguish",
    "how are they different",
    "how do they differ",
    "what is different",
    "compare documents",
    "compare files",
)

_METADATA_TERMS = (
    "how many docs",
    "how many documents",
    "uploaded docs",
    "uploaded files",
    "list documents",
    "list files",
    "what files",
    "what documents",
    "document names",
    "file names",
    "which files",
    "which documents",
)

_GENERAL_TERMS = (
    "hi",
    "hello",
    "hey",
    "how are you",
    "who are you",
    "what can you do",
    "help",
    "thanks",
    "thank you",
)

_FOLLOWUP_TERMS = (
    "what about",
    "tell me more",
    "explain more",
    "and",
    "then",
    "compare further",
    "second one",
    "first one",
    "that document",
    "this document",
    "that one",
    "this one",
    "more about it",
    "elaborate",
    "continue",
)


def is_comparison_query(query: str) -> bool:
    """Return ``True`` when the query asks to compare/contrast documents."""

    query = query.lower()
    return any(term in query for term in _COMPARISON_TERMS)


def is_metadata_query(query: str) -> bool:
    """Return ``True`` for questions about the uploaded document set itself."""

    query = query.lower()
    return any(term in query for term in _METADATA_TERMS)


def is_general_chat(query: str) -> bool:
    """Return ``True`` for greetings / small talk (exact match)."""

    query = query.lower().strip()
    return any(term == query for term in _GENERAL_TERMS)


def is_followup_query(query: str) -> bool:
    """Return ``True`` when the query likely depends on prior context."""

    query = query.lower().strip()
    return any(term in query for term in _FOLLOWUP_TERMS)
