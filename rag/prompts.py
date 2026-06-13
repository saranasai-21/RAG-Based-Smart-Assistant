"""Prompt templates used across the assistant.

Keeping prompts in one place makes them easy to review, version and test
(asserting that the user query and context are always injected).
"""

from __future__ import annotations

from collections.abc import Mapping


def build_qa_prompt(context: str, sources_text: str, query: str, history_text: str = "") -> str:
    """Prompt for grounded question answering over retrieved context.

    When ``history_text`` is supplied it is included so the model can resolve
    references to the ongoing conversation (lightweight conversational memory).
    """

    history_block = ""
    if history_text.strip():
        history_block = f"""
CONVERSATION HISTORY (for context only, do not treat as factual source):
{history_text}
"""

    return f"""
You are an enterprise AI document assistant.

Rules:
- answer ONLY using provided context
- NEVER hallucinate
- provide concise factual answers
- synthesize intelligently
- cite sources naturally
{history_block}
CONTEXT:
{context}

AVAILABLE SOURCES:
{sources_text}

QUESTION:
{query}

ANSWER:
"""


def build_comparison_prompt(query: str, document_summaries: Mapping[str, str]) -> str:
    """Prompt for holistic, document-wide comparison."""

    summaries_text = ""
    for doc, summary in document_summaries.items():
        summaries_text += f"""

==================================================
DOCUMENT: {doc}
==================================================

{summary}

"""

    return f"""
You are an advanced AI document comparison expert.

Compare ENTIRE DOCUMENTS,
NOT isolated chunks/pages.

Your task:
- explain overall purpose
- explain similarities
- explain differences
- compare themes and concepts
- compare technologies and focus areas
- provide holistic synthesis

USER QUERY:
{query}

DOCUMENT SUMMARIES:
{summaries_text}

FINAL COMPARISON:
"""


def build_summary_prompt(text: str, filename: str = "") -> str:
    """Prompt for a structured, document-wide summary."""

    return f"""
You are an expert document analyst.

Analyze this ENTIRE document carefully.

Your job:
- identify the main subject
- identify the purpose
- identify major topics
- identify technologies/tools/concepts
- identify important sections
- identify the domain
- summarize holistically
- DO NOT summarize page-by-page
- think document-wide

DOCUMENT NAME:
{filename}

DOCUMENT:
{text}

STRUCTURED SUMMARY:
"""


def build_multi_query_prompt(query: str) -> str:
    """Prompt asking the LLM for alternative semantic search queries."""

    return f"""
Generate 3 alternative semantic search queries.

Original Query:
{query}

Alternative Queries:
"""


def build_followup_rewrite_prompt(query: str, history_text: str) -> str:
    """Prompt to rewrite a follow-up into a standalone question."""

    return f"""
You are an AI assistant.

Convert the follow-up query into a fully standalone question.

Use conversation history for context.

IMPORTANT:
- preserve meaning
- make it self-contained
- concise
- DO NOT answer
- ONLY rewrite

CHAT HISTORY:
{history_text}

FOLLOW-UP QUERY:
{query}

REWRITTEN QUERY:
"""


def build_relevant_docs_prompt(query: str, docs_text: str) -> str:
    """Prompt to identify which documents are relevant to the query."""

    return f"""
You are an AI assistant.

Identify which uploaded documents are relevant
to answering the user query.

Return ONLY document filenames separated by commas.

If multiple documents are relevant,
return multiple filenames.

USER QUERY:
{query}

DOCUMENTS:
{docs_text}

RELEVANT DOCUMENTS:
"""
