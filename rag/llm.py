"""LLM helpers: client construction and prompt-driven utilities."""

from __future__ import annotations

from collections.abc import Mapping

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

class MissingAPIKeyError(Exception):
    pass


class MultiLLMRouter:
    def __init__(self, groq_key_1=None, groq_key_2=None, gemini_key=None):
        settings = get_settings()
        self.groq_1 = self._init_groq(groq_key_1 or os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY"))
        self.groq_2 = self._init_groq(groq_key_2 or os.getenv("GROQ_API_KEY_2"))
        self.gemini = self._init_gemini(gemini_key or os.getenv("GEMINI_API_KEY"))
        
    def _init_groq(self, key):
        if not key: return None
        try:
            from langchain_groq import ChatGroq
            settings = get_settings()
            return ChatGroq(model=settings.groq_model, api_key=key, temperature=settings.llm_temperature)
        except Exception:
            return None
            
    def _init_gemini(self, key):
        if not key: return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            settings = get_settings()
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, temperature=settings.llm_temperature)
        except Exception:
            return None
            
    def invoke(self, messages):
        groq_result = None
        if self.groq_1:
            try: groq_result = self.groq_1.invoke(messages).content
            except Exception as e: logger.warning(f"Groq 1 failed: {e}")
                
        if not groq_result and self.groq_2:
            try: groq_result = self.groq_2.invoke(messages).content
            except Exception as e: logger.warning(f"Groq 2 failed: {e}")
                
        if not groq_result:
            if self.gemini:
                try: return self.gemini.invoke(messages)
                except Exception: pass
            raise RuntimeError("All LLMs failed or no keys configured.")
            
        if self.gemini:
            try:
                from langchain_core.messages import HumanMessage
                refine_prompt = f"Please refine and polish the following text, keeping all citations intact:\n\n{groq_result}"
                return self.gemini.invoke([HumanMessage(content=refine_prompt)])
            except Exception as e:
                logger.warning(f"Gemini fine-tuning failed: {e}")
                
        from langchain_core.messages import AIMessage
        return AIMessage(content=groq_result)
        
    def stream(self, messages):
        groq_result = None
        if self.groq_1:
            try: groq_result = self.groq_1.invoke(messages).content
            except Exception as e: logger.warning(f"Groq 1 failed: {e}")
                
        if not groq_result and self.groq_2:
            try: groq_result = self.groq_2.invoke(messages).content
            except Exception as e: logger.warning(f"Groq 2 failed: {e}")
                
        if not groq_result:
            if self.gemini:
                try:
                    for chunk in self.gemini.stream(messages): yield chunk
                    return
                except Exception: pass
            raise RuntimeError("All LLMs failed or no keys configured.")
            
        if self.gemini:
            try:
                from langchain_core.messages import HumanMessage
                refine_prompt = f"Please refine and polish the following text, keeping all citations intact:\n\n{groq_result}"
                for chunk in self.gemini.stream([HumanMessage(content=refine_prompt)]): yield chunk
                return
            except Exception as e:
                logger.warning(f"Gemini fine-tuning failed: {e}")
                
        from langchain_core.messages import AIMessageChunk
        words = groq_result.split(" ")
        for w in words: yield AIMessageChunk(content=w + " ")

def load_llm(api_key: str | None = None):
    import os
    return MultiLLMRouter(groq_key_1=api_key)


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
