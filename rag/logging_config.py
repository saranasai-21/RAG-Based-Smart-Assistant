"""Logging helpers for the RAG assistant."""

from __future__ import annotations

import logging

from rag.config import get_settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Optional log level name. Falls back to the configured
            ``LOG_LEVEL`` setting when omitted.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""

    setup_logging()
    return logging.getLogger(name)
