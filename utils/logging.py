"""Minimal logging setup so training runs and API requests look the same."""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: str | None = None) -> None:
    """Attach a single stream handler to the root logger (idempotent)."""
    global _configured
    if _configured:
        return
    resolved = (level or os.getenv("RECOMSENSE_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=resolved, format=_LOG_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    configure_logging()
    return logging.getLogger(name)
