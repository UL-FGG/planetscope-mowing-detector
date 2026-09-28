"""General utilities."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_directory(path: Path) -> Path:
    """Create and return a directory."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(parts: Iterable[str], length: int = 12) -> str:
    """Create a short deterministic hash from strings."""
    text = "|".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
