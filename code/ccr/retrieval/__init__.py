"""Deterministic retrieval components for CCR experiments."""

from .bm25 import INDEX_VERSION, build_index, retrieve

__all__ = ["INDEX_VERSION", "build_index", "retrieve"]
