"""Repository path discovery for installed code and extracted reviewer archives."""

from __future__ import annotations

from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    anchor = (start or Path(__file__)).resolve()
    for candidate in (anchor, *anchor.parents):
        if (candidate / "code").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(f"could not locate CCR repository root from {anchor}")
