"""Build the frozen HybridQA BM25 corpus index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccr.retrieval.bm25 import build_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_index(args.passages, args.db, args.manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
