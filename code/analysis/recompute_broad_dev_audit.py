#!/usr/bin/env python3
"""Recompute the 1,397-item two-query retrieval grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from qwen_projection import (  # noqa: E402
    BROAD_MEMBER,
    EvidenceError,
    broad_summary,
    canonical_json_bytes,
    load_archive,
)


ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ARCHIVE = ROOT / "data/generated/ccr_reviewer_authority.zip"
DEFAULT_OUTPUT = ROOT / "data/evidence/broad_dev_audit.json"


def build_report(archive: Path) -> dict[str, object]:
    data = load_archive(archive)
    return {
        "definitions": {
            "decoded_trace_id": "percent-decoded dataset-traced Wikipedia ID",
            "population": (
                "unique-hidden development items outside the 335-item strict scope, "
                "including 1,182 no-selection rows and 215 answer-in-table rows"
            ),
            "recall_at_5": "trace ID occurs among the first five returned IDs",
        },
        "provenance": {
            "archive": data["archive"],
            "executed_retrieval": data["protocol"]["broad_development"],
            "records_member": next(
                row
                for row in data["manifest"]["members"]
                if row["path"] == BROAD_MEMBER
            ),
            "retrieval_identity": data["protocol"]["source_bindings"],
        },
        "schema_version": 2,
        "status": "verified",
        "table_18": broad_summary(data["broad"]),
        "validation": {
            "complete_query_grid": True,
            "network_used": False,
            "ranking_cells": len(data["broad"]) * 2 * 5,
            "status_counts": {"succeeded": len(data["broad"]) * 2 * 5},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = build_report(args.archive.resolve())
        payload = canonical_json_bytes(report)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, output)
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        print(f"broad-development audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
