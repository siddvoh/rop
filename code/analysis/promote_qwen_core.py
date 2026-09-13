#!/usr/bin/env python3
"""Verify and export the paper-scoped Qwen selection records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from qwen_projection import (  # noqa: E402
    EvidenceError,
    canonical_json_bytes,
    load_archive,
    qwen_groups,
    sha256_bytes,
)


ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ARCHIVE = ROOT / "data/generated/ccr_reviewer_authority.zip"
DEFAULT_OUTPUT = ROOT / "data/evidence/qwen_core_promotion"


def summary(data: dict[str, Any]) -> dict[str, Any]:
    rows = data["qwen"]
    resolvable = [row for row in rows if row["selected_id_resolvable"]]
    distinct = [row for row in rows if not row["selected_matches_trace"]]
    distinct_resolvable = [row for row in distinct if row["selected_id_resolvable"]]
    return {
        "body_bm25": qwen_groups(rows, "body_bm25"),
        "cohort": {
            "all_rows": len(rows),
            "corpus_resolvable_selected_ids": len(resolvable),
            "object_distinct_rows": len(distinct),
            "resolvable_object_distinct_rows": len(distinct_resolvable),
            "unresolvable_question_ids": [
                row["question_id"] for row in rows if not row["selected_id_resolvable"]
            ],
        },
        "exact_decoded_field": qwen_groups(rows, "exact_decoded_field"),
        "final_hybrid": qwen_groups(rows, "final_hybrid"),
        "query_coverage": {
            "decoded_id_query_equals_generated_fact": sum(
                row["body_bm25"]["query_source"] == "model_fact_same_match_query"
                for row in rows
            ),
            "decoded_id_raw_equals_generated_fact": sum(
                row["body_bm25"]["raw_equals_generated_fact"] for row in rows
            ),
            "trace_label_fallback_rows": sum(
                row["body_bm25"]["query_source"]
                == "trace_label_same_selected_id"
                for row in rows
            ),
        },
        "schema_version": 2,
        "status": "verified",
        "status_counts": {
            "body_bm25": {"succeeded": len(rows)},
            "final_hybrid": {"succeeded": len(rows)},
            "selection": {"succeeded": len(rows)},
            "selection_outside_candidate_table": sum(
                not row["selection"]["selected_id_in_candidate_table"] for row in rows
            ),
        },
    }


def manifest(data: dict[str, Any], items: bytes, report: bytes) -> dict[str, Any]:
    return {
        "archive": data["archive"],
        "executed_identity": data["protocol"]["qwen"],
        "outputs": {
            "items.jsonl": {
                "bytes": len(items),
                "record_count": 335,
                "sha256": sha256_bytes(items),
            },
            "summary.json": {
                "bytes": len(report),
                "sha256": sha256_bytes(report),
            },
        },
        "schema_version": 2,
        "status": "verified",
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def promote(archive: Path, output: Path) -> dict[str, Any]:
    data = load_archive(archive)
    items = data["qwen_payload"]
    report = canonical_json_bytes(summary(data))
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "items.jsonl", items)
    atomic_write(output / "summary.json", report)
    result = manifest(data, items, report)
    atomic_write(output / "manifest.json", canonical_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = promote(args.archive.resolve(), args.output_dir.resolve())
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        print(f"Qwen evidence verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
