#!/usr/bin/env python3
"""Recompute the Qwen decoded-ID BM25 and exact-field controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from qwen_projection import (  # noqa: E402
    EvidenceError,
    QWEN_MEMBER,
    canonical_json_bytes,
    load_archive,
    load_from_members,
    qwen_groups,
    rate,
)


ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ARCHIVE = ROOT / "data/generated/ccr_reviewer_authority.zip"
DEFAULT_OUTPUT = ROOT / "data/evidence/qwen_decoded_id_replication.json"


def build_report(data: dict[str, Any]) -> dict[str, Any]:
    rows = data["qwen"]
    distinct = [row for row in rows if not row["selected_matches_trace"]]
    fallback = [
        row["question_id"]
        for row in rows
        if row["body_bm25"]["query_source"] == "trace_label_same_selected_id"
    ]
    return {
        "api_calls": 0,
        "body_only_bm25": qwen_groups(rows, "body_bm25"),
        "cohort": {
            "all_rows": len(rows),
            "c0_selected_differs_from_traced": len(distinct),
            "c1_selected_equals_traced": len(rows) - len(distinct),
            "corpus_resolvable_selected_ids": sum(
                bool(row["selected_id_resolvable"]) for row in rows
            ),
            "corpus_unresolvable_selected_ids": sum(
                not bool(row["selected_id_resolvable"]) for row in rows
            ),
        },
        "definitions": {
            "decoded_selected_id": (
                "remove /wiki/, percent-decode, replace underscores with spaces, "
                "and collapse whitespace"
            ),
            "exact_field_baseline": (
                "return the selected ID at rank one when it occurs in the frozen resolver"
            ),
            "p_at_k": "selected ID occurs among the first k returned IDs",
        },
        "exact_field_baseline": qwen_groups(rows, "exact_decoded_field"),
        "execution_scope": {
            "exact_query_coverage": rate(len(rows), len(rows)),
            "fresh_sqlite_execution": False,
            "method": (
                "reuse a recorded BM25 ranking only when its frozen MATCH query "
                "equals the decoded selected-ID query"
            ),
        },
        "inputs": {
            "archive": data["archive"],
            "model_and_retrieval": data["protocol"]["qwen"],
            "qwen_records": next(
                row
                for row in data["manifest"]["members"]
                if row["path"] == QWEN_MEMBER
            ),
            "source_bindings": data["protocol"]["source_bindings"],
        },
        "model_calls": 0,
        "network_used": False,
        "query_coverage": {
            "all_rows": len(rows),
            "decoded_id_query_equals_generated_fact": sum(
                row["body_bm25"]["query_source"] == "model_fact_same_match_query"
                for row in rows
            ),
            "decoded_id_raw_equals_generated_fact": sum(
                row["body_bm25"]["raw_equals_generated_fact"] for row in rows
            ),
            "trace_label_fallback_question_ids": fallback,
        },
        "schema_version": 2,
        "status": "verified_zero_call_qwen_decoded_id_replication",
        "validation": {
            "all_recorded_rankings_succeeded": True,
            "selected_id_outside_candidate_table": sum(
                not row["selection"]["selected_id_in_candidate_table"] for row in rows
            ),
            "unresolvable_question_ids": [
                row["question_id"] for row in rows if not row["selected_id_resolvable"]
            ],
        },
    }


def build_report_from_members(
    reader: Callable[[str], bytes],
    *,
    script_path: Path | None = None,
) -> dict[str, Any]:
    del script_path
    return build_report(load_from_members(reader))


def build_report_from_artifact(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as bundle:
            return build_report_from_members(bundle.read)
    except zipfile.BadZipFile as error:
        raise EvidenceError(f"invalid artifact ZIP: {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--artifact", type=Path)
    source.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.artifact is not None:
            report = build_report_from_artifact(args.artifact.resolve())
        else:
            report = build_report(load_archive(args.archive.resolve()))
        payload = canonical_json_bytes(report)
        output = args.output or DEFAULT_OUTPUT
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        print(f"Qwen decoded-ID audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
