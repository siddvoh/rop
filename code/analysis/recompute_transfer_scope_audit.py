#!/usr/bin/env python3
"""Recompute supplied-ID and Qwen selected-ID retrieval checks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Callable
from urllib.parse import unquote
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from qwen_projection import (  # noqa: E402
    EvidenceError,
    LINKS_MEMBER,
    canonical_json_bytes,
    jsonl,
    load_archive,
    load_from_members,
    qwen_group,
    rate,
    sha256_bytes,
)


ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ARCHIVE = ROOT / "data/generated/ccr_reviewer_authority.zip"
DEFAULT_LINKS = (
    ROOT
    / "data/results/released/hybridqa_natural_v3_retriever_sensitivity_v1/links.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/evidence/transfer_scope_audit.json"
BUCKETS = ("unique_hidden_no_selection", "unique_hidden_answer_in_table")
DEVELOPMENT_RETRIEVERS = ("bm25", "hybrid_rrf_bge_reranker_v2_m3")
CUTOFFS = (1, 5, 20)


def title(link: str) -> str:
    return " ".join(unquote(link.removeprefix("/wiki/")).replace("_", " ").split())


def development_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, Any] = {}
    for bucket in BUCKETS:
        subset = [row for row in rows if row["bucket"] == bucket]
        retrievers = {}
        for retriever in DEVELOPMENT_RETRIEVERS:
            counts = {}
            for cutoff in CUTOFFS:
                hits = sum(
                    row["trace_id"]
                    in row["rankings"][retriever]["decoded_trace_id"]["top20"][
                        :cutoff
                    ]
                    for row in subset
                )
                counts[str(cutoff)] = rate(hits, len(subset))
            retrievers[retriever] = {
                "preservation_by_cutoff": counts,
                "records": len(subset),
            }
        buckets[bucket] = {"records": len(subset), "retrievers": retrievers}
    return {
        "buckets": buckets,
        "records": len(rows),
        "target": "dataset-traced ID",
        "query": "decoded dataset-traced ID",
    }


def qwen_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partitions = {
        "all": rows,
        "c1_selected_id_matches_traced_id": [
            row for row in rows if row["selected_matches_trace"]
        ],
        "c0_selected_id_differs_from_traced_id": [
            row for row in rows if not row["selected_matches_trace"]
        ],
    }
    groups = {}
    for name, subset in partitions.items():
        metrics = qwen_group(subset, "final_hybrid")
        by_cutoff = {
            cutoff: values["p_selected_id_preserved_on_resolvable_rows"]
            for cutoff, values in metrics["by_cutoff"].items()
        }
        joint = metrics["by_cutoff"]["5"][
            "selected_vs_traced_joint_on_resolvable_rows"
        ]
        unresolvable = [row for row in subset if not row["selected_id_resolvable"]]
        groups[name] = {
            "all_rows": len(subset),
            "corpus_resolvable_selected_ids": metrics[
                "corpus_resolvable_selected_ids"
            ],
            "excluded_unresolvable_selected_ids": {
                "count": len(unresolvable),
                "traced_id_return_at_5": sum(
                    row["trace_id"] in row["final_hybrid"]["top20"][:5]
                    for row in unresolvable
                ),
            },
            "preservation_by_cutoff_on_resolvable_selected_ids": by_cutoff,
            "selected_vs_traced_return_joint_at_5_on_resolvable_selected_ids": joint,
        }
    return {
        "groups": groups,
        "query": "Qwen model-written retrieval fact",
        "target": "Qwen selected ID",
    }


def title_report(links_payload: bytes) -> dict[str, Any]:
    rows = jsonl(links_payload, LINKS_MEMBER)
    links = [row.get("link") for row in rows]
    require_links = all(isinstance(link, str) and link.startswith("/wiki/") for link in links)
    if not require_links or len(links) != len(set(links)):
        raise EvidenceError("resolver inventory is invalid")
    titles = [title(link) for link in links]
    counts = Counter(titles)
    collisions = [count for count in counts.values() if count > 1]
    return {
        "all_title_strings_unique": not collisions,
        "collision_title_count": len(collisions),
        "distinct_title_count": len(counts),
        "identifier_count": len(links),
        "links_sha256": sha256_bytes(links_payload),
        "maximum_title_multiplicity": max(counts.values()),
    }


def build_report(data: dict[str, Any], links_payload: bytes) -> dict[str, Any]:
    titles = title_report(links_payload)
    if titles["identifier_count"] != 286_270 or not titles["all_title_strings_unique"]:
        raise EvidenceError("identifier-title inventory changed")
    return {
        "api_calls": 0,
        "definitions": {
            "p_at_k": "selected ID occurs among the first k returned IDs",
            "h_at_k": "dataset-traced ID occurs among the first k returned IDs",
            "title_projection": (
                "remove /wiki/, percent-decode, replace underscores with spaces, "
                "and collapse whitespace"
            ),
        },
        "identifier_title_uniqueness": titles,
        "inputs": {
            "archive": data["archive"],
            "development_retrieval": data["protocol"]["broad_development"],
            "model_and_retrieval": data["protocol"]["qwen"],
            "source_bindings": data["protocol"]["source_bindings"],
        },
        "model_calls": 0,
        "network_used": False,
        "qwen_selected_id_preservation": qwen_report(data["qwen"]),
        "schema_version": 2,
        "status": "verified_zero_call_transfer_scope_audit",
        "supplied_traced_id_projection_preservation": development_report(
            data["broad"]
        ),
    }


def build_report_from_members(reader: Callable[[str], bytes]) -> dict[str, Any]:
    try:
        links_payload = reader(LINKS_MEMBER)
    except (KeyError, OSError) as error:
        raise EvidenceError(f"missing artifact member: {LINKS_MEMBER}") from error
    return build_report(load_from_members(reader), links_payload)


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
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if args.artifact is not None:
            report = build_report_from_artifact(args.artifact.resolve())
        else:
            report = build_report(
                load_archive(args.archive.resolve()), args.links.resolve().read_bytes()
            )
        payload = canonical_json_bytes(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        print(f"transfer-scope audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
