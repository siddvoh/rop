#!/usr/bin/env python3
"""Replay a deterministic rank-and-identifier taxonomy of BM25 omissions.

The released bridge ledger records the visible-label BM25 ranking to depth 20.
The released passage inventory records the exact identifier universe but not
the indexed passage text. This audit therefore reports the result supported by
the released bytes: it partitions every corpus-resolvable selected ID
omitted at rank 5 by (a) its observed rank, when present at ranks 6--20, and
(b) the frozen BM25 query-term relation between its label and the identifiers
shown in the top five, when absent from the recorded top 20.

The latter categories are descriptive identifier-title diagnostics.  They are
not a causal account of the BM25 score because passage text and per-result
BM25 scores are not present in the released bridge ledger.  No model, API,
network, clock, or random input is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Callable, Mapping
from urllib.parse import unquote
import zipfile


SCHEMA_VERSION = 1
MODELS = ("claude", "openai", "gemini")
MODEL_LABELS = {"claude": "Claude", "openai": "GPT", "gemini": "Gemini"}
TOP_K = 5
RECORDED_DEPTH = 20

BRIDGE_MEMBER = (
    "data/results/released/hybridqa_natural_v2/"
    "bridge_confirmatory/details.jsonl"
)
LINKS_MEMBER = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1/links.jsonl"
)
BM25_MANIFEST_MEMBER = (
    "data/results/released/hybridqa_natural_v1/bm25_v1/"
    "index_manifest.json"
)
PREPARE_MANIFEST_MEMBER = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1/prepare_manifest.json"
)
BM25_CODE_MEMBER = "code/src/ccr/retrieval/bm25.py"
BRIDGE_RUN_MANIFEST_MEMBER = (
    "data/results/released/hybridqa_natural_v2/"
    "bridge_confirmatory/run_manifest.json"
)

# Bind every byte used by this audit.  Passage text is deliberately not listed:
# it is not a member of the released reviewer artifact.
EXPECTED_MEMBERS: dict[str, dict[str, Any]] = {
    BRIDGE_MEMBER: {
        "bytes": 4_991_925,
        "records": 1_800,
        "sha256": "abf0027a3f61aa338eae4a259b8751f7adecc5ea7f79a08f25ad11b179a79fc5",
    },
    LINKS_MEMBER: {
        "bytes": 15_250_846,
        "records": 286_270,
        "sha256": "6e9488f4a09fbd5be96ffff23f4e956195e0c231932137ba6894bfcdfe2fb303",
    },
    BM25_MANIFEST_MEMBER: {
        "bytes": 599,
        "sha256": "86a8f7032988d3c7a0b7776db249b17f51dd8bedd717e76e8325e339c4277283",
    },
    PREPARE_MANIFEST_MEMBER: {
        "bytes": 1_007,
        "sha256": "fc1d4ba92c4f4a127525495fc20ffa1f69938178e8d9ff10f0131459168fa3b0",
    },
    BM25_CODE_MEMBER: {
        "bytes": 8_140,
        "sha256": "55dcd1da941803e6ef3a2e48e191913de44ad971814d01d8d54b98b84bf074d3",
    },
    BRIDGE_RUN_MANIFEST_MEMBER: {
        "bytes": 743,
        "sha256": "e34355963f3142f9d9ae2f3d162258d6b86cd757ffcea1a50b442eeb0f154a4c",
    },
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

MECHANISM_CATEGORIES = (
    "cutoff_competition_selected_id_rank_6_to_20",
    "query_document_nonmatch_exhaustive_result_set_below_20",
    "full_top20_selected_id_absent_rank_or_nonmatch_unresolved",
)

IDENTIFIER_TITLE_CATEGORIES = (
    "selected_id_rank_6_to_20",
    "absent_top20_same_query_term_set_in_top5_identifier",
    "absent_top20_all_query_terms_in_one_top5_identifier",
    "absent_top20_partial_query_term_overlap_in_top5_identifiers",
    "absent_top20_no_query_term_overlap_in_top5_identifiers",
)

# Independent post-specification replay targets.  These make a changed join,
# cutoff, tokenizer, or category priority stop the replay.
EXPECTED_CATEGORY_COUNTS = {
    "claude": {
        "selected_id_rank_6_to_20": 69,
        "absent_top20_same_query_term_set_in_top5_identifier": 3,
        "absent_top20_all_query_terms_in_one_top5_identifier": 43,
        "absent_top20_partial_query_term_overlap_in_top5_identifiers": 34,
        "absent_top20_no_query_term_overlap_in_top5_identifiers": 16,
    },
    "openai": {
        "selected_id_rank_6_to_20": 59,
        "absent_top20_same_query_term_set_in_top5_identifier": 3,
        "absent_top20_all_query_terms_in_one_top5_identifier": 43,
        "absent_top20_partial_query_term_overlap_in_top5_identifiers": 32,
        "absent_top20_no_query_term_overlap_in_top5_identifiers": 14,
    },
    "gemini": {
        "selected_id_rank_6_to_20": 63,
        "absent_top20_same_query_term_set_in_top5_identifier": 3,
        "absent_top20_all_query_terms_in_one_top5_identifier": 43,
        "absent_top20_partial_query_term_overlap_in_top5_identifiers": 30,
        "absent_top20_no_query_term_overlap_in_top5_identifiers": 11,
    },
}

EXPECTED_MECHANISM_COUNTS = {
    "claude": {
        "cutoff_competition_selected_id_rank_6_to_20": 69,
        "query_document_nonmatch_exhaustive_result_set_below_20": 5,
        "full_top20_selected_id_absent_rank_or_nonmatch_unresolved": 91,
    },
    "openai": {
        "cutoff_competition_selected_id_rank_6_to_20": 59,
        "query_document_nonmatch_exhaustive_result_set_below_20": 5,
        "full_top20_selected_id_absent_rank_or_nonmatch_unresolved": 87,
    },
    "gemini": {
        "cutoff_competition_selected_id_rank_6_to_20": 63,
        "query_document_nonmatch_exhaustive_result_set_below_20": 4,
        "full_top20_selected_id_absent_rank_or_nonmatch_unresolved": 83,
    },
}

EXPECTED_MODEL_TOTALS = {
    "claude": {
        "records": 600,
        "valid_outputs": 600,
        "corpus_resolvable_selected_ids": 599,
        "corpus_unresolvable_selected_ids": 1,
        "top5_preserved": 434,
        "top5_omitted": 165,
        "absent_top20": 96,
    },
    "openai": {
        "records": 600,
        "valid_outputs": 594,
        "corpus_resolvable_selected_ids": 594,
        "corpus_unresolvable_selected_ids": 0,
        "top5_preserved": 443,
        "top5_omitted": 151,
        "absent_top20": 92,
    },
    "gemini": {
        "records": 600,
        "valid_outputs": 600,
        "corpus_resolvable_selected_ids": 599,
        "corpus_unresolvable_selected_ids": 1,
        "top5_preserved": 449,
        "top5_omitted": 150,
        "absent_top20": 87,
    },
}


class AuditError(RuntimeError):
    """Raised when a bound input or taxonomy invariant does not verify."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_rows(payload: bytes, member: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"{member}: not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{member}:{line_number}: invalid JSON") from exc
        require(isinstance(row, dict), f"{member}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def _json_object(payload: bytes, member: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{member}: invalid JSON object") from exc
    require(isinstance(value, dict), f"{member}: JSON value is not an object")
    return value


def _member_reader(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> Callable[[str], bytes]:
    if callable(source):
        return source

    def read(member: str) -> bytes:
        try:
            payload = source[member]
        except KeyError as exc:
            raise AuditError(f"missing source member: {member}") from exc
        require(isinstance(payload, bytes), f"{member}: source must be bytes")
        return payload

    return read


def _validate_sources(
    source: Mapping[str, bytes] | Callable[[str], bytes],
    expected_members: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    require(
        set(expected_members) == set(EXPECTED_MEMBERS),
        "expected member set differs from the seven declared audit inputs",
    )
    read = _member_reader(source)
    payloads: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for member in sorted(expected_members):
        try:
            payload = read(member)
        except (KeyError, FileNotFoundError) as exc:
            raise AuditError(f"missing source member: {member}") from exc
        expected = expected_members[member]
        digest = sha256_bytes(payload)
        require(len(payload) == expected["bytes"], f"{member}: byte count drift")
        require(digest == expected["sha256"], f"{member}: SHA-256 drift")
        details: dict[str, Any] = {"bytes": len(payload), "sha256": digest}
        if "records" in expected:
            records = sum(bool(line.strip()) for line in payload.splitlines())
            require(records == expected["records"], f"{member}: record count drift")
            details["records"] = records
        payloads[member] = payload
        metadata[member] = details
    return payloads, metadata


def _link_label(link: str) -> str:
    value = unquote(link.removeprefix("/wiki/")).replace("_", " ")
    return " ".join(value.split())


def _query_term_set(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1
    )


def _selected_rank(selected_link: str, ranking: list[str]) -> int | None:
    return next(
        (rank for rank, link in enumerate(ranking, 1) if link == selected_link),
        None,
    )


def _classify_omission(
    *, selected_link: str, visible_label: str, ranking: list[str]
) -> str:
    """Return one mutually exclusive category for an observed top-5 omission."""

    rank = _selected_rank(selected_link, ranking)
    require(rank is None or rank > TOP_K, "classification input is not a top-5 omission")
    if rank is not None:
        require(rank <= RECORDED_DEPTH, "selected rank exceeds recorded depth")
        return IDENTIFIER_TITLE_CATEGORIES[0]

    query_terms = _query_term_set(visible_label)
    require(bool(query_terms), "visible label has an empty frozen BM25 query term set")
    top5_term_sets = [
        _query_term_set(_link_label(link)) for link in ranking[:TOP_K]
    ]
    if any(candidate == query_terms for candidate in top5_term_sets):
        return IDENTIFIER_TITLE_CATEGORIES[1]
    if any(query_terms < candidate for candidate in top5_term_sets):
        return IDENTIFIER_TITLE_CATEGORIES[2]
    if any(query_terms & candidate for candidate in top5_term_sets):
        return IDENTIFIER_TITLE_CATEGORIES[3]
    return IDENTIFIER_TITLE_CATEGORIES[4]


def _classify_mechanism(
    *, selected_link: str, ranking: list[str]
) -> str:
    """Partition an omission by what the frozen top-20 result set proves."""

    rank = _selected_rank(selected_link, ranking)
    require(rank is None or rank > TOP_K, "classification input is not a top-5 omission")
    if rank is not None:
        require(rank <= RECORDED_DEPTH, "selected rank exceeds recorded depth")
        return MECHANISM_CATEGORIES[0]
    if len(ranking) < RECORDED_DEPTH:
        # retrieve() returns every matching document when fewer than LIMIT rows
        # satisfy the OR query.  An absent target therefore matched no retained
        # query token under the frozen FTS5 execution contract.
        return MECHANISM_CATEGORIES[1]
    return MECHANISM_CATEGORIES[2]


def _load_inventory(rows: list[dict[str, Any]]) -> set[str]:
    require(len(rows) == 286_270, "passage identifier inventory must have 286,270 rows")
    inventory: set[str] = set()
    for expected_pid, row in enumerate(rows):
        require(set(row) == {"pid", "link"}, f"links:{expected_pid + 1}: schema drift")
        require(row.get("pid") == expected_pid, f"links:{expected_pid + 1}: pid drift")
        link = row.get("link")
        require(
            isinstance(link, str) and link.startswith("/wiki/"),
            f"links:{expected_pid + 1}: invalid link",
        )
        require(link not in inventory, f"links:{expected_pid + 1}: duplicate link")
        inventory.add(link)
    return inventory


def _load_bridge(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) == 1_800, "bridge ledger must have 1,800 rows")
    seen: set[tuple[str, str]] = set()
    model_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    result: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        question_id = row.get("question_id")
        model = row.get("model")
        require(
            isinstance(question_id, str) and question_id,
            f"bridge:{row_number}: invalid question_id",
        )
        require(model in MODELS, f"bridge:{row_number}: unexpected model")
        identity = (question_id, model)
        require(identity not in seen, f"bridge:{row_number}: duplicate identity")
        seen.add(identity)
        model_counts[model] += 1
        require(row.get("status") == "succeeded", f"bridge:{identity}: status drift")
        require(type(row.get("valid_json")) is bool, f"bridge:{identity}: valid_json drift")
        if row["valid_json"]:
            valid_counts[model] += 1
            selected_link = row.get("selected_link")
            visible_label = row.get("primary_predicted_query")
            ranking = row.get("predicted_ranking")
            require(
                isinstance(selected_link, str) and selected_link.startswith("/wiki/"),
                f"bridge:{identity}: invalid selected link",
            )
            require(
                visible_label == _link_label(selected_link),
                f"bridge:{identity}: visible label drift",
            )
            require(
                isinstance(ranking, list)
                and len(ranking) <= RECORDED_DEPTH
                and all(isinstance(link, str) and link.startswith("/wiki/") for link in ranking),
                f"bridge:{identity}: invalid BM25 ranking",
            )
            require(len(ranking) == len(set(ranking)), f"bridge:{identity}: duplicate ranking link")
            require(
                bool(_query_term_set(visible_label)),
                f"bridge:{identity}: empty BM25 query term set",
            )
        else:
            require(model == "openai", f"bridge:{identity}: unexpected invalid model")
            require(
                row.get("selected_link") is None
                and row.get("primary_predicted_query") is None
                and row.get("predicted_ranking") is None,
                f"bridge:{identity}: invalid row has selection or ranking",
            )
        result.append(row)
    require(
        model_counts == Counter({model: 600 for model in MODELS}),
        f"bridge model counts drifted: {dict(model_counts)}",
    )
    require(
        valid_counts == Counter({"claude": 600, "openai": 594, "gemini": 600}),
        f"bridge valid counts drifted: {dict(valid_counts)}",
    )
    return result


def _validate_manifests(
    bm25_manifest: Mapping[str, Any], prepare_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    require(bm25_manifest.get("status") == "complete", "BM25 manifest is not complete")
    require(
        bm25_manifest.get("document_count") == 286_270,
        "BM25 document count drift",
    )
    require(
        bm25_manifest.get("query")
        == {
            "token_pattern": "[a-z0-9]+",
            "minimum_token_length": 2,
            "deduplicate_tokens": True,
            "operator": "OR",
        },
        "BM25 query tokenizer drift",
    )
    require(
        bm25_manifest.get("ranking")
        == {
            "implementation": "SQLite FTS5 bm25",
            "document_tokenizer": "unicode61",
            "tie_break": "ascending FTS rowid",
        },
        "BM25 ranking specification drift",
    )
    require(prepare_manifest.get("status") == "prepared", "prepare manifest status drift")
    require(
        prepare_manifest.get("document_count") == 286_270,
        "prepare manifest document count drift",
    )
    require(
        prepare_manifest.get("links_sha256") == EXPECTED_MEMBERS[LINKS_MEMBER]["sha256"],
        "prepare manifest links binding drift",
    )
    require(
        prepare_manifest.get("bm25_manifest_sha256")
        == EXPECTED_MEMBERS[BM25_MANIFEST_MEMBER]["sha256"],
        "prepare manifest BM25 binding drift",
    )
    for field in ("bm25_db_sha256", "bm25_source_inventory_sha256", "collection_sha256"):
        value = prepare_manifest.get(field)
        require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"prepare manifest {field} drift",
        )
    require(
        prepare_manifest["bm25_source_inventory_sha256"]
        == bm25_manifest.get("source_inventory_sha256"),
        "BM25 source inventory cross-binding drift",
    )
    return {
        "bm25_database_sha256": prepare_manifest["bm25_db_sha256"],
        "passage_collection_sha256": prepare_manifest["collection_sha256"],
        "passage_source_inventory_sha256": prepare_manifest[
            "bm25_source_inventory_sha256"
        ],
        "passage_identifier_inventory_sha256": prepare_manifest["links_sha256"],
        "document_count": prepare_manifest["document_count"],
    }


def _validate_execution_contract(
    bm25_code: bytes,
    bridge_run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the released BM25 implementation and run-manifest binding."""

    try:
        bm25_text = bm25_code.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError("BM25 execution source is not valid UTF-8") from exc
    for fragment in (
        "match = query_terms(query)",
        "WHERE passage_fts MATCH ?",
        "ORDER BY score ASC, rowid ASC",
        "LIMIT ?",
    ):
        require(fragment in bm25_text, f"BM25 execution contract missing {fragment!r}")
    require(
        bridge_run_manifest.get("index_manifest_sha256")
        == EXPECTED_MEMBERS[BM25_MANIFEST_MEMBER]["sha256"],
        "bridge run/index manifest binding drift",
    )
    for field in ("result_set_sha256", "bridge_implementation_sha256"):
        value = bridge_run_manifest.get(field)
        require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"bridge run manifest {field} drift",
        )
    return {
        "bm25_retrieval_code_sha256": sha256_bytes(bm25_code),
        "bridge_run_result_set_sha256": bridge_run_manifest["result_set_sha256"],
        "retrieval_depth": RECORDED_DEPTH,
        "no_score_threshold": True,
        "exhaustive_when_returned_rows_below_depth": True,
    }


def _rate(count: int, total: int) -> dict[str, Any]:
    require(total > 0 and 0 <= count <= total, "invalid rate")
    return {"count": count, "total": total, "rate": count / total}


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
    *,
    expected_members: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the exact report from the six hash-bound released members."""

    expected = EXPECTED_MEMBERS if expected_members is None else expected_members
    payloads, source_metadata = _validate_sources(source, expected)
    bridge = _load_bridge(_jsonl_rows(payloads[BRIDGE_MEMBER], BRIDGE_MEMBER))
    inventory = _load_inventory(_jsonl_rows(payloads[LINKS_MEMBER], LINKS_MEMBER))
    corpus_binding = _validate_manifests(
        _json_object(payloads[BM25_MANIFEST_MEMBER], BM25_MANIFEST_MEMBER),
        _json_object(payloads[PREPARE_MANIFEST_MEMBER], PREPARE_MANIFEST_MEMBER),
    )
    execution_contract = _validate_execution_contract(
        payloads[BM25_CODE_MEMBER],
        _json_object(
            payloads[BRIDGE_RUN_MANIFEST_MEMBER], BRIDGE_RUN_MANIFEST_MEMBER
        ),
    )

    models: dict[str, Any] = {}
    aggregate_mechanisms: Counter[str] = Counter()
    aggregate_title_diagnostics: Counter[str] = Counter()
    aggregate_eligible = 0
    aggregate_preserved = 0
    for model in MODELS:
        model_rows = [row for row in bridge if row["model"] == model]
        valid_rows = [row for row in model_rows if row["valid_json"]]
        eligible = [row for row in valid_rows if row["selected_link"] in inventory]
        unresolved = [row for row in valid_rows if row["selected_link"] not in inventory]
        preserved = [
            row
            for row in eligible
            if row["selected_link"] in row["predicted_ranking"][:TOP_K]
        ]
        omissions = [
            row
            for row in eligible
            if row["selected_link"] not in row["predicted_ranking"][:TOP_K]
        ]
        mechanism_counts: Counter[str] = Counter()
        title_diagnostic_counts: Counter[str] = Counter()
        rank_counts: Counter[int] = Counter()
        absent_top20 = 0
        for row in omissions:
            mechanism = _classify_mechanism(
                selected_link=row["selected_link"],
                ranking=row["predicted_ranking"],
            )
            title_diagnostic = _classify_omission(
                selected_link=row["selected_link"],
                visible_label=row["primary_predicted_query"],
                ranking=row["predicted_ranking"],
            )
            mechanism_counts[mechanism] += 1
            title_diagnostic_counts[title_diagnostic] += 1
            rank = _selected_rank(row["selected_link"], row["predicted_ranking"])
            if rank is None:
                absent_top20 += 1
            else:
                rank_counts[rank] += 1

        observed_mechanisms = {
            category: mechanism_counts[category]
            for category in MECHANISM_CATEGORIES
        }
        require(
            observed_mechanisms == EXPECTED_MECHANISM_COUNTS[model],
            f"{model}: mechanism count target drift: {observed_mechanisms}",
        )
        observed_title_diagnostics = {
            category: title_diagnostic_counts[category]
            for category in IDENTIFIER_TITLE_CATEGORIES
        }
        require(
            observed_title_diagnostics == EXPECTED_CATEGORY_COUNTS[model],
            f"{model}: title diagnostic target drift: {observed_title_diagnostics}",
        )
        totals = {
            "records": len(model_rows),
            "valid_outputs": len(valid_rows),
            "corpus_resolvable_selected_ids": len(eligible),
            "corpus_unresolvable_selected_ids": len(unresolved),
            "top5_preserved": len(preserved),
            "top5_omitted": len(omissions),
            "absent_top20": absent_top20,
        }
        require(
            totals == EXPECTED_MODEL_TOTALS[model],
            f"{model}: model totals drift: {totals}",
        )
        require(
            sum(mechanism_counts.values()) == len(omissions),
            f"{model}: mechanism taxonomy does not partition omissions",
        )
        require(
            sum(title_diagnostic_counts.values()) == len(omissions),
            f"{model}: title diagnostics do not partition omissions",
        )
        require(
            mechanism_counts[MECHANISM_CATEGORIES[0]] + absent_top20
            == len(omissions),
            f"{model}: coarse rank partition drift",
        )
        require(
            len(preserved) + len(omissions) == len(eligible),
            f"{model}: preservation partition drift",
        )

        models[model] = {
            "display_name": MODEL_LABELS[model],
            **totals,
            "top5_preservation_on_corpus_resolvable_selected_ids": _rate(
                len(preserved), len(eligible)
            ),
            "top5_omission_on_corpus_resolvable_selected_ids": _rate(
                len(omissions), len(eligible)
            ),
            "coarse_observed_rank_partition": {
                "selected_id_observed_at_rank_6_to_20": mechanism_counts[
                    MECHANISM_CATEGORIES[0]
                ],
                "selected_id_absent_from_recorded_top20": absent_top20,
                "total_top5_omissions": len(omissions),
            },
            "mutually_exclusive_mechanism_categories": {
                category: {
                    "count": mechanism_counts[category],
                    "fraction_of_top5_omissions": mechanism_counts[category]
                    / len(omissions),
                }
                for category in MECHANISM_CATEGORIES
            },
            "secondary_noncausal_identifier_title_categories": {
                category: {
                    "count": title_diagnostic_counts[category],
                    "fraction_of_top5_omissions": title_diagnostic_counts[category]
                    / len(omissions),
                }
                for category in IDENTIFIER_TITLE_CATEGORIES
            },
            "rank_6_to_20_distribution": {
                str(rank): rank_counts[rank]
                for rank in range(TOP_K + 1, RECORDED_DEPTH + 1)
            },
            "partition_verified": True,
        }
        aggregate_mechanisms.update(mechanism_counts)
        aggregate_title_diagnostics.update(title_diagnostic_counts)
        aggregate_eligible += len(eligible)
        aggregate_preserved += len(preserved)

    aggregate_omissions = aggregate_eligible - aggregate_preserved
    expected_aggregate_title_diagnostics = {
        "selected_id_rank_6_to_20": 191,
        "absent_top20_same_query_term_set_in_top5_identifier": 9,
        "absent_top20_all_query_terms_in_one_top5_identifier": 129,
        "absent_top20_partial_query_term_overlap_in_top5_identifiers": 96,
        "absent_top20_no_query_term_overlap_in_top5_identifiers": 41,
    }
    require(
        {
            category: aggregate_title_diagnostics[category]
            for category in IDENTIFIER_TITLE_CATEGORIES
        }
        == expected_aggregate_title_diagnostics,
        "pooled title diagnostic target drift",
    )
    expected_aggregate_mechanisms = {
        "cutoff_competition_selected_id_rank_6_to_20": 191,
        "query_document_nonmatch_exhaustive_result_set_below_20": 14,
        "full_top20_selected_id_absent_rank_or_nonmatch_unresolved": 261,
    }
    require(
        {
            category: aggregate_mechanisms[category]
            for category in MECHANISM_CATEGORIES
        }
        == expected_aggregate_mechanisms,
        "pooled mechanism target drift",
    )
    require(
        aggregate_eligible == 1_792
        and aggregate_preserved == 1_326
        and aggregate_omissions == 466,
        "pooled preservation totals drift",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_bm25_omission_rank_and_identifier_taxonomy",
        "api_calls": 0,
        "network_used": False,
        "definition_and_support_boundary": {
            "audit_unit": "one frozen valid (question_id, model) selected ID that resolves in links.jsonl",
            "target_event": "selected ID omitted from the visible-label BM25 top five",
            "recorded_ranking_depth": RECORDED_DEPTH,
            "taxonomy_type": "deterministic mechanism partition with an explicit unresolved residual",
            "deterministic_mechanism_partition_supported": True,
            "complete_causal_bm25_scoring_attribution_supported": False,
            "support_boundary": (
                "The released testbed binds the passage collection by SHA-256 and releases "
                "the identifier inventory and top-20 link rankings. It does not release the "
                "indexed passage text or per-result BM25 scores for these bridge rows. A selected "
                "ID at ranks 6--20 is a verified cutoff-competition omission. When fewer than 20 "
                "documents satisfy the frozen OR query, the result set is exhaustive, so an absent "
                "selected ID is a verified query-document nonmatch. When all 20 slots are filled "
                "and the ID is absent, rank-below-20 and nonmatch cannot be distinguished. Identifier-"
                "title term relations are reported separately and are not scoring explanations."
            ),
            "available_inputs": [
                "exact selected identifiers",
                "deterministic visible labels",
                "top-20 BM25 identifier rankings",
                "complete 286,270-identifier passage inventory",
                "frozen BM25 query tokenizer and collection/database hashes",
            ],
            "unavailable_inputs_required_for_causal_scoring_attribution": [
                "indexed passage text",
                "per-result BM25 scores for bridge visible-label rows",
                "per-term document frequencies and term contributions",
            ],
        },
        "mechanism_category_meaning": {
            MECHANISM_CATEGORIES[0]: "selected ID matches the query but is removed by the top-five cutoff because its exact recorded rank is 6 through 20",
            MECHANISM_CATEGORIES[1]: "fewer than 20 documents match the OR query and the selected ID is absent, proving that its indexed passage matches none of the retained query terms",
            MECHANISM_CATEGORIES[2]: "all 20 returned slots are filled and the selected ID is absent; the released depth cannot distinguish rank below 20 from no query-term match",
        },
        "secondary_identifier_title_category_priority_and_meaning": {
            IDENTIFIER_TITLE_CATEGORIES[0]: "selected ID is present in the recorded ranking at an exact rank from 6 through 20",
            IDENTIFIER_TITLE_CATEGORIES[1]: "selected ID is absent top 20 and a different top-5 identifier title has the same frozen BM25 query-term set",
            IDENTIFIER_TITLE_CATEGORIES[2]: "no same-set case; one top-5 identifier title contains every query term plus at least one additional term",
            IDENTIFIER_TITLE_CATEGORIES[3]: "no earlier case; at least one query term occurs in a top-5 identifier title",
            IDENTIFIER_TITLE_CATEGORIES[4]: "no query term occurs in any top-5 identifier title",
        },
        "frozen_bm25_query_rule": {
            "token_pattern": "[a-z0-9]+",
            "minimum_token_length": 2,
            "deduplicate_tokens": True,
            "operator": "OR",
            "title_relation_uses": "unordered set of retained terms",
        },
        "inputs": {
            "members": source_metadata,
            "corpus_binding": corpus_binding,
            "execution_contract": execution_contract,
        },
        "models": models,
        "pooled_model_records": {
            "corpus_resolvable_selected_ids": aggregate_eligible,
            "top5_preserved": aggregate_preserved,
            "top5_omitted": aggregate_omissions,
            "mechanism_categories": {
                category: aggregate_mechanisms[category]
                for category in MECHANISM_CATEGORIES
            },
            "secondary_noncausal_identifier_title_categories": {
                category: aggregate_title_diagnostics[category]
                for category in IDENTIFIER_TITLE_CATEGORIES
            },
            "partition_verified": True,
        },
    }


def build_report_from_archive(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact not found: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist()]
            duplicates = [name for name, count in Counter(names).items() if count > 1]
            require(not duplicates, f"artifact has duplicate members: {duplicates}")

            def read(member: str) -> bytes:
                try:
                    return archive.read(member)
                except KeyError as exc:
                    raise AuditError(f"artifact missing member: {member}") from exc

            return build_report_from_members(read)
    except zipfile.BadZipFile as exc:
        raise AuditError(f"invalid artifact ZIP: {path}") from exc


def build_report_from_directory(path: Path) -> dict[str, Any]:
    root = path.resolve()
    require(root.is_dir(), f"artifact directory not found: {path}")

    def read(member: str) -> bytes:
        relative = Path(*PurePosixPath(member).parts)
        source = root / relative
        require(source.is_file() and not source.is_symlink(), f"artifact missing member: {member}")
        return source.read_bytes()

    return build_report_from_members(read)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path)
    source.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = (
            build_report_from_archive(args.artifact)
            if args.artifact is not None
            else build_report_from_directory(args.artifact_root)
        )
        payload = canonical_json_bytes(report)
        if args.output is None:
            sys.stdout.buffer.write(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
    except (AuditError, OSError) as exc:
        print(f"BM25 omission audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
