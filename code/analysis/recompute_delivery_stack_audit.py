#!/usr/bin/env python3
"""Replay selection-conditioned traced-passage delivery across five stacks.

This audit is deliberately standard-library only and makes no model, provider,
or network calls.  It joins the frozen bridge selector ledger to the primary
BM25 answer ledger and the four released retriever-sensitivity ledgers by the
exact ``(question_id, model)`` key.  The denominator is a correct selection of
the dataset-traced table-to-passage edge.  Delivery means that the same traced
passage occurs in the top five results for the frozen deterministic label of
the selected link.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 2
MODELS = ("claude", "openai", "gemini")
MODEL_LABELS = {
    "claude": "Claude",
    "openai": "GPT",
    "gemini": "Gemini",
}
EXPECTED_QUESTIONS = 600
EXPECTED_ROWS_PER_MODEL = 600
TOP_K = 5

BRIDGE_MEMBER = (
    "data/results/released/hybridqa_natural_v2/"
    "bridge_confirmatory/details.jsonl"
)
ANSWER_MEMBER = (
    "data/results/released/hybridqa_natural_v3/"
    "answer_confirmatory/details.jsonl"
)
ASSOCIATION_MEMBER = (
    "data/results/released/hybridqa_natural_v3/"
    "answer_confirmatory/association_rows.jsonl"
)
SENSITIVITY_ROOT = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1"
)
STACK_MEMBERS = {
    "bm25": ANSWER_MEMBER,
    "bge_m3": f"{SENSITIVITY_ROOT}/bge_m3/details.jsonl",
    "colbert_v2": f"{SENSITIVITY_ROOT}/colbert_v2/details.jsonl",
    "hybrid_rrf": f"{SENSITIVITY_ROOT}/hybrid_rrf/details.jsonl",
    "hybrid_rrf_bge_reranker_v2_m3": (
        f"{SENSITIVITY_ROOT}/hybrid_rrf_bge_reranker_v2_m3/details.jsonl"
    ),
}
STACK_LABELS = {
    "bm25": "BM25",
    "bge_m3": "BGE-M3",
    "colbert_v2": "ColBERTv2",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_rrf_bge_reranker_v2_m3": "Hybrid RRF + BGE reranker",
}
FINAL_STACK = "hybrid_rrf_bge_reranker_v2_m3"

EXPECTED_MEMBERS: dict[str, dict[str, Any]] = {
    BRIDGE_MEMBER: {
        "bytes": 4_991_925,
        "records": 1_800,
        "sha256": "abf0027a3f61aa338eae4a259b8751f7adecc5ea7f79a08f25ad11b179a79fc5",
    },
    ANSWER_MEMBER: {
        "bytes": 11_600_206,
        "records": 6_600,
        "sha256": "0e4877be9bb7983f407c632d0c8bc28cf73b4787f53cc330945db2b9d0cc4a72",
    },
    ASSOCIATION_MEMBER: {
        "bytes": 202_618,
        "records": 1_800,
        "sha256": "8432fafd71f2ac35d034f5858029fdc15969748e7cca4d355be4b6c837c6f8a1",
    },
    STACK_MEMBERS["bge_m3"]: {
        "bytes": 23_539_533,
        "records": 2_400,
        "sha256": "ce73c83e1ac0ef6b77f129b8f23060dcfb4fceb02c1070cc5454ff8889c317ec",
    },
    STACK_MEMBERS["colbert_v2"]: {
        "bytes": 21_114_400,
        "records": 2_400,
        "sha256": "6411ec10ab55373e2c745c652678f26529045fc33ea9dd5b5ca2205213bd7bc2",
    },
    STACK_MEMBERS["hybrid_rrf"]: {
        "bytes": 5_308_408,
        "records": 2_400,
        "sha256": "916fc87fb3865dc4a13e13997286028b0bc71a6a456f70648d86b683005af1d7",
    },
    STACK_MEMBERS[FINAL_STACK]: {
        "bytes": 6_894_412,
        "records": 2_400,
        "sha256": "0550dc6690e3e44191474ae489905565b0081e887fd5836ea76d01edd9969a81",
    },
}


class AuditError(RuntimeError):
    """Raised when frozen bytes, identities, or derivations do not verify."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_rows(data: bytes, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"{member}: not valid UTF-8") from exc
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{member}:{line_number}: invalid JSON") from exc
        require(
            isinstance(row, dict),
            f"{member}:{line_number}: JSONL record must be an object",
        )
        rows.append(row)
    return rows


def _member_reader(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> Callable[[str], bytes]:
    if callable(source):
        return source

    def read(member: str) -> bytes:
        try:
            value = source[member]
        except KeyError as exc:
            raise AuditError(f"missing source member: {member}") from exc
        require(isinstance(value, bytes), f"{member}: source value must be bytes")
        return value

    return read


def _validate_sources(
    source: Mapping[str, bytes] | Callable[[str], bytes],
    expected_members: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    read = _member_reader(source)
    data: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    require(
        set(expected_members) == set(EXPECTED_MEMBERS),
        "expected member set must equal the seven declared audit inputs",
    )
    for member in sorted(expected_members):
        try:
            payload = read(member)
        except (KeyError, FileNotFoundError) as exc:
            raise AuditError(f"missing source member: {member}") from exc
        expected = expected_members[member]
        digest = sha256_bytes(payload)
        record_count = sum(bool(line.strip()) for line in payload.splitlines())
        require(len(payload) == expected["bytes"], f"{member}: byte count drift")
        require(digest == expected["sha256"], f"{member}: SHA-256 drift")
        require(
            record_count == expected["records"],
            f"{member}: JSONL record count drift",
        )
        data[member] = payload
        metadata[member] = {
            "bytes": len(payload),
            "records": record_count,
            "sha256": digest,
        }
    return data, metadata


def _identity(row: Mapping[str, Any], context: str) -> tuple[str, str]:
    question_id = row.get("question_id")
    model = row.get("model")
    require(
        isinstance(question_id, str) and question_id,
        f"{context}: invalid question_id",
    )
    require(model in MODELS, f"{context}: unexpected model {model!r}")
    return question_id, model


def _index_rows(
    rows: list[dict[str, Any]], context: str
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        identity = _identity(row, f"{context}:{row_number}")
        require(identity not in index, f"{context}: duplicate identity {identity}")
        index[identity] = row
    return index


def _load_bridge(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    require(len(rows) == 1_800, "bridge ledger must contain 1,800 rows")
    index = _index_rows(rows, "bridge")
    counts = Counter(model for _, model in index)
    require(
        counts == Counter({model: EXPECTED_ROWS_PER_MODEL for model in MODELS}),
        f"bridge model counts changed: {dict(counts)}",
    )
    require(
        len({question_id for question_id, _ in index}) == EXPECTED_QUESTIONS,
        "bridge question count changed",
    )
    for identity, row in index.items():
        require(row.get("status") == "succeeded", f"bridge {identity}: status drift")
        require(type(row.get("valid_json")) is bool, f"bridge {identity}: invalid valid_json")
        selection = row.get("selection_correct")
        require(
            selection is None or type(selection) is bool,
            f"bridge {identity}: invalid selection_correct",
        )
        gold_link = row.get("gold_link")
        selected_link = row.get("selected_link")
        query = row.get("primary_predicted_query")
        require(
            isinstance(gold_link, str) and gold_link.startswith("/wiki/"),
            f"bridge {identity}: invalid gold_link",
        )
        require(
            selected_link is None
            or (isinstance(selected_link, str) and selected_link.startswith("/wiki/")),
            f"bridge {identity}: invalid selected_link",
        )
        if selection is None:
            require(
                query is None,
                f"bridge {identity}: invalid selector row unexpectedly has a query",
            )
        else:
            require(
                isinstance(query, str) and query.strip(),
                f"bridge {identity}: invalid primary_predicted_query",
            )
        if selection is True:
            require(
                row["valid_json"] is True and selected_link == gold_link,
                f"bridge {identity}: correct-selection fields disagree",
            )
        if selection is None:
            require(
                row["valid_json"] is False,
                f"bridge {identity}: unknown selection must be invalid JSON",
            )
    return index


def _load_bm25(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    require(len(rows) == 6_600, "answer ledger must contain 6,600 rows")
    conditions = Counter(row.get("condition") for row in rows)
    require(
        conditions
        == Counter(
            {
                "question_only": 1_800,
                "predicted_fact": 1_800,
                "query_rewrite": 1_800,
                "no_kb": 600,
                "oracle_passage": 600,
            }
        ),
        f"answer condition counts changed: {dict(conditions)}",
    )
    predicted_rows = [row for row in rows if row.get("condition") == "predicted_fact"]
    return _index_rows(predicted_rows, "bm25 predicted_fact")


def _load_sensitivity(
    rows: list[dict[str, Any]], stack: str
) -> dict[tuple[str, str], dict[str, Any]]:
    require(len(rows) == 2_400, f"{stack}: expected 2,400 rows")
    conditions = Counter(row.get("condition") for row in rows)
    require(
        conditions == Counter({"predicted_fact": 1_800, "question_only": 600}),
        f"{stack}: condition counts changed: {dict(conditions)}",
    )
    question_rows = [row for row in rows if row.get("condition") == "question_only"]
    require(
        all(row.get("model") is None for row in question_rows),
        f"{stack}: question-only rows must be model-shared",
    )
    predicted_rows = [row for row in rows if row.get("condition") == "predicted_fact"]
    return _index_rows(predicted_rows, f"{stack} predicted_fact")


def _ranking_links(row: Mapping[str, Any], stack: str, identity: tuple[str, str]) -> list[str]:
    ranking = row.get("ranking")
    require(isinstance(ranking, list), f"{stack} {identity}: ranking must be a list")
    if row.get("status") != "succeeded":
        require(not ranking, f"{stack} {identity}: failed ranking must be empty")
        return []
    if stack == "bm25":
        require(
            all(isinstance(link, str) and link.startswith("/wiki/") for link in ranking),
            f"{stack} {identity}: invalid ranking link",
        )
        return list(ranking)
    links: list[str] = []
    for expected_rank, result in enumerate(ranking, 1):
        require(isinstance(result, dict), f"{stack} {identity}: result must be an object")
        require(
            result.get("rank") == expected_rank,
            f"{stack} {identity}: ranking order drift at {expected_rank}",
        )
        link = result.get("link")
        require(
            isinstance(link, str) and link.startswith("/wiki/"),
            f"{stack} {identity}: invalid result link",
        )
        links.append(link)
    return links


def _rate_record(denominator: int, delivered: int) -> dict[str, Any]:
    require(denominator > 0, "conditional denominator must be positive")
    require(0 <= delivered <= denominator, "delivery count outside denominator")
    missed = denominator - delivered
    return {
        "correct_selection_denominator": denominator,
        "delivered_at_5": delivered,
        "delivery_rate": delivered / denominator,
        "selection_without_delivery": missed,
        "selection_without_delivery_rate": missed / denominator,
    }


def _transition_record(values: Counter[tuple[bool, bool]]) -> dict[str, int]:
    return {
        "bm25_miss_final_miss": values[(False, False)],
        "bm25_miss_final_hit": values[(False, True)],
        "bm25_hit_final_miss": values[(True, False)],
        "bm25_hit_final_hit": values[(True, True)],
    }


def _recorded_rank(
    links: list[str], target: str, stack: str, identity: tuple[str, str]
) -> int | None:
    """Return the one-based rank in a released list, or ``None`` if censored."""

    matches = [rank for rank, link in enumerate(links, 1) if link == target]
    require(
        len(matches) <= 1,
        f"{stack} {identity}: traced link occurs more than once in ranking",
    )
    return matches[0] if matches else None


def _final_stack_omission_tail(
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
    stack_rows: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    delivery: Mapping[str, Mapping[tuple[str, str], bool]],
) -> dict[str, Any]:
    """Describe what the two released top-20 lists show below the k=5 cutoff."""

    records: list[dict[str, Any]] = []
    for identity in sorted(bridge):
        bridge_row = bridge[identity]
        if (
            bridge_row.get("selection_correct") is not True
            or delivery[FINAL_STACK][identity]
        ):
            continue

        gold_link = bridge_row["gold_link"]
        rrf_links = _ranking_links(
            stack_rows["hybrid_rrf"][identity], "hybrid_rrf", identity
        )
        final_links = _ranking_links(
            stack_rows[FINAL_STACK][identity], FINAL_STACK, identity
        )
        require(
            len(rrf_links) == 20 and len(final_links) == 20,
            f"final-stack tail {identity}: released rankings must each contain 20 results",
        )
        rrf_rank = _recorded_rank(rrf_links, gold_link, "hybrid_rrf", identity)
        final_rank = _recorded_rank(final_links, gold_link, FINAL_STACK, identity)
        require(
            final_rank is None or final_rank > TOP_K,
            f"final-stack tail {identity}: top-five delivery disagrees with rank",
        )
        if rrf_rank is None and final_rank is None:
            category = "outside_both_recorded_top_20"
        else:
            category = "located_below_final_top_5_cutoff"
        records.append(
            {
                "question_id": identity[0],
                "model": identity[1],
                "gold_link": gold_link,
                "hybrid_rrf_recorded_rank": rrf_rank,
                "final_stack_recorded_rank": final_rank,
                "category": category,
            }
        )

    model_counts = Counter(record["model"] for record in records)
    transition_counts = Counter(
        (
            record["hybrid_rrf_recorded_rank"],
            record["final_stack_recorded_rank"],
        )
        for record in records
        if record["category"] == "located_below_final_top_5_cutoff"
    )
    outside_both = sum(
        record["category"] == "outside_both_recorded_top_20"
        for record in records
    )

    return {
        "scope": (
            "correct-selection records where the final hybrid-RRF-plus-BGE-"
            "reranker stack does not deliver the dataset-traced passage at k=5"
        ),
        "recorded_depth": {
            "hybrid_rrf": 20,
            FINAL_STACK: 20,
        },
        "rank_censoring_boundary": (
            "a null recorded rank means only that the traced passage is absent "
            "from that stack's released top-20 list; it is neither a verified "
            "nonmatch nor evidence that the passage was absent from any reranker "
            "candidate pool"
        ),
        "summary": {
            "selection_without_delivery_records": len(records),
            "unique_question_ids": len(
                {record["question_id"] for record in records}
            ),
            "model_records": {model: model_counts[model] for model in MODELS},
            "located_below_final_top_5_cutoff": len(records) - outside_both,
            "outside_both_recorded_top_20": outside_both,
            "rank_transitions": [
                {
                    "hybrid_rrf_recorded_rank": rrf_rank,
                    "final_stack_recorded_rank": final_rank,
                    "records": count,
                }
                for (rrf_rank, final_rank), count in sorted(
                    transition_counts.items()
                )
            ],
        },
        "records": records,
    }


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
    *,
    expected_members: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the deterministic report from exact member bytes."""

    expected = EXPECTED_MEMBERS if expected_members is None else expected_members
    payloads, source_metadata = _validate_sources(source, expected)
    bridge = _load_bridge(_jsonl_rows(payloads[BRIDGE_MEMBER], BRIDGE_MEMBER))
    stack_rows: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        "bm25": _load_bm25(_jsonl_rows(payloads[ANSWER_MEMBER], ANSWER_MEMBER))
    }
    for stack in STACK_MEMBERS:
        if stack == "bm25":
            continue
        member = STACK_MEMBERS[stack]
        stack_rows[stack] = _load_sensitivity(
            _jsonl_rows(payloads[member], member), stack
        )

    bridge_keys = set(bridge)
    for stack, index in stack_rows.items():
        require(
            set(index) == bridge_keys,
            f"{stack}: question/model key set differs from bridge ledger",
        )
        for identity, row in index.items():
            bridge_row = bridge[identity]
            require(
                row.get("gold_link") == bridge_row["gold_link"],
                f"{stack} {identity}: gold_link join mismatch",
            )
            if stack == "bm25":
                plan = row.get("plan")
                require(isinstance(plan, dict), f"{stack} {identity}: missing plan")
                query = plan.get("retrieval_query")
            else:
                query = row.get("text")
            require(
                query == bridge_row.get("primary_predicted_query"),
                f"{stack} {identity}: frozen query join mismatch",
            )

    delivery: dict[str, dict[tuple[str, str], bool]] = {}
    for stack, index in stack_rows.items():
        stack_delivery: dict[tuple[str, str], bool] = {}
        for identity, row in index.items():
            links = _ranking_links(row, stack, identity)
            stack_delivery[identity] = bridge[identity]["gold_link"] in links[:TOP_K]
        delivery[stack] = stack_delivery

    association_rows = _jsonl_rows(
        payloads[ASSOCIATION_MEMBER], ASSOCIATION_MEMBER
    )
    require(
        len(association_rows) == 1_800,
        "association join must contain 1,800 rows",
    )
    associations = _index_rows(association_rows, "association rows")
    require(
        set(associations) == bridge_keys,
        "association key set differs from bridge ledger",
    )
    for identity, association in associations.items():
        selection = bridge[identity].get("selection_correct") is True
        require(
            association.get("selected_equals_trace") is selection,
            f"association {identity}: selection replay mismatch",
        )
        require(
            association.get("predicted_retrieval_hit")
            is delivery["bm25"][identity],
            f"association {identity}: BM25 delivery replay mismatch",
        )

    selected_counts = {
        model: sum(
            identity[1] == model and row.get("selection_correct") is True
            for identity, row in bridge.items()
        )
        for model in MODELS
    }
    unknown_selection = {
        model: sum(
            identity[1] == model and row.get("selection_correct") is None
            for identity, row in bridge.items()
        )
        for model in MODELS
    }
    results: dict[str, Any] = {}
    for stack in STACK_MEMBERS:
        models: dict[str, Any] = {}
        pooled_denominator = 0
        pooled_delivered = 0
        for model in MODELS:
            identities = [
                identity
                for identity, row in bridge.items()
                if identity[1] == model and row.get("selection_correct") is True
            ]
            require(
                all(stack_rows[stack][identity].get("status") == "succeeded" for identity in identities),
                f"{stack}/{model}: a correct-selection retrieval row did not succeed",
            )
            delivered_count = sum(delivery[stack][identity] for identity in identities)
            models[model] = {
                "display_name": MODEL_LABELS[model],
                **_rate_record(len(identities), delivered_count),
            }
            pooled_denominator += len(identities)
            pooled_delivered += delivered_count
        results[stack] = {
            "display_name": STACK_LABELS[stack],
            "models": models,
            "pooled_descriptive": _rate_record(pooled_denominator, pooled_delivered),
        }

    transitions: dict[str, Any] = {"models": {}}
    pooled_transitions: Counter[tuple[bool, bool]] = Counter()
    for model in MODELS:
        model_transitions: Counter[tuple[bool, bool]] = Counter()
        for identity, row in bridge.items():
            if identity[1] != model or row.get("selection_correct") is not True:
                continue
            pair = (delivery["bm25"][identity], delivery[FINAL_STACK][identity])
            model_transitions[pair] += 1
            pooled_transitions[pair] += 1
        transitions["models"][model] = {
            "display_name": MODEL_LABELS[model],
            **_transition_record(model_transitions),
        }
    transitions["pooled_descriptive"] = _transition_record(pooled_transitions)

    final_stack_omission_tail = _final_stack_omission_tail(
        bridge, stack_rows, delivery
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_replay",
        "api_calls": 0,
        "definition": {
            "audit_unit": "one frozen (question_id, model) bridge-selection record",
            "join_key": ["question_id", "model"],
            "selection_denominator": (
                "bridge selection_correct is the JSON boolean true: the selector "
                "chose the dataset-traced table-to-passage link"
            ),
            "delivery": (
                "the bridge gold_link occurs in the first five results for the "
                "frozen primary_predicted_query under the named retrieval stack"
            ),
            "unknown_selection_policy": (
                "invalid/unknown selector outputs are outside the conditional "
                "correct-selection denominator, not counted as correct selections"
            ),
            "failed_retrieval_policy": (
                "a failed or invalid retrieval would be a delivery miss; all "
                "correct-selection retrieval rows in these sources succeeded"
            ),
            "reference_scope": (
                "delivery concerns the one dataset-traced passage, not all "
                "semantically sufficient evidence"
            ),
            "analysis_scope": (
                "retrospective descriptive accounting on the fixed enriched cohort; "
                "not a causal or deployment-generalization estimate"
            ),
            "pooled_scope": (
                "pooled counts are descriptive only because the same 600 question "
                "IDs repeat across model families"
            ),
            "top_k": TOP_K,
        },
        "inputs": {
            "container_policy": (
                "bind exact scientific member bytes; do not bind the mutable outer "
                "reviewer-artifact ZIP hash"
            ),
            "members": source_metadata,
        },
        "join_verification": {
            "bridge_rows": len(bridge),
            "unique_question_model_keys": len(bridge_keys),
            "unique_question_ids": len({question_id for question_id, _ in bridge_keys}),
            "rows_per_model": {model: EXPECTED_ROWS_PER_MODEL for model in MODELS},
            "correct_selection_denominators": selected_counts,
            "unknown_selection_rows": unknown_selection,
            "predicted_query_rows_per_stack": {
                stack: len(index) for stack, index in stack_rows.items()
            },
            "exact_question_model_keyset_match": True,
            "exact_gold_link_match": True,
            "exact_frozen_query_match": True,
            "bm25_association_join_match": True,
            "all_correct_selection_retrieval_rows_succeeded": True,
        },
        "results": results,
        "paired_bm25_to_hybrid_rerank": transitions,
        "final_stack_top5_omission_tail": final_stack_omission_tail,
    }


def build_report_from_archive(artifact: Path) -> dict[str, Any]:
    require(artifact.is_file(), f"artifact not found: {artifact}")
    try:
        with zipfile.ZipFile(artifact) as archive:
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
        raise AuditError(f"invalid artifact ZIP: {artifact}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="reviewer artifact ZIP containing the frozen released ledgers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the canonical JSON report (stdout if omitted)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report_from_archive(args.artifact)
        payload = canonical_json_bytes(report)
        if args.output is None:
            sys.stdout.buffer.write(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
    except (AuditError, OSError) as exc:
        print(f"delivery-stack audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
