#!/usr/bin/env python3
"""Replay mismatch-specific P-by-H cells for every frozen search stack.

The audit uses the exact released selector and ranking ledgers.  Its population
is the 329 hosted-model rows on which the selector and dataset trace resolve to
distinct passage PIDs under the frozen identifier resolver.  The replay binds
    the resolver, verifies zero raw-identifier alias collapses, and stops if
any returned ID used through rank 20 does not resolve.  For each of the five
recorded visible-label executors and each frozen cutoff in {1, 5, 20}, it
reports the joint counts of selected-object preservation (P) and traced-passage
delivery (H), by hosted-model family and pooled.

The same question can contribute one row per hosted-model family, so the report
also binds question-cluster multiplicity and cell-level unique-question counts.
It verifies the relevant frozen prompt and configuration bindings. The prompt
fixed ``selected_link`` as one exact shown Wikipedia key whose page passage was
needed; the configs fixed its decoded-label query, executor roster, output
depth, and reported cutoffs. Those inputs did not define the later top-k
selected-object profile or the mismatch-specific P/H join, so the report keeps
the retrospective analysis boundary explicit. No model, retrieval, API,
network, credential, random, or clock input is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping
import zipfile


_IDENTITY_PATH = Path(__file__).with_name("recompute_identity_handoff.py")
_IDENTITY_SPEC = importlib.util.spec_from_file_location(
    "mismatch_stack_cutoffs_identity_handoff", _IDENTITY_PATH
)
if _IDENTITY_SPEC is None or _IDENTITY_SPEC.loader is None:
    raise RuntimeError(f"cannot load exact-identity replay at {_IDENTITY_PATH}")
identity = importlib.util.module_from_spec(_IDENTITY_SPEC)
_IDENTITY_SPEC.loader.exec_module(identity)


SCHEMA_VERSION = 1
CUTOFFS = (1, 5, 20)
STACKS = (
    "visible_label_bm25",
    "visible_label_bge_m3",
    "visible_label_colbert_v2",
    "visible_label_hybrid_rrf",
    "visible_label_hybrid_reranker",
)
STACK_LABELS = {
    "visible_label_bm25": "Body-only BM25",
    "visible_label_bge_m3": "BGE-M3",
    "visible_label_colbert_v2": "ColBERTv2",
    "visible_label_hybrid_rrf": "Hybrid RRF",
    "visible_label_hybrid_reranker": "Hybrid RRF + BGE reranker",
}

CONFIG_V2_MEMBER = "code/configs/experiments/hybridqa_natural_v2.yaml"
BRIDGE_IMPLEMENTATION_MEMBER = "code/src/ccr/retrieval/hybridqa_bridge.py"
BRIDGE_RUN_MANIFEST_MEMBER = (
    "data/results/released/hybridqa_natural_v2/"
    "bridge_confirmatory/run_manifest.json"
)
SENSITIVITY_CONFIG_MEMBER = (
    "code/configs/experiments/hybridqa_retriever_sensitivity_v1.yaml"
)
SENSITIVITY_PROVENANCE_MEMBER = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1/provenance.json"
)

RANKING_MEMBERS = (
    identity.BRIDGE_MEMBER,
    identity.STACK_MEMBERS["visible_label_bge_m3"],
    identity.STACK_MEMBERS["visible_label_colbert_v2"],
    identity.STACK_MEMBERS["visible_label_hybrid_rrf"],
    identity.STACK_MEMBERS["visible_label_hybrid_reranker"],
    identity.LINKS_MEMBER,
)
RELEVANT_MEMBERS = (
    *RANKING_MEMBERS,
    CONFIG_V2_MEMBER,
    BRIDGE_IMPLEMENTATION_MEMBER,
    BRIDGE_RUN_MANIFEST_MEMBER,
    SENSITIVITY_CONFIG_MEMBER,
    SENSITIVITY_PROVENANCE_MEMBER,
)

EXTRA_MEMBER_BINDINGS: dict[str, dict[str, Any]] = {
    CONFIG_V2_MEMBER: {
        "bytes": 3_437,
        "records": 90,
        "sha256": "378d0257f568dc6306c6455b528ed56ae76208e4af9b726bbadc4e51ddf358d0",
    },
    BRIDGE_IMPLEMENTATION_MEMBER: {
        "bytes": 2_729,
        "records": 55,
        "sha256": "5e0f0fa578a741e29e8f91dd5f683ba6ba1afa96f8f3beaf230ec0729e4c879b",
    },
    BRIDGE_RUN_MANIFEST_MEMBER: {
        "bytes": 743,
        "records": 11,
        "sha256": "e34355963f3142f9d9ae2f3d162258d6b86cd757ffcea1a50b442eeb0f154a4c",
    },
    SENSITIVITY_CONFIG_MEMBER: {
        "bytes": 2_534,
        "records": 76,
        "sha256": "5673d2cca5fef9a9d49515ff40175c8b7464b93ab948d5b346afc2a106574cb7",
    },
    SENSITIVITY_PROVENANCE_MEMBER: {
        "bytes": 2_761,
        "records": 72,
        "sha256": "282deccdf7f4dd318fab6051af2956ca6a915d5fa1c4deffb7589f51284864ec",
    },
}

# Fixed targets for (records, P1H1, P1H0, P0H1, P0H0).
EXPECTED_CELLS = {
    "visible_label_bm25": {
        1: {
            "claude": (143, 0, 83, 2, 58),
            "openai": (85, 0, 54, 2, 29),
            "gemini": (101, 0, 67, 1, 33),
            "pooled": (329, 0, 204, 5, 120),
        },
        5: {
            "claude": (143, 15, 88, 2, 38),
            "openai": (85, 13, 54, 0, 18),
            "gemini": (101, 14, 68, 1, 18),
            "pooled": (329, 42, 210, 3, 74),
        },
        20: {
            "claude": (143, 22, 95, 0, 26),
            "openai": (85, 20, 50, 0, 15),
            "gemini": (101, 19, 69, 0, 13),
            "pooled": (329, 61, 214, 0, 54),
        },
    },
    "visible_label_bge_m3": {
        1: {
            "claude": (143, 0, 115, 0, 28),
            "openai": (85, 0, 72, 0, 13),
            "gemini": (101, 0, 82, 0, 19),
            "pooled": (329, 0, 269, 0, 60),
        },
        5: {
            "claude": (143, 11, 125, 0, 7),
            "openai": (85, 10, 73, 0, 2),
            "gemini": (101, 10, 86, 0, 5),
            "pooled": (329, 31, 284, 0, 14),
        },
        20: {
            "claude": (143, 16, 122, 0, 5),
            "openai": (85, 16, 68, 0, 1),
            "gemini": (101, 14, 86, 0, 1),
            "pooled": (329, 46, 276, 0, 7),
        },
    },
    "visible_label_colbert_v2": {
        1: {
            "claude": (143, 0, 127, 0, 16),
            "openai": (85, 0, 82, 0, 3),
            "gemini": (101, 0, 94, 0, 7),
            "pooled": (329, 0, 303, 0, 26),
        },
        5: {
            "claude": (143, 13, 126, 0, 4),
            "openai": (85, 13, 72, 0, 0),
            "gemini": (101, 13, 87, 0, 1),
            "pooled": (329, 39, 285, 0, 5),
        },
        20: {
            "claude": (143, 25, 116, 0, 2),
            "openai": (85, 21, 64, 0, 0),
            "gemini": (101, 20, 81, 0, 0),
            "pooled": (329, 66, 261, 0, 2),
        },
    },
    "visible_label_hybrid_rrf": {
        1: {
            "claude": (143, 0, 97, 1, 45),
            "openai": (85, 0, 67, 0, 18),
            "gemini": (101, 0, 80, 0, 21),
            "pooled": (329, 0, 244, 1, 84),
        },
        5: {
            "claude": (143, 12, 113, 0, 18),
            "openai": (85, 11, 64, 0, 10),
            "gemini": (101, 11, 81, 0, 9),
            "pooled": (329, 34, 258, 0, 37),
        },
        20: {
            "claude": (143, 20, 116, 0, 7),
            "openai": (85, 20, 61, 0, 4),
            "gemini": (101, 18, 81, 0, 2),
            "pooled": (329, 58, 258, 0, 13),
        },
    },
    "visible_label_hybrid_reranker": {
        1: {
            "claude": (143, 0, 129, 0, 14),
            "openai": (85, 0, 80, 0, 5),
            "gemini": (101, 0, 94, 0, 7),
            "pooled": (329, 0, 303, 0, 26),
        },
        5: {
            "claude": (143, 14, 127, 0, 2),
            "openai": (85, 17, 68, 0, 0),
            "gemini": (101, 17, 83, 0, 1),
            "pooled": (329, 48, 278, 0, 3),
        },
        20: {
            "claude": (143, 26, 116, 0, 1),
            "openai": (85, 24, 61, 0, 0),
            "gemini": (101, 23, 78, 0, 0),
            "pooled": (329, 73, 255, 0, 1),
        },
    },
}

EXPECTED = {
    "records": 329,
    "records_by_model": {"claude": 143, "openai": 85, "gemini": 101},
    "raw_identifier_differences": 331,
    "raw_differences_with_resolvable_selected_id": 329,
    "raw_differences_with_unresolvable_selected_id": 2,
    "canonical_object_distinct_records": 329,
    "raw_differences_collapsing_to_same_canonical_pid": 0,
    "canonical_pair_mapping_sha256": (
        "ca2c78d3a92fb23dc7974cf43565c675c4a2094495569cb4e508c120c24ee496"
    ),
    "unique_questions": 160,
    "question_multiplicity": {"exactly_1": 55, "exactly_2": 41, "exactly_3": 64},
    "v2_pre_result_config_sha256": (
        "ddea892f58918eb33601c014a379980ab0a75f997dc04afd424ec8776643cae6"
    ),
    "v2_pre_result_git_commit": "f3aeb6c",
    "sensitivity_executed_config_sha256": (
        "dd9e942ae411c60e916ac26d17e5f2357a4b270345d6524da91f2883779ba0a7"
    ),
}

# Independent fixed targets for the complete five-executor pairwise grid.
# Headline gaps are oriented final hybrid reranker minus body-only BM25.
EXPECTED_PAIRWISE_SUMMARY = {
    1: {
        "sign_agreement": 0,
        "sign_disagreement": 7,
        "tie_involved": 3,
        "both_metrics_tied": 1,
        "preservation_tie_only": 0,
        "trace_delivery_tie_only": 2,
        "headline_preservation_gap_count": 99,
        "headline_trace_delivery_gap_count": -5,
    },
    5: {
        "sign_agreement": 6,
        "sign_disagreement": 4,
        "tie_involved": 0,
        "both_metrics_tied": 0,
        "preservation_tie_only": 0,
        "trace_delivery_tie_only": 0,
        "headline_preservation_gap_count": 74,
        "headline_trace_delivery_gap_count": 3,
    },
    20: {
        "sign_agreement": 7,
        "sign_disagreement": 3,
        "tie_involved": 0,
        "both_metrics_tied": 0,
        "preservation_tie_only": 0,
        "trace_delivery_tie_only": 0,
        "headline_preservation_gap_count": 53,
        "headline_trace_delivery_gap_count": 12,
    },
}

EXPECTED_PAIRED_HEADLINE_AT_5 = {
    "selected_object": (252, 0, 74, 3),
    "dataset_trace": (34, 11, 14, 270),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise identity.AuditError(message)


def _reader(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> Callable[[str], bytes]:
    if callable(source):
        return source

    def read(member: str) -> bytes:
        try:
            payload = source[member]
        except KeyError as exc:
            raise identity.AuditError(f"missing source member: {member}") from exc
        require(isinstance(payload, bytes), f"{member}: source must be bytes")
        return payload

    return read


def _expected_binding(member: str) -> Mapping[str, Any]:
    if member in EXTRA_MEMBER_BINDINGS:
        return EXTRA_MEMBER_BINDINGS[member]
    return identity.EXPECTED_MEMBERS[member]


def _validate_sources(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    read = _reader(source)
    payloads: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for member in sorted(RELEVANT_MEMBERS):
        try:
            payload = read(member)
        except (KeyError, FileNotFoundError) as exc:
            raise identity.AuditError(f"missing source member: {member}") from exc
        expected = _expected_binding(member)
        digest = hashlib.sha256(payload).hexdigest()
        records = sum(bool(line.strip()) for line in payload.splitlines())
        require(len(payload) == expected["bytes"], f"{member}: byte count drift")
        require(digest == expected["sha256"], f"{member}: SHA-256 drift")
        require(records == expected["records"], f"{member}: line count drift")
        payloads[member] = payload
        metadata[member] = {
            "bytes": len(payload),
            "records": records,
            "sha256": digest,
        }
    return payloads, metadata


def _json_object(payload: bytes, member: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise identity.AuditError(f"{member}: invalid JSON") from exc
    require(isinstance(value, dict), f"{member}: root must be an object")
    return value


def _replace_exactly_once(payload: bytes, old: bytes, new: bytes, label: str) -> bytes:
    require(payload.count(old) == 1, f"{label}: expected exactly one portable-path match")
    return payload.replace(old, new)


def _protocol_binding_report(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    v2_public = payloads[CONFIG_V2_MEMBER]
    v2_pre_result = _replace_exactly_once(
        v2_public,
        b"implementation: code/src/ccr/retrieval/hybridqa_bridge.py",
        b"implementation: ccr_retrieval/hybridqa_bridge.py",
        "v2 config",
    )
    v2_digest = hashlib.sha256(v2_pre_result).hexdigest()
    bridge_manifest = _json_object(
        payloads[BRIDGE_RUN_MANIFEST_MEMBER], BRIDGE_RUN_MANIFEST_MEMBER
    )
    require(
        v2_digest == EXPECTED["v2_pre_result_config_sha256"],
        "v2 pre-result config reconstruction drifted",
    )
    require(
        bridge_manifest.get("experiment_config_sha256") == v2_digest,
        "bridge run manifest does not bind the reconstructed v2 config",
    )
    require(
        bridge_manifest.get("pre_result_git_commit")
        == EXPECTED["v2_pre_result_git_commit"],
        "bridge pre-result commit declaration drifted",
    )

    sensitivity_public = payloads[SENSITIVITY_CONFIG_MEMBER]
    sensitivity_executed = sensitivity_public
    execution_root = b"/" + b"content" + b"/"
    for old, new in (
        (
            b"cohort: data/released/hybridqa_natural_v3/confirmatory_cohort.json",
            b"cohort: " + execution_root
            + b"ccr_retriever_payload/data/released/"
            b"hybridqa_natural_v3/confirmatory_cohort.json",
        ),
        (
            b"bridge_details: data/results/released/hybridqa_natural_v2/"
            b"bridge_confirmatory/details.jsonl",
            b"bridge_details: " + execution_root
            + b"ccr_retriever_payload/results/released/"
            b"hybridqa_natural_v2/bridge_confirmatory/details.jsonl",
        ),
        (
            b"bm25_db: external_inputs/hybridqa_bm25.sqlite",
            b"bm25_db: " + execution_root
            + b"ccr_frozen_inputs/hybridqa_bm25.sqlite",
        ),
        (
            b"bm25_manifest: external_inputs/hybridqa_bm25.manifest.json",
            b"bm25_manifest: " + execution_root
            + b"ccr_frozen_inputs/"
            b"hybridqa_bm25.manifest.json",
        ),
    ):
        sensitivity_executed = _replace_exactly_once(
            sensitivity_executed, old, new, "sensitivity config"
        )
    sensitivity_digest = hashlib.sha256(sensitivity_executed).hexdigest()
    provenance = _json_object(
        payloads[SENSITIVITY_PROVENANCE_MEMBER], SENSITIVITY_PROVENANCE_MEMBER
    )
    require(
        sensitivity_digest == EXPECTED["sensitivity_executed_config_sha256"],
        "executed sensitivity config reconstruction drifted",
    )
    require(
        provenance.get("source_frozen_config_sha256") == sensitivity_digest,
        "sensitivity provenance does not bind the executed config",
    )
    require(
        provenance.get("status") == "independently_verified_and_promoted",
        "sensitivity provenance status drifted",
    )

    v2_text = v2_public.decode("utf-8")
    bridge_implementation_text = payloads[BRIDGE_IMPLEMENTATION_MEMBER].decode(
        "utf-8"
    )
    sensitivity_text = sensitivity_public.decode("utf-8")
    for fragment in (
        "status: frozen_after_development_before_confirmatory_model_outcomes",
        "  - bm25\n",
        "  - hybrid_rrf_bge_reranker_v2_m3\n",
        "  primary_k: 5\n",
        "  report_k: [1, 5, 20]\n",
        "    retrieve_k: 20\n",
    ):
        require(fragment in v2_text, f"v2 protocol declaration missing {fragment!r}")
    for fragment in (
        "status: frozen_before_neural_retriever_execution",
        "  output_k: 20\n",
        "  report_k:\n  - 1\n  - 5\n  - 20\n",
        "  hybrid_rrf_bge_reranker_v2_m3: "
        "bge_v2_m3_rerank_of_top_100_hybrid_rrf_candidates\n",
        "  multiplicity_family: all_4_retrievers_x_3_models_x_3_recall_cutoffs\n",
        "  selective_omission_forbidden: true\n",
    ):
        require(
            fragment in sensitivity_text,
            f"sensitivity protocol declaration missing {fragment!r}",
        )
    for fragment in (
        '"required": ["selected_link", "retrieval_fact"]',
        "table-linked Wikipedia page whose passage is needed to answer the question",
        "Choose the single shown link that best bridges the table to the external",
        "Set selected_link to the exact /wiki/...",
        "Set retrieval_fact to a concise human-readable name for that same",
        "entity, suitable as a search query.",
    ):
        require(
            fragment in bridge_implementation_text,
            f"bridge prompt declaration missing {fragment!r}",
        )
    for forbidden in (
        "selected-object preservation",
        "mismatch-specific",
        "P_by_H",
        "P-by-H",
    ):
        require(
            forbidden not in v2_text
            and forbidden not in bridge_implementation_text
            and forbidden not in sensitivity_text,
            f"unexpected frozen-protocol P/H declaration {forbidden!r}",
        )
    for forbidden in (
        "hard selection",
        "hard_selection",
        "must survive",
        "must appear",
        "preserve_at",
        "retention obligation",
    ):
        require(
            forbidden not in v2_text.lower()
            and forbidden not in bridge_implementation_text.lower()
            and forbidden not in sensitivity_text.lower(),
            f"unexpected frozen-protocol preservation obligation {forbidden!r}",
        )

    return {
        "verified_inputs": {
            "bridge_protocol": {
                "portable_member_sha256": hashlib.sha256(v2_public).hexdigest(),
                "selector_implementation_member": BRIDGE_IMPLEMENTATION_MEMBER,
                "selector_implementation_sha256": hashlib.sha256(
                    payloads[BRIDGE_IMPLEMENTATION_MEMBER]
                ).hexdigest(),
                "pre_result_config_sha256": v2_digest,
                "run_manifest_config_sha256_matches": True,
                "producer_recorded_pre_result_git_commit": bridge_manifest[
                    "pre_result_git_commit"
                ],
                "declared_status": (
                    "frozen_after_development_before_confirmatory_model_outcomes"
                ),
            },
            "retriever_sensitivity_protocol": {
                "portable_member_sha256": hashlib.sha256(
                    sensitivity_public
                ).hexdigest(),
                "executed_config_sha256": sensitivity_digest,
                "promoted_provenance_config_sha256_matches": True,
                "promoted_provenance_status": provenance["status"],
                "declared_status": "frozen_before_neural_retriever_execution",
            },
        },
        "result_scope": {
            "executor_roster_declared_in_bridge_protocol": list(STACKS),
            "bridge_primary_cutoff_declared": 5,
            "bridge_report_cutoffs_declared": list(CUTOFFS),
            "sensitivity_output_depth_declared": 20,
            "sensitivity_report_cutoffs_declared": list(CUTOFFS),
            "sensitivity_comparison_family_declared": (
                "all four neural/hybrid retrievers x three hosted-model families "
                "x three traced-passage-recall cutoffs, with BM25 recorded as the "
                "baseline in the same release"
            ),
            "rankings_precede_this_join_by_data_dependency": True,
            "selected_link_semantics_declared_before_selector_calls": True,
            "selected_link_declared_as_single_exact_shown_wikipedia_key": True,
            "selected_link_declared_as_page_whose_passage_is_needed": True,
            "decoded_selected_link_declared_as_primary_predicted_query": True,
            "selected_link_top_k_obligation_in_frozen_protocol": False,
            "mismatch_p_h_analysis_in_frozen_protocol": False,
            "profile_application": "retrospective_application_of_prospective_profile",
            "protocol_and_analysis_scope": (
                "The frozen prompt defines selected_link as one exact shown "
                "Wikipedia key whose page passage was needed. The configurations "
                "bind its decoded-label query, the body-only BM25 baseline, "
                "BGE-M3, ColBERTv2, hybrid RRF, the final hybrid reranker, output "
                "depth 20, and report cutoffs 1/5/20. The selected-object top-k "
                "obligation and mismatch-specific P/H join belong to the paper's "
                "later profile evaluation, which is applied retrospectively to "
                "the frozen rankings."
            ),
        },
        "temporal_evidence_boundary": (
            "The protocol records and pre-result commit declaration are hash-bound. "
            "The exact rankings are inputs to this deterministic join; the release "
            "does not claim an independently authenticated wall-clock timestamp for "
            "the later P/H analysis."
        ),
    }


def _cells(cells: Counter[tuple[bool, bool]], records: int) -> dict[str, Any]:
    values = {
        "records": records,
        "selected_id_preserved_traced_passage_hit": cells[(True, True)],
        "selected_id_preserved_traced_passage_miss": cells[(True, False)],
        "selected_id_not_preserved_traced_passage_hit": cells[(False, True)],
        "selected_id_not_preserved_traced_passage_miss": cells[(False, False)],
    }
    require(sum(value for key, value in values.items() if key != "records") == records,
            "P/H cells do not sum to the denominator")
    preserved = values["selected_id_preserved_traced_passage_hit"] + values[
        "selected_id_preserved_traced_passage_miss"
    ]
    traced = values["selected_id_preserved_traced_passage_hit"] + values[
        "selected_id_not_preserved_traced_passage_hit"
    ]
    return {
        **values,
        "selected_id_preservation_rate": preserved / records,
        "traced_passage_hit_rate": traced / records,
    }


def _cell_tuple(report: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        report["records"],
        report["selected_id_preserved_traced_passage_hit"],
        report["selected_id_preserved_traced_passage_miss"],
        report["selected_id_not_preserved_traced_passage_hit"],
        report["selected_id_not_preserved_traced_passage_miss"],
    )


def _metric_count(report: Mapping[str, Any], metric: str) -> int:
    if metric == "preservation":
        return (
            report["selected_id_preserved_traced_passage_hit"]
            + report["selected_id_preserved_traced_passage_miss"]
        )
    if metric == "trace_delivery":
        return (
            report["selected_id_preserved_traced_passage_hit"]
            + report["selected_id_not_preserved_traced_passage_hit"]
        )
    raise ValueError(f"unknown metric: {metric}")


def _paired_headline_at_5(
    subset: list[tuple[str, str]],
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
    inventory: Mapping[str, int],
    canonical_rankings: Mapping[str, Mapping[tuple[str, str], list[int]]],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    bm25 = canonical_rankings["visible_label_bm25"]
    reranker = canonical_rankings["visible_label_hybrid_reranker"]
    for label, target_field in (
        ("selected_object", "selected_link"),
        ("dataset_trace", "gold_link"),
    ):
        transitions: Counter[tuple[bool, bool]] = Counter()
        for key in subset:
            target = inventory[bridge[key][target_field]]
            transitions[(target in bm25[key][:5], target in reranker[key][:5])] += 1
        counts = (
            transitions[(True, True)],
            transitions[(True, False)],
            transitions[(False, True)],
            transitions[(False, False)],
        )
        require(
            counts == EXPECTED_PAIRED_HEADLINE_AT_5[label],
            f"{label}: paired BM25-to-reranker transitions drifted",
        )
        reports[label] = {
            "bm25_hit_reranker_hit": counts[0],
            "bm25_hit_reranker_miss": counts[1],
            "bm25_miss_reranker_hit": counts[2],
            "bm25_miss_reranker_miss": counts[3],
            "gains": counts[2],
            "regressions": counts[1],
            "net_change": counts[2] - counts[1],
        }
    return {
        "cutoff": 5,
        "records": len(subset),
        "orientation": "final_hybrid_reranker_minus_body_only_bm25",
        **reports,
    }


def _comparison_sign(gap_count: int) -> str:
    if gap_count > 0:
        return "left_higher"
    if gap_count < 0:
        return "right_higher"
    return "tie"


def _sign_relation(preservation_gap: int, trace_delivery_gap: int) -> str:
    if preservation_gap == 0 and trace_delivery_gap == 0:
        return "both_metrics_tied"
    if preservation_gap == 0:
        return "preservation_tie_only"
    if trace_delivery_gap == 0:
        return "trace_delivery_tie_only"
    if (preservation_gap > 0) == (trace_delivery_gap > 0):
        return "sign_agreement"
    return "sign_disagreement"


def _metric_comparison(
    left_count: int, right_count: int, records: int
) -> dict[str, Any]:
    gap_count = left_count - right_count
    return {
        "left_count": left_count,
        "right_count": right_count,
        "gap_count_left_minus_right": gap_count,
        "gap_percentage_points_left_minus_right": 100 * gap_count / records,
        "sign": _comparison_sign(gap_count),
    }


def _order_tiers(
    counts: Mapping[str, int], records: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for stack in STACKS:
        grouped[counts[stack]].append(stack)
    return [
        {
            "count": count,
            "rate": count / records,
            "stacks": grouped[count],
        }
        for count in sorted(grouped, reverse=True)
    ]


def _executor_order_and_pairwise_comparisons(
    stack_reports: Mapping[str, Any], records: int
) -> dict[str, Any]:
    """Summarize exact pooled P/H orders and every executor pair."""

    cutoff_reports: dict[str, Any] = {}
    aggregate = Counter()
    reversal_effects: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        pooled = {
            stack: stack_reports[stack]["cutoffs"][str(cutoff)][
                "pooled_model_question_records"
            ]
            for stack in STACKS
        }
        preservation_counts = {
            stack: _metric_count(values, "preservation")
            for stack, values in pooled.items()
        }
        trace_counts = {
            stack: _metric_count(values, "trace_delivery")
            for stack, values in pooled.items()
        }

        comparisons: list[dict[str, Any]] = []
        relation_counts = Counter()
        for left, right in itertools.combinations(STACKS, 2):
            p_gap = preservation_counts[left] - preservation_counts[right]
            h_gap = trace_counts[left] - trace_counts[right]
            relation = _sign_relation(p_gap, h_gap)
            relation_counts[relation] += 1
            comparisons.append(
                {
                    "left_stack": left,
                    "right_stack": right,
                    "preservation": _metric_comparison(
                        preservation_counts[left],
                        preservation_counts[right],
                        records,
                    ),
                    "trace_delivery": _metric_comparison(
                        trace_counts[left], trace_counts[right], records
                    ),
                    "sign_relation": relation,
                }
            )
            if relation == "sign_disagreement":
                reversal_effects.append(
                    {
                        "cutoff": cutoff,
                        "left_stack": left,
                        "right_stack": right,
                        "absolute_preservation_gap_count": abs(p_gap),
                        "absolute_preservation_gap_percentage_points": (
                            100 * abs(p_gap) / records
                        ),
                        "absolute_trace_delivery_gap_count": abs(h_gap),
                        "absolute_trace_delivery_gap_percentage_points": (
                            100 * abs(h_gap) / records
                        ),
                    }
                )

        summary = {
            "comparisons": len(comparisons),
            "sign_agreement": relation_counts["sign_agreement"],
            "sign_disagreement": relation_counts["sign_disagreement"],
            "tie_involved": (
                relation_counts["both_metrics_tied"]
                + relation_counts["preservation_tie_only"]
                + relation_counts["trace_delivery_tie_only"]
            ),
            "both_metrics_tied": relation_counts["both_metrics_tied"],
            "preservation_tie_only": relation_counts[
                "preservation_tie_only"
            ],
            "trace_delivery_tie_only": relation_counts[
                "trace_delivery_tie_only"
            ],
        }
        expected = EXPECTED_PAIRWISE_SUMMARY[cutoff]
        require(
            summary["comparisons"] == 10,
            f"cutoff {cutoff}: incomplete executor pair grid",
        )
        for field in (
            "sign_agreement",
            "sign_disagreement",
            "tie_involved",
            "both_metrics_tied",
            "preservation_tie_only",
            "trace_delivery_tie_only",
        ):
            require(
                summary[field] == expected[field],
                f"cutoff {cutoff}: pairwise {field} drifted",
            )
            aggregate[field] += summary[field]

        bm25 = STACKS[0]
        reranker = STACKS[-1]
        headline_p_gap = preservation_counts[reranker] - preservation_counts[bm25]
        headline_h_gap = trace_counts[reranker] - trace_counts[bm25]
        require(
            headline_p_gap == expected["headline_preservation_gap_count"],
            f"cutoff {cutoff}: headline preservation gap drifted",
        )
        require(
            headline_h_gap == expected["headline_trace_delivery_gap_count"],
            f"cutoff {cutoff}: headline trace-delivery gap drifted",
        )
        headline = {
            "orientation": "final_hybrid_reranker_minus_body_only_bm25",
            "preservation_gap_count": headline_p_gap,
            "preservation_gap_percentage_points": 100 * headline_p_gap / records,
            "trace_delivery_gap_count": headline_h_gap,
            "trace_delivery_gap_percentage_points": 100 * headline_h_gap / records,
            "sign_relation": _sign_relation(headline_p_gap, headline_h_gap),
        }

        cutoff_reports[str(cutoff)] = {
            "preservation_order_high_to_low": _order_tiers(
                preservation_counts, records
            ),
            "trace_delivery_order_high_to_low": _order_tiers(
                trace_counts, records
            ),
            "pairwise_sign_summary": summary,
            "pairwise_comparisons": comparisons,
            "headline_bm25_to_final_reranker": headline,
        }

    require(len(reversal_effects) == 14, "reversal effect inventory drifted")
    preservation_effects = [
        row["absolute_preservation_gap_percentage_points"]
        for row in reversal_effects
    ]
    trace_effects = [
        row["absolute_trace_delivery_gap_percentage_points"]
        for row in reversal_effects
    ]
    reversal_effect_summary = {
        "comparisons": len(reversal_effects),
        "comparisons_by_cutoff": {
            str(cutoff): sum(
                row["cutoff"] == cutoff for row in reversal_effects
            )
            for cutoff in CUTOFFS
        },
        "absolute_preservation_gap_percentage_points": {
            "minimum": min(preservation_effects),
            "median": statistics.median(preservation_effects),
            "maximum": max(preservation_effects),
        },
        "absolute_trace_delivery_gap_percentage_points": {
            "minimum": min(trace_effects),
            "median": statistics.median(trace_effects),
            "maximum": max(trace_effects),
        },
        "trace_delivery_gap_count_at_most_five": {
            "numerator": sum(
                row["absolute_trace_delivery_gap_count"] <= 5
                for row in reversal_effects
            ),
            "denominator": len(reversal_effects),
        },
        "comparisons_detail": reversal_effects,
        "interpretation_boundary": (
            "These are descriptive absolute gaps for the 14 sign-disagreeing "
            "cells in the complete frozen grid. They do not estimate reversal "
            "prevalence or practical deployment impact."
        ),
    }

    return {
        "orientation": (
            "pairwise gaps are left_stack minus right_stack; each metric sign "
            "therefore describes the left executor relative to the right"
        ),
        "cutoffs": cutoff_reports,
        "all_cutoffs_pairwise_sign_summary": {
            "comparisons": 30,
            "sign_agreement": aggregate["sign_agreement"],
            "sign_disagreement": aggregate["sign_disagreement"],
            "tie_involved": aggregate["tie_involved"],
            "both_metrics_tied": aggregate["both_metrics_tied"],
            "preservation_tie_only": aggregate["preservation_tie_only"],
            "trace_delivery_tie_only": aggregate[
                "trace_delivery_tie_only"
            ],
        },
        "sign_disagreement_effect_magnitudes": reversal_effect_summary,
        "interpretation_boundary": (
            "At k=5 and k=20 the BM25-to-final-reranker contrast has the same "
            "direction under P and H, matching 6/10 and 7/10 executor pairs, "
            "respectively. At k=1 its directions disagree, as do 7/10 pairs. "
            "The headline is therefore a valid pairwise contrast, not a claim "
            "that P and H induce the same complete executor ordering."
        ),
    }


def _question_set_sha256(question_ids: set[str]) -> str:
    payload = "".join(f"{question_id}\n" for question_id in sorted(question_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _question_model_map_sha256(
    question_models: Mapping[str, set[str]],
) -> str:
    payload = "".join(
        f"{question_id}\t{','.join(sorted(models))}\n"
        for question_id, models in sorted(question_models.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_pair_mapping_sha256(
    subset: list[tuple[str, str]],
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
    inventory: Mapping[str, int],
) -> str:
    """Bind the selected/traced identifiers and their resolved passage PIDs."""

    digest = hashlib.sha256()
    for question_id, model in subset:
        row = bridge[(question_id, model)]
        selected_link = row["selected_link"]
        traced_link = row["gold_link"]
        record = {
            "model": model,
            "question_id": question_id,
            "selected_link": selected_link,
            "selected_pid": inventory[selected_link],
            "traced_link": traced_link,
            "traced_pid": inventory[traced_link],
        }
        digest.update(
            (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _canonicalize_audited_ranking(
    ranking: list[str],
    inventory: Mapping[str, int],
    context: str,
) -> list[int]:
    """Resolve every returned identifier used by an audited cutoff."""

    canonical: list[int] = []
    for rank, identifier in enumerate(ranking[: max(CUTOFFS)], 1):
        require(
            identifier in inventory,
            f"{context}: returned ID at audited rank {rank} does not resolve",
        )
        canonical.append(inventory[identifier])
    return canonical


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> dict[str, Any]:
    """Build a canonical path-free report from exact frozen member bytes."""

    payloads, source_metadata = _validate_sources(source)
    protocol_binding = _protocol_binding_report(payloads)
    bridge = identity._load_bridge(
        identity._jsonl_rows(payloads[identity.BRIDGE_MEMBER], identity.BRIDGE_MEMBER)
    )
    inventory = identity._load_links(
        identity._jsonl_rows(payloads[identity.LINKS_MEMBER], identity.LINKS_MEMBER)
    )
    rankings: dict[str, dict[tuple[str, str], list[str]]] = {
        "visible_label_bm25": {
            key: list(row["predicted_ranking"]) if row["valid_json"] else []
            for key, row in bridge.items()
        }
    }
    for stack in STACKS[1:]:
        member = identity.STACK_MEMBERS[stack]
        rankings[stack] = identity._load_sensitivity(
            identity._jsonl_rows(payloads[member], member), bridge, stack
        )

    raw_differences = sorted(
        key
        for key, row in bridge.items()
        if row.get("selection_correct") is False
    )
    raw_differences_with_unresolvable_selected_id = [
        key
        for key in raw_differences
        if bridge[key].get("selected_link") not in inventory
    ]
    selected_resolvable_differences = [
        key
        for key in raw_differences
        if bridge[key].get("selected_link") in inventory
    ]
    gold_unresolvable = [
        key
        for key in selected_resolvable_differences
        if bridge[key].get("gold_link") not in inventory
    ]
    require(not gold_unresolvable, "a traced ID in the mismatch audit does not resolve")
    canonical_alias_collapses = [
        key
        for key in selected_resolvable_differences
        if inventory[bridge[key]["selected_link"]]
        == inventory[bridge[key]["gold_link"]]
    ]
    subset = [
        key
        for key in selected_resolvable_differences
        if inventory[bridge[key]["selected_link"]]
        != inventory[bridge[key]["gold_link"]]
    ]
    require(
        len(raw_differences) == EXPECTED["raw_identifier_differences"],
        "raw selected/traced identifier difference count drifted",
    )
    require(
        len(selected_resolvable_differences)
        == EXPECTED["raw_differences_with_resolvable_selected_id"],
        "resolvable raw-difference count drifted",
    )
    require(
        len(raw_differences_with_unresolvable_selected_id)
        == EXPECTED["raw_differences_with_unresolvable_selected_id"],
        "unresolvable raw-difference count drifted",
    )
    require(
        len(canonical_alias_collapses)
        == EXPECTED["raw_differences_collapsing_to_same_canonical_pid"],
        "raw identifier differences collapsed to a common canonical object",
    )
    require(
        len(subset) == EXPECTED["canonical_object_distinct_records"],
        "canonical object-distinct denominator drifted",
    )
    canonical_pair_mapping_sha256 = _canonical_pair_mapping_sha256(
        subset, bridge, inventory
    )
    require(
        canonical_pair_mapping_sha256
        == EXPECTED["canonical_pair_mapping_sha256"],
        "canonical selected/traced object mapping drifted",
    )

    canonical_rankings = {
        stack: {
            key: _canonicalize_audited_ranking(
                stack_rankings[key], inventory, f"{stack} {key}"
            )
            for key in subset
        }
        for stack, stack_rankings in rankings.items()
    }
    paired_headline_at_5 = _paired_headline_at_5(
        subset, bridge, inventory, canonical_rankings
    )
    returned_id_occurrences_checked = sum(
        len(ranking)
        for stack_rankings in canonical_rankings.values()
        for ranking in stack_rankings.values()
    )
    records_by_model = Counter(model for _, model in subset)
    question_models: dict[str, set[str]] = defaultdict(set)
    for question_id, model in subset:
        question_models[question_id].add(model)
    multiplicity = Counter(len(models) for models in question_models.values())
    require(len(subset) == EXPECTED["records"], "mismatch record denominator drifted")
    require(
        {model: records_by_model[model] for model in identity.MODELS}
        == EXPECTED["records_by_model"],
        "mismatch per-model denominators drifted",
    )
    require(
        len(question_models) == EXPECTED["unique_questions"],
        "mismatch question-cluster denominator drifted",
    )
    observed_multiplicity = {
        f"exactly_{value}": multiplicity[value] for value in (1, 2, 3)
    }
    require(
        observed_multiplicity == EXPECTED["question_multiplicity"],
        "mismatch question multiplicity drifted",
    )

    stack_reports: dict[str, Any] = {}
    for stack in STACKS:
        cutoff_reports: dict[str, Any] = {}
        for cutoff in CUTOFFS:
            pooled_cells: Counter[tuple[bool, bool]] = Counter()
            cell_questions: dict[tuple[bool, bool], set[str]] = defaultdict(set)
            per_model: dict[str, Any] = {}
            for model in identity.MODELS:
                cells: Counter[tuple[bool, bool]] = Counter()
                for key in subset:
                    question_id, row_model = key
                    if row_model != model:
                        continue
                    bridge_row = bridge[key]
                    ranking = canonical_rankings[stack][key][:cutoff]
                    cell = (
                        inventory[bridge_row["selected_link"]] in ranking,
                        inventory[bridge_row["gold_link"]] in ranking,
                    )
                    cells[cell] += 1
                    pooled_cells[cell] += 1
                    cell_questions[cell].add(question_id)
                model_report = _cells(cells, records_by_model[model])
                require(
                    _cell_tuple(model_report)
                    == EXPECTED_CELLS[stack][cutoff][model],
                    f"{stack}@{cutoff}/{model}: P/H cells drifted",
                )
                per_model[model] = {
                    "display_name": identity.MODEL_LABELS[model],
                    **model_report,
                }
            pooled_report = _cells(pooled_cells, len(subset))
            require(
                _cell_tuple(pooled_report)
                == EXPECTED_CELLS[stack][cutoff]["pooled"],
                f"{stack}@{cutoff}: pooled P/H cells drifted",
            )
            cutoff_reports[str(cutoff)] = {
                "per_model": per_model,
                "pooled_model_question_records": pooled_report,
                "unique_question_clusters_contributing_to_each_cell": {
                    "selected_id_preserved_traced_passage_hit": len(
                        cell_questions[(True, True)]
                    ),
                    "selected_id_preserved_traced_passage_miss": len(
                        cell_questions[(True, False)]
                    ),
                    "selected_id_not_preserved_traced_passage_hit": len(
                        cell_questions[(False, True)]
                    ),
                    "selected_id_not_preserved_traced_passage_miss": len(
                        cell_questions[(False, False)]
                    ),
                },
            }
        stack_reports[stack] = {
            "display_name": STACK_LABELS[stack],
            "cutoffs": cutoff_reports,
        }

    repeated_questions = multiplicity[2] + multiplicity[3]
    repeated_records = 2 * multiplicity[2] + 3 * multiplicity[3]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_all_stack_mismatch_p_h_replay",
        "api_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "network_used": False,
        "credentials_used": False,
        "random_used": False,
        "clock_used": False,
        "definition": {
            "record_unit": "one frozen (question_id, hosted_model) record",
            "subset": (
                "valid selector outputs whose selected and dataset-traced IDs "
                "both resolve to distinct passage PIDs under the frozen resolver"
            ),
            "canonical_object_identity": (
                "zero-based passage PID in the exact frozen links.jsonl resolver"
            ),
            "P_at_k": (
                "canonical selected-object PID occurs in the canonicalized named "
                "recorded top-k ranking"
            ),
            "H_at_k": (
                "canonical dataset-traced-object PID occurs in the same "
                "canonicalized named recorded top-k ranking"
            ),
            "cutoffs": list(CUTOFFS),
            "stack_order": list(STACKS),
            "scope": (
                "retrospective deterministic accounting on frozen rankings; no "
                "new reader outcome, causal claim, or deployment prevalence claim"
            ),
        },
        "inputs": {
            "container_policy": "bind exact scientific member bytes, not the outer ZIP",
            "members": source_metadata,
        },
        "canonical_identity_verification": {
            "resolver": {
                "member": identity.LINKS_MEMBER,
                "member_sha256": source_metadata[identity.LINKS_MEMBER]["sha256"],
                "object_identity": "zero-based passage PID",
                "inventory_records": len(inventory),
                "unique_identifier_to_pid_mapping": True,
                "unique_contiguous_pids": True,
            },
            "raw_identifier_differences": len(raw_differences),
            "raw_differences_with_resolvable_selected_id": len(
                selected_resolvable_differences
            ),
            "raw_differences_with_unresolvable_selected_id": len(
                raw_differences_with_unresolvable_selected_id
            ),
            "gold_unresolvable_among_selected_resolvable_differences": len(
                gold_unresolvable
            ),
            "canonical_object_distinct_records": len(subset),
            "canonical_object_distinct_records_by_model": {
                model: records_by_model[model] for model in identity.MODELS
            },
            "raw_differences_collapsing_to_same_canonical_pid": len(
                canonical_alias_collapses
            ),
            "canonical_pair_mapping_sha256": canonical_pair_mapping_sha256,
            "audited_cutoffs": list(CUTOFFS),
            "returned_id_occurrences_checked_through_max_cutoff": (
                returned_id_occurrences_checked
            ),
            "all_returned_ids_at_audited_cutoffs_resolved": True,
        },
        "denominators_and_dependence": {
            "paired_model_question_records": len(subset),
            "records_by_model": {
                model: records_by_model[model] for model in identity.MODELS
            },
            "unique_question_clusters": len(question_models),
            "question_model_multiplicity": observed_multiplicity,
            "question_id_set_sha256": _question_set_sha256(set(question_models)),
            "question_model_map_sha256": _question_model_map_sha256(question_models),
            "questions_with_multiple_hosted_model_records": {
                "numerator": repeated_questions,
                "denominator": len(question_models),
                "rate": repeated_questions / len(question_models),
            },
            "records_from_questions_with_multiple_hosted_model_records": {
                "numerator": repeated_records,
                "denominator": len(subset),
                "rate": repeated_records / len(subset),
            },
            "dependence_warning": (
                "pooled counts are descriptive: 329 model-question rows come from "
                "160 questions, and rows sharing a question ID are not independent"
            ),
        },
        "protocol_binding": protocol_binding,
        "stacks": stack_reports,
        "paired_bm25_to_final_reranker_at_5": paired_headline_at_5,
        "executor_order_and_pairwise_comparisons": (
            _executor_order_and_pairwise_comparisons(
                stack_reports, len(subset)
            )
        ),
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
                    raise identity.AuditError(
                        f"artifact missing member: {member}"
                    ) from exc

            return build_report_from_members(read)
    except zipfile.BadZipFile as exc:
        raise identity.AuditError(f"invalid artifact ZIP: {path}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = identity.canonical_json_bytes(build_report_from_archive(args.artifact))
        if args.output is None:
            sys.stdout.buffer.write(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
    except (identity.AuditError, OSError) as exc:
        print(f"all-stack mismatch P/H audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
