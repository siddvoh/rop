#!/usr/bin/env python3
"""Replay full-cohort retrieval-stack decisions under P and H.

This audit answers whether returned-object preservation (P) and traced-passage
delivery (H) select different retrieval stacks on the complete frozen hosted-
model cohort, rather than only on the object-distinct mismatch subset.  It
uses all 1,800 planned ``(question_id, hosted_model)`` rows, all five recorded
visible-label stacks, and cutoffs 1, 5, and 20.  Six schema-invalid selector
outputs and two corpus-unresolvable selected IDs remain in the primary
denominator; an unavailable selected object is a P miss, and the recorded
empty ranking for an upstream-invalid row is an H miss.

The primary all-planned result is reported separately per model family and
under pooled-record, equal-model, and equal-question aggregation.  The latter
three coincide algebraically in this balanced 600-question x three-family
design, but are recomputed independently.  Valid-selector and resolvable-
selection populations are retained as explicit sensitivity analyses.  The
report compares both the maximizing stack set (the selection decision) and the
complete ten-pair preference grid.  No model, retrieval, API, network,
credential, random, or clock input is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
import zipfile


_MISMATCH_PATH = Path(__file__).with_name("recompute_mismatch_stack_cutoffs.py")
_MISMATCH_SPEC = importlib.util.spec_from_file_location(
    "full_cohort_stack_decisions_mismatch", _MISMATCH_PATH
)
if _MISMATCH_SPEC is None or _MISMATCH_SPEC.loader is None:
    raise RuntimeError(f"cannot load mismatch replay at {_MISMATCH_PATH}")
mismatch = importlib.util.module_from_spec(_MISMATCH_SPEC)
_MISMATCH_SPEC.loader.exec_module(mismatch)
identity = mismatch.identity


SCHEMA_VERSION = 1
CUTOFFS = (1, 5, 20)
STACKS = mismatch.STACKS
STACK_LABELS = mismatch.STACK_LABELS
PAIRS = tuple(itertools.combinations(STACKS, 2))

RELEVANT_MEMBERS = (
    identity.BRIDGE_MEMBER,
    identity.STACK_MEMBERS["visible_label_bge_m3"],
    identity.STACK_MEMBERS["visible_label_colbert_v2"],
    identity.STACK_MEMBERS["visible_label_hybrid_rrf"],
    identity.STACK_MEMBERS["visible_label_hybrid_reranker"],
    identity.LINKS_MEMBER,
)

EXPECTED = {
    "planned_records": 1_800,
    "unique_questions": 600,
    "records_by_model": {"claude": 600, "openai": 600, "gemini": 600},
    "selector_valid_records": 1_794,
    "selector_invalid_records": 6,
    "selector_invalid_by_model": {"openai": 6},
    "selected_id_resolvable_records": 1_792,
    "selected_id_unresolvable_valid_records": 2,
    "selected_id_unresolvable_valid_by_model": {"claude": 1, "gemini": 1},
    "canonical_selected_traced_match_records": 1_463,
    "canonical_selected_traced_mismatch_records": 329,
    "question_set_sha256": (
        "a10dbbe79799fe21035e94e0953bea195c5622db9ac41b7277bdb734cdab69b7"
    ),
    "returned_id_occurrences_checked_through_rank_20": 176_997,
    "outcome_projection_sha256": (
        "92428cadc07b07dd646af5766f5feff51e936fe2bd9bb1da2c7a275527a489c0"
    ),
}

EXPECTED_PRIMARY_WINNERS = {
    1: ["visible_label_colbert_v2"],
    5: ["visible_label_hybrid_reranker"],
    20: ["visible_label_hybrid_reranker"],
}

EXPECTED_MEMBERS = {
    member: identity.EXPECTED_MEMBERS[member] for member in RELEVANT_MEMBERS
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


def _validate_sources(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    read = _reader(source)
    payloads: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    require(
        set(EXPECTED_MEMBERS) == set(RELEVANT_MEMBERS),
        "declared full-cohort member set drifted",
    )
    for member in sorted(RELEVANT_MEMBERS):
        try:
            payload = read(member)
        except (KeyError, FileNotFoundError) as exc:
            raise identity.AuditError(f"missing source member: {member}") from exc
        expected = EXPECTED_MEMBERS[member]
        digest = hashlib.sha256(payload).hexdigest()
        records = sum(bool(line.strip()) for line in payload.splitlines())
        require(len(payload) == expected["bytes"], f"{member}: byte count drift")
        require(digest == expected["sha256"], f"{member}: SHA-256 drift")
        require(records == expected["records"], f"{member}: record count drift")
        payloads[member] = payload
        metadata[member] = {
            "bytes": len(payload),
            "records": records,
            "sha256": digest,
        }
    return payloads, metadata


def _fraction(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "rate": float(value),
        "percentage": 100.0 * float(value),
    }


def _question_set_sha256(question_ids: set[str] | list[str]) -> str:
    payload = "".join(f"{question_id}\n" for question_id in sorted(question_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _outcome_projection_sha256(
    keys: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> str:
    lines = []
    for question_id, model in sorted(keys):
        bits = []
        key = (question_id, model)
        for cutoff in CUTOFFS:
            for stack in STACKS:
                preserved, delivered = outcomes[(key, stack, cutoff)]
                bits.extend((str(int(preserved)), str(int(delivered))))
        lines.append(f"{question_id}\t{model}\t{''.join(bits)}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _rates_from_keys(
    keys: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> dict[tuple[str, int, str], Fraction]:
    require(bool(keys), "cannot compute rates from an empty population")
    counts: Counter[tuple[str, int, str]] = Counter()
    for key in keys:
        for stack in STACKS:
            for cutoff in CUTOFFS:
                preserved, delivered = outcomes[(key, stack, cutoff)]
                counts[(stack, cutoff, "preservation")] += preserved
                counts[(stack, cutoff, "trace_delivery")] += delivered
    return {
        name: Fraction(counts[name], len(keys))
        for name in itertools.product(
            STACKS, CUTOFFS, ("preservation", "trace_delivery")
        )
    }


def _order_tiers(
    rates: Mapping[tuple[str, int, str], Fraction],
    cutoff: int,
    metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[Fraction, list[str]] = defaultdict(list)
    for stack in STACKS:
        grouped[rates[(stack, cutoff, metric)]].append(stack)
    return [
        {
            "value": _fraction(value),
            "stacks": sorted(grouped[value], key=STACKS.index),
        }
        for value in sorted(grouped, reverse=True)
    ]


def _pair_relation(p_gap: Fraction, h_gap: Fraction) -> str:
    if p_gap == 0 and h_gap == 0:
        return "both_metrics_tied"
    if p_gap == 0:
        return "preservation_tie_only"
    if h_gap == 0:
        return "trace_delivery_tie_only"
    if (p_gap > 0) == (h_gap > 0):
        return "same_preference"
    return "opposite_preference"


def _decision_report(
    rates: Mapping[tuple[str, int, str], Fraction],
) -> dict[str, Any]:
    cutoffs: dict[str, Any] = {}
    winner_changes = 0
    complete_order_changes = 0
    opposite_preferences = 0
    for cutoff in CUTOFFS:
        p_tiers = _order_tiers(rates, cutoff, "preservation")
        h_tiers = _order_tiers(rates, cutoff, "trace_delivery")
        p_winners = p_tiers[0]["stacks"]
        h_winners = h_tiers[0]["stacks"]
        same_winners = p_winners == h_winners
        same_order = [tier["stacks"] for tier in p_tiers] == [
            tier["stacks"] for tier in h_tiers
        ]
        winner_changes += not same_winners
        complete_order_changes += not same_order

        comparisons = []
        relations: Counter[str] = Counter()
        for left, right in PAIRS:
            p_gap = rates[(left, cutoff, "preservation")] - rates[
                (right, cutoff, "preservation")
            ]
            h_gap = rates[(left, cutoff, "trace_delivery")] - rates[
                (right, cutoff, "trace_delivery")
            ]
            relation = _pair_relation(p_gap, h_gap)
            relations[relation] += 1
            comparisons.append(
                {
                    "left_stack": left,
                    "right_stack": right,
                    "preservation_gap_left_minus_right": _fraction(p_gap),
                    "trace_delivery_gap_left_minus_right": _fraction(h_gap),
                    "preference_relation": relation,
                }
            )
        opposite_preferences += relations["opposite_preference"]
        cutoffs[str(cutoff)] = {
            "preservation_order_high_to_low": p_tiers,
            "trace_delivery_order_high_to_low": h_tiers,
            "preservation_maximizing_stacks": p_winners,
            "trace_delivery_maximizing_stacks": h_winners,
            "stack_selection_decision_changes": not same_winners,
            "complete_ordering_changes": not same_order,
            "pairwise_preference_summary": {
                key: relations[key]
                for key in (
                    "same_preference",
                    "opposite_preference",
                    "preservation_tie_only",
                    "trace_delivery_tie_only",
                    "both_metrics_tied",
                )
            },
            "pairwise_comparisons": comparisons,
        }
    return {
        "cutoffs": cutoffs,
        "summary": {
            "cutoffs": len(CUTOFFS),
            "stack_selection_decision_changes": winner_changes,
            "complete_ordering_changes": complete_order_changes,
            "opposite_pairwise_preferences": opposite_preferences,
        },
    }


def _equal_model_rates(
    keys: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> dict[tuple[str, int, str], Fraction]:
    per_model = {
        model: _rates_from_keys([key for key in keys if key[1] == model], outcomes)
        for model in identity.MODELS
    }
    return {
        name: sum(
            (per_model[model][name] for model in identity.MODELS), Fraction(0)
        )
        / len(identity.MODELS)
        for name in next(iter(per_model.values()))
    }


def _equal_question_rates(
    keys: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> dict[tuple[str, int, str], Fraction]:
    question_models: dict[str, list[str]] = defaultdict(list)
    for question_id, model in keys:
        question_models[question_id].append(model)
    rates: dict[tuple[str, int, str], Fraction] = {}
    for stack in STACKS:
        for cutoff in CUTOFFS:
            for metric_index, metric in enumerate(("preservation", "trace_delivery")):
                question_means = [
                    Fraction(
                        sum(
                            outcomes[((question_id, model), stack, cutoff)][metric_index]
                            for model in models
                        ),
                        len(models),
                    )
                    for question_id, models in question_models.items()
                ]
                rates[(stack, cutoff, metric)] = sum(
                    question_means, Fraction(0)
                ) / len(question_means)
    return rates


def _population_report(
    keys: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
    *,
    include_per_model: bool,
) -> dict[str, Any]:
    question_counts = Counter(question_id for question_id, _ in keys)
    model_counts = Counter(model for _, model in keys)
    pooled = _rates_from_keys(keys, outcomes)
    equal_model = _equal_model_rates(keys, outcomes)
    equal_question = _equal_question_rates(keys, outcomes)
    report: dict[str, Any] = {
        "denominators": {
            "model_question_records": len(keys),
            "unique_questions": len(question_counts),
            "records_by_model": {
                model: model_counts[model] for model in identity.MODELS
            },
            "question_record_multiplicity": {
                str(count): sum(value == count for value in question_counts.values())
                for count in sorted(set(question_counts.values()))
            },
        },
        "pooled_record_weighted": _decision_report(pooled),
        "model_equal_weighted": _decision_report(equal_model),
        "question_equal_weighted": _decision_report(equal_question),
        "aggregate_weightings_identical": (
            pooled == equal_model == equal_question
        ),
    }
    if include_per_model:
        report["per_model_family"] = {
            model: {
                "display_name": identity.MODEL_LABELS[model],
                "records": model_counts[model],
                **_decision_report(
                    _rates_from_keys(
                        [key for key in keys if key[1] == model], outcomes
                    )
                ),
            }
            for model in identity.MODELS
        }
    return report


def _noneligible_outcome_accounting(
    keys: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> dict[str, Any]:
    return {
        stack: {
            str(cutoff): {
                "records": len(keys),
                "P_hits": sum(
                    outcomes[(key, stack, cutoff)][0] for key in keys
                ),
                "H_hits": sum(
                    outcomes[(key, stack, cutoff)][1] for key in keys
                ),
            }
            for cutoff in CUTOFFS
        }
        for stack in STACKS
    }


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> dict[str, Any]:
    payloads, source_metadata = _validate_sources(source)
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

    keys = sorted(bridge)
    require(
        len(keys) == EXPECTED["planned_records"],
        "full-cohort record count drifted",
    )
    question_ids = {question_id for question_id, _ in keys}
    require(
        len(question_ids) == EXPECTED["unique_questions"],
        "full-cohort question count drifted",
    )
    model_counts = Counter(model for _, model in keys)
    require(
        {model: model_counts[model] for model in identity.MODELS}
        == EXPECTED["records_by_model"],
        "full-cohort model balance drifted",
    )
    require(
        all(bridge[key]["gold_link"] in inventory for key in keys),
        "a full-cohort traced ID does not resolve",
    )

    canonical_rankings = {
        stack: {
            key: mismatch._canonicalize_audited_ranking(
                stack_rankings[key][: max(CUTOFFS)],
                inventory,
                f"{stack} {key}",
            )
            for key in keys
        }
        for stack, stack_rankings in rankings.items()
    }
    checked_returned_ids = sum(
        len(ranking)
        for stack_rankings in canonical_rankings.values()
        for ranking in stack_rankings.values()
    )

    outcomes: dict[
        tuple[tuple[str, str], str, int], tuple[bool, bool]
    ] = {}
    for key in keys:
        row = bridge[key]
        selected_pid = inventory.get(row.get("selected_link"))
        traced_pid = inventory[row["gold_link"]]
        for stack in STACKS:
            for cutoff in CUTOFFS:
                ranking = canonical_rankings[stack][key][:cutoff]
                outcomes[(key, stack, cutoff)] = (
                    selected_pid is not None and selected_pid in ranking,
                    traced_pid in ranking,
                )

    valid_keys = [key for key in keys if bridge[key]["valid_json"]]
    resolvable_keys = [
        key for key in keys if bridge[key].get("selected_link") in inventory
    ]
    invalid_keys = [key for key in keys if not bridge[key]["valid_json"]]
    unresolvable_keys = [
        key
        for key in valid_keys
        if bridge[key].get("selected_link") not in inventory
    ]
    mismatch_keys = [
        key
        for key in resolvable_keys
        if inventory[bridge[key]["selected_link"]]
        != inventory[bridge[key]["gold_link"]]
    ]
    mismatch_set = set(mismatch_keys)
    matched_keys = [key for key in resolvable_keys if key not in mismatch_set]

    observed_accounting = {
        "planned_records": len(keys),
        "unique_questions": len(question_ids),
        "records_by_model": {
            model: model_counts[model] for model in identity.MODELS
        },
        "selector_valid_records": len(valid_keys),
        "selector_invalid_records": len(invalid_keys),
        "selector_invalid_by_model": dict(
            sorted(Counter(model for _, model in invalid_keys).items())
        ),
        "selected_id_resolvable_records": len(resolvable_keys),
        "selected_id_unresolvable_valid_records": len(unresolvable_keys),
        "selected_id_unresolvable_valid_by_model": dict(
            sorted(Counter(model for _, model in unresolvable_keys).items())
        ),
        "canonical_selected_traced_match_records": len(matched_keys),
        "canonical_selected_traced_mismatch_records": len(mismatch_keys),
        "question_set_sha256": _question_set_sha256(question_ids),
        "returned_id_occurrences_checked_through_rank_20": checked_returned_ids,
    }
    for field, expected_value in EXPECTED.items():
        if field == "outcome_projection_sha256":
            continue
        require(
            observed_accounting[field] == expected_value,
            f"full-cohort {field} drifted",
        )

    outcome_projection_sha256 = _outcome_projection_sha256(keys, outcomes)
    require(
        outcome_projection_sha256 == EXPECTED["outcome_projection_sha256"],
        "full-cohort P/H outcome projection drifted",
    )

    invalid_outcomes = Counter(
        outcomes[(key, stack, cutoff)]
        for key in invalid_keys
        for stack in STACKS
        for cutoff in CUTOFFS
    )
    require(
        invalid_outcomes == Counter({(False, False): len(invalid_keys) * 15}),
        "upstream-invalid rows do not remain misses for every stack/cutoff",
    )

    all_planned = _population_report(
        keys, outcomes, include_per_model=True
    )
    selector_valid = _population_report(
        valid_keys, outcomes, include_per_model=True
    )
    selected_resolvable = _population_report(
        resolvable_keys, outcomes, include_per_model=True
    )
    mismatch_only = _population_report(
        mismatch_keys, outcomes, include_per_model=True
    )

    for weighting in (
        "pooled_record_weighted",
        "model_equal_weighted",
        "question_equal_weighted",
    ):
        primary = all_planned[weighting]
        require(
            primary["summary"]
            == {
                "cutoffs": 3,
                "stack_selection_decision_changes": 0,
                "complete_ordering_changes": 0,
                "opposite_pairwise_preferences": 0,
            },
            f"primary full-cohort decision summary drifted for {weighting}",
        )
        for cutoff in CUTOFFS:
            section = primary["cutoffs"][str(cutoff)]
            require(
                section["preservation_maximizing_stacks"]
                == EXPECTED_PRIMARY_WINNERS[cutoff],
                f"primary P winner drifted for {weighting}@{cutoff}",
            )
            require(
                section["trace_delivery_maximizing_stacks"]
                == EXPECTED_PRIMARY_WINNERS[cutoff],
                f"primary H winner drifted for {weighting}@{cutoff}",
            )
    for model in identity.MODELS:
        require(
            all_planned["per_model_family"][model]["summary"]
            == {
                "cutoffs": 3,
                "stack_selection_decision_changes": 0,
                "complete_ordering_changes": 0,
                "opposite_pairwise_preferences": 0,
            },
            f"primary full-cohort decision summary drifted for {model}",
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_full_cohort_p_h_stack_decision_replay",
        "api_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "network_used": False,
        "credentials_used": False,
        "random_used": False,
        "clock_used": False,
        "definition": {
            "record_unit": "one frozen (question_id, hosted_model) record",
            "primary_population": (
                "all 1,800 planned records; invalid or corpus-unresolvable "
                "selected IDs are P misses, and upstream-invalid empty rankings "
                "are H misses"
            ),
            "P_at_k": (
                "the canonical selected-object PID occurs in the named recorded "
                "top-k ranking; false when the selected ID is absent or unresolvable"
            ),
            "H_at_k": (
                "the canonical dataset-traced-object PID occurs in the same "
                "named recorded top-k ranking"
            ),
            "stack_selection_decision": (
                "the set of stacks attaining the largest exact rate for the "
                "named metric, population, aggregation, and cutoff"
            ),
            "complete_ordering_change": (
                "the exact high-to-low stack tier partition differs between P and H"
            ),
            "cutoffs": list(CUTOFFS),
            "stacks": list(STACKS),
            "scope": (
                "retrospective deterministic accounting on frozen rankings; no "
                "reader outcome, semantic-correctness, causal, or deployment claim"
            ),
        },
        "inputs": {
            "container_policy": "bind exact scientific member bytes, not the outer ZIP",
            "members": source_metadata,
            "outcome_projection_sha256": outcome_projection_sha256,
        },
        "cohort_accounting": {
            "planned_records": len(keys),
            "unique_questions": len(question_ids),
            "question_set_sha256": _question_set_sha256(question_ids),
            "records_by_model": {
                model: model_counts[model] for model in identity.MODELS
            },
            "selector_valid_records": len(valid_keys),
            "selector_invalid_records": len(invalid_keys),
            "selector_invalid_by_model": dict(
                sorted(Counter(model for _, model in invalid_keys).items())
            ),
            "selected_id_resolvable_records": len(resolvable_keys),
            "selected_id_unresolvable_valid_records": len(unresolvable_keys),
            "selected_id_unresolvable_valid_by_model": dict(
                sorted(Counter(model for _, model in unresolvable_keys).items())
            ),
            "canonical_selected_traced_match_records": len(matched_keys),
            "canonical_selected_traced_mismatch_records": len(mismatch_keys),
            "all_traced_ids_resolve": True,
            "returned_id_occurrences_checked_through_rank_20": checked_returned_ids,
            "all_returned_ids_through_rank_20_resolve": True,
            "invalid_rows_are_p_and_h_misses_for_every_stack_cutoff": True,
            "noneligible_outcome_accounting": {
                "selector_invalid": _noneligible_outcome_accounting(
                    invalid_keys, outcomes
                ),
                "selected_id_unresolvable_valid": _noneligible_outcome_accounting(
                    unresolvable_keys, outcomes
                ),
            },
        },
        "full_cohort_all_planned": all_planned,
        "selector_valid_sensitivity": selector_valid,
        "selected_id_resolvable_sensitivity": selected_resolvable,
        "mismatch_only_reference": mismatch_only,
        "decision_relevance": {
            "full_cohort_stack_selection_changes": all_planned[
                "pooled_record_weighted"
            ]["summary"]["stack_selection_decision_changes"],
            "full_cohort_complete_ordering_changes": all_planned[
                "pooled_record_weighted"
            ]["summary"]["complete_ordering_changes"],
            "full_cohort_opposite_pairwise_preferences": all_planned[
                "pooled_record_weighted"
            ]["summary"]["opposite_pairwise_preferences"],
            "mismatch_only_stack_selection_changes": mismatch_only[
                "pooled_record_weighted"
            ]["summary"]["stack_selection_decision_changes"],
            "mismatch_only_complete_ordering_changes": mismatch_only[
                "pooled_record_weighted"
            ]["summary"]["complete_ordering_changes"],
            "mismatch_only_opposite_pairwise_preferences": mismatch_only[
                "pooled_record_weighted"
            ]["summary"]["opposite_pairwise_preferences"],
            "result_summary": (
                "Across the full 1,800-record cohort, P and H produce the same "
                "stack selection, complete ordering, and pairwise preferences at "
                "cutoffs 1, 5, and 20 within every hosted-model family and under "
                "pooled-record, equal-model, and equal-question aggregation. On "
                "the conditioned 329-record canonical-mismatch subset, the "
                "maximizing stack set differs once: at cutoff 1, P ties ColBERTv2 "
                "with the final hybrid reranker while H selects body-only BM25. "
                "This is a conditional diagnostic result rather than a changed "
                "full-cohort stack choice or reader-outcome result."
            ),
        },
    }


def build_report_from_archive(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact not found: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist()]
            duplicates = [
                name for name, count in Counter(names).items() if count > 1
            ]
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
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report_from_archive(args.artifact)
        payload = identity.canonical_json_bytes(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    except (identity.AuditError, OSError, ValueError) as exc:
        print(f"full-cohort P/H stack audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "full_cohort": report["decision_relevance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
