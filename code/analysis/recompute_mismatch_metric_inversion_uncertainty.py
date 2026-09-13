#!/usr/bin/env python3
"""Replay question-cluster uncertainty for the mismatch P/H inversion.

This audit uses only the exact frozen selector, resolver, and ranking ledgers
consumed by ``recompute_mismatch_stack_cutoffs.py``.  It addresses two related
descriptive questions on the 329 canonical-object-distinct model-question
records:

* how stable are the body-only BM25 to BGE-M3 P and H gaps at cutoffs 5 and
  20 when the 160 unique questions, rather than 329 repeated model rows, are
  the resampling unit; and
* how stable is the number of opposite-sign P/H comparisons when all 30
  executor-pair-by-cutoff cells are recomputed jointly in every draw.

Question IDs are sampled with replacement and every hosted-model row belonging
to a sampled question travels with it.  The pseudorandom streams are fully
specified by fixed SHA-256-derived seeds.  These are post-selection stability
summaries conditional on the frozen mismatch stratum, not population confidence
intervals, binomial trials over grid cells, causal estimates, or deployment
prevalence estimates.  No model, retrieval, API, network, credential, or clock
input is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping
import zipfile


_EXACT_STATISTICS_PATH = Path(__file__).with_name("exact_statistics.py")
_exact_statistics_spec = importlib.util.spec_from_file_location(
    "_rop_exact_statistics", _EXACT_STATISTICS_PATH
)
exact_statistics = importlib.util.module_from_spec(_exact_statistics_spec)
_exact_statistics_spec.loader.exec_module(exact_statistics)
sample_standard_deviation = exact_statistics.sample_standard_deviation


_BASE_PATH = Path(__file__).with_name("recompute_mismatch_stack_cutoffs.py")
_BASE_SPEC = importlib.util.spec_from_file_location(
    "mismatch_metric_inversion_uncertainty_base", _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load mismatch replay at {_BASE_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)
identity = base.identity


SCHEMA_VERSION = 1
RELEVANT_MEMBERS = base.RELEVANT_MEMBERS
BOOTSTRAP_SAMPLES = 10_000
BASE_SEED = 20_260_716
CONTRAST_LABEL = "body_only_bm25_vs_bge_m3_mismatch_question_cluster_v1"
GRID_LABEL = "mismatch_grid_direction_reversal_question_cluster_v1"
TARGET_CUTOFFS = (5, 20)
TARGET_STACKS = ("visible_label_bm25", "visible_label_bge_m3")
PAIRS = tuple(itertools.combinations(base.STACKS, 2))
UNIT_SCALE = 6  # exactly represents averages over 1, 2, or 3 model rows


EXPECTED = {
    "records": 329,
    "unique_questions": 160,
    "records_by_model": {"claude": 143, "openai": 85, "gemini": 101},
    "question_multiplicity": {"exactly_1": 55, "exactly_2": 41, "exactly_3": 64},
    "question_set_sha256": (
        "5f451a619152521d3af146d54afc9153dcb06f2745b34ddf6eae25e27b1c0aad"
    ),
    "outcome_projection_sha256": (
        "3678b65eda73775ccd50aa7a27ef5682cebc538dd8f5311f47102dbecec80cb5"
    ),
    "derived_contrast_seed": 16_709_038_045_988_039_055,
    "derived_grid_seed": 14_507_245_148_007_068_347,
    "equal_question_gap_fractions": {
        "k5_P": Fraction(199, 960),
        "k5_H": Fraction(-1, 32),
        "k20_P": Fraction(131, 960),
        "k20_H": Fraction(-17, 480),
    },
    "record_weighted_gap_fractions": {
        "k5_P": Fraction(63, 329),
        "k5_H": Fraction(-14, 329),
        "k20_P": Fraction(47, 329),
        "k20_H": Fraction(-15, 329),
    },
    "equal_question_interval_rounded": {
        "k5_P": [14.5, 27.2],
        "k5_H": [-7.5, 0.8],
        "k20_P": [8.8, 19.0],
        "k20_H": [-7.7, 0.4],
    },
    "record_weighted_interval_rounded": {
        "k5_P": [12.6, 26.1],
        "k5_H": [-9.5, 0.6],
        "k20_P": [8.9, 20.2],
        "k20_H": [-9.8, 0.3],
    },
    "opposite_direction_draws": {"5": 9267, "20": 9547},
    "grid_point_relations": {
        "same_direction": 13,
        "opposite_direction": 14,
        "tie_involved": 3,
    },
    "grid_reversal_frequency": {
        2: 4,
        3: 11,
        4: 31,
        5: 60,
        6: 113,
        7: 193,
        8: 257,
        9: 456,
        10: 794,
        11: 1331,
        12: 1426,
        13: 1595,
        14: 1897,
        15: 1240,
        16: 473,
        17: 102,
        18: 14,
        19: 2,
        20: 1,
    },
    "contrast_draw_projection_sha256": (
        "e43ac836b42baa6b7b81b218a273a32f21976486796ff912cc153afbc2eb9551"
    ),
    "grid_draw_projection_sha256": (
        "f14a9f6aca5b16b6df77298e9070ece2c6e693c330740bb0d9adcc48552c4f39"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise identity.AuditError(message)


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _question_set_sha256(question_ids: list[str]) -> str:
    payload = "".join(f"{question_id}\n" for question_id in sorted(question_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _projection_sha256(
    subset: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> str:
    lines = []
    for question_id, model in subset:
        key = (question_id, model)
        bits = []
        for cutoff in base.CUTOFFS:
            for stack in base.STACKS:
                preserved, delivered = outcomes[(key, stack, cutoff)]
                bits.extend((str(int(preserved)), str(int(delivered))))
        lines.append(f"{question_id}\t{model}\t{''.join(bits)}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _percentile(sorted_values: list[float], probability: float) -> float:
    require(bool(sorted_values), "cannot take a percentile of an empty list")
    require(0.0 <= probability <= 1.0, "percentile outside [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]


def _fraction_report(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "rate_delta": float(value),
        "percentage_points": 100.0 * float(value),
    }


def _draw_summary(values: list[float], point: Fraction) -> dict[str, Any]:
    require(len(values) == BOOTSTRAP_SAMPLES, "bootstrap draw count drifted")
    interval = _interval(values)
    return {
        "point": _fraction_report(point),
        "bootstrap_standard_error_percentage_points": sample_standard_deviation(values),
        "percentile_95_stability_interval_percentage_points": interval,
        "paper_rounded_one_decimal": [round(value, 1) for value in interval],
    }


def _load_frozen_projection(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> tuple[
    dict[str, dict[str, Any]],
    list[tuple[str, str]],
    dict[str, list[tuple[str, str]]],
    dict[tuple[tuple[str, str], str, int], tuple[bool, bool]],
]:
    payloads, source_metadata = base._validate_sources(source)
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
    for stack in base.STACKS[1:]:
        member = identity.STACK_MEMBERS[stack]
        rankings[stack] = identity._load_sensitivity(
            identity._jsonl_rows(payloads[member], member), bridge, stack
        )

    raw_differences = sorted(
        key for key, row in bridge.items() if row.get("selection_correct") is False
    )
    subset = [
        key
        for key in raw_differences
        if bridge[key].get("selected_link") in inventory
        and bridge[key].get("gold_link") in inventory
        and inventory[bridge[key]["selected_link"]]
        != inventory[bridge[key]["gold_link"]]
    ]
    require(len(subset) == EXPECTED["records"], "mismatch denominator drifted")

    question_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    records_by_model = Counter()
    for key in subset:
        question_keys[key[0]].append(key)
        records_by_model[key[1]] += 1
    multiplicity = Counter(len(values) for values in question_keys.values())
    observed_multiplicity = {
        f"exactly_{value}": multiplicity[value] for value in (1, 2, 3)
    }
    require(
        len(question_keys) == EXPECTED["unique_questions"],
        "unique-question denominator drifted",
    )
    require(
        {model: records_by_model[model] for model in identity.MODELS}
        == EXPECTED["records_by_model"],
        "model denominators drifted",
    )
    require(
        observed_multiplicity == EXPECTED["question_multiplicity"],
        "question multiplicity drifted",
    )
    require(
        _question_set_sha256(list(question_keys))
        == EXPECTED["question_set_sha256"],
        "question set drifted",
    )

    canonical_rankings = {
        stack: {
            key: base._canonicalize_audited_ranking(
                stack_rankings[key], inventory, f"{stack} {key}"
            )
            for key in subset
        }
        for stack, stack_rankings in rankings.items()
    }
    outcomes: dict[
        tuple[tuple[str, str], str, int], tuple[bool, bool]
    ] = {}
    for key in subset:
        row = bridge[key]
        selected_pid = inventory[row["selected_link"]]
        traced_pid = inventory[row["gold_link"]]
        for stack in base.STACKS:
            for cutoff in base.CUTOFFS:
                ranking = canonical_rankings[stack][key][:cutoff]
                outcomes[(key, stack, cutoff)] = (
                    selected_pid in ranking,
                    traced_pid in ranking,
                )
    projection_hash = _projection_sha256(subset, outcomes)
    require(
        projection_hash == EXPECTED["outcome_projection_sha256"],
        "outcome projection drifted",
    )
    return source_metadata, subset, dict(question_keys), outcomes


def _question_units(
    question_keys: Mapping[str, list[tuple[str, str]]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> dict[tuple[str, str, int, str], int]:
    units: dict[tuple[str, str, int, str], int] = {}
    for question_id, keys in question_keys.items():
        scale = UNIT_SCALE // len(keys)
        require(scale * len(keys) == UNIT_SCALE, "unsupported question multiplicity")
        for stack in base.STACKS:
            for cutoff in base.CUTOFFS:
                for metric_index, metric in enumerate(("P", "H")):
                    units[(question_id, stack, cutoff, metric)] = scale * sum(
                        int(outcomes[(key, stack, cutoff)][metric_index]) for key in keys
                    )
    return units


def _contrast_bootstrap(
    question_ids: list[str],
    question_keys: Mapping[str, list[tuple[str, str]]],
    units: Mapping[tuple[str, str, int, str], int],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> dict[str, Any]:
    bm25, bge = TARGET_STACKS
    labels = tuple(
        (cutoff, metric) for cutoff in TARGET_CUTOFFS for metric in ("P", "H")
    )
    unit_delta = {
        (question_id, cutoff, metric): (
            units[(question_id, bge, cutoff, metric)]
            - units[(question_id, bm25, cutoff, metric)]
        )
        for question_id in question_ids
        for cutoff, metric in labels
    }
    row_delta = {
        (question_id, cutoff, metric): [
            int(outcomes[(key, bge, cutoff)][0 if metric == "P" else 1])
            - int(outcomes[(key, bm25, cutoff)][0 if metric == "P" else 1])
            for key in question_keys[question_id]
        ]
        for question_id in question_ids
        for cutoff, metric in labels
    }
    equal_points = {
        (cutoff, metric): Fraction(
            sum(unit_delta[(question_id, cutoff, metric)] for question_id in question_ids),
            UNIT_SCALE * len(question_ids),
        )
        for cutoff, metric in labels
    }
    record_points = {
        (cutoff, metric): Fraction(
            sum(
                value
                for question_id in question_ids
                for value in row_delta[(question_id, cutoff, metric)]
            ),
            sum(len(question_keys[question_id]) for question_id in question_ids),
        )
        for cutoff, metric in labels
    }
    require(
        {f"k{k}_{metric}": value for (k, metric), value in equal_points.items()}
        == EXPECTED["equal_question_gap_fractions"],
        "equal-question contrast points drifted",
    )
    require(
        {f"k{k}_{metric}": value for (k, metric), value in record_points.items()}
        == EXPECTED["record_weighted_gap_fractions"],
        "record-weighted contrast points drifted",
    )

    derived_seed = _derived_seed(CONTRAST_LABEL)
    require(
        derived_seed == EXPECTED["derived_contrast_seed"],
        "contrast seed derivation drifted",
    )
    rng = random.Random(derived_seed)
    equal_draws = {label: [] for label in labels}
    record_draws = {label: [] for label in labels}
    opposite_direction = Counter()
    digest = hashlib.sha256()
    for draw_index in range(BOOTSTRAP_SAMPLES):
        sampled = [
            question_ids[rng.randrange(len(question_ids))]
            for _ in range(len(question_ids))
        ]
        row_denominator = sum(len(question_keys[question_id]) for question_id in sampled)
        exact_values = []
        for cutoff, metric in labels:
            q_numerator = sum(
                unit_delta[(question_id, cutoff, metric)] for question_id in sampled
            )
            r_numerator = sum(
                value
                for question_id in sampled
                for value in row_delta[(question_id, cutoff, metric)]
            )
            equal_draws[(cutoff, metric)].append(
                100.0 * q_numerator / (UNIT_SCALE * len(question_ids))
            )
            record_draws[(cutoff, metric)].append(
                100.0 * r_numerator / row_denominator
            )
            exact_values.extend((q_numerator, r_numerator))
        for cutoff in TARGET_CUTOFFS:
            if (
                exact_values[labels.index((cutoff, "P")) * 2] > 0
                and exact_values[labels.index((cutoff, "H")) * 2] < 0
            ):
                opposite_direction[cutoff] += 1
        digest.update(
            (
                f"{draw_index}\t{row_denominator}\t"
                + "\t".join(str(value) for value in exact_values)
                + "\n"
            ).encode("utf-8")
        )

    equal_report = {
        f"k{cutoff}_{metric}": _draw_summary(
            equal_draws[(cutoff, metric)], equal_points[(cutoff, metric)]
        )
        for cutoff, metric in labels
    }
    record_report = {
        f"k{cutoff}_{metric}": _draw_summary(
            record_draws[(cutoff, metric)], record_points[(cutoff, metric)]
        )
        for cutoff, metric in labels
    }
    require(
        {
            key: value["paper_rounded_one_decimal"]
            for key, value in equal_report.items()
        }
        == EXPECTED["equal_question_interval_rounded"],
        "equal-question bootstrap intervals drifted",
    )
    require(
        {
            key: value["paper_rounded_one_decimal"]
            for key, value in record_report.items()
        }
        == EXPECTED["record_weighted_interval_rounded"],
        "record-weighted bootstrap intervals drifted",
    )
    observed_opposite = {
        str(cutoff): opposite_direction[cutoff] for cutoff in TARGET_CUTOFFS
    }
    require(
        observed_opposite == EXPECTED["opposite_direction_draws"],
        "opposite-direction bootstrap counts drifted",
    )
    draw_hash = digest.hexdigest()
    if EXPECTED["contrast_draw_projection_sha256"] != "TO_BE_FILLED":
        require(
            draw_hash == EXPECTED["contrast_draw_projection_sha256"],
            "contrast draw projection drifted",
        )
    return {
        "method": (
            "paired question-cluster bootstrap; sample 160 unique question IDs "
            "with replacement and carry every contributing hosted-model row "
            "for each sampled question; linear percentile interval"
        ),
        "samples": BOOTSTRAP_SAMPLES,
        "base_seed": BASE_SEED,
        "seed_label": CONTRAST_LABEL,
        "seed_derivation": f"SHA-256({BASE_SEED}:{CONTRAST_LABEL}) first 64 bits",
        "derived_seed": derived_seed,
        "draw_projection_sha256": draw_hash,
        "equal_question": equal_report,
        "complementary_record_weighted": record_report,
        "opposite_direction_draws": {
            str(cutoff): {
                "numerator": opposite_direction[cutoff],
                "denominator": BOOTSTRAP_SAMPLES,
                "rate": opposite_direction[cutoff] / BOOTSTRAP_SAMPLES,
            }
            for cutoff in TARGET_CUTOFFS
        },
    }


def _relation(preservation_gap: int, trace_gap: int) -> str:
    if preservation_gap == 0 and trace_gap == 0:
        return "both_metrics_tied"
    if preservation_gap == 0:
        return "preservation_tie_only"
    if trace_gap == 0:
        return "trace_delivery_tie_only"
    if (preservation_gap > 0) == (trace_gap > 0):
        return "same_direction"
    return "opposite_direction"


def _grid_relations(
    sampled_questions: list[str],
    units: Mapping[tuple[str, str, int, str], int],
) -> dict[tuple[int, str, str], str]:
    relations = {}
    for cutoff in base.CUTOFFS:
        for left, right in PAIRS:
            p_gap = sum(
                units[(question_id, left, cutoff, "P")]
                - units[(question_id, right, cutoff, "P")]
                for question_id in sampled_questions
            )
            h_gap = sum(
                units[(question_id, left, cutoff, "H")]
                - units[(question_id, right, cutoff, "H")]
                for question_id in sampled_questions
            )
            relations[(cutoff, left, right)] = _relation(p_gap, h_gap)
    return relations


def _grid_bootstrap(
    question_ids: list[str],
    units: Mapping[tuple[str, str, int, str], int],
) -> dict[str, Any]:
    point_relations = _grid_relations(question_ids, units)
    point_counts = Counter(point_relations.values())
    point_summary = {
        "same_direction": point_counts["same_direction"],
        "opposite_direction": point_counts["opposite_direction"],
        "tie_involved": sum(
            point_counts[name]
            for name in (
                "both_metrics_tied",
                "preservation_tie_only",
                "trace_delivery_tie_only",
            )
        ),
    }
    require(
        point_summary == EXPECTED["grid_point_relations"],
        "point grid relation summary drifted",
    )
    point_reversals = sorted(
        cell
        for cell, relation in point_relations.items()
        if relation == "opposite_direction"
    )
    require(len(point_reversals) == 14, "point reversal inventory drifted")

    derived_seed = _derived_seed(GRID_LABEL)
    require(
        derived_seed == EXPECTED["derived_grid_seed"],
        "grid seed derivation drifted",
    )
    rng = random.Random(derived_seed)
    reversal_counts = []
    frequency = Counter()
    original_reversal_retention = Counter()
    digest = hashlib.sha256()
    relation_codes = {
        "same_direction": "S",
        "opposite_direction": "O",
        "both_metrics_tied": "B",
        "preservation_tie_only": "P",
        "trace_delivery_tie_only": "H",
    }
    ordered_cells = sorted(point_relations)
    for draw_index in range(BOOTSTRAP_SAMPLES):
        sampled = [
            question_ids[rng.randrange(len(question_ids))]
            for _ in range(len(question_ids))
        ]
        relations = _grid_relations(sampled, units)
        count = sum(value == "opposite_direction" for value in relations.values())
        reversal_counts.append(count)
        frequency[count] += 1
        for cell in point_reversals:
            original_reversal_retention[cell] += (
                relations[cell] == "opposite_direction"
            )
        digest.update(
            (
                f"{draw_index}\t{count}\t"
                + "".join(relation_codes[relations[cell]] for cell in ordered_cells)
                + "\n"
            ).encode("utf-8")
        )
    observed_frequency = {key: frequency[key] for key in sorted(frequency)}
    require(
        observed_frequency == EXPECTED["grid_reversal_frequency"],
        "grid reversal-count distribution drifted",
    )
    draw_hash = digest.hexdigest()
    if EXPECTED["grid_draw_projection_sha256"] != "TO_BE_FILLED":
        require(
            draw_hash == EXPECTED["grid_draw_projection_sha256"],
            "grid draw projection drifted",
        )
    interval = _interval([float(value) for value in reversal_counts])
    retention_rows = [
        {
            "cutoff": cutoff,
            "left_stack": left,
            "right_stack": right,
            "draws_retaining_opposite_direction": original_reversal_retention[cell],
            "retention_rate": (
                original_reversal_retention[cell] / BOOTSTRAP_SAMPLES
            ),
        }
        for cell in point_reversals
        for cutoff, left, right in (cell,)
    ]
    retention_values = [row["retention_rate"] for row in retention_rows]
    return {
        "method": (
            "question-cluster bootstrap that recomputes all five stacks, three "
            "cutoffs, and 30 dependent pairwise P/H sign relations jointly in "
            "every draw"
        ),
        "samples": BOOTSTRAP_SAMPLES,
        "base_seed": BASE_SEED,
        "seed_label": GRID_LABEL,
        "seed_derivation": f"SHA-256({BASE_SEED}:{GRID_LABEL}) first 64 bits",
        "derived_seed": derived_seed,
        "draw_projection_sha256": draw_hash,
        "point_grid": point_summary,
        "reversal_count_stability": {
            "point": point_summary["opposite_direction"],
            "bootstrap_mean": statistics.mean(reversal_counts),
            "bootstrap_standard_error": sample_standard_deviation(reversal_counts),
            "percentile_95_stability_interval": interval,
            "minimum": min(reversal_counts),
            "maximum": max(reversal_counts),
            "frequency": {str(key): value for key, value in observed_frequency.items()},
            "draws_equal_to_point": {
                "numerator": frequency[point_summary["opposite_direction"]],
                "denominator": BOOTSTRAP_SAMPLES,
                "rate": frequency[point_summary["opposite_direction"]]
                / BOOTSTRAP_SAMPLES,
            },
            "draws_at_least_point": {
                "numerator": sum(
                    count >= point_summary["opposite_direction"]
                    for count in reversal_counts
                ),
                "denominator": BOOTSTRAP_SAMPLES,
                "rate": sum(
                    count >= point_summary["opposite_direction"]
                    for count in reversal_counts
                )
                / BOOTSTRAP_SAMPLES,
            },
        },
        "original_reversal_cell_stability": {
            "cells": retention_rows,
            "minimum_retention_rate": min(retention_values),
            "median_retention_rate": statistics.median(retention_values),
            "maximum_retention_rate": max(retention_values),
        },
    }


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> dict[str, Any]:
    source_metadata, subset, question_keys, outcomes = _load_frozen_projection(source)
    question_ids = sorted(question_keys)
    units = _question_units(question_keys, outcomes)
    contrast = _contrast_bootstrap(
        question_ids, question_keys, units, outcomes
    )
    grid = _grid_bootstrap(question_ids, units)
    multiplicity = Counter(len(values) for values in question_keys.values())
    records_by_model = Counter(model for _, model in subset)
    repeated_questions = multiplicity[2] + multiplicity[3]
    repeated_records = 2 * multiplicity[2] + 3 * multiplicity[3]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_mismatch_metric_inversion_uncertainty",
        "api_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "network_used": False,
        "credentials_used": False,
        "clock_used": False,
        "random_used": True,
        "randomness": (
            "deterministic local Python random.Random streams from fixed "
            "SHA-256-derived seeds; no system entropy"
        ),
        "definition": {
            "record_unit": "one frozen (question_id, hosted_model) record",
            "cluster_unit": "one unique question ID with all contributing model rows",
            "subset": (
                "valid selector rows whose selected and traced identifiers resolve "
                "to distinct canonical passage PIDs"
            ),
            "P_at_k": "selected-object PID occurs in the recorded top-k ranking",
            "H_at_k": "dataset-traced-object PID occurs in the same ranking",
            "contrast": "BGE-M3 minus body-only BM25 at cutoffs 5 and 20",
            "grid": "all ten unordered stack pairs at cutoffs 1, 5, and 20",
            "question_equal_weighting": (
                "average over available model rows within each question and then "
                "weight all 160 questions equally"
            ),
        },
        "inputs": {
            "container_policy": "bind exact scientific member bytes, not the outer ZIP",
            "members": source_metadata,
            "outcome_projection_sha256": EXPECTED["outcome_projection_sha256"],
        },
        "denominators_and_dependence": {
            "model_question_records": len(subset),
            "unique_question_clusters": len(question_keys),
            "records_by_model": {
                model: records_by_model[model] for model in identity.MODELS
            },
            "question_model_multiplicity": {
                "exactly_1": multiplicity[1],
                "exactly_2": multiplicity[2],
                "exactly_3": multiplicity[3],
            },
            "questions_with_multiple_model_rows": {
                "numerator": repeated_questions,
                "denominator": len(question_keys),
                "rate": repeated_questions / len(question_keys),
            },
            "records_from_questions_with_multiple_model_rows": {
                "numerator": repeated_records,
                "denominator": len(subset),
                "rate": repeated_records / len(subset),
            },
            "question_set_sha256": EXPECTED["question_set_sha256"],
        },
        "bm25_to_bge_m3_contrast": contrast,
        "joint_grid_reversal_stability": grid,
        "interpretation_boundaries": [
            (
                "The percentile ranges are post-selection stability intervals "
                "conditional on the fixed mismatch stratum, not confirmatory "
                "population confidence intervals."
            ),
            (
                "The 30 grid cells are complete and mutually dependent, not 30 "
                "sampled independent trials; a binomial interval for 14/30 is invalid."
            ),
            (
                "A paired permutation p-value is not reported because stack labels "
                "were not randomized and the highlighted contrast was selected "
                "retrospectively."
            ),
            (
                "Resampling observed questions does not correct retrospective "
                "contrast selection or estimate interface, failure, or deployment "
                "prevalence."
            ),
        ],
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
        print(f"mismatch metric inversion uncertainty audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
