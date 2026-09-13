#!/usr/bin/env python3
"""Replay robustness of the complete mismatch-specific executor ordering grid.

The primary mismatch audit pools 329 ``(question_id, hosted_model)`` records
from 160 questions.  This no-call companion recomputes all 30 executor-pair and
cutoff comparisons (five executors, three cutoffs) under four deterministic
weightings:

* separately within each hosted-model family;
* with every unique question given equal total weight;
* with every hosted-model family given equal total weight; and
* within the 64-question intersection that contributes a mismatch record for
  all three model families.

All inputs are the same hash-locked selector and ranking ledgers used by
``recompute_mismatch_stack_cutoffs.py``.  No model, retrieval, API, network,
credential, random, or clock input is used.  The 30 comparisons are the
complete fixed grid, not a random sample of comparisons; the report therefore
gives exact descriptive counts rather than a binomial interval over pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
import zipfile


_BASE_PATH = Path(__file__).with_name("recompute_mismatch_stack_cutoffs.py")
_BASE_SPEC = importlib.util.spec_from_file_location(
    "mismatch_ordering_robustness_base", _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load mismatch stack replay at {_BASE_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)
identity = base.identity


SCHEMA_VERSION = 1
RELEVANT_MEMBERS = base.RELEVANT_MEMBERS
PAIRS = tuple(itertools.combinations(base.STACKS, 2))

EXPECTED = {
    "records": 329,
    "unique_questions": 160,
    "records_by_model": {"claude": 143, "openai": 85, "gemini": 101},
    "question_multiplicity": {"exactly_1": 55, "exactly_2": 41, "exactly_3": 64},
    "question_set_sha256": (
        "5f451a619152521d3af146d54afc9153dcb06f2745b34ddf6eae25e27b1c0aad"
    ),
    "common_question_set_sha256": (
        "f95bacc36501b89e926aa1f7b388c073451217a7241e25cf31018ecbcbab1848"
    ),
    "projection_sha256": (
        "3678b65eda73775ccd50aa7a27ef5682cebc538dd8f5311f47102dbecec80cb5"
    ),
}

# Exact complete-grid summaries.  Keys not shown have count zero.
EXPECTED_SUMMARIES = {
    "claude": {
        "1": {"sign_disagreement": 7, "trace_delivery_tie_only": 3},
        "5": {"sign_agreement": 5, "sign_disagreement": 5},
        "20": {"sign_agreement": 7, "sign_disagreement": 3},
    },
    "openai": {
        "1": {"sign_disagreement": 4, "trace_delivery_tie_only": 6},
        "5": {
            "sign_agreement": 5,
            "sign_disagreement": 3,
            "preservation_tie_only": 1,
            "trace_delivery_tie_only": 1,
        },
        "20": {
            "sign_agreement": 6,
            "sign_disagreement": 2,
            "preservation_tie_only": 1,
            "trace_delivery_tie_only": 1,
        },
    },
    "gemini": {
        "1": {
            "sign_disagreement": 4,
            "trace_delivery_tie_only": 5,
            "both_metrics_tied": 1,
        },
        "5": {
            "sign_agreement": 5,
            "sign_disagreement": 4,
            "preservation_tie_only": 1,
        },
        "20": {
            "sign_agreement": 6,
            "sign_disagreement": 3,
            "preservation_tie_only": 1,
        },
    },
    "equal_question": {
        "1": {"sign_disagreement": 7, "trace_delivery_tie_only": 3},
        "5": {"sign_agreement": 6, "sign_disagreement": 4},
        "20": {"sign_agreement": 7, "sign_disagreement": 3},
    },
    "equal_model": {
        "1": {"sign_disagreement": 7, "trace_delivery_tie_only": 3},
        "5": {"sign_agreement": 6, "sign_disagreement": 4},
        "20": {"sign_agreement": 7, "sign_disagreement": 3},
    },
    "claude_common_64": {
        "1": {"sign_disagreement": 4, "trace_delivery_tie_only": 6},
        "5": {"sign_agreement": 5, "sign_disagreement": 5},
        "20": {
            "sign_agreement": 6,
            "sign_disagreement": 3,
            "both_metrics_tied": 1,
        },
    },
    "openai_common_64": {
        "1": {
            "sign_disagreement": 4,
            "trace_delivery_tie_only": 5,
            "both_metrics_tied": 1,
        },
        "5": {
            "sign_agreement": 5,
            "sign_disagreement": 3,
            "preservation_tie_only": 1,
            "trace_delivery_tie_only": 1,
        },
        "20": {
            "sign_agreement": 6,
            "sign_disagreement": 2,
            "preservation_tie_only": 1,
            "trace_delivery_tie_only": 1,
        },
    },
    "gemini_common_64": {
        "1": {"sign_disagreement": 4, "trace_delivery_tie_only": 6},
        "5": {"sign_agreement": 5, "sign_disagreement": 5},
        "20": {
            "sign_agreement": 4,
            "sign_disagreement": 3,
            "preservation_tie_only": 3,
        },
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise identity.AuditError(message)


def _question_set_sha256(question_ids: set[str] | list[str]) -> str:
    payload = "".join(f"{question_id}\n" for question_id in sorted(question_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relation(preservation_gap: Fraction, trace_gap: Fraction) -> str:
    if preservation_gap == 0 and trace_gap == 0:
        return "both_metrics_tied"
    if preservation_gap == 0:
        return "preservation_tie_only"
    if trace_gap == 0:
        return "trace_delivery_tie_only"
    if (preservation_gap > 0) == (trace_gap > 0):
        return "sign_agreement"
    return "sign_disagreement"


def _fraction_report(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "rate": float(value),
        "percentage_points": 100.0 * float(value),
    }


def _ordering_report(
    rates: Mapping[tuple[str, int, str], Fraction], expected_name: str
) -> dict[str, Any]:
    cutoffs: dict[str, Any] = {}
    aggregate = Counter()
    for cutoff in base.CUTOFFS:
        comparisons = []
        relations = Counter()
        for left, right in PAIRS:
            preservation_gap = (
                rates[(left, cutoff, "preservation")]
                - rates[(right, cutoff, "preservation")]
            )
            trace_gap = (
                rates[(left, cutoff, "trace_delivery")]
                - rates[(right, cutoff, "trace_delivery")]
            )
            relation = _relation(preservation_gap, trace_gap)
            relations[relation] += 1
            aggregate[relation] += 1
            comparisons.append(
                {
                    "left_stack": left,
                    "right_stack": right,
                    "preservation_gap_left_minus_right": _fraction_report(
                        preservation_gap
                    ),
                    "trace_delivery_gap_left_minus_right": _fraction_report(
                        trace_gap
                    ),
                    "sign_relation": relation,
                }
            )
        summary = {key: relations[key] for key in sorted(relations)}
        require(sum(summary.values()) == 10, f"{expected_name}@{cutoff}: incomplete grid")
        require(
            summary == EXPECTED_SUMMARIES[expected_name][str(cutoff)],
            f"{expected_name}@{cutoff}: sign summary drifted",
        )
        cutoffs[str(cutoff)] = {
            "pairwise_sign_summary": summary,
            "pairwise_comparisons": comparisons,
        }
    all_summary = {key: aggregate[key] for key in sorted(aggregate)}
    require(sum(all_summary.values()) == 30, f"{expected_name}: incomplete grid")
    return {
        "cutoffs": cutoffs,
        "all_cutoffs_pairwise_sign_summary": all_summary,
    }


def _rates_from_keys(
    keys: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> dict[tuple[str, int, str], Fraction]:
    require(bool(keys), "cannot compute rates from an empty key set")
    counts: Counter[tuple[str, int, str]] = Counter()
    for key in keys:
        for stack in base.STACKS:
            for cutoff in base.CUTOFFS:
                preserved, delivered = outcomes[(key, stack, cutoff)]
                counts[(stack, cutoff, "preservation")] += preserved
                counts[(stack, cutoff, "trace_delivery")] += delivered
    return {name: Fraction(counts[name], len(keys)) for name in counts}


def _projection_sha256(
    subset: list[tuple[str, str]],
    outcomes: Mapping[tuple[tuple[str, str], str, int], tuple[bool, bool]],
) -> str:
    lines = []
    for question_id, model in subset:
        bits = []
        key = (question_id, model)
        for cutoff in base.CUTOFFS:
            for stack in base.STACKS:
                preserved, delivered = outcomes[(key, stack, cutoff)]
                bits.extend((str(int(preserved)), str(int(delivered))))
        lines.append(f"{question_id}\t{model}\t{''.join(bits)}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> dict[str, Any]:
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
        key
        for key, row in bridge.items()
        if row.get("selection_correct") is False
    )
    selected_resolvable = [
        key
        for key in raw_differences
        if bridge[key].get("selected_link") in inventory
    ]
    require(
        all(bridge[key].get("gold_link") in inventory for key in selected_resolvable),
        "a traced ID in the mismatch audit does not resolve",
    )
    subset = [
        key
        for key in selected_resolvable
        if inventory[bridge[key]["selected_link"]]
        != inventory[bridge[key]["gold_link"]]
    ]
    require(len(subset) == EXPECTED["records"], "mismatch denominator drifted")
    question_models: dict[str, set[str]] = defaultdict(set)
    for question_id, model in subset:
        question_models[question_id].add(model)
    require(
        len(question_models) == EXPECTED["unique_questions"],
        "unique-question denominator drifted",
    )
    records_by_model = Counter(model for _, model in subset)
    observed_by_model = {
        model: records_by_model[model] for model in identity.MODELS
    }
    require(
        observed_by_model == EXPECTED["records_by_model"],
        "per-model denominators drifted",
    )
    multiplicity = Counter(len(models) for models in question_models.values())
    observed_multiplicity = {
        f"exactly_{value}": multiplicity[value] for value in (1, 2, 3)
    }
    require(
        observed_multiplicity == EXPECTED["question_multiplicity"],
        "question multiplicity drifted",
    )
    require(
        _question_set_sha256(set(question_models))
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
        bridge_row = bridge[key]
        for stack in base.STACKS:
            for cutoff in base.CUTOFFS:
                ranking = canonical_rankings[stack][key][:cutoff]
                outcomes[(key, stack, cutoff)] = (
                    inventory[bridge_row["selected_link"]] in ranking,
                    inventory[bridge_row["gold_link"]] in ranking,
                )
    require(
        _projection_sha256(subset, outcomes) == EXPECTED["projection_sha256"],
        "ordering projection drifted",
    )

    per_model = {}
    model_rates = {}
    for model in identity.MODELS:
        keys = [key for key in subset if key[1] == model]
        rates = _rates_from_keys(keys, outcomes)
        model_rates[model] = rates
        per_model[model] = {
            "display_name": identity.MODEL_LABELS[model],
            "unique_questions": len(keys),
            **_ordering_report(rates, model),
        }

    equal_question_rates: dict[tuple[str, int, str], Fraction] = {}
    for stack in base.STACKS:
        for cutoff in base.CUTOFFS:
            for metric_index, metric in enumerate(
                ("preservation", "trace_delivery")
            ):
                question_means = []
                for question_id, models in question_models.items():
                    question_means.append(
                        Fraction(
                            sum(
                                outcomes[((question_id, model), stack, cutoff)][
                                    metric_index
                                ]
                                for model in models
                            ),
                            len(models),
                        )
                    )
                equal_question_rates[(stack, cutoff, metric)] = sum(
                    question_means, start=Fraction(0)
                ) / len(question_means)

    equal_model_rates = {
        name: sum(
            (model_rates[model][name] for model in identity.MODELS),
            start=Fraction(0),
        )
        / len(identity.MODELS)
        for name in next(iter(model_rates.values()))
    }

    common_questions = sorted(
        question_id
        for question_id, models in question_models.items()
        if len(models) == len(identity.MODELS)
    )
    require(len(common_questions) == 64, "common-question denominator drifted")
    require(
        _question_set_sha256(common_questions)
        == EXPECTED["common_question_set_sha256"],
        "common-question set drifted",
    )
    common_per_model = {}
    for model in identity.MODELS:
        rates = _rates_from_keys(
            [(question_id, model) for question_id in common_questions], outcomes
        )
        common_per_model[model] = {
            "display_name": identity.MODEL_LABELS[model],
            "unique_questions": len(common_questions),
            **_ordering_report(rates, f"{model}_common_64"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_mismatch_ordering_robustness",
        "api_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "network_used": False,
        "credentials_used": False,
        "random_used": False,
        "clock_used": False,
        "definition": {
            "record_unit": (
                "one frozen (question_id, hosted_model) record whose selected "
                "and traced IDs resolve to distinct canonical passage PIDs"
            ),
            "per_model_unit": (
                "one unique question within a hosted-model family; a family has "
                "at most one record for each question"
            ),
            "equal_question_estimand": (
                "average P or H within each question over only its available "
                "model-family mismatch records, then weight all 160 questions equally"
            ),
            "equal_model_estimand": (
                "average the three family-specific P or H rates with weight one-third"
            ),
            "common_support_estimand": (
                "repeat each family-specific grid on the 64 questions contributing "
                "mismatch records for all three families"
            ),
            "comparison_grid": (
                "all ten unordered executor pairs at each cutoff in {1,5,20}"
            ),
            "sign_disagreement": (
                "the nonzero P gap and nonzero H gap have opposite signs"
            ),
        },
        "inputs": {
            "container_policy": "bind exact scientific member bytes, not the outer ZIP",
            "members": source_metadata,
            "outcome_projection_sha256": EXPECTED["projection_sha256"],
        },
        "denominators": {
            "model_question_records": len(subset),
            "unique_questions": len(question_models),
            "records_by_model": observed_by_model,
            "question_model_multiplicity": observed_multiplicity,
            "common_all_three_model_questions": len(common_questions),
            "question_set_sha256": EXPECTED["question_set_sha256"],
            "common_question_set_sha256": EXPECTED[
                "common_question_set_sha256"
            ],
        },
        "per_model_family": per_model,
        "question_equal_weighted": {
            "unique_questions": len(question_models),
            **_ordering_report(equal_question_rates, "equal_question"),
        },
        "model_equal_weighted": {
            "hosted_model_families": len(identity.MODELS),
            **_ordering_report(equal_model_rates, "equal_model"),
        },
        "common_question_intersection": {
            "unique_questions": len(common_questions),
            "per_model_family": common_per_model,
        },
        "uncertainty_boundary": (
            "The 30 pair-cutoff comparisons are the complete fixed grid; "
            "they are not 30 sampled independent trials, so a binomial interval "
            "for the reversal count would be inapplicable. The separate frozen "
            "question-cluster report supplies dependence-aware intervals for the "
            "headline preservation-gap magnitude."
        ),
        "interpretation": (
            "Equal-question and equal-family weighting both reproduce 14 of 30 "
            "sign disagreements. Every family separately shows disagreements, "
            "with 15 for Claude, 9 for GPT, and 11 for Gemini; on common support "
            "the counts are 12, 9, and 12. The pooled result is therefore not an "
            "artifact of one family or of repeated-question row weights, although "
            "the exact count is family-dependent and ties are retained explicitly."
        ),
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
        print(f"mismatch ordering robustness audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
