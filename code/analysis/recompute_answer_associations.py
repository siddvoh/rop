#!/usr/bin/env python3
"""Replay the paper's equal-five-passage answer associations.

The released answer ledger contains decoded-ID BM25, model-written rewrite,
and original-question reader arms. This audit
retains only selector/traced-ID matches and, for each arm pair, only rows where
both recorded reader contexts contain exactly five passages.  It then asks
whether paired exact-match outcomes follow whichever arm delivers the shared
selected/traced passage.

The analysis measures association across recorded contexts. The separate
deletion experiment supplies the passage-removal intervention. No model, API,
credential, clock, or network is used here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping
import zipfile


_IDENTITY_PATH = Path(__file__).with_name("recompute_identity_handoff.py")
_IDENTITY_SPEC = importlib.util.spec_from_file_location(
    "answer_association_shared_loaders", _IDENTITY_PATH
)
if _IDENTITY_SPEC is None or _IDENTITY_SPEC.loader is None:
    raise RuntimeError(f"cannot load identity replay at {_IDENTITY_PATH}")
identity = importlib.util.module_from_spec(_IDENTITY_SPEC)
_IDENTITY_SPEC.loader.exec_module(identity)


SCHEMA_VERSION = 1
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_716

PROMPT_MEMBER = "code/src/ccr/runtime/hybridqa_answers.py"
CONFIG_MEMBER = "code/configs/experiments/hybridqa_natural_v3.yaml"
RELEVANT_MEMBERS = (
    identity.BRIDGE_MEMBER,
    identity.ANSWER_MEMBER,
    PROMPT_MEMBER,
    CONFIG_MEMBER,
)
EXPECTED_MEMBERS: dict[str, dict[str, Any]] = {
    identity.BRIDGE_MEMBER: {
        "bytes": 4_991_925,
        "nonempty_lines": 1_800,
        "sha256": (
            "abf0027a3f61aa338eae4a259b8751f7adecc5ea7f79a08f25ad11b179a79fc5"
        ),
    },
    identity.ANSWER_MEMBER: {
        "bytes": 11_600_206,
        "nonempty_lines": 6_600,
        "sha256": (
            "0e4877be9bb7983f407c632d0c8bc28cf73b4787f53cc330945db2b9d0cc4a72"
        ),
    },
    PROMPT_MEMBER: {
        "bytes": 6_040,
        "nonempty_lines": 153,
        "sha256": (
            "ca16bb6b73d5d96d15f5ca2da6839f4e8ce647c0a6b8fbfd3ee854fbc36679c6"
        ),
    },
    CONFIG_MEMBER: {
        "bytes": 3_054,
        "nonempty_lines": 84,
        "sha256": (
            "42ea9f7975a9c34e43630b12255aec50bae5e13cbfab3499b6bfd0ece987bbd4"
        ),
    },
}

ARM_CONDITIONS = {
    "decoded_id_bm25": "predicted_fact",
    "rewrite": "query_rewrite",
    "question": "question_only",
}
PAIR_ORDER = (
    ("decoded_id_bm25", "question"),
    ("decoded_id_bm25", "rewrite"),
    ("rewrite", "question"),
)

# Each model target is:
# (eligible, P00, P01, P10, P11,
#  direction-01 outcomes 11/10/01/00,
#  direction-10 outcomes 11/10/01/00).
EXPECTED_PAIR_MODEL_COUNTS: dict[str, dict[str, tuple[int, ...]]] = {
    "decoded_id_bm25__vs__rewrite": {
        "claude": (435, 64, 56, 22, 293, 24, 0, 20, 12, 7, 8, 0, 7),
        "openai": (482, 68, 60, 21, 333, 32, 0, 13, 15, 10, 3, 1, 7),
        "gemini": (474, 106, 21, 20, 327, 9, 0, 11, 1, 7, 8, 0, 5),
    },
    "decoded_id_bm25__vs__question": {
        "claude": (435, 106, 14, 245, 70, 7, 0, 3, 4, 88, 109, 3, 45),
        "openai": (482, 114, 14, 275, 79, 3, 0, 5, 6, 148, 63, 6, 58),
        "gemini": (474, 113, 14, 270, 77, 7, 0, 4, 3, 151, 71, 8, 40),
    },
    "rewrite__vs__question": {
        "claude": (456, 85, 6, 283, 82, 5, 0, 1, 0, 106, 122, 2, 53),
        "openai": (509, 88, 6, 322, 93, 5, 0, 1, 0, 177, 78, 8, 59),
        "gemini": (481, 115, 14, 272, 80, 8, 0, 3, 3, 158, 76, 6, 32),
    },
}

EXPECTED_C1_COUNTS = {"claude": 456, "openai": 509, "gemini": 498}
EXPECTED_ARM_EXACT_FIVE = {
    "claude": {
        "decoded_id_bm25": (435, 435),
        "rewrite": (456, 456),
        "question": (456, 456),
    },
    "openai": {
        "decoded_id_bm25": (482, 481),
        "rewrite": (509, 508),
        "question": (509, 501),
    },
    "gemini": {
        "decoded_id_bm25": (474, 474),
        "rewrite": (481, 481),
        "question": (498, 498),
    },
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
    for member in sorted(RELEVANT_MEMBERS):
        try:
            payload = read(member)
        except (KeyError, FileNotFoundError) as exc:
            raise identity.AuditError(f"missing source member: {member}") from exc
        expected = EXPECTED_MEMBERS[member]
        digest = hashlib.sha256(payload).hexdigest()
        lines = sum(bool(line.strip()) for line in payload.splitlines())
        require(len(payload) == expected["bytes"], f"{member}: byte count drift")
        require(digest == expected["sha256"], f"{member}: SHA-256 drift")
        require(
            lines == expected["nonempty_lines"],
            f"{member}: nonempty-line count drift",
        )
        payloads[member] = payload
        metadata[member] = {
            "bytes": len(payload),
            "nonempty_lines": lines,
            "sha256": digest,
        }

    prompt = payloads[PROMPT_MEMBER]
    config = payloads[CONFIG_MEMBER]
    for fragment in (
        b"visible_bridge_fact: str | None = None",
        b"Selected table bridge:",
        b"Passage {index}",
    ):
        require(fragment in prompt, f"prompt contract fragment missing: {fragment!r}")
    for fragment in (
        b"evidence_top_k: 5",
        b"passage_char_limit: null",
        b"evidence_order: ascending_retrieval_rank",
        b"max_output_tokens: 4096",
    ):
        require(fragment in config, f"configuration fragment missing: {fragment!r}")
    return payloads, metadata


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(denominator > 0 and 0 <= numerator <= denominator, "invalid rate")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator,
    }


def _rate_or_null(numerator: int, denominator: int) -> dict[str, Any]:
    require(denominator >= 0 and 0 <= numerator <= denominator, "invalid rate")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        identity._percentile(ordered, 0.025),
        identity._percentile(ordered, 0.975),
    ]


def _item_bootstrap(
    deltas: list[int], *, label: str, estimand: str
) -> dict[str, Any]:
    require(
        bool(deltas) and all(value in {-1, 0, 1} for value in deltas),
        f"{label}: invalid paired deltas",
    )
    rng = random.Random(_derived_seed(label))
    count = len(deltas)
    draws = [
        100.0
        * sum(deltas[rng.randrange(count)] for _ in range(count))
        / count
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    interval = _interval(draws)
    return {
        "method": (
            "paired item bootstrap with replacement within the fixed audit subset; "
            "linear percentile interval"
        ),
        "estimand": estimand,
        "samples": BOOTSTRAP_SAMPLES,
        "seed_derivation": f"SHA-256({BOOTSTRAP_SEED}:{label}) first 64 bits",
        "point_estimate_percentage_points": 100.0 * sum(deltas) / count,
        "percentile_95_interval_percentage_points": interval,
        "paper_rounded_one_decimal": [round(value, 1) for value in interval],
        "sorted_record_order": True,
    }


def _question_cluster_bootstrap(
    rows: list[tuple[str, str, int]], *, label: str, estimand: str
) -> dict[str, Any]:
    require(bool(rows), f"{label}: no rows for question-cluster bootstrap")
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for question_id, model, delta in rows:
        require(model in identity.MODELS, f"{label}: invalid model")
        require(delta in {-1, 0, 1}, f"{label}: invalid delta")
        grouped[question_id].append((model, delta))
    question_ids = sorted(grouped)
    require(
        all(len(values) == len(set(model for model, _ in values)) for values in grouped.values()),
        f"{label}: duplicate model in a question cluster",
    )

    observed_by_model: dict[str, list[int]] = {model: [] for model in identity.MODELS}
    for _, model, delta in rows:
        observed_by_model[model].append(delta)
    require(
        all(observed_by_model[model] for model in identity.MODELS),
        f"{label}: every model must contribute",
    )
    pooled_point = 100.0 * sum(delta for _, _, delta in rows) / len(rows)
    model_points = {
        model: 100.0 * sum(values) / len(values)
        for model, values in observed_by_model.items()
    }
    macro_point = math.fsum(model_points.values()) / len(model_points)

    rng = random.Random(_derived_seed(label))
    pooled_draws: list[float] = []
    macro_draws: list[float] = []
    rejected_draws = 0
    while len(pooled_draws) < BOOTSTRAP_SAMPLES:
        totals = {model: 0 for model in identity.MODELS}
        counts = {model: 0 for model in identity.MODELS}
        for _ in question_ids:
            selected = question_ids[rng.randrange(len(question_ids))]
            for model, delta in grouped[selected]:
                totals[model] += delta
                counts[model] += 1
        if any(counts[model] == 0 for model in identity.MODELS):
            rejected_draws += 1
            require(
                rejected_draws <= BOOTSTRAP_SAMPLES,
                f"{label}: too many cluster draws without all models",
            )
            continue
        pooled_count = sum(counts.values())
        pooled_draws.append(100.0 * sum(totals.values()) / pooled_count)
        macro_draws.append(
            math.fsum(100.0 * totals[model] / counts[model] for model in identity.MODELS)
            / len(identity.MODELS)
        )

    pooled_interval = _interval(pooled_draws)
    macro_interval = _interval(macro_draws)
    return {
        "method": (
            "question-cluster bootstrap conditional on the fixed audit subset; "
            "resample its unique question IDs and carry all hosted-model rows "
            "for a sampled question together"
        ),
        "estimand": estimand,
        "samples": BOOTSTRAP_SAMPLES,
        "seed_derivation": f"SHA-256({BOOTSTRAP_SEED}:{label}) first 64 bits",
        "unique_question_clusters": len(question_ids),
        "rejected_draws_missing_a_model": rejected_draws,
        "pooled_record_weighted": {
            "point_estimate_percentage_points": pooled_point,
            "percentile_95_interval_percentage_points": pooled_interval,
            "paper_rounded_one_decimal": [round(value, 1) for value in pooled_interval],
        },
        "model_macro_average": {
            "model_point_estimates_percentage_points": model_points,
            "point_estimate_percentage_points": macro_point,
            "percentile_95_interval_percentage_points": macro_interval,
            "paper_rounded_one_decimal": [round(value, 1) for value in macro_interval],
        },
    }


def _outcomes(
    pairs: list[tuple[int, int]], *, first_name: str, second_name: str
) -> dict[str, Any]:
    require(
        all(first in (0, 1) and second in (0, 1) for first, second in pairs),
        "paired outcomes must be binary",
    )
    cells = Counter(pairs)
    first_only = cells[(1, 0)]
    second_only = cells[(0, 1)]
    return {
        "first_name": first_name,
        "second_name": second_name,
        "both_exact": cells[(1, 1)],
        "first_only_exact": first_only,
        "second_only_exact": second_only,
        "neither_exact": cells[(0, 0)],
        "paired_records": len(pairs),
        "discordant_pairs": first_only + second_only,
        "exact_two_sided_sign_test": identity._exact_two_sided_sign_test(
            first_only, second_only
        ),
    }


def _cells(report: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(report["both_exact"]),
        int(report["first_only_exact"]),
        int(report["second_only_exact"]),
        int(report["neither_exact"]),
    )


def _projection_sha256(rows: list[dict[str, Any]]) -> str:
    payload = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in sorted(
            rows,
            key=lambda row: (
                str(row.get("question_id", "")),
                str(row.get("model", "")),
                str(row.get("arm_a", "")),
                str(row.get("arm_b", "")),
            ),
        )
    )
    return hashlib.sha256(payload).hexdigest()


def _pair_key(arm_a: str, arm_b: str) -> str:
    return f"{arm_a}__vs__{arm_b}"


def _validate_plan_contract(
    answers: Mapping[tuple[str, str, str], Mapping[str, Any]],
    matching_keys: list[tuple[str, str]],
) -> None:
    no_hint = ("predicted_fact", "query_rewrite", "question_only")
    for question_id, model in matching_keys:
        for condition in no_hint:
            plan = answers[(question_id, model, condition)].get("plan")
            require(isinstance(plan, dict), f"{model}/{question_id}/{condition}: plan missing")
            require(
                plan.get("visible_bridge_fact") is None,
                f"{model}/{question_id}/{condition}: unexpected reader-visible hint",
            )


def _arm_inventory(
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
    answers: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in identity.MODELS:
        keys = sorted(
            key
            for key, row in bridge.items()
            if key[1] == model and row.get("selection_correct") is True
        )
        require(len(keys) == EXPECTED_C1_COUNTS[model], f"{model}: C=1 count drift")
        arms: dict[str, Any] = {}
        for arm, condition in ARM_CONDITIONS.items():
            rows = [answers[(question_id, model, condition)] for question_id, _ in keys]
            succeeded = sum(row["status"] == "succeeded" for row in rows)
            passage_counts = Counter(
                len(row["evidence_links"])
                for row in rows
                if row["status"] == "succeeded"
            )
            exact_five = sum(
                row["status"] == "succeeded" and len(row["evidence_links"]) == 5
                for row in rows
            )
            valid_exact_five = sum(
                row["status"] == "succeeded"
                and len(row["evidence_links"]) == 5
                and row["valid_json"]
                for row in rows
            )
            require(
                (exact_five, valid_exact_five) == EXPECTED_ARM_EXACT_FIVE[model][arm],
                f"{model}/{arm}: exact-five inventory drift",
            )
            arms[arm] = {
                "condition": condition,
                "C1_records": len(rows),
                "succeeded_records": succeeded,
                "succeeded_passage_count_distribution": {
                    str(count): passage_counts[count] for count in sorted(passage_counts)
                },
                "exactly_five_passage_records": exact_five,
                "valid_json_among_exactly_five": valid_exact_five,
                "invalid_json_retained_as_exact_zero_among_exactly_five": (
                    exact_five - valid_exact_five
                ),
            }
        result[model] = {
            "display_name": identity.MODEL_LABELS[model],
            "selected_id_equals_traced_id_records": len(keys),
            "arms": arms,
        }
    return result


def _pair_rows(
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
    answers: Mapping[tuple[str, str, str], Mapping[str, Any]],
    arm_a: str,
    arm_b: str,
) -> list[dict[str, Any]]:
    condition_a = ARM_CONDITIONS[arm_a]
    condition_b = ARM_CONDITIONS[arm_b]
    rows: list[dict[str, Any]] = []
    for key in sorted(bridge):
        question_id, model = key
        bridge_row = bridge[key]
        if bridge_row.get("selection_correct") is not True:
            continue
        selected_link = bridge_row["selected_link"]
        require(
            selected_link == bridge_row["gold_link"],
            f"{model}/{question_id}: C=1 identity drift",
        )
        first = answers[(question_id, model, condition_a)]
        second = answers[(question_id, model, condition_b)]
        if first["status"] != "succeeded" or second["status"] != "succeeded":
            continue
        if len(first["evidence_links"]) != 5 or len(second["evidence_links"]) != 5:
            continue
        evidence_a = tuple(first["evidence_links"])
        evidence_b = tuple(second["evidence_links"])
        rows.append(
            {
                "question_id": question_id,
                "model": model,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "P_a": int(selected_link in evidence_a),
                "P_b": int(selected_link in evidence_b),
                "exact_a": int(first["exact"]),
                "exact_b": int(second["exact"]),
                "valid_json_a": bool(first["valid_json"]),
                "valid_json_b": bool(second["valid_json"]),
                "evidence_a": list(evidence_a),
                "evidence_b": list(evidence_b),
                "top5_set_overlap": len(set(evidence_a) & set(evidence_b)),
            }
        )
    return rows


def _direction_report(rows: list[dict[str, Any]], transition: tuple[int, int]) -> dict[str, Any]:
    selected = [row for row in rows if (row["P_a"], row["P_b"]) == transition]
    pairs = [(row["exact_a"], row["exact_b"]) for row in selected]
    return {
        "P_a": transition[0],
        "P_b": transition[1],
        "records": len(selected),
        "arm_a_em": _rate_or_null(sum(first for first, _ in pairs), len(pairs)),
        "arm_b_em": _rate_or_null(sum(second for _, second in pairs), len(pairs)),
        "arm_b_minus_arm_a_percentage_points": (
            None
            if not pairs
            else 100.0 * sum(second - first for first, second in pairs) / len(pairs)
        ),
        "paired_outcomes": _outcomes(
            pairs, first_name="arm_a", second_name="arm_b"
        ),
    }


def _delivery_aligned_report(
    rows: list[dict[str, Any]], *, label: str, include_item_interval: bool
) -> dict[str, Any]:
    selected = [row for row in rows if row["P_a"] != row["P_b"]]
    pairs: list[tuple[int, int]] = []
    deltas: list[int] = []
    for row in selected:
        if row["P_a"]:
            hit, miss = row["exact_a"], row["exact_b"]
        else:
            hit, miss = row["exact_b"], row["exact_a"]
        pairs.append((hit, miss))
        deltas.append(hit - miss)
    report: dict[str, Any] = {
        "records": len(selected),
        "hit_context_em": _rate(sum(hit for hit, _ in pairs), len(pairs)),
        "miss_context_em": _rate(sum(miss for _, miss in pairs), len(pairs)),
        "hit_minus_miss_percentage_points": 100.0 * sum(deltas) / len(deltas),
        "paired_outcomes": _outcomes(
            pairs, first_name="hit_context", second_name="miss_context"
        ),
        "top5_set_overlap_distribution": {
            str(overlap): count
            for overlap, count in sorted(
                Counter(row["top5_set_overlap"] for row in selected).items()
            )
        },
    }
    if include_item_interval:
        report["paired_item_resampling"] = _item_bootstrap(
            deltas,
            label=label,
            estimand=(
                "exact match in the arm that delivers the selected/traced passage "
                "minus exact match in the arm that omits it, percentage points"
            ),
        )
    return report


def _pair_report(
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
    answers: Mapping[tuple[str, str, str], Mapping[str, Any]],
    arm_a: str,
    arm_b: str,
) -> dict[str, Any]:
    key = _pair_key(arm_a, arm_b)
    rows = _pair_rows(bridge, answers, arm_a, arm_b)
    models: dict[str, Any] = {}
    for model in identity.MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        transitions = Counter((row["P_a"], row["P_b"]) for row in model_rows)
        direction_01 = _direction_report(model_rows, (0, 1))
        direction_10 = _direction_report(model_rows, (1, 0))
        observed = (
            len(model_rows),
            transitions[(0, 0)],
            transitions[(0, 1)],
            transitions[(1, 0)],
            transitions[(1, 1)],
            *_cells(direction_01["paired_outcomes"]),
            *_cells(direction_10["paired_outcomes"]),
        )
        require(
            observed == EXPECTED_PAIR_MODEL_COUNTS[key][model],
            f"{key}/{model}: frozen pair counts drifted: {observed}",
        )
        models[model] = {
            "display_name": identity.MODEL_LABELS[model],
            "strict_matched_five_records": len(model_rows),
            "P_transitions": {
                "P0_P0": transitions[(0, 0)],
                "P0_P1": transitions[(0, 1)],
                "P1_P0": transitions[(1, 0)],
                "P1_P1": transitions[(1, 1)],
            },
            "arm_a_miss_arm_b_hit": direction_01,
            "arm_a_hit_arm_b_miss": direction_10,
            "delivery_aligned": _delivery_aligned_report(
                model_rows,
                label=f"{key}:{model}:delivery_aligned",
                include_item_interval=True,
            ),
        }

    direction_01 = _direction_report(rows, (0, 1))
    direction_10 = _direction_report(rows, (1, 0))
    switched = [row for row in rows if row["P_a"] != row["P_b"]]
    cluster_rows = []
    for row in switched:
        delta = (
            row["exact_a"] - row["exact_b"]
            if row["P_a"]
            else row["exact_b"] - row["exact_a"]
        )
        cluster_rows.append((row["question_id"], row["model"], delta))
    pooled_delivery = _delivery_aligned_report(
        rows, label=f"{key}:pooled:delivery_aligned", include_item_interval=False
    )
    pooled_delivery["question_cluster_resampling"] = _question_cluster_bootstrap(
        cluster_rows,
        label=f"{key}:question_cluster:delivery_aligned",
        estimand=(
            "exact match in the arm that delivers the selected/traced passage "
            "minus exact match in the arm that omits it, percentage points"
        ),
    )

    if key == "decoded_id_bm25__vs__rewrite":
        for name, transition, directional in (
            ("arm_a_miss_arm_b_hit", (0, 1), direction_01),
            ("arm_a_hit_arm_b_miss", (1, 0), direction_10),
        ):
            directional_rows = [
                row for row in rows if (row["P_a"], row["P_b"]) == transition
            ]
            directional["question_cluster_resampling"] = _question_cluster_bootstrap(
                [
                    (
                        row["question_id"],
                        row["model"],
                        row["exact_b"] - row["exact_a"],
                    )
                    for row in directional_rows
                ],
                label=f"{key}:question_cluster:{name}",
                estimand="arm B exact match minus arm A exact match, percentage points",
            )
            for model in identity.MODELS:
                model_directional = [
                    row for row in directional_rows if row["model"] == model
                ]
                models[model][name]["paired_item_resampling"] = _item_bootstrap(
                    [row["exact_b"] - row["exact_a"] for row in model_directional],
                    label=f"{key}:{model}:{name}",
                    estimand="arm B exact match minus arm A exact match, percentage points",
                )

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "condition_a": ARM_CONDITIONS[arm_a],
        "condition_b": ARM_CONDITIONS[arm_b],
        "strict_eligibility": (
            "C=1, both calls succeeded, and both evidence lists contain exactly five passages"
        ),
        "eligible_projection_sha256": _projection_sha256(rows),
        "models": models,
        "pooled_descriptive_records": len(rows),
        "pooled_P_transitions": {
            "P0_P0": sum(values[1] for values in EXPECTED_PAIR_MODEL_COUNTS[key].values()),
            "P0_P1": sum(values[2] for values in EXPECTED_PAIR_MODEL_COUNTS[key].values()),
            "P1_P0": sum(values[3] for values in EXPECTED_PAIR_MODEL_COUNTS[key].values()),
            "P1_P1": sum(values[4] for values in EXPECTED_PAIR_MODEL_COUNTS[key].values()),
        },
        "pooled_arm_a_miss_arm_b_hit": direction_01,
        "pooled_arm_a_hit_arm_b_miss": direction_10,
        "pooled_delivery_aligned": pooled_delivery,
    }


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> dict[str, Any]:
    payloads, source_metadata = _validate_sources(source)
    bridge = identity._load_bridge(
        identity._jsonl_rows(payloads[identity.BRIDGE_MEMBER], identity.BRIDGE_MEMBER)
    )
    answers = identity._load_answers(
        identity._jsonl_rows(payloads[identity.ANSWER_MEMBER], identity.ANSWER_MEMBER),
        bridge,
    )
    matching_keys = sorted(
        key for key, row in bridge.items() if row.get("selection_correct") is True
    )
    require(len(matching_keys) == sum(EXPECTED_C1_COUNTS.values()), "pooled C=1 count drift")
    _validate_plan_contract(answers, matching_keys)

    pair_reports = {
        _pair_key(arm_a, arm_b): _pair_report(
            bridge, answers, arm_a, arm_b
        )
        for arm_a, arm_b in PAIR_ORDER
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_answer_associations",
        "api_calls": 0,
        "network_used": False,
        "definitions": {
            "record_unit": "one frozen (question_id, hosted-model family) record",
            "aligned_identity_subset": (
                "selector-valid rows whose selected /wiki/ ID exactly equals the "
                "dataset-traced /wiki/ ID"
            ),
            "strict_matched_five": (
                "both named answer calls succeeded and each ordered evidence list "
                "contains exactly five passages"
            ),
            "P": (
                "whether the shared selected/traced ID occurs in the named arm's "
                "ordered top-five evidence list"
            ),
            "membership_difference": "one arm has P=1 and the other has P=0",
            "membership_aligned_difference": (
                "exact match in the P=1 arm minus exact match in the P=0 arm"
            ),
            "answer_scoring": (
                "frozen exact field with terminal schema-invalid answers scored zero"
            ),
        },
        "inputs": {
            "container_policy": "bind exact scientific and prompt-contract member bytes",
            "members": source_metadata,
        },
        "prompt_and_budget_contract": {
            "evidence_passages_per_strict_arm": 5,
            "passage_character_limit": None,
            "reader_output_max_tokens": 4_096,
            "evidence_order": "ascending retrieval rank",
            "decoded_id_bm25_reader_visible_hint": False,
            "rewrite_reader_visible_hint": False,
            "question_reader_visible_hint": False,
            "total_input_tokens_matched": False,
            "reason_total_input_tokens_not_matched": (
                "full passage texts have different lengths even when passage count is five"
            ),
        },
        "join_verification": {
            "bridge_records": len(bridge),
            "answer_records": len(answers),
            "selected_id_equals_traced_id_records": len(matching_keys),
            "unique_question_ids": len({question_id for question_id, _ in bridge}),
            "exact_retained_answer_inventory": True,
            "same_question_table_model_reader_join": True,
            "invalid_terminal_answers_not_dropped": True,
        },
        "aligned_identity_arm_inventory": _arm_inventory(bridge, answers),
        "pairwise_matched_five": pair_reports,
        "interpretation": {
            "estimand": "answer association on aligned-identity membership-difference rows",
            "context_factors_recorded": [
                "passage identities",
                "ranks",
                "passage lengths",
                "distractors",
                "input length"
            ],
            "companion_test": "the paper's Gemini deletion experiment supplies the passage-removal estimate"
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
        print(f"answer-association replay failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
