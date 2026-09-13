#!/usr/bin/env python3
"""Replay mismatch preservation gaps with question-cluster uncertainty.

The audit is restricted to the same frozen hosted-model records used by the
mismatch-overlap analysis: the selector output is valid and corpus-resolvable,
and its selected and dataset-traced identifiers resolve to distinct canonical
passage objects. For each paired record, it compares selected-object preservation at rank five
under the released body-only BM25 ranking and the released hybrid RRF plus BGE
reranking ranking.

The primary addition is dependence-aware accounting.  A question can
contribute one record for each hosted-model family.  The report therefore
gives both the original record-weighted gap and an equal-question macro gap,
then resamples unique question IDs so every hosted-model row belonging to a
sampled question travels together.

No model, retrieval, API, network, credential, or clock input is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import math
from pathlib import Path
import random
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


_MISMATCH_PATH = Path(__file__).with_name("recompute_mismatch_overlap.py")
_MISMATCH_SPEC = importlib.util.spec_from_file_location(
    "mismatch_preservation_gap_overlap", _MISMATCH_PATH
)
if _MISMATCH_SPEC is None or _MISMATCH_SPEC.loader is None:
    raise RuntimeError(f"cannot load mismatch replay at {_MISMATCH_PATH}")
mismatch = importlib.util.module_from_spec(_MISMATCH_SPEC)
_MISMATCH_SPEC.loader.exec_module(mismatch)
identity = mismatch.identity


SCHEMA_VERSION = 1
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_716
BOOTSTRAP_LABEL = (
    "body_only_bm25_vs_rrf_rerank_mismatch_question_cluster_v1"
)
RELEVANT_MEMBERS = mismatch.RELEVANT_MEMBERS

# Fixed targets for the frozen projection.
EXPECTED = {
    "records": 329,
    "unique_questions": 160,
    "question_multiplicity": (55, 41, 64),
    "records_by_model": {"claude": 143, "openai": 85, "gemini": 101},
    "body_only_bm25_preserved": 252,
    "rrf_rerank_preserved": 326,
    "improved_records": 74,
    "regressed_records": 0,
    "tied_records": 255,
    "positive_gap_questions": 43,
    "zero_gap_questions": 117,
    "negative_gap_questions": 0,
    "record_projection_sha256": (
        "dc2db0702609c3834b15d83edabbbf8eca5c6ddbfc923e5b2ec0a3c864b57816"
    ),
    "question_projection_sha256": (
        "6a0042d5e245a00aa19671c2082b708fdb8870b767367afab21fbea645176666"
    ),
    "question_set_sha256": (
        "5f451a619152521d3af146d54afc9153dcb06f2745b34ddf6eae25e27b1c0aad"
    ),
    "derived_seed": 16_771_578_087_979_229_916,
    "question_macro_interval_rounded": [17.8, 30.4],
    "record_weighted_interval_rounded": [16.3, 29.3],
}




def require(condition: bool, message: str) -> None:
    if not condition:
        raise identity.AuditError(message)


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _percentile(sorted_values: list[float], probability: float) -> float:
    require(bool(sorted_values), "cannot take percentile of empty values")
    require(0.0 <= probability <= 1.0, "percentile outside [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]


def _record_projection_sha256(
    rows: list[tuple[str, str, int, int]],
) -> str:
    payload = "".join(
        f"{question_id}\t{model}\t{bm25}\t{rerank}\t{rerank - bm25}\n"
        for question_id, model, bm25, rerank in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _question_projection_sha256(
    grouped: Mapping[str, list[tuple[str, int, int]]],
) -> str:
    payload = "".join(
        f"{question_id}\t"
        + ",".join(
            f"{model}:{bm25}:{rerank}"
            for model, bm25, rerank in sorted(values)
        )
        + "\n"
        for question_id, values in sorted(grouped.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _question_set_sha256(question_ids: list[str]) -> str:
    payload = "".join(f"{question_id}\n" for question_id in sorted(question_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(denominator > 0, "rate denominator must be positive")
    require(0 <= numerator <= denominator, "invalid rate numerator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator,
    }


def _summarize_draws(values: list[float], point: float) -> dict[str, Any]:
    require(len(values) == BOOTSTRAP_SAMPLES, "bootstrap draw count drifted")
    interval = _interval(values)
    return {
        "point_estimate_percentage_points": point,
        "bootstrap_standard_error_percentage_points": sample_standard_deviation(values),
        "percentile_95_interval_percentage_points": interval,
        "paper_rounded_one_decimal": [round(value, 1) for value in interval],
    }


def _cluster_bootstrap(
    grouped: Mapping[str, list[tuple[str, int, int]]],
) -> dict[str, Any]:
    question_ids = sorted(grouped)
    require(
        len(question_ids) == EXPECTED["unique_questions"],
        "question-cluster denominator drifted",
    )

    question_bm25 = {
        question_id: sum(row[1] for row in values) / len(values)
        for question_id, values in grouped.items()
    }
    question_rerank = {
        question_id: sum(row[2] for row in values) / len(values)
        for question_id, values in grouped.items()
    }
    question_delta = {
        question_id: question_rerank[question_id] - question_bm25[question_id]
        for question_id in question_ids
    }

    question_bm25_point = 100.0 * math.fsum(question_bm25.values()) / len(question_ids)
    question_rerank_point = (
        100.0 * math.fsum(question_rerank.values()) / len(question_ids)
    )
    question_gap_point = 100.0 * math.fsum(question_delta.values()) / len(question_ids)

    all_rows = [row for question_id in question_ids for row in grouped[question_id]]
    record_bm25_point = 100.0 * sum(row[1] for row in all_rows) / len(all_rows)
    record_rerank_point = 100.0 * sum(row[2] for row in all_rows) / len(all_rows)
    record_gap_point = record_rerank_point - record_bm25_point

    rng = random.Random(_derived_seed(BOOTSTRAP_LABEL))
    q_bm25_draws: list[float] = []
    q_rerank_draws: list[float] = []
    q_gap_draws: list[float] = []
    record_gap_draws: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [
            question_ids[rng.randrange(len(question_ids))]
            for _ in range(len(question_ids))
        ]
        q_bm25_draws.append(
            100.0 * math.fsum(question_bm25[value] for value in sampled) / len(sampled)
        )
        q_rerank_draws.append(
            100.0 * math.fsum(question_rerank[value] for value in sampled) / len(sampled)
        )
        q_gap_draws.append(
            100.0 * math.fsum(question_delta[value] for value in sampled) / len(sampled)
        )
        sampled_rows = [row for value in sampled for row in grouped[value]]
        record_gap_draws.append(
            100.0
            * sum(row[2] - row[1] for row in sampled_rows)
            / len(sampled_rows)
        )

    return {
        "method": (
            "paired question-cluster bootstrap conditional on the fixed mismatch "
            "subset; resample 160 unique question IDs with replacement and carry "
            "all contributing hosted-model rows for each sampled question together; "
            "linear 2.5th and 97.5th percentile interval"
        ),
        "samples": BOOTSTRAP_SAMPLES,
        "base_seed": BOOTSTRAP_SEED,
        "seed_derivation": (
            f"SHA-256({BOOTSTRAP_SEED}:{BOOTSTRAP_LABEL}) first 64 bits"
        ),
        "derived_seed": _derived_seed(BOOTSTRAP_LABEL),
        "sorted_question_order": True,
        "question_equal_weighted": {
            "body_only_bm25_preservation": _summarize_draws(
                q_bm25_draws, question_bm25_point
            ),
            "rrf_rerank_preservation": _summarize_draws(
                q_rerank_draws, question_rerank_point
            ),
            "paired_preservation_gap": _summarize_draws(
                q_gap_draws, question_gap_point
            ),
        },
        "complementary_record_weighted_gap": _summarize_draws(
            record_gap_draws, record_gap_point
        ),
    }


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> dict[str, Any]:
    """Build the path-free deterministic report from frozen member bytes."""

    payloads, source_metadata = mismatch._validate_relevant_sources(source)
    bridge = identity._load_bridge(
        identity._jsonl_rows(
            payloads[identity.BRIDGE_MEMBER], identity.BRIDGE_MEMBER
        )
    )
    inventory = identity._load_links(
        identity._jsonl_rows(payloads[identity.LINKS_MEMBER], identity.LINKS_MEMBER)
    )
    rerankings = identity._load_sensitivity(
        identity._jsonl_rows(
            payloads[identity.STACK_MEMBERS["visible_label_hybrid_reranker"]],
            identity.STACK_MEMBERS["visible_label_hybrid_reranker"],
        ),
        bridge,
        "visible_label_hybrid_reranker",
    )

    rows: list[tuple[str, str, int, int]] = []
    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for key, row in sorted(bridge.items()):
        question_id, model = key
        if not (
            row.get("selection_correct") is False
            and row.get("selected_link") in inventory
        ):
            continue
        require(
            row["selected_link"] != row["gold_link"],
            f"{question_id}/{model}: mismatch subset contains equal IDs",
        )
        bm25 = int(
            row["selected_link"] in row["predicted_ranking"][: identity.TOP_K]
        )
        rerank = int(
            row["selected_link"] in rerankings[key][: identity.TOP_K]
        )
        rows.append((question_id, model, bm25, rerank))
        grouped[question_id].append((model, bm25, rerank))

    records_by_model = Counter(model for _, model, _, _ in rows)
    multiplicity = Counter(len(values) for values in grouped.values())
    improvements = sum(rerank > bm25 for _, _, bm25, rerank in rows)
    regressions = sum(rerank < bm25 for _, _, bm25, rerank in rows)
    ties = len(rows) - improvements - regressions
    question_deltas = {
        question_id: sum(rerank - bm25 for _, bm25, rerank in values) / len(values)
        for question_id, values in grouped.items()
    }

    require(len(rows) == EXPECTED["records"], "record denominator drifted")
    require(
        len(grouped) == EXPECTED["unique_questions"],
        "unique-question denominator drifted",
    )
    require(
        tuple(multiplicity[value] for value in (1, 2, 3))
        == EXPECTED["question_multiplicity"],
        "question multiplicity drifted",
    )
    require(dict(records_by_model) == EXPECTED["records_by_model"], "model counts drifted")
    require(
        sum(bm25 for _, _, bm25, _ in rows)
        == EXPECTED["body_only_bm25_preserved"],
        "BM25 preservation count drifted",
    )
    require(
        sum(rerank for _, _, _, rerank in rows)
        == EXPECTED["rrf_rerank_preserved"],
        "reranker preservation count drifted",
    )
    require(
        (improvements, regressions, ties)
        == (
            EXPECTED["improved_records"],
            EXPECTED["regressed_records"],
            EXPECTED["tied_records"],
        ),
        "paired record outcome counts drifted",
    )
    require(
        (
            sum(value > 0 for value in question_deltas.values()),
            sum(value == 0 for value in question_deltas.values()),
            sum(value < 0 for value in question_deltas.values()),
        )
        == (
            EXPECTED["positive_gap_questions"],
            EXPECTED["zero_gap_questions"],
            EXPECTED["negative_gap_questions"],
        ),
        "question gap direction counts drifted",
    )

    record_hash = _record_projection_sha256(rows)
    question_hash = _question_projection_sha256(grouped)
    question_set_hash = _question_set_sha256(list(grouped))
    require(
        record_hash == EXPECTED["record_projection_sha256"],
        "record projection SHA-256 drifted",
    )
    require(
        question_hash == EXPECTED["question_projection_sha256"],
        "question projection SHA-256 drifted",
    )
    require(
        question_set_hash == EXPECTED["question_set_sha256"],
        "question set SHA-256 drifted",
    )

    model_reports: dict[str, Any] = {}
    for model in identity.MODELS:
        subset = [row for row in rows if row[1] == model]
        denominator = len(subset)
        bm25_count = sum(row[2] for row in subset)
        rerank_count = sum(row[3] for row in subset)
        model_reports[model] = {
            "display_name": identity.MODEL_LABELS[model],
            "records": denominator,
            "body_only_bm25_preservation": _rate(bm25_count, denominator),
            "rrf_rerank_preservation": _rate(rerank_count, denominator),
            "paired_gap_percentage_points": (
                100.0 * (rerank_count - bm25_count) / denominator
            ),
        }

    record_count = len(rows)
    bm25_count = sum(row[2] for row in rows)
    rerank_count = sum(row[3] for row in rows)
    question_ids = sorted(grouped)
    question_bm25_rate = math.fsum(
        sum(row[1] for row in grouped[question_id]) / len(grouped[question_id])
        for question_id in question_ids
    ) / len(question_ids)
    question_rerank_rate = math.fsum(
        sum(row[2] for row in grouped[question_id]) / len(grouped[question_id])
        for question_id in question_ids
    ) / len(question_ids)

    repeated_questions = multiplicity[2] + multiplicity[3]
    repeated_records = 2 * multiplicity[2] + 3 * multiplicity[3]
    bootstrap = _cluster_bootstrap(grouped)
    require(
        bootstrap["derived_seed"] == EXPECTED["derived_seed"],
        "derived bootstrap seed drifted",
    )
    require(
        bootstrap["question_equal_weighted"]["paired_preservation_gap"]
        ["paper_rounded_one_decimal"]
        == EXPECTED["question_macro_interval_rounded"],
        "question-macro bootstrap interval drifted",
    )
    require(
        bootstrap["complementary_record_weighted_gap"]
        ["paper_rounded_one_decimal"]
        == EXPECTED["record_weighted_interval_rounded"],
        "record-weighted cluster interval drifted",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_mismatch_preservation_gap_uncertainty",
        "api_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "network_used": False,
        "credentials_used": False,
        "clock_used": False,
        "definition": {
            "record_unit": "one frozen (question_id, hosted_model) record",
            "subset": (
                "valid selections whose selected and dataset-traced IDs both "
                "resolve to distinct passage PIDs under the frozen resolver"
            ),
            "canonical_object_identity": (
                "zero-based passage PID under the exact one-to-one links.jsonl resolver"
            ),
            "paired_executors": {
                "body_only_bm25": (
                    "the released predicted_ranking field, queried from the "
                    "selected ID's visible label over passage body text"
                ),
                "rrf_rerank": (
                    "the released hybrid RRF plus BGE-reranker-v2-m3 ranking "
                    "for the same visible-label query"
                ),
            },
            "outcome": (
                "canonical selected-object PID occurs in the canonicalized "
                "recorded top five"
            ),
            "cutoff": identity.TOP_K,
            "gap_direction": "RRF-plus-rerank preservation minus body-only BM25",
            "question_macro": (
                "within each contributing question, average the binary outcome "
                "over its one to three hosted-model records; then weight all "
                "160 contributing questions equally"
            ),
            "scope": (
                "conditional retrospective uncertainty for this fixed mismatch "
                "subset; it does not estimate deployment prevalence or model-call "
                "uncertainty"
            ),
        },
        "inputs": {
            "container_policy": "bind exact scientific member bytes, not the outer ZIP",
            "members": source_metadata,
            "record_projection_sha256": record_hash,
            "question_projection_sha256": question_hash,
            "question_id_set_sha256": question_set_hash,
        },
        "denominators_and_dependence": {
            "paired_model_question_records": record_count,
            "unique_question_clusters": len(grouped),
            "records_by_model": {
                model: records_by_model[model] for model in identity.MODELS
            },
            "question_model_multiplicity": {
                "exactly_1": multiplicity[1],
                "exactly_2": multiplicity[2],
                "exactly_3": multiplicity[3],
            },
            "questions_with_multiple_hosted_model_records": _rate(
                repeated_questions, len(grouped)
            ),
            "records_from_questions_with_multiple_hosted_model_records": _rate(
                repeated_records, record_count
            ),
            "dependence_warning": (
                "the 329 model-question rows are paired across executors but are "
                "not 329 independent questions; all rows sharing a question ID "
                "must remain together in uncertainty resampling"
            ),
        },
        "paired_record_weighted": {
            "body_only_bm25_preservation": _rate(bm25_count, record_count),
            "rrf_rerank_preservation": _rate(rerank_count, record_count),
            "paired_gap_percentage_points": (
                100.0 * (rerank_count - bm25_count) / record_count
            ),
            "paired_outcomes": {
                "rrf_rerank_only_preserves": improvements,
                "body_only_bm25_only_preserves": regressions,
                "same_preservation_state": ties,
            },
            "per_model": model_reports,
            "model_macro_average_gap_percentage_points": math.fsum(
                values["paired_gap_percentage_points"]
                for values in model_reports.values()
            )
            / len(model_reports),
        },
        "question_equal_weighted_macro": {
            "unique_questions": len(grouped),
            "body_only_bm25_preservation_rate": question_bm25_rate,
            "rrf_rerank_preservation_rate": question_rerank_rate,
            "paired_gap_percentage_points": (
                100.0 * (question_rerank_rate - question_bm25_rate)
            ),
            "question_gap_direction": {
                "positive": sum(value > 0 for value in question_deltas.values()),
                "zero": sum(value == 0 for value in question_deltas.values()),
                "negative": sum(value < 0 for value in question_deltas.values()),
            },
        },
        "question_cluster_resampling": bootstrap,
        "interpretation": (
            "The equal-question macro is the dependence-robust descriptive "
            "summary requested here. The complementary record-weighted cluster "
            "interval preserves the paper's 329-record estimand while correcting "
            "its uncertainty unit from records to questions."
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
        payload = identity.canonical_json_bytes(
            build_report_from_archive(args.artifact)
        )
        if args.output is None:
            sys.stdout.buffer.write(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
    except (identity.AuditError, OSError) as exc:
        print(f"mismatch preservation-gap audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
