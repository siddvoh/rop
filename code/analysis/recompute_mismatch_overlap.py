#!/usr/bin/env python3
"""Replay question-level overlap for differing selected/traced IDs offline.

The audit uses only frozen selector rows, the frozen hybrid-reranker rankings,
and the frozen passage-link inventory already bound by the exact-identity
handoff replay.  The inventory is a one-to-one link-to-PID resolver, so the
script's exact link membership tests are byte-equivalent to canonical PID
membership.  For corpus-resolvable selections whose canonical object differs
from the dataset-traced object, it cross-tabulates returned-object preservation
(P) and trace delivery (H) at cutoff five.  It then counts how often the same
question contributes an H record in exactly one, two, or three hosted-model
families.

No model, API, network, clock, or random input is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
import zipfile

_IDENTITY_PATH = Path(__file__).with_name("recompute_identity_handoff.py")
_IDENTITY_SPEC = importlib.util.spec_from_file_location(
    "mismatch_overlap_identity_handoff", _IDENTITY_PATH
)
if _IDENTITY_SPEC is None or _IDENTITY_SPEC.loader is None:
    raise RuntimeError(f"cannot load exact-identity replay at {_IDENTITY_PATH}")
identity = importlib.util.module_from_spec(_IDENTITY_SPEC)
_IDENTITY_SPEC.loader.exec_module(identity)


SCHEMA_VERSION = 1
STACKS = (
    "visible_label_bm25",
    "visible_label_hybrid_reranker",
)
RELEVANT_MEMBERS = (
    identity.BRIDGE_MEMBER,
    identity.STACK_MEMBERS["visible_label_hybrid_reranker"],
    identity.LINKS_MEMBER,
)

# Independent, pre-output targets. The replay stops if the joined
# question/model rows or the question-level overlap change.
EXPECTED = {
    "visible_label_bm25": {
        "per_model_cells": {
            "claude": (143, 15, 88, 2, 38),
            "openai": (85, 13, 54, 0, 18),
            "gemini": (101, 14, 68, 1, 18),
        },
        "pooled_cells": (329, 42, 210, 3, 74),
        "traced_hit_records": 45,
        "traced_hit_unique_questions": 19,
        "traced_hit_question_multiplicity": (4, 4, 11),
        "preserved_traced_hit_records": 42,
        "preserved_traced_hit_unique_questions": 17,
        "preserved_traced_hit_question_multiplicity": (3, 3, 11),
        "all_contributing_traced_hits_preserved_questions": 17,
    },
    "visible_label_hybrid_reranker": {
        "per_model_cells": {
            "claude": (143, 14, 127, 0, 2),
            "openai": (85, 17, 68, 0, 0),
            "gemini": (101, 17, 83, 0, 1),
        },
        "pooled_cells": (329, 48, 278, 0, 3),
        "traced_hit_records": 48,
        "traced_hit_unique_questions": 22,
        "traced_hit_question_multiplicity": (7, 4, 11),
        "preserved_traced_hit_records": 48,
        "preserved_traced_hit_unique_questions": 22,
        "preserved_traced_hit_question_multiplicity": (7, 4, 11),
        "all_contributing_traced_hits_preserved_questions": 22,
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


def _validate_relevant_sources(
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
        expected = identity.EXPECTED_MEMBERS[member]
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


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(denominator > 0, "rate denominator must be positive")
    require(0 <= numerator <= denominator, "invalid rate numerator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator,
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


def _multiplicity_summary(
    question_models: Mapping[str, set[str]],
) -> dict[str, Any]:
    require(question_models, "question-level traced-hit set is empty")
    require(
        all(models and models <= set(identity.MODELS) for models in question_models.values()),
        "unexpected model family in question overlap",
    )
    multiplicity = Counter(len(models) for models in question_models.values())
    unique_questions = len(question_models)
    model_question_records = sum(
        count * number_of_models for number_of_models, count in multiplicity.items()
    )
    repeated_questions = sum(
        count for number_of_models, count in multiplicity.items() if number_of_models >= 2
    )
    repeated_records = sum(
        count * number_of_models
        for number_of_models, count in multiplicity.items()
        if number_of_models >= 2
    )
    model_sets = {
        model: {
            question_id
            for question_id, models in question_models.items()
            if model in models
        }
        for model in identity.MODELS
    }
    pairwise = {
        f"{first}_and_{second}": len(model_sets[first] & model_sets[second])
        for index, first in enumerate(identity.MODELS)
        for second in identity.MODELS[index + 1 :]
    }
    return {
        "model_question_records": model_question_records,
        "unique_question_ids": unique_questions,
        "question_id_set_sha256": _question_set_sha256(set(question_models)),
        "question_model_map_sha256": _question_model_map_sha256(question_models),
        "model_family_multiplicity": {
            "exactly_1": multiplicity[1],
            "exactly_2": multiplicity[2],
            "exactly_3": multiplicity[3],
        },
        "questions_appearing_in_multiple_model_families": repeated_questions,
        "question_share_appearing_in_multiple_model_families": _rate(
            repeated_questions, unique_questions
        ),
        "records_from_questions_appearing_in_multiple_model_families": repeated_records,
        "record_share_from_questions_appearing_in_multiple_model_families": _rate(
            repeated_records, model_question_records
        ),
        "pairwise_question_overlap": pairwise,
        "all_three_model_families": len(set.intersection(*model_sets.values())),
    }


def _cell_dict(cells: Counter[tuple[bool, bool]], records: int) -> dict[str, int]:
    return {
        "records": records,
        "selected_id_preserved_traced_passage_hit": cells[(True, True)],
        "selected_id_preserved_traced_passage_miss": cells[(True, False)],
        "selected_id_not_preserved_traced_passage_hit": cells[(False, True)],
        "selected_id_not_preserved_traced_passage_miss": cells[(False, False)],
    }


def _cell_tuple(cells: Mapping[str, int]) -> tuple[int, int, int, int, int]:
    return (
        cells["records"],
        cells["selected_id_preserved_traced_passage_hit"],
        cells["selected_id_preserved_traced_passage_miss"],
        cells["selected_id_not_preserved_traced_passage_hit"],
        cells["selected_id_not_preserved_traced_passage_miss"],
    )


def _disagreement(cells: Mapping[str, int]) -> dict[str, Any]:
    """Summarize records where selected-object execution and trace delivery differ."""
    records = cells["records"]
    count = (
        cells["selected_id_preserved_traced_passage_miss"]
        + cells["selected_id_not_preserved_traced_passage_hit"]
    )
    return {
        "definition": "P differs from H on the same frozen record",
        "count": count,
        "denominator": records,
        "rate": count / records,
    }


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> dict[str, Any]:
    """Build the deterministic question-overlap report from frozen bytes."""

    payloads, source_metadata = _validate_relevant_sources(source)
    bridge = identity._load_bridge(
        identity._jsonl_rows(
            payloads[identity.BRIDGE_MEMBER], identity.BRIDGE_MEMBER
        )
    )
    inventory = identity._load_links(
        identity._jsonl_rows(payloads[identity.LINKS_MEMBER], identity.LINKS_MEMBER)
    )
    stack_rankings = {
        "visible_label_bm25": {
            key: list(row["predicted_ranking"]) if row["valid_json"] else []
            for key, row in bridge.items()
        },
        "visible_label_hybrid_reranker": identity._load_sensitivity(
            identity._jsonl_rows(
                payloads[identity.STACK_MEMBERS["visible_label_hybrid_reranker"]],
                identity.STACK_MEMBERS["visible_label_hybrid_reranker"],
            ),
            bridge,
            "visible_label_hybrid_reranker",
        ),
    }

    stack_reports: dict[str, Any] = {}
    for stack in STACKS:
        rankings = stack_rankings[stack]
        pooled_cells: Counter[tuple[bool, bool]] = Counter()
        per_model: dict[str, Any] = {}
        traced_hit_rows: dict[str, dict[str, bool]] = defaultdict(dict)
        preserved_traced_hit_question_models: dict[str, set[str]] = defaultdict(set)

        for model in identity.MODELS:
            subset = sorted(
                key
                for key, row in bridge.items()
                if key[1] == model
                and row.get("selection_correct") is False
                and row.get("selected_link") in inventory
            )
            cells: Counter[tuple[bool, bool]] = Counter()
            for key in subset:
                question_id, _ = key
                row = bridge[key]
                ranking = rankings[key][: identity.TOP_K]
                preserved = row["selected_link"] in ranking
                traced_hit = row["gold_link"] in ranking
                cells[(preserved, traced_hit)] += 1
                pooled_cells[(preserved, traced_hit)] += 1
                if traced_hit:
                    require(
                        model not in traced_hit_rows[question_id],
                        f"duplicate traced-hit row for {question_id}/{model}",
                    )
                    traced_hit_rows[question_id][model] = preserved
                    if preserved:
                        preserved_traced_hit_question_models[question_id].add(model)

            cell_report = _cell_dict(cells, len(subset))
            observed = _cell_tuple(cell_report)
            require(
                observed == EXPECTED[stack]["per_model_cells"][model],
                f"{stack}/{model}: P-by-H cells drifted: {observed}",
            )
            per_model[model] = {
                "display_name": identity.MODEL_LABELS[model],
                "p_x_h_at_5": cell_report,
                "selected_trace_disagreement": _disagreement(cell_report),
                "preservation_given_traced_hit": _rate(
                    cell_report["selected_id_preserved_traced_passage_hit"],
                    cell_report["selected_id_preserved_traced_passage_hit"]
                    + cell_report["selected_id_not_preserved_traced_passage_hit"],
                ),
            }

        pooled_report = _cell_dict(pooled_cells, sum(pooled_cells.values()))
        pooled_observed = _cell_tuple(pooled_report)
        require(
            pooled_observed == EXPECTED[stack]["pooled_cells"],
            f"{stack}: pooled P-by-H cells drifted: {pooled_observed}",
        )

        traced_hit_question_models = {
            question_id: set(model_map)
            for question_id, model_map in traced_hit_rows.items()
        }
        traced_overlap = _multiplicity_summary(traced_hit_question_models)
        preserved_overlap = _multiplicity_summary(
            preserved_traced_hit_question_models
        )
        all_preserved = sum(
            all(model_map.values()) for model_map in traced_hit_rows.values()
        )
        none_preserved = sum(
            not any(model_map.values()) for model_map in traced_hit_rows.values()
        )
        mixed_preservation = len(traced_hit_rows) - all_preserved - none_preserved

        expected = EXPECTED[stack]
        require(
            traced_overlap["model_question_records"]
            == expected["traced_hit_records"],
            f"{stack}: traced-hit record count drifted",
        )
        require(
            traced_overlap["unique_question_ids"]
            == expected["traced_hit_unique_questions"],
            f"{stack}: traced-hit unique-question count drifted",
        )
        require(
            tuple(
                traced_overlap["model_family_multiplicity"][f"exactly_{value}"]
                for value in (1, 2, 3)
            )
            == expected["traced_hit_question_multiplicity"],
            f"{stack}: traced-hit question multiplicity drifted",
        )
        require(
            preserved_overlap["model_question_records"]
            == expected["preserved_traced_hit_records"],
            f"{stack}: preserved/traced-hit record count drifted",
        )
        require(
            preserved_overlap["unique_question_ids"]
            == expected["preserved_traced_hit_unique_questions"],
            f"{stack}: preserved/traced-hit unique-question count drifted",
        )
        require(
            tuple(
                preserved_overlap["model_family_multiplicity"][f"exactly_{value}"]
                for value in (1, 2, 3)
            )
            == expected["preserved_traced_hit_question_multiplicity"],
            f"{stack}: preserved/traced-hit question multiplicity drifted",
        )
        require(
            all_preserved
            == expected["all_contributing_traced_hits_preserved_questions"],
            f"{stack}: question-collapsed preservation count drifted",
        )

        stack_reports[stack] = {
            "per_model": per_model,
            "pooled_model_question_records": {
                "p_x_h_at_5": pooled_report,
                "selected_trace_disagreement": _disagreement(pooled_report),
                "preservation_given_traced_hit": _rate(
                    pooled_report["selected_id_preserved_traced_passage_hit"],
                    pooled_report["selected_id_preserved_traced_passage_hit"]
                    + pooled_report[
                        "selected_id_not_preserved_traced_passage_hit"
                    ],
                ),
                "independence_warning": (
                    "descriptive only: the same question may contribute one record "
                    "for each hosted-model family"
                ),
            },
            "traced_hit_question_overlap": traced_overlap,
            "preserved_and_traced_hit_question_overlap": preserved_overlap,
            "question_collapsed_preservation_given_traced_hit": {
                "all_contributing_traced_hit_rows_preserved": _rate(
                    all_preserved, len(traced_hit_rows)
                ),
                "no_contributing_traced_hit_rows_preserved": none_preserved,
                "mixed_preservation_across_contributing_model_families": (
                    mixed_preservation
                ),
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_no_call_mismatch_question_overlap_replay",
        "api_calls": 0,
        "network_used": False,
        "definition": {
            "record_unit": "one frozen (question_id, model) record",
            "question_overlap_unit": (
                "one question_id, with contributing hosted-model families deduplicated"
            ),
            "subset": (
                "valid selections whose selected and dataset-traced IDs both "
                "resolve to distinct passage PIDs under the frozen resolver"
            ),
            "canonical_object_identity": (
                "zero-based passage PID in the exact frozen links.jsonl resolver; "
                "its one-to-one link map makes exact link membership equivalent"
            ),
            "P": (
                "canonical selected-object PID occurs in the canonicalized "
                "recorded visible-label top five"
            ),
            "H": (
                "canonical dataset-traced-object PID occurs in the same "
                "canonicalized recorded top five"
            ),
            "cutoff": identity.TOP_K,
            "scope": (
                "retrospective deterministic replay of frozen rows; no new reader "
                "call and no claim that pooled model-question records are independent"
            ),
        },
        "inputs": {
            "container_policy": "bind exact scientific member bytes, not the outer ZIP",
            "members": source_metadata,
        },
        "stacks": stack_reports,
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
        print(f"mismatch-overlap audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
