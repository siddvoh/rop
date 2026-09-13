#!/usr/bin/env python3
"""Replay selected/traced target divergence under non-decoded BM25 queries.

The main mismatch audit uses the deterministic visible title of the selected
identifier as its query.  This audit asks whether the observed distinction
between selected-object preservation (P) and traced-passage delivery (H) is
only an artifact of that injective construction.  It reuses three already
frozen alternatives: the selector's separately generated ``retrieval_fact``,
the direct query-rewrite baseline, and the original question.  No query is
issued and no ranking is recomputed.

The population is the same 329 hosted-model records used by the all-stack
mismatch audit: selected and traced identifiers both resolve to distinct PIDs
in the frozen passage inventory.  Query-rewrite's one upstream-invalid row is
retained with its recorded empty ranking for the prespecified intent-to-treat
view, and a complete-case sensitivity is reported separately.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping
import zipfile


SCHEMA_VERSION = 1
MODELS = ("claude", "openai", "gemini")
MODEL_LABELS = {"claude": "Claude", "openai": "GPT", "gemini": "Gemini"}
CUTOFFS = (1, 5, 20)

BRIDGE_MEMBER = (
    "data/results/released/hybridqa_natural_v2/"
    "bridge_confirmatory/details.jsonl"
)
ANSWER_MEMBER = (
    "data/results/released/hybridqa_natural_v3/"
    "answer_confirmatory/details.jsonl"
)
LINKS_MEMBER = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1/links.jsonl"
)
BRIDGE_CODE_MEMBER = "code/src/ccr/retrieval/hybridqa_bridge.py"
REWRITE_CODE_MEMBER = "code/src/ccr/retrieval/hybridqa_query_rewrite.py"
BM25_CODE_MEMBER = "code/src/ccr/retrieval/bm25.py"
CONFIG_V2_MEMBER = "code/configs/experiments/hybridqa_natural_v2.yaml"
CONFIG_V3_MEMBER = "code/configs/experiments/hybridqa_natural_v3.yaml"
BRIDGE_MANIFEST_MEMBER = (
    "data/results/released/hybridqa_natural_v2/"
    "bridge_confirmatory/run_manifest.json"
)
REWRITE_MANIFEST_MEMBER = (
    "data/results/released/hybridqa_natural_v3/"
    "query_rewrite_confirmatory/run_manifest.json"
)

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
    LINKS_MEMBER: {
        "bytes": 15_250_846,
        "records": 286_270,
        "sha256": "6e9488f4a09fbd5be96ffff23f4e956195e0c231932137ba6894bfcdfe2fb303",
    },
    BRIDGE_CODE_MEMBER: {
        "bytes": 2_729,
        "records": 55,
        "sha256": "5e0f0fa578a741e29e8f91dd5f683ba6ba1afa96f8f3beaf230ec0729e4c879b",
    },
    REWRITE_CODE_MEMBER: {
        "bytes": 1_448,
        "records": 37,
        "sha256": "d7dda8a26ad434a73ae676ae9fa602ac0a2db5f3a97b916c634b5af2edd8fe9a",
    },
    BM25_CODE_MEMBER: {
        "bytes": 8_140,
        "records": 202,
        "sha256": "55dcd1da941803e6ef3a2e48e191913de44ad971814d01d8d54b98b84bf074d3",
    },
    CONFIG_V2_MEMBER: {
        "bytes": 3_437,
        "records": 90,
        "sha256": "378d0257f568dc6306c6455b528ed56ae76208e4af9b726bbadc4e51ddf358d0",
    },
    CONFIG_V3_MEMBER: {
        "bytes": 3_054,
        "records": 84,
        "sha256": "42ea9f7975a9c34e43630b12255aec50bae5e13cbfab3499b6bfd0ece987bbd4",
    },
    BRIDGE_MANIFEST_MEMBER: {
        "bytes": 743,
        "records": 11,
        "sha256": "e34355963f3142f9d9ae2f3d162258d6b86cd757ffcea1a50b442eeb0f154a4c",
    },
    REWRITE_MANIFEST_MEMBER: {
        "bytes": 902,
        "records": 12,
        "sha256": "318a510df8a4dfa1be26dab877e902cdca39175f807e5a2cd3621a91f92426ad",
    },
}

QUERY_FORMS = (
    "decoded_selected_title_bm25",
    "model_generated_retrieval_fact_bm25",
    "direct_query_rewrite_bm25",
    "question_only_bm25",
)

# (records, P1H1, P1H0, P0H1, P0H0). These fixed targets were
# calculated independently before this replay implementation was written.
EXPECTED_CELLS: dict[str, dict[int, dict[str, tuple[int, int, int, int, int]]]] = {
    "decoded_selected_title_bm25": {
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
    "model_generated_retrieval_fact_bm25": {
        1: {
            "claude": (143, 0, 86, 3, 54),
            "openai": (85, 0, 54, 2, 29),
            "gemini": (101, 0, 67, 1, 33),
            "pooled": (329, 0, 207, 6, 116),
        },
        5: {
            "claude": (143, 16, 91, 3, 33),
            "openai": (85, 13, 54, 0, 18),
            "gemini": (101, 14, 68, 1, 18),
            "pooled": (329, 43, 213, 4, 69),
        },
        20: {
            "claude": (143, 23, 98, 0, 22),
            "openai": (85, 20, 50, 0, 15),
            "gemini": (101, 19, 69, 0, 13),
            "pooled": (329, 62, 217, 0, 50),
        },
    },
    "direct_query_rewrite_bm25": {
        1: {
            "claude": (143, 0, 54, 15, 74),
            "openai": (85, 0, 37, 15, 33),
            "gemini": (101, 0, 43, 7, 51),
            "pooled": (329, 0, 134, 37, 158),
        },
        5: {
            "claude": (143, 21, 55, 14, 53),
            "openai": (85, 16, 33, 11, 25),
            "gemini": (101, 11, 52, 10, 28),
            "pooled": (329, 48, 140, 35, 106),
        },
        20: {
            "claude": (143, 35, 57, 18, 33),
            "openai": (85, 30, 28, 12, 15),
            "gemini": (101, 16, 54, 12, 19),
            "pooled": (329, 81, 139, 42, 67),
        },
    },
    "question_only_bm25": {
        1: {
            "claude": (143, 0, 13, 5, 125),
            "openai": (85, 0, 17, 3, 65),
            "gemini": (101, 0, 26, 2, 73),
            "pooled": (329, 0, 56, 10, 263),
        },
        5: {
            "claude": (143, 5, 23, 17, 98),
            "openai": (85, 6, 21, 5, 53),
            "gemini": (101, 8, 34, 6, 53),
            "pooled": (329, 19, 78, 28, 204),
        },
        20: {
            "claude": (143, 11, 26, 23, 83),
            "openai": (85, 8, 23, 6, 48),
            "gemini": (101, 14, 38, 6, 43),
            "pooled": (329, 33, 87, 35, 174),
        },
    },
}

EXPECTED_FACT_FORM_STATS = {
    "claude": (600, 590, 2, 2, 156, 150),
    "openai": (594, 583, 0, 0, 11, 2),
    "gemini": (600, 587, 0, 0, 9, 6),
    "pooled": (1_794, 723, 136, 125, 176, 158),
}
EXPECTED_C0_FACT_EQUIVALENCE = {
    "claude": (143, 104, 105, 107, 110),
    "openai": (85, 85, 85, 85, 85),
    "gemini": (101, 100, 101, 101, 101),
    "pooled": (329, 289, 291, 293, 296),
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class AuditError(RuntimeError):
    """Raised when a bound input or replay invariant changes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def query_terms(text: str) -> str:
    """Mirror the exact frozen SQLite-FTS query canonicalization."""

    tokens = [
        token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1
    ]
    return " OR ".join(dict.fromkeys(tokens))


def _jsonl(payload: bytes, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{member}:{number}: invalid JSON") from exc
        require(isinstance(row, dict), f"{member}:{number}: row is not an object")
        rows.append(row)
    return rows


def _validate_sources(
    source: Mapping[str, bytes] | Callable[[str], bytes],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    read = source.__getitem__ if isinstance(source, Mapping) else source
    payloads: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for member in sorted(EXPECTED_MEMBERS):
        try:
            payload = read(member)
        except (KeyError, FileNotFoundError) as exc:
            raise AuditError(f"missing source member: {member}") from exc
        expected = EXPECTED_MEMBERS[member]
        records = sum(bool(line.strip()) for line in payload.splitlines())
        require(len(payload) == expected["bytes"], f"{member}: byte count drift")
        require(records == expected["records"], f"{member}: record count drift")
        require(
            sha256_bytes(payload) == expected["sha256"],
            f"{member}: SHA-256 drift",
        )
        payloads[member] = payload
        metadata[member] = {
            "bytes": len(payload),
            "records": records,
            "sha256": sha256_bytes(payload),
        }
    return payloads, metadata


def _load_inventory(rows: list[dict[str, Any]]) -> dict[str, int]:
    require(len(rows) == 286_270, "link inventory size drifted")
    inventory: dict[str, int] = {}
    pids: set[int] = set()
    for row in rows:
        link, pid = row.get("link"), row.get("pid")
        require(
            isinstance(link, str) and link.startswith("/wiki/"),
            "invalid inventory link",
        )
        require(type(pid) is int and pid >= 0, "invalid inventory PID")
        require(link not in inventory and pid not in pids, "resolver is not one-to-one")
        inventory[link] = pid
        pids.add(pid)
    require(pids == set(range(len(rows))), "inventory PIDs are not contiguous")
    return inventory


def _load_bridge(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    require(len(rows) == 1_800, "bridge row count drifted")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("question_id"), row.get("model"))
        require(
            isinstance(key[0], str) and key[1] in MODELS,
            "invalid bridge identity",
        )
        require(key not in result, f"duplicate bridge identity: {key}")
        require(row.get("status") == "succeeded", f"bridge status drift: {key}")
        require(type(row.get("valid_json")) is bool, f"bridge validity drift: {key}")
        if row["valid_json"]:
            selected, gold = row.get("selected_link"), row.get("gold_link")
            require(
                isinstance(selected, str) and selected.startswith("/wiki/"),
                f"invalid selected ID: {key}",
            )
            require(
                isinstance(gold, str) and gold.startswith("/wiki/"),
                f"invalid traced ID: {key}",
            )
            require(
                row.get("selection_correct") is (selected == gold),
                f"selection identity drift: {key}",
            )
            for field in ("retrieval_fact", "primary_predicted_query"):
                require(
                    isinstance(row.get(field), str) and row[field].strip(),
                    f"invalid {field}: {key}",
                )
            for field in ("predicted_ranking", "diagnostic_model_fact_ranking"):
                ranking = row.get(field)
                require(
                    isinstance(ranking, list)
                    and len(ranking) <= 20
                    and len(ranking) == len(set(ranking))
                    and all(
                        isinstance(link, str) and link.startswith("/wiki/")
                        for link in ranking
                    ),
                    f"invalid {field}: {key}",
                )
        result[key] = row
    require(
        Counter(model for _, model in result) == Counter({m: 600 for m in MODELS}),
        "bridge model counts drifted",
    )
    return result


def _load_answers(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    require(len(rows) == 6_600, "answer row count drifted")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("question_id"), row.get("model"), row.get("condition"))
        require(
            isinstance(key[0], str) and key[1] in MODELS and isinstance(key[2], str),
            "invalid answer identity",
        )
        require(key not in result, f"duplicate answer identity: {key}")
        ranking = row.get("ranking")
        require(
            isinstance(ranking, list)
            and len(ranking) <= 20
            and len(ranking) == len(set(ranking))
            and all(
                isinstance(link, str) and link.startswith("/wiki/")
                for link in ranking
            ),
            f"invalid answer ranking: {key}",
        )
        result[key] = row
    return result


def _cell_report(cells: Counter[tuple[bool, bool]], records: int) -> dict[str, Any]:
    values = {
        "records": records,
        "p1_h1": cells[(True, True)],
        "p1_h0": cells[(True, False)],
        "p0_h1": cells[(False, True)],
        "p0_h0": cells[(False, False)],
    }
    require(sum(values[key] for key in ("p1_h1", "p1_h0", "p0_h1", "p0_h0")) == records,
            "P/H cells do not sum to denominator")
    p_count = values["p1_h1"] + values["p1_h0"]
    h_count = values["p1_h1"] + values["p0_h1"]
    disagreement = values["p1_h0"] + values["p0_h1"]
    return {
        **values,
        "selected_object_preserved": {
            "numerator": p_count,
            "denominator": records,
            "rate": p_count / records,
        },
        "traced_passage_delivered": {
            "numerator": h_count,
            "denominator": records,
            "rate": h_count / records,
        },
        "p_h_disagreement": {
            "numerator": disagreement,
            "denominator": records,
            "rate": disagreement / records,
        },
        "p_minus_h_percentage_points": 100 * (p_count - h_count) / records,
    }


def _cell_tuple(report: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple(report[key] for key in ("records", "p1_h1", "p1_h0", "p0_h1", "p0_h0"))  # type: ignore[return-value]


def _fact_form_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in (*MODELS, "pooled"):
        subset = rows if group == "pooled" else [row for row in rows if row["model"] == group]
        raw_by_selected: dict[str, set[str]] = defaultdict(set)
        terms_by_selected: dict[str, set[str]] = defaultdict(set)
        for row in subset:
            raw_by_selected[row["selected_link"]].add(row["retrieval_fact"])
            terms_by_selected[row["selected_link"]].add(query_terms(row["retrieval_fact"]))
        observed = (
            len(subset),
            len(raw_by_selected),
            sum(len(values) > 1 for values in raw_by_selected.values()),
            sum(len(values) > 1 for values in terms_by_selected.values()),
            sum(row["retrieval_fact"] != row["primary_predicted_query"] for row in subset),
            sum(
                query_terms(row["retrieval_fact"])
                != query_terms(row["primary_predicted_query"])
                for row in subset
            ),
        )
        require(observed == EXPECTED_FACT_FORM_STATS[group], f"fact-form stats drift: {group}")
        result[group] = {
            "records": observed[0],
            "unique_selected_ids": observed[1],
            "selected_ids_with_multiple_exact_retrieval_facts": observed[2],
            "selected_ids_with_multiple_executed_bm25_term_queries": observed[3],
            "retrieval_fact_differs_from_decoded_title": observed[4],
            "executed_fact_terms_differ_from_decoded_title_terms": observed[5],
        }
    return result


def build_report_from_members(
    source: Mapping[str, bytes] | Callable[[str], bytes],
    *,
    script_path: Path | None = None,
) -> dict[str, Any]:
    payloads, source_metadata = _validate_sources(source)
    inventory = _load_inventory(_jsonl(payloads[LINKS_MEMBER], LINKS_MEMBER))
    bridge = _load_bridge(_jsonl(payloads[BRIDGE_MEMBER], BRIDGE_MEMBER))
    answers = _load_answers(_jsonl(payloads[ANSWER_MEMBER], ANSWER_MEMBER))

    # Bind the semantics needed to distinguish constructed and non-constructed
    # queries, in addition to binding the source files byte-for-byte.
    bridge_code = payloads[BRIDGE_CODE_MEMBER].decode("utf-8")
    rewrite_code = payloads[REWRITE_CODE_MEMBER].decode("utf-8")
    bm25_code = payloads[BM25_CODE_MEMBER].decode("utf-8")
    config_v3 = payloads[CONFIG_V3_MEMBER].decode("utf-8")
    require('"retrieval_fact": {"type": "string"}' in bridge_code, "bridge schema drifted")
    require("same\nentity, suitable as a search query" in bridge_code, "retrieval-fact prompt drifted")
    require("def build_query_rewrite_messages(\n    question: str, table: dict[str, Any]" in rewrite_code,
            "query-rewrite input signature drifted")
    require("selected_link" not in rewrite_code and "gold_link" not in rewrite_code,
            "query-rewrite code unexpectedly receives identifier fields")
    require("The query may name an entity, relation, or both" in rewrite_code,
            "query-rewrite free-form prompt drifted")
    require('TOKEN_PATTERN = re.compile(r"[a-z0-9]+")' in bm25_code,
            "BM25 token rule drifted")
    require('return " OR ".join(dict.fromkeys(tokens))' in bm25_code,
            "BM25 query canonicalization drifted")
    require("model_input: [question, full_table_title_headers_cells_and_explicit_links]" in config_v3,
            "query-rewrite model input drifted")
    require(
        "hosted_all_models:\n    - question_only\n    - predicted_fact\n    - query_rewrite"
        in config_v3,
        "query-rewrite condition drifted",
    )

    raw_differences = sorted(
        key for key, row in bridge.items() if row.get("selection_correct") is False
    )
    unresolvable_selected = [
        key for key in raw_differences if bridge[key].get("selected_link") not in inventory
    ]
    subset = [
        key
        for key in raw_differences
        if bridge[key].get("selected_link") in inventory
        and bridge[key].get("gold_link") in inventory
        and inventory[bridge[key]["selected_link"]] != inventory[bridge[key]["gold_link"]]
    ]
    require(len(raw_differences) == 331, "raw mismatch count drifted")
    require(len(unresolvable_selected) == 2, "unresolvable selected-ID count drifted")
    require(len(subset) == 329, "canonical C0 population drifted")
    records_by_model = Counter(model for _, model in subset)
    require(records_by_model == Counter({"claude": 143, "openai": 85, "gemini": 101}),
            "C0 model denominators drifted")

    # Extract the four frozen rankings.  Query-rewrite and question-only rows
    # were answer conditions, but their rankings are independent frozen BM25
    # outputs and can be joined without invoking the reader.
    rankings: dict[str, dict[tuple[str, str], list[str]]] = {}
    rankings["decoded_selected_title_bm25"] = {
        key: list(bridge[key]["predicted_ranking"]) for key in subset
    }
    rankings["model_generated_retrieval_fact_bm25"] = {
        key: list(bridge[key]["diagnostic_model_fact_ranking"]) for key in subset
    }
    for form, condition in (
        ("direct_query_rewrite_bm25", "query_rewrite"),
        ("question_only_bm25", "question_only"),
    ):
        rankings[form] = {}
        for question_id, model in subset:
            row = answers[(question_id, model, condition)]
            require(row.get("status") in {"succeeded", "upstream_invalid"},
                    f"{condition} status drift: {(question_id, model)}")
            if condition == "query_rewrite":
                if row["status"] == "succeeded":
                    require(
                        isinstance(row.get("query_rewrite"), str)
                        and row["query_rewrite"].strip()
                        and row.get("plan", {}).get("retrieval_query") == row["query_rewrite"],
                        f"query-rewrite execution binding drift: {(question_id, model)}",
                    )
                else:
                    require(not row["ranking"], "invalid rewrite must have empty ranking")
            rankings[form][(question_id, model)] = list(row["ranking"])

    returned_occurrences = 0
    for form in QUERY_FORMS:
        for key, ranking in rankings[form].items():
            for rank, link in enumerate(ranking[: max(CUTOFFS)], 1):
                require(link in inventory, f"{form} {key}: unresolved returned ID at rank {rank}")
                returned_occurrences += 1

    reports: dict[str, Any] = {}
    row_digest_records: list[dict[str, Any]] = []
    for form in QUERY_FORMS:
        by_cutoff: dict[str, Any] = {}
        for cutoff in CUTOFFS:
            per_model: dict[str, Any] = {}
            pooled_cells: Counter[tuple[bool, bool]] = Counter()
            for model in MODELS:
                cells: Counter[tuple[bool, bool]] = Counter()
                for key in subset:
                    if key[1] != model:
                        continue
                    row = bridge[key]
                    canonical_ranking = [inventory[link] for link in rankings[form][key][:cutoff]]
                    cell = (
                        inventory[row["selected_link"]] in canonical_ranking,
                        inventory[row["gold_link"]] in canonical_ranking,
                    )
                    cells[cell] += 1
                    pooled_cells[cell] += 1
                report = _cell_report(cells, records_by_model[model])
                require(
                    _cell_tuple(report) == EXPECTED_CELLS[form][cutoff][model],
                    f"{form}@{cutoff}/{model}: P/H cells drifted",
                )
                per_model[model] = {"display_name": MODEL_LABELS[model], **report}
            pooled = _cell_report(pooled_cells, len(subset))
            require(
                _cell_tuple(pooled) == EXPECTED_CELLS[form][cutoff]["pooled"],
                f"{form}@{cutoff}: pooled P/H cells drifted",
            )
            by_cutoff[str(cutoff)] = {
                "per_model": per_model,
                "pooled_model_question_records": pooled,
            }
        reports[form] = {"cutoffs": by_cutoff}

    for key in subset:
        row = bridge[key]
        digest_record: dict[str, Any] = {
            "question_id": key[0],
            "model": key[1],
            "selected_pid": inventory[row["selected_link"]],
            "traced_pid": inventory[row["gold_link"]],
            "forms": {},
        }
        for form in QUERY_FORMS:
            full = [inventory[link] for link in rankings[form][key]]
            digest_record["forms"][form] = {
                str(cutoff): [
                    inventory[row["selected_link"]] in full[:cutoff],
                    inventory[row["gold_link"]] in full[:cutoff],
                ]
                for cutoff in CUTOFFS
            }
        row_digest_records.append(digest_record)
    row_digest_sha256 = sha256_bytes(
        json.dumps(
            row_digest_records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    # Show directly that retrieval_fact was not a deterministic title decoder.
    valid_bridge = [row for row in bridge.values() if row["valid_json"]]
    fact_stats = _fact_form_stats(valid_bridge)
    c0_fact_equivalence: dict[str, Any] = {}
    for group in (*MODELS, "pooled"):
        keys = subset if group == "pooled" else [key for key in subset if key[1] == group]
        observed = (
            len(keys),
            sum(bridge[key]["retrieval_fact"] == bridge[key]["primary_predicted_query"] for key in keys),
            sum(
                query_terms(bridge[key]["retrieval_fact"])
                == query_terms(bridge[key]["primary_predicted_query"])
                for key in keys
            ),
            sum(
                bridge[key]["diagnostic_model_fact_ranking"]
                == bridge[key]["predicted_ranking"]
                for key in keys
            ),
            sum(
                bridge[key]["diagnostic_model_fact_ranking"][:5]
                == bridge[key]["predicted_ranking"][:5]
                for key in keys
            ),
        )
        require(observed == EXPECTED_C0_FACT_EQUIVALENCE[group], f"C0 fact equivalence drift: {group}")
        c0_fact_equivalence[group] = {
            "records": observed[0],
            "exact_raw_fact_equals_decoded_title": observed[1],
            "executed_term_query_equals_decoded_title_terms": observed[2],
            "full_top20_ranking_equals_decoded_title_ranking": observed[3],
            "ordered_top5_equals_decoded_title_top5": observed[4],
        }

    # The original question is an empirical many-to-one query with respect to
    # U: the same exact query is reused across providers that selected distinct
    # objects. It is direct finite-sample evidence of non-injectivity.
    question_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in subset:
        answer = answers[(key[0], key[1], "question_only")]
        query = answer.get("plan", {}).get("retrieval_query")
        require(isinstance(query, str) and query, f"question query missing: {key}")
        question_groups[query].append(key)
    require(len(question_groups) == 160, "exact question-query cluster count drifted")
    for query, keys in question_groups.items():
        require(
            len({tuple(rankings["question_only_bm25"][key]) for key in keys}) == 1,
            f"deterministic question ranking differs within exact query: {query!r}",
        )
    multi_target_queries = {
        query: {inventory[bridge[key]["selected_link"]] for key in keys}
        for query, keys in question_groups.items()
    }
    multi_target_queries = {
        query: targets for query, targets in multi_target_queries.items() if len(targets) > 1
    }
    multi_target_records = sum(len(question_groups[query]) for query in multi_target_queries)
    target_multiplicity = Counter(len(targets) for targets in multi_target_queries.values())
    require(len(multi_target_queries) == 40, "many-to-one question count drifted")
    require(multi_target_records == 105, "many-to-one question record count drifted")
    require(target_multiplicity == Counter({2: 33, 3: 7}), "question target multiplicity drifted")

    rewrite_status = Counter(
        answers[(key[0], key[1], "query_rewrite")]["status"] for key in subset
    )
    require(rewrite_status == Counter({"succeeded": 328, "upstream_invalid": 1}),
            "query-rewrite C0 status counts drifted")
    complete_keys = [
        key
        for key in subset
        if answers[(key[0], key[1], "query_rewrite")]["status"] == "succeeded"
    ]
    complete_cells: Counter[tuple[bool, bool]] = Counter()
    for key in complete_keys:
        row = bridge[key]
        ranking = [inventory[link] for link in rankings["direct_query_rewrite_bm25"][key][:5]]
        complete_cells[(inventory[row["selected_link"]] in ranking,
                        inventory[row["gold_link"]] in ranking)] += 1
    complete_case = _cell_report(complete_cells, len(complete_keys))
    require(_cell_tuple(complete_case) == (328, 48, 140, 35, 105),
            "query-rewrite complete-case cells drifted")

    question_models: dict[str, set[str]] = defaultdict(set)
    for question_id, model in subset:
        question_models[question_id].add(model)
    multiplicity = Counter(len(models) for models in question_models.values())
    require(multiplicity == Counter({1: 55, 2: 41, 3: 64}),
            "question/model multiplicity drifted")

    reports["decoded_selected_title_bm25"]["query_semantics"] = {
        "selected_id_is_query_input": True,
        "construction": "deterministic decoded visible title B(U)",
        "role": "reference condition challenged as injective",
    }
    reports["model_generated_retrieval_fact_bm25"]["query_semantics"] = {
        "selected_id_is_query_input": False,
        "construction": (
            "retrieval_fact is a separate free-text field co-generated with selected_link "
            "from the question and full rendered table; it is not computed by the title decoder"
        ),
        "prompt_constraint": "asked to name the same entity, so it remains identity-oriented",
        "all_valid_record_form_audit": fact_stats,
        "c0_equivalence_to_decoded_title": c0_fact_equivalence,
    }
    reports["direct_query_rewrite_bm25"]["query_semantics"] = {
        "selected_id_is_query_input": False,
        "construction": (
            "a separate model call receives only question and full rendered table and may "
            "name an entity, relation, or both; selected_link is absent from its implementation"
        ),
        "intent_to_treat_status_counts": dict(sorted(rewrite_status.items())),
        "complete_case_at_5": complete_case,
    }
    reports["question_only_bm25"]["query_semantics"] = {
        "selected_id_is_query_input": False,
        "construction": "the original question string, shared across provider-model rows",
        "exact_query_clusters": len(question_groups),
        "queries_mapping_to_multiple_selected_objects": len(multi_target_queries),
        "records_in_those_many_to_one_queries": multi_target_records,
        "distinct_selected_object_multiplicity": {
            "two": target_multiplicity[2],
            "three": target_multiplicity[3],
        },
        "same_exact_query_always_has_same_recorded_ranking": True,
    }

    verifier = Path(__file__) if script_path is None else script_path
    require(verifier.is_file(), f"replay script is missing: {verifier}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_zero_call_lossy_query_target_divergence",
        "api_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "network_used": False,
        "credentials_used": False,
        "random_used": False,
        "clock_used": False,
        "definition": {
            "record_unit": "one frozen (question_id, hosted_model) selector record",
            "population": (
                "valid selector outputs whose selected and dataset-traced identifiers "
                "both resolve to distinct PIDs under the frozen passage resolver"
            ),
            "P_at_k": "selected-object PID occurs in the named recorded BM25 top-k ranking",
            "H_at_k": "dataset-traced passage PID occurs in that same ranking",
            "query_scope": (
                "four already-recorded body-only BM25 query forms; no ranking is rerun"
            ),
        },
        "inputs": source_metadata,
        "replay": {
            "script": "scripts/recompute_lossy_query_target_divergence.py",
            "script_sha256": sha256_bytes(verifier.read_bytes()),
            "row_level_target_and_cell_digest_sha256": row_digest_sha256,
        },
        "population_verification": {
            "raw_selected_traced_identifier_differences": len(raw_differences),
            "unresolvable_selected_identifiers": len(unresolvable_selected),
            "canonical_object_distinct_records": len(subset),
            "records_by_model": {model: records_by_model[model] for model in MODELS},
            "unique_question_clusters": len(question_models),
            "question_model_multiplicity": {
                "exactly_1": multiplicity[1],
                "exactly_2": multiplicity[2],
                "exactly_3": multiplicity[3],
            },
            "returned_id_occurrences_resolved_through_20": returned_occurrences,
            "all_returned_ids_resolve": True,
            "row_level_target_and_cell_digest_sha256": row_digest_sha256,
        },
        "query_forms": reports,
        "licensed_conclusions": {
            "not_only_decoded_id_construction": (
                "P and H remain different empirical events under the co-generated "
                "retrieval_fact, the selected-ID-independent direct rewrite, and the "
                "many-to-one original-question query; divergence is therefore not "
                "created solely by deterministic title decoding B(U)."
            ),
            "direct_rewrite_result": (
                "At k=5 under direct query rewrite, P and H disagree on 175/329 "
                "intent-to-treat model-question records (53.2%); the exact cells are "
                "P1H1=48, P1H0=140, P0H1=35, P0H0=106."
            ),
            "original_question_many_to_one_result": (
                "The original-question query is exactly shared by records with distinct "
                "selected objects for 40/160 question clusters (105/329 records); at "
                "k=5 its P/H disagreement is 106/329 (32.2%)."
            ),
            "retrieval_fact_result": (
                "At k=5 under model-generated retrieval_fact, P/H disagreement is "
                "217/329 (66.0%), close to 213/329 (64.7%) under decoded titles."
            ),
        },
        "limitations": {
            "retrospective": True,
            "same_corpus": "all results use the same frozen HybridQA passage corpus and body-only BM25 executor",
            "dependence": "329 model-question records arise from 160 questions and pooled rows are descriptive",
            "retrieval_fact_identity_orientation": (
                "the retrieval_fact prompt asks for a human-readable name for the same "
                "selected entity; 291/329 C0 executed term queries equal the decoded-title terms"
            ),
            "query_rewrite_is_not_a_handoff_conformance_test": (
                "direct rewrite bypasses selected_link, so it tests whether target "
                "divergence survives a U-independent query, not whether a downstream "
                "component violated an obligation to preserve U"
            ),
            "question_only_is_not_a_handoff_conformance_test": (
                "the original question likewise does not consume U and is only a lossy-query control"
            ),
            "no_causal_answer_claim": True,
            "no_cross_corpus_or_deployment_claim": True,
        },
    }


def build_report_from_archive(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact not found: {path}")
    try:
        with zipfile.ZipFile(path) as bundle:
            names = [info.filename for info in bundle.infolist()]
            require(len(names) == len(set(names)), "artifact contains duplicate members")

            def read(member: str) -> bytes:
                try:
                    return bundle.read(member)
                except KeyError as exc:
                    raise AuditError(f"artifact missing member: {member}") from exc

            return build_report_from_members(read)
    except zipfile.BadZipFile as exc:
        raise AuditError(f"invalid artifact ZIP: {path}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = canonical_json_bytes(build_report_from_archive(args.artifact))
        if args.output is None:
            sys.stdout.buffer.write(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
    except (AuditError, OSError) as exc:
        print(f"lossy-query target-divergence audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
