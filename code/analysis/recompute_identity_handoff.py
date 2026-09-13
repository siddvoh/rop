#!/usr/bin/env python3
"""Shared loaders for the paper's HybridQA retrieval and answer replays."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from urllib.parse import unquote
from typing import Any, Callable, Mapping


MODELS = ("claude", "openai", "gemini")
MODEL_LABELS = {"claude": "Claude", "openai": "GPT", "gemini": "Gemini"}
PRIMARY_ANSWER_CONDITIONS = (
    "question_only",
    "predicted_fact",
    "query_rewrite",
)
GEMINI_PROVENANCE_CONDITIONS = ("no_kb", "oracle_passage")
ANSWER_CONDITIONS_BY_MODEL = {
    "claude": PRIMARY_ANSWER_CONDITIONS,
    "openai": PRIMARY_ANSWER_CONDITIONS,
    "gemini": (*PRIMARY_ANSWER_CONDITIONS, *GEMINI_PROVENANCE_CONDITIONS),
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
SENSITIVITY_ROOT = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1"
)
STACK_MEMBERS = {
    "visible_label_bm25": ANSWER_MEMBER,
    "visible_label_bge_m3": f"{SENSITIVITY_ROOT}/bge_m3/details.jsonl",
    "visible_label_colbert_v2": f"{SENSITIVITY_ROOT}/colbert_v2/details.jsonl",
    "visible_label_hybrid_rrf": f"{SENSITIVITY_ROOT}/hybrid_rrf/details.jsonl",
    "visible_label_hybrid_reranker": (
        f"{SENSITIVITY_ROOT}/hybrid_rrf_bge_reranker_v2_m3/details.jsonl"
    ),
}
LINKS_MEMBER = f"{SENSITIVITY_ROOT}/links.jsonl"

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
    STACK_MEMBERS["visible_label_bge_m3"]: {
        "bytes": 23_539_533,
        "records": 2_400,
        "sha256": "ce73c83e1ac0ef6b77f129b8f23060dcfb4fceb02c1070cc5454ff8889c317ec",
    },
    STACK_MEMBERS["visible_label_colbert_v2"]: {
        "bytes": 21_114_400,
        "records": 2_400,
        "sha256": "6411ec10ab55373e2c745c652678f26529045fc33ea9dd5b5ca2205213bd7bc2",
    },
    STACK_MEMBERS["visible_label_hybrid_rrf"]: {
        "bytes": 5_308_408,
        "records": 2_400,
        "sha256": "916fc87fb3865dc4a13e13997286028b0bc71a6a456f70648d86b683005af1d7",
    },
    STACK_MEMBERS["visible_label_hybrid_reranker"]: {
        "bytes": 6_894_412,
        "records": 2_400,
        "sha256": "0550dc6690e3e44191474ae489905565b0081e887fd5836ea76d01edd9969a81",
    },
    LINKS_MEMBER: {
        "bytes": 15_250_846,
        "records": 286_270,
        "sha256": "6e9488f4a09fbd5be96ffff23f4e956195e0c231932137ba6894bfcdfe2fb303",
    },
}


class AuditError(RuntimeError):
    """Raised when a released input or a deterministic join changes."""


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
        raise AuditError(f"{member}: invalid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{member}:{line_number}: invalid JSON") from exc
        require(isinstance(row, dict), f"{member}:{line_number}: expected object")
        rows.append(row)
    return rows


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
        records = sum(bool(line.strip()) for line in payload.splitlines())
        require(len(payload) == expected["bytes"], f"{member}: byte count changed")
        require(digest == expected["sha256"], f"{member}: SHA-256 changed")
        require(records == expected["records"], f"{member}: record count changed")
        payloads[member] = payload
        metadata[member] = {
            "bytes": len(payload),
            "records": records,
            "sha256": digest,
        }
    return payloads, metadata


def _link_label(link: str) -> str:
    value = unquote(link.removeprefix("/wiki/")).replace("_", " ")
    return " ".join(value.split())


def _identity(row: Mapping[str, Any], context: str) -> tuple[str, str]:
    question_id = row.get("question_id")
    model = row.get("model")
    require(isinstance(question_id, str) and question_id, f"{context}: bad question_id")
    require(model in MODELS, f"{context}: unexpected model {model!r}")
    return question_id, model


def _valid_link(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("/wiki/")


def _plain_ranking(value: Any, context: str, *, failed: bool = False) -> list[str]:
    require(isinstance(value, list), f"{context}: ranking must be a list")
    require(all(_valid_link(link) for link in value), f"{context}: invalid link")
    require(len(value) == len(set(value)), f"{context}: duplicate link")
    if failed:
        require(not value, f"{context}: failed ranking must be empty")
    else:
        require(len(value) <= 20, f"{context}: ranking exceeds cutoff 20")
    return list(value)


def _load_bridge(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    require(len(rows) == 1_800, "bridge ledger must contain 1,800 rows")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    model_counts: Counter[str] = Counter()
    prior: tuple[str, str] | None = None
    for row_number, row in enumerate(rows, 1):
        identity = _identity(row, f"bridge:{row_number}")
        canonical_order = (identity[1], identity[0])
        require(identity not in index, f"bridge: duplicate identity {identity}")
        require(prior is None or canonical_order > prior, "bridge order changed")
        prior = canonical_order
        require(row.get("status") == "succeeded", f"bridge {identity}: status changed")
        require(type(row.get("valid_json")) is bool, f"bridge {identity}: bad validity")
        require(_valid_link(row.get("gold_link")), f"bridge {identity}: bad traced link")
        _plain_ranking(row.get("question_ranking"), f"bridge {identity} question")
        _plain_ranking(row.get("gold_fact_ranking"), f"bridge {identity} label")
        if row["valid_json"]:
            require(type(row.get("selection_correct")) is bool, f"bridge {identity}: bad match field")
            require(_valid_link(row.get("selected_link")), f"bridge {identity}: bad selected link")
            require(isinstance(row.get("retrieval_fact"), str) and row["retrieval_fact"].strip(), f"bridge {identity}: missing retrieval fact")
            require(row.get("primary_predicted_query") == _link_label(row["selected_link"]), f"bridge {identity}: decoded query changed")
            _plain_ranking(row.get("predicted_ranking"), f"bridge {identity} decoded")
            _plain_ranking(row.get("diagnostic_model_fact_ranking"), f"bridge {identity} generated")
            require(row["selection_correct"] is (row["selected_link"] == row["gold_link"]), f"bridge {identity}: match field disagrees")
        else:
            require(identity[1] == "openai", f"bridge {identity}: unexpected invalid model")
            for field in (
                "selection_correct",
                "selected_link",
                "retrieval_fact",
                "primary_predicted_query",
                "predicted_ranking",
                "diagnostic_model_fact_ranking",
            ):
                require(row.get(field) is None, f"bridge {identity}: invalid row has {field}")
        index[identity] = row
        model_counts[identity[1]] += 1
    require(model_counts == Counter({model: 600 for model in MODELS}), "bridge model counts changed")
    require(len({question_id for question_id, _ in index}) == EXPECTED_QUESTIONS, "bridge question count changed")
    return index


def _load_answers(
    rows: list[dict[str, Any]],
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    require(len(rows) == 6_600, "answer ledger must contain 6,600 rows")
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    for row_number, row in enumerate(rows, 1):
        question_id, model = _identity(row, f"answer:{row_number}")
        condition = row.get("condition")
        require(condition in ANSWER_CONDITIONS_BY_MODEL[model], f"answer:{row_number}: condition changed")
        identity = (question_id, model, condition)
        require(identity not in index, f"answer: duplicate identity {identity}")
        bridge_row = bridge.get((question_id, model))
        require(bridge_row is not None, f"answer {identity}: missing bridge join")
        require(row.get("gold_link") == bridge_row.get("gold_link"), f"answer {identity}: traced-link join changed")
        require(row.get("status") in {"succeeded", "upstream_invalid"}, f"answer {identity}: status changed")
        require(type(row.get("valid_json")) is bool, f"answer {identity}: bad validity")
        require(row.get("exact") in (0, 1), f"answer {identity}: bad exact score")
        require(isinstance(row.get("f1"), (int, float)) and 0 <= row["f1"] <= 1, f"answer {identity}: bad F1")
        ranking = row.get("ranking")
        evidence = row.get("evidence_links")
        require(isinstance(ranking, list) and all(_valid_link(link) for link in ranking), f"answer {identity}: bad ranking")
        require(isinstance(evidence, list) and all(_valid_link(link) for link in evidence), f"answer {identity}: bad evidence")
        require(len(ranking) == len(set(ranking)), f"answer {identity}: duplicate ranking link")
        if row["status"] == "upstream_invalid":
            require(not row["valid_json"] and row["exact"] == 0 and not ranking and not evidence, f"answer {identity}: unissued accounting changed")
        else:
            if condition == "no_kb":
                require(not ranking, f"answer {identity}: no-kb ranking changed")
            elif condition == "oracle_passage":
                require(len(ranking) == 1, f"answer {identity}: oracle ranking changed")
            else:
                require(len(ranking) <= 20, f"answer {identity}: ranking exceeds cutoff 20")
            require(evidence == ranking[:TOP_K], f"answer {identity}: evidence/ranking mismatch")
        if condition == "predicted_fact":
            if bridge_row["valid_json"]:
                require(row["status"] == "succeeded", f"answer {identity}: valid selector did not issue")
                require(row.get("predicted_fact") == bridge_row["primary_predicted_query"], f"answer {identity}: decoded label changed")
                require(row.get("plan", {}).get("retrieval_query") == bridge_row["primary_predicted_query"], f"answer {identity}: plan query changed")
                require(ranking == bridge_row["predicted_ranking"], f"answer {identity}: BM25 ranking changed")
            else:
                require(row["status"] == "upstream_invalid" and not ranking, f"answer {identity}: invalid dependency changed")
        if condition == "oracle_passage":
            require(model == "gemini", f"answer {identity}: oracle scope changed")
            require(row["status"] == "succeeded" and ranking == [bridge_row["gold_link"]] and evidence == [bridge_row["gold_link"]], f"answer {identity}: oracle passage changed")
        if condition == "no_kb":
            require(model == "gemini", f"answer {identity}: no-kb scope changed")
        if condition == "query_rewrite" and row["status"] == "succeeded":
            require(isinstance(row.get("query_rewrite"), str) and row["query_rewrite"].strip(), f"answer {identity}: rewrite missing")
            require(row.get("plan", {}).get("retrieval_query") == row["query_rewrite"], f"answer {identity}: rewrite plan changed")
        index[identity] = row
        counts[(model, condition)] += 1
    expected = Counter(
        (model, condition)
        for model in MODELS
        for condition in ANSWER_CONDITIONS_BY_MODEL[model]
        for _ in range(EXPECTED_ROWS_PER_MODEL)
    )
    require(counts == expected, "answer model/condition inventory changed")
    return index


def _sensitivity_ranking(value: Any, context: str) -> list[str]:
    require(isinstance(value, list) and len(value) in {20, 100}, f"{context}: ranking length changed")
    links: list[str] = []
    for expected_rank, result in enumerate(value, 1):
        require(isinstance(result, dict), f"{context}: result must be an object")
        require(result.get("rank") == expected_rank, f"{context}: rank sequence changed")
        link = result.get("link")
        require(_valid_link(link), f"{context}: invalid link")
        links.append(link)
    require(len(links) == len(set(links)), f"{context}: duplicate link")
    return links


def _load_sensitivity(
    rows: list[dict[str, Any]],
    bridge: Mapping[tuple[str, str], Mapping[str, Any]],
    stack: str,
) -> dict[tuple[str, str], list[str]]:
    require(len(rows) == 2_400, f"{stack} ledger must contain 2,400 rows")
    conditions = Counter(row.get("condition") for row in rows)
    require(conditions == Counter({"predicted_fact": 1_800, "question_only": 600}), f"{stack}: condition counts changed")
    predicted: dict[tuple[str, str], list[str]] = {}
    question_ids: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        condition = row.get("condition")
        question_id = row.get("question_id")
        require(isinstance(question_id, str) and question_id, f"{stack}:{row_number}: bad question_id")
        require(_valid_link(row.get("gold_link")), f"{stack}:{row_number}: bad traced link")
        if condition == "question_only":
            require(row.get("model") is None, f"{stack}:{row_number}: shared row has model")
            require(question_id not in question_ids, f"{stack}: duplicate question row")
            question_ids.add(question_id)
            require(row.get("status") == "succeeded", f"{stack}: question status changed")
            _sensitivity_ranking(row.get("ranking"), f"{stack} question {question_id}")
            continue
        identity = _identity(row, f"{stack}:{row_number}")
        require(identity not in predicted, f"{stack}: duplicate identity {identity}")
        bridge_row = bridge.get(identity)
        require(bridge_row is not None, f"{stack} {identity}: missing bridge join")
        require(row.get("gold_link") == bridge_row["gold_link"], f"{stack} {identity}: traced-link join changed")
        if bridge_row["valid_json"]:
            require(row.get("status") == "succeeded", f"{stack} {identity}: status changed")
            require(row.get("text") == bridge_row["primary_predicted_query"], f"{stack} {identity}: query changed")
            predicted[identity] = _sensitivity_ranking(row.get("ranking"), f"{stack} {identity}")
        else:
            require(row.get("status") == "upstream_invalid" and row.get("text") is None and row.get("ranking") == [], f"{stack} {identity}: invalid dependency changed")
            predicted[identity] = []
    require(question_ids == {question_id for question_id, _ in bridge}, f"{stack}: question inventory changed")
    require(set(predicted) == set(bridge), f"{stack}: predicted key set changed")
    return predicted


def _load_links(rows: list[dict[str, Any]]) -> dict[str, int]:
    require(len(rows) == 286_270, "passage inventory must contain 286,270 rows")
    inventory: dict[str, int] = {}
    for expected_pid, row in enumerate(rows):
        require(set(row) == {"pid", "link"}, f"links:{expected_pid + 1}: schema changed")
        require(row.get("pid") == expected_pid, f"links:{expected_pid + 1}: pid changed")
        link = row.get("link")
        require(_valid_link(link), f"links:{expected_pid + 1}: invalid link")
        require(link not in inventory, f"links: duplicate link {link}")
        inventory[link] = expected_pid
    return inventory


def _exact_two_sided_sign_test(first_only: int, second_only: int) -> dict[str, Any]:
    require(first_only >= 0 and second_only >= 0, "negative discordant count")
    count = first_only + second_only
    if count == 0:
        probability = Fraction(1, 1)
    else:
        tail = sum(math.comb(count, k) for k in range(min(first_only, second_only) + 1))
        probability = min(Fraction(1, 1), Fraction(2 * tail, 2**count))
    return {
        "method": "two-sided exact binomial sign test with p=0.5; ties omitted",
        "discordant_n": count,
        "smaller_discordant_cell": min(first_only, second_only),
        "p_value_fraction": f"{probability.numerator}/{probability.denominator}",
        "p_value": float(probability),
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    require(bool(sorted_values), "cannot take percentile of empty values")
    require(0 <= probability <= 1, "percentile probability outside [0,1]")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
