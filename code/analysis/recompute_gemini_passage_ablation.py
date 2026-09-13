#!/usr/bin/env python3
"""Recompute the Gemini passage-ablation results from public records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import random
import re
import string
from typing import Any, Iterable


RELEASE_ID = "gemini_passage_ablation_openrouter_byok_v1"
SOURCE_EXPERIMENT_ID = "gemini_passage_ablation_v1"
MODEL = "google/gemini-3.1-pro-preview"
SOURCE_MODEL = "gemini-3.1-pro-preview"
ROUTED_PROVIDER = "Google AI Studio"
PROVIDER_PIN = ["google-ai-studio"]
EXPECTED_MAIN_TASKS = 561
EXPECTED_PREFLIGHT_TASKS = 3
EXPECTED_COHORT = 64
EXPECTED_SHAM = 59
EXPECTED_WIRE_ATTEMPTS = 570
EXPECTED_STATUS = Counter({"succeeded_valid": 537, "failed_terminal": 24})
EXPECTED_ARMS = Counter({"intact": 192, "trace_drop": 192, "sham_drop": 177})
EXPECTED_FAILURE_ARMS = Counter({"intact": 13, "trace_drop": 8, "sham_drop": 3})
EXPECTED_FAILURE_KINDS = Counter(
    {
        "provider_error_503_high_demand": 11,
        "provider_error_504_message_524": 12,
        "invalid_response_json": 1,
    }
)
EXPECTED_ATTEMPT_KINDS = Counter(
    {"model_output": 537, "provider_error": 32, "invalid_json": 1}
)
REPEATS = (1, 2, 3)
ARMS = ("intact", "trace_drop", "sham_drop")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_714
EXPECTED_FILES = {
    "analysis_summary.json",
    "batch_claims.json",
    "cohort.json",
    "executed_protocol.json",
    "execution_route_config.json",
    "main_results.jsonl",
    "preflight_marker.json",
    "preflight_results.jsonl",
    "source_items.jsonl",
    "tasks.jsonl",
}
EXPECTED_RESULT_COLUMNS = {
    "arm",
    "attempts",
    "economic_cost_usd",
    "input_tokens",
    "is_byok",
    "openrouter_cost_usd",
    "outcome_reason",
    "output_tokens",
    "parsed_answer",
    "question_id",
    "reference",
    "repeat_index",
    "request_json",
    "request_sha256",
    "requested_model",
    "response_model",
    "routed_provider",
    "schedule_index",
    "schema_version",
    "source_raw_result_sha256",
    "source_release_result_sha256",
    "source_request_json_sha256",
    "status",
    "task_id",
    "task_sha256",
    "upstream_inference_cost_usd",
    "valid_json",
    "wire_attempt_count",
}
EXPECTED_ATTEMPT_COLUMNS = {
    "attempt_number",
    "economic_cost_usd",
    "http_status",
    "is_byok",
    "openrouter_cost_usd",
    "parser_visible_response_sha256",
    "parser_visible_response_text",
    "response_kind",
    "response_model",
    "routed_provider",
    "source_request_body_sha256",
    "source_response_body_sha256",
    "upstream_inference_cost_usd",
}
EXPECTED_VERTEX_CONFIG = {
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
    "response_schema": {
        "additionalProperties": False,
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "type": "object",
    },
    "thinking_config": {"thinking_level": "MEDIUM"},
}
FORBIDDEN_JSON_FIELDS = {
    "base64",
    "finished_at",
    "headers",
    "latency_sec",
    "raw_response",
    "request_id",
    "response_id",
    "started_at",
    "timestamp",
    "updated_at",
}
SECRET_KEYS = {
    "access-token",
    "access_token",
    "api-key",
    "api_key",
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
}
HEX_64 = re.compile(r"[0-9a-f]{64}")
UNIX_MACHINE_PATH = re.compile(
    rb"(?ix)(?:^|[\s\"'=:(])/(?:users|home|root|content|private|tmp|var|etc|opt|srv|mnt|media|volumes)(?:/|\b)"
)
WINDOWS_DRIVE_PATH = re.compile(rb"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[/\\]")
EMAIL = re.compile(rb"(?i)[a-z0-9._%+\-]{1,64}@[a-z0-9.\-]{1,255}\.[a-z]{2,24}")
ACCOUNT_ID = re.compile(rb"user_[A-Za-z0-9]+")
EVIDENCE_MARKER = "\n\nEvidence:\n"
RETURN_MARKER = "\n\nReturn a JSON object with one string field named answer."
PASSAGE_HEADER = re.compile(r"(?m)^Passage ([1-9][0-9]*) — (.*?) \[([^\]\n]+)\]\n")
ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCTUATION = frozenset(string.punctuation)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "evidence" / RELEASE_ID


class VerificationError(RuntimeError):
    """Raised when released evidence does not satisfy a checked invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def strict_loads(text: str, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise VerificationError(f"{label}: invalid strict JSON") from error


def load_json(path: Path) -> Any:
    try:
        return strict_loads(path.read_text(encoding="utf-8"), path.name)
    except OSError as error:
        raise VerificationError(f"cannot read {path.name}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            require(
                raw.endswith(b"\n") and raw.strip(),
                f"{path.name}:{line_number}: partial or blank line",
            )
            try:
                text = raw[:-1].decode("utf-8")
            except UnicodeDecodeError as error:
                raise VerificationError(
                    f"{path.name}:{line_number}: invalid UTF-8"
                ) from error
            row = strict_loads(text, f"{path.name}:{line_number}")
            require(isinstance(row, dict), f"{path.name}:{line_number}: row is not an object")
            require(text == canonical_json(row), f"{path.name}:{line_number}: row is not canonical JSON")
            rows.append(row)
    return rows


def _scan_bytes(payload: bytes, label: str) -> None:
    require(UNIX_MACHINE_PATH.search(payload) is None, f"{label}: machine path retained")
    require(WINDOWS_DRIVE_PATH.search(payload) is None, f"{label}: Windows path retained")
    require(EMAIL.search(payload) is None, f"{label}: email address retained")
    require(ACCOUNT_ID.search(payload) is None, f"{label}: account identifier retained")


def _walk_public_json(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            require(lowered not in SECRET_KEYS, f"{label}: credential field retained")
            require(lowered not in FORBIDDEN_JSON_FIELDS, f"{label}: transport field {key!r} retained")
            _walk_public_json(child, label)
    elif isinstance(value, list):
        for child in value:
            _walk_public_json(child, label)


def _float_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _equal(left: Any, right: Any, label: str) -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        require(type(left) is type(right) and left == right, f"{label} changed")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        require(_float_equal(float(left), float(right)), f"{label} changed")
        return
    require(type(left) is type(right), f"{label}: JSON type changed")
    if isinstance(left, dict):
        require(set(left) == set(right), f"{label}: fields changed")
        for key in left:
            _equal(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, list):
        require(len(left) == len(right), f"{label}: length changed")
        for index, (first, second) in enumerate(zip(left, right)):
            _equal(first, second, f"{label}[{index}]")
    else:
        require(left == right, f"{label} changed")


def verify_inventory(root: Path) -> dict[str, Any]:
    manifest_path = root / "release_manifest.json"
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == 2, "release schema changed")
    require(manifest.get("release_id") == RELEASE_ID, "release ID changed")
    actual = {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and path.name != manifest_path.name
    }
    require(set(actual) == EXPECTED_FILES, "experiment inventory changed")
    declared = manifest.get("files")
    require(isinstance(declared, dict) and set(declared) == set(actual), "manifest inventory changed")
    for name, path in actual.items():
        record = declared[name]
        require(record["bytes"] == path.stat().st_size, f"{name}: byte count changed")
        require(record["sha256"] == sha256_file(path), f"{name}: SHA-256 changed")
        if name.endswith(".jsonl"):
            require(record["records"] == len(load_jsonl(path)), f"{name}: row count changed")
        payload = path.read_bytes()
        _scan_bytes(payload, name)
        if name.endswith(".json"):
            _walk_public_json(load_json(path), name)
        elif name.endswith(".jsonl"):
            for index, row in enumerate(load_jsonl(path), 1):
                _walk_public_json(row, f"{name}:{index}")
    replay = manifest["replay"]
    require(replay["path"] == "scripts/recompute_gemini_passage_ablation.py", "replay path changed")
    require(replay["sha256"] == sha256_file(Path(__file__)), "replay script hash changed")
    require(
        replay["network_required"] is False
        and replay["credentials_required"] is False
        and replay["third_party_packages_required"] is False,
        "offline replay declaration changed",
    )
    accounting = manifest["accounting"]
    require(
        accounting["main"]
        == {
            "failed_terminal": 24,
            "logical_calls": 561,
            "openrouter_cost_usd": 0.0,
            "succeeded_valid": 537,
            "upstream_economic_cost_usd": 4.3857,
            "wire_attempts": 570,
        },
        "main accounting declaration changed",
    )
    require(
        accounting["preflight"]
        == {
            "excluded_from_analysis": True,
            "logical_calls": 3,
            "openrouter_cost_usd": 0.0,
            "succeeded_valid": 3,
            "upstream_economic_cost_usd": 0.024284,
            "wire_attempts": 3,
        }
        and _float_equal(accounting["total_upstream_economic_cost_usd"], 4.409984),
        "preflight or total accounting declaration changed",
    )
    return manifest


def normalize_answer(answer: str) -> str:
    lowered = answer.lower()
    without_punctuation = "".join(character for character in lowered if character not in PUNCTUATION)
    without_articles = ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def compute_exact(gold: str, prediction: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(prediction))


def compute_f1(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split() if gold else []
    prediction_tokens = normalize_answer(prediction).split() if prediction else []
    shared = sum((Counter(gold_tokens) & Counter(prediction_tokens)).values())
    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    if shared == 0:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _parse_exact_answer(text: Any) -> tuple[str | None, bool]:
    if not isinstance(text, str):
        return None, False
    try:
        value = strict_loads(text, "model answer")
    except VerificationError:
        return None, False
    if (
        not isinstance(value, dict)
        or set(value) != {"answer"}
        or not isinstance(value["answer"], str)
        or not value["answer"].strip()
    ):
        return None, False
    return value["answer"], True


def _parse_evidence(contents: str) -> tuple[str, str, list[dict[str, Any]]]:
    require(contents.count(EVIDENCE_MARKER) == 1, "prompt has an ambiguous Evidence marker")
    require(contents.count(RETURN_MARKER) == 1, "prompt has an ambiguous return marker")
    evidence_start = contents.index(EVIDENCE_MARKER) + len(EVIDENCE_MARKER)
    return_start = contents.index(RETURN_MARKER, evidence_start)
    prefix = contents[:evidence_start]
    body = contents[evidence_start:return_start]
    suffix = contents[return_start:]
    matches = list(PASSAGE_HEADER.finditer(body))
    require(bool(matches) and matches[0].start() == 0, "evidence body does not start with a passage")
    ranks = [int(match.group(1)) for match in matches]
    require(ranks == sorted(set(ranks)), "passage ranks are not ordered and unique")
    blocks: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        if index + 1 < len(matches):
            next_start = matches[index + 1].start()
            require(body[next_start - 2 : next_start] == "\n\n", "passage separator changed")
            end = next_start - 2
        else:
            end = len(body)
        raw = body[match.start() : end]
        blocks.append(
            {
                "link": match.group(3),
                "rank": int(match.group(1)),
                "raw": raw,
                "rendered_length": len(raw),
                "sha256": sha256_bytes(raw.encode("utf-8")),
                "title": match.group(2),
            }
        )
    return prefix, suffix, blocks


def _delete_block(prefix: str, suffix: str, blocks: list[dict[str, Any]], removed: dict[str, Any]) -> str:
    require(sum(block == removed for block in blocks) == 1, "removed passage is not unique")
    return prefix + "\n\n".join(block["raw"] for block in blocks if block != removed) + suffix


def _public_block(block: dict[str, Any]) -> dict[str, Any]:
    return {key: block[key] for key in ("link", "rank", "rendered_length", "sha256", "title")}


def _validate_source_request(request: Any) -> None:
    require(isinstance(request, dict), "source request is not an object")
    require(set(request) == {"model", "system_instruction", "contents", "config"}, "source request fields changed")
    require(request["model"] == SOURCE_MODEL, "source model changed")
    require(request["config"] == EXPECTED_VERTEX_CONFIG, "source request config changed")
    require(isinstance(request["system_instruction"], str) and request["system_instruction"], "system instruction changed")
    require(
        isinstance(request["contents"], str)
        and request["contents"].startswith("[USER]\n")
        and request["contents"].count("[USER]\n") == 1,
        "source user marker changed",
    )


def verify_protocol_and_tasks(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol = load_json(root / "executed_protocol.json")
    route = load_json(root / "execution_route_config.json")
    require(protocol["experiment_id"] == RELEASE_ID, "protocol experiment ID changed")
    require(
        protocol["model_route"]
        == {
            "allow_fallbacks": False,
            "byok_required": True,
            "direct_google_api_usage": False,
            "model": MODEL,
            "openrouter_cost_required_usd": 0.0,
            "provider_pin": PROVIDER_PIN,
            "routed_provider": ROUTED_PROVIDER,
            "transport": "OpenRouter BYOK",
        },
        "protocol model route changed",
    )
    require(
        protocol["request"]["maximum_output_tokens"] == 4096
        and protocol["request"]["reasoning_effort"] == "medium"
        and protocol["request"]["temperature"] == 1.0
        and protocol["request"]["seed"] is None,
        "protocol generation controls changed",
    )
    require(
        protocol["execution_settings"]
        == {
            "concurrency": 561,
            "logical_calls": 561,
            "maximum_wire_attempts_per_call": 2,
            "one_barrier_parallel_launch": True,
            "preflight_calls": 3,
        },
        "protocol execution settings changed",
    )
    require(
        set(protocol["setting_provenance"])
        == {"fixed_before_hosted_execution", "set_for_this_hosted_execution"},
        "setting provenance changed",
    )
    require(
        route["model"]["name"] == MODEL
        and route["model"]["transport"] == "OpenRouter"
        and route["model"]["direct_google_api_usage"] is False
        and route["model"]["is_byok_required"] is True
        and route["model"]["provider_pin"] == PROVIDER_PIN
        and route["model"]["routed_provider"] == ROUTED_PROVIDER
        and route["model"]["allow_fallbacks"] is False,
        "route configuration changed",
    )

    cohort = load_json(root / "cohort.json")
    source_items = load_jsonl(root / "source_items.jsonl")
    tasks = load_jsonl(root / "tasks.jsonl")
    require(isinstance(cohort, list) and len(cohort) == EXPECTED_COHORT and len(set(cohort)) == EXPECTED_COHORT, "cohort membership changed")
    require(len(source_items) == EXPECTED_COHORT, "source item count changed")
    item_by_id = {row["question_id"]: row for row in source_items}
    require(len(item_by_id) == EXPECTED_COHORT and set(item_by_id) == set(cohort), "source items differ from cohort")

    sham_count = 0
    for question_id, item in item_by_id.items():
        require(item["schema_version"] == 1, f"{question_id}: source item schema changed")
        require(item["reference_normalized"] == normalize_answer(item["reference"]), f"{question_id}: reference normalization changed")
        requests = item["requests"]
        require(set(requests) in ({"intact", "trace_drop"}, {"intact", "trace_drop", "sham_drop"}), f"{question_id}: arm set changed")
        for request in requests.values():
            _validate_source_request(request)
        intact = requests["intact"]
        require(sha256_bytes(canonical_json(intact).encode()) == item["source"]["request_json_sha256"], f"{question_id}: intact request hash changed")
        require(sha256_bytes(intact["contents"].encode()) == item["source"]["contents_sha256"], f"{question_id}: intact content hash changed")
        prefix, suffix, blocks = _parse_evidence(intact["contents"])
        traced_matches = [block for block in blocks if block["link"] == item["gold_link"]]
        require(len(traced_matches) == 1, f"{question_id}: traced passage is not unique")
        traced = traced_matches[0]
        require(_public_block(traced) == item["trace_block"], f"{question_id}: traced passage metadata changed")
        require(requests["trace_drop"]["contents"] == _delete_block(prefix, suffix, blocks, traced), f"{question_id}: traced passage deletion changed")
        candidates = [
            block
            for block in blocks
            if block["link"] != traced["link"]
            and normalize_answer(item["reference"]) not in normalize_answer(block["raw"])
        ]
        expected_sham = (
            min(candidates, key=lambda block: (abs(block["rendered_length"] - traced["rendered_length"]), block["rank"]))
            if candidates
            else None
        )
        if expected_sham is None:
            require(item["sham_block"] is None and "sham_drop" not in requests, f"{question_id}: sham eligibility changed")
        else:
            sham_count += 1
            metadata = {**_public_block(expected_sham), "answer_safe": True}
            metadata["length_distance_from_trace"] = abs(expected_sham["rendered_length"] - traced["rendered_length"])
            require(item["sham_block"] == metadata, f"{question_id}: sham passage metadata changed")
            require(requests["sham_drop"]["contents"] == _delete_block(prefix, suffix, blocks, expected_sham), f"{question_id}: sham passage deletion changed")
    require(sham_count == EXPECTED_SHAM, "sham eligibility count changed")

    require(len(tasks) == EXPECTED_MAIN_TASKS, "task count changed")
    require(Counter(task["arm"] for task in tasks) == EXPECTED_ARMS, "task arm counts changed")
    require([task["schedule_index"] for task in tasks] == list(range(1, EXPECTED_MAIN_TASKS + 1)), "schedule indexes changed")
    require(len({task["task_id"] for task in tasks}) == EXPECTED_MAIN_TASKS, "task IDs are not unique")
    queues: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        question_id = task["question_id"]
        item = item_by_id[question_id]
        arm = task["arm"]
        repeat_index = task["repeat_index"]
        require(repeat_index in REPEATS, f"{task['task_id']}: repeat changed")
        require(task["request"] == item["requests"][arm], f"{task['task_id']}: request differs from source item")
        request_sha = sha256_bytes(canonical_json(task["request"]).encode())
        require(task["request_sha256"] == request_sha, f"{task['task_id']}: request hash changed")
        require(task["contents_sha256"] == sha256_bytes(task["request"]["contents"].encode()), f"{task['task_id']}: content hash changed")
        identity = {
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "question_id": question_id,
            "arm": arm,
            "repeat_index": repeat_index,
            "request_sha256": request_sha,
        }
        require(task["task_id"] == sha256_bytes(canonical_json(identity).encode()), f"{task['task_id']}: task identity changed")
        schedule_key = sha256_bytes(f"{SOURCE_EXPERIMENT_ID}|schedule-v1|{repeat_index}|{arm}|{question_id}".encode())
        require(task["schedule_key"] == schedule_key, f"{task['task_id']}: schedule key changed")
        require(
            task["reference"] == item["reference"]
            and task["gold_link"] == item["gold_link"]
            and task["source_task_key"] == item["source"]["task_key"]
            and task["source_prompt_sha256"] == item["source"]["prompt_sha256"]
            and task["source_request_json_sha256"] == item["source"]["request_json_sha256"],
            f"{task['task_id']}: source binding changed",
        )
        queues[(repeat_index, arm)].append(task)
    for queue in queues.values():
        queue.sort(key=lambda task: (task["schedule_key"], task["question_id"]))
    lanes: list[tuple[int, str]] = []
    for repeat_index in REPEATS:
        shift = repeat_index - 1
        rotated = ARMS[shift:] + ARMS[:shift]
        lanes.extend((repeat_index, arm) for arm in rotated)
    expected_schedule: list[str] = []
    for round_index in range(max(len(queue) for queue in queues.values())):
        for lane in lanes:
            queue = queues.get(lane, [])
            if round_index < len(queue):
                expected_schedule.append(queue[round_index]["task_id"])
    require(expected_schedule == [task["task_id"] for task in tasks], "prehashed schedule changed")
    return tasks, item_by_id


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and HEX_64.fullmatch(value) is not None


def verify_results(
    rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    *,
    mode: str,
    selected_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    task_by_id = {task["task_id"]: task for task in tasks}
    expected_ids = [task["task_id"] for task in tasks] if mode == "main" else list(selected_task_ids or [])
    require([row["task_id"] for row in rows] == expected_ids, f"{mode}: result population or order changed")
    status_counts: Counter[str] = Counter()
    arm_counts: Counter[str] = Counter()
    failure_arms: Counter[str] = Counter()
    failure_kinds: Counter[str] = Counter()
    attempt_kinds: Counter[str] = Counter()
    http_statuses: Counter[int] = Counter()
    wire_attempt_count = 0
    input_tokens = 0
    output_tokens = 0
    economic_cost = 0.0
    openrouter_cost = 0.0
    scored: list[dict[str, Any]] = []
    for row in rows:
        task_id = row["task_id"]
        task = task_by_id.get(task_id)
        require(task is not None, f"{mode}: unknown task {task_id}")
        require(set(row) == EXPECTED_RESULT_COLUMNS and row["schema_version"] == 2, f"{task_id}: public result schema changed")
        require(row["task_sha256"] == sha256_bytes(canonical_json(task).encode()), f"{task_id}: task hash changed")
        require(row["request_json"] == canonical_json(task["request"]), f"{task_id}: visible request changed")
        require(row["request_sha256"] == task["request_sha256"], f"{task_id}: request hash changed")
        require(row["source_request_json_sha256"] == task["source_request_json_sha256"], f"{task_id}: source request hash changed")
        for field in ("source_raw_result_sha256", "source_release_result_sha256"):
            require(_valid_hash(row[field]), f"{task_id}: {field} is not a SHA-256")
        for field in ("schedule_index", "question_id", "arm", "repeat_index", "reference"):
            require(row[field] == task[field], f"{task_id}: {field} changed")
        require(row["requested_model"] == MODEL, f"{task_id}: requested model changed")

        attempts = row["attempts"]
        require(isinstance(attempts, list) and 1 <= len(attempts) <= 2, f"{task_id}: attempt inventory changed")
        require(len(attempts) == row["wire_attempt_count"], f"{task_id}: attempt count changed")
        for index, attempt in enumerate(attempts, 1):
            require(set(attempt) == EXPECTED_ATTEMPT_COLUMNS, f"{task_id}: attempt schema changed")
            require(attempt["attempt_number"] == index, f"{task_id}: attempt order changed")
            require(_valid_hash(attempt["source_request_body_sha256"]), f"{task_id}: request body hash changed")
            require(_valid_hash(attempt["source_response_body_sha256"]), f"{task_id}: response body hash changed")
            text = attempt["parser_visible_response_text"]
            require(isinstance(text, str), f"{task_id}: parser-visible response is not text")
            require(sha256_bytes(text.encode()) == attempt["parser_visible_response_sha256"], f"{task_id}: parser-visible response hash changed")
            kind = attempt["response_kind"]
            require(kind in {"model_output", "provider_error", "invalid_json"}, f"{task_id}: unknown response kind")
            attempt_kinds[kind] += 1
            http_statuses[attempt["http_status"]] += 1
            if kind == "provider_error":
                envelope = strict_loads(text, f"{task_id}:provider-error")
                require(isinstance(envelope, dict) and set(envelope) == {"error"} and isinstance(envelope["error"], dict), f"{task_id}: provider error envelope changed")
            elif kind == "invalid_json":
                try:
                    strict_loads(text, f"{task_id}:invalid-response")
                except VerificationError:
                    pass
                else:
                    raise VerificationError(f"{task_id}: invalid response now parses")
            if index < len(attempts):
                require(kind == "provider_error" and attempt["http_status"] == 503, f"{task_id}: retry trigger changed")
                require(attempt["economic_cost_usd"] == 0.0, f"{task_id}: retry cost changed")

        final = attempts[-1]
        status = row["status"]
        status_counts[status] += 1
        arm_counts[row["arm"]] += 1
        if status == "succeeded_valid":
            answer, answer_valid = _parse_exact_answer(final["parser_visible_response_text"])
            require(answer_valid and final["response_kind"] == "model_output", f"{task_id}: valid answer changed")
            require(
                row["parsed_answer"] == answer
                and row["valid_json"] is True
                and row["outcome_reason"] is None,
                f"{task_id}: parsed answer fields changed",
            )
            require(
                row["response_model"] == MODEL
                and row["routed_provider"] == ROUTED_PROVIDER
                and row["is_byok"] is True
                and row["openrouter_cost_usd"] == 0.0
                and row["upstream_inference_cost_usd"] is not None,
                f"{task_id}: successful route fields changed",
            )
            require(
                final["response_model"] == MODEL
                and final["routed_provider"] == ROUTED_PROVIDER
                and final["is_byok"] is True
                and final["openrouter_cost_usd"] == 0.0
                and _float_equal(final["economic_cost_usd"], row["economic_cost_usd"])
                and _float_equal(final["upstream_inference_cost_usd"], row["upstream_inference_cost_usd"]),
                f"{task_id}: final attempt accounting changed",
            )
            prediction = answer
        elif status == "failed_terminal":
            failure_arms[row["arm"]] += 1
            failure_kinds[row["outcome_reason"]] += 1
            require(
                row["outcome_reason"] in EXPECTED_FAILURE_KINDS
                and row["parsed_answer"] is None
                and row["valid_json"] is False
                and row["response_model"] is None
                and row["routed_provider"] is None
                and row["is_byok"] is None
                and row["openrouter_cost_usd"] is None
                and row["upstream_inference_cost_usd"] is None
                and row["economic_cost_usd"] == 0.0
                and row["input_tokens"] == 0
                and row["output_tokens"] == 0,
                f"{task_id}: unsuccessful outcome fields changed",
            )
            if row["outcome_reason"] == "invalid_response_json":
                require(final["response_kind"] == "invalid_json", f"{task_id}: invalid JSON evidence changed")
            else:
                require(final["response_kind"] == "provider_error", f"{task_id}: provider error evidence changed")
                code = strict_loads(final["parser_visible_response_text"], task_id)["error"]["code"]
                require(code in {503, 504}, f"{task_id}: provider error code changed")
            prediction = None
        else:
            raise VerificationError(f"{task_id}: unknown terminal status")
        wire_attempt_count += len(attempts)
        input_tokens += int(row["input_tokens"])
        output_tokens += int(row["output_tokens"])
        economic_cost += float(row["economic_cost_usd"])
        openrouter_cost += float(row["openrouter_cost_usd"] or 0.0)
        scored.append(
            {
                "arm": row["arm"],
                "exact": compute_exact(row["reference"], prediction) if prediction is not None else 0,
                "f1": compute_f1(row["reference"], prediction) if prediction is not None else 0.0,
                "prediction_normalized": normalize_answer(prediction) if prediction is not None else None,
                "question_id": row["question_id"],
                "repeat_index": row["repeat_index"],
                "status": status,
            }
        )
    if mode == "main":
        require(status_counts == EXPECTED_STATUS, "main status counts changed")
        require(arm_counts == EXPECTED_ARMS, "main arm counts changed")
        require(failure_arms == EXPECTED_FAILURE_ARMS, "main failure arm counts changed")
        require(failure_kinds == EXPECTED_FAILURE_KINDS, "main failure kinds changed")
        require(attempt_kinds == EXPECTED_ATTEMPT_KINDS, "main attempt kinds changed")
        require(http_statuses == Counter({200: 561, 503: 9}), "main HTTP status accounting changed")
        require(wire_attempt_count == EXPECTED_WIRE_ATTEMPTS, "main wire-attempt count changed")
        require(input_tokens == 1_287_900 and output_tokens == 150_825, "main token totals changed")
        require(_float_equal(economic_cost, 4.3857) and openrouter_cost == 0.0, "main cost totals changed")
    else:
        require(status_counts == Counter({"succeeded_valid": 3}), "preflight status counts changed")
        require(attempt_kinds == Counter({"model_output": 3}), "preflight attempt kinds changed")
        require(wire_attempt_count == 3, "preflight attempt count changed")
        require(input_tokens == 5752 and output_tokens == 1065, "preflight token totals changed")
        require(_float_equal(economic_cost, 0.024284) and openrouter_cost == 0.0, "preflight cost totals changed")
    return {
        "arm_counts": dict(sorted(arm_counts.items())),
        "attempt_kind_counts": dict(sorted(attempt_kinds.items())),
        "economic_cost_usd": economic_cost,
        "failure_arm_counts": dict(sorted(failure_arms.items())),
        "failure_kind_counts": dict(sorted(failure_kinds.items())),
        "input_tokens": input_tokens,
        "openrouter_cost_usd": openrouter_cost,
        "output_tokens": output_tokens,
        "scored": scored,
        "status_counts": dict(sorted(status_counts.items())),
        "wire_attempt_count": wire_attempt_count,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _exact_two_sided_sign_test(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _holm(p_values: Iterable[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted[name] = running
    return adjusted


def _item_arm_records(scored: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(row["question_id"], row["arm"])].append(row)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["repeat_index"])
        require([row["repeat_index"] for row in rows] == [1, 2, 3], f"{key}: repeat coverage changed")
        exacts = [int(row["exact"]) for row in rows]
        f1s = [float(row["f1"]) for row in rows]
        normalized = [row["prediction_normalized"] for row in rows]
        valid_f1_repeats = [float(row["f1"]) for row in rows if row["status"] == "succeeded_valid"]
        records[key] = {
            "all_three_same_valid_normalized_prediction": all(value is not None for value in normalized) and len(set(normalized)) == 1,
            "arm": key[1],
            "em_item_mean": sum(exacts) / 3,
            "exact_count": sum(exacts),
            "f1_item_mean": sum(f1s) / 3,
            "majority_exact": int(sum(exacts) >= 2),
            "question_id": key[0],
            "unique_valid_normalized_predictions": len({value for value in normalized if value is not None}),
            "valid_count": len(valid_f1_repeats),
            "valid_f1_repeats": valid_f1_repeats,
        }
    require(
        Counter(arm for _, arm in records)
        == Counter({"intact": EXPECTED_COHORT, "trace_drop": EXPECTED_COHORT, "sham_drop": EXPECTED_SHAM}),
        "item-arm coverage changed",
    )
    return records


def _bootstrap(pairs: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    generator = random.Random(BOOTSTRAP_SEED)
    em_values: list[float] = []
    f1_values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indexes = generator.choices(range(len(pairs)), k=len(pairs))
        em_values.append(sum(pairs[index][0] - pairs[index][2] for index in indexes) / len(indexes))
        f1_values.append(sum(pairs[index][1] - pairs[index][3] for index in indexes) / len(indexes))
    return {
        "confidence_level": 0.95,
        "em_difference_95": [_percentile(em_values, 0.025), _percentile(em_values, 0.975)],
        "f1_difference_95": [_percentile(f1_values, 0.025), _percentile(f1_values, 0.975)],
        "method": "paired item bootstrap with replacement; percentile interval",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _contrast(
    records: dict[tuple[str, str], dict[str, Any]],
    *,
    name: str,
    first_arm: str,
    second_arm: str,
) -> dict[str, Any]:
    question_ids = sorted(question_id for question_id, arm in records if arm == first_arm and (question_id, second_arm) in records)
    pairs = [
        (
            records[(question_id, first_arm)]["em_item_mean"],
            records[(question_id, first_arm)]["f1_item_mean"],
            records[(question_id, second_arm)]["em_item_mean"],
            records[(question_id, second_arm)]["f1_item_mean"],
        )
        for question_id in question_ids
    ]
    first_only = sum(
        records[(question_id, first_arm)]["majority_exact"] == 1
        and records[(question_id, second_arm)]["majority_exact"] == 0
        for question_id in question_ids
    )
    second_only = sum(
        records[(question_id, first_arm)]["majority_exact"] == 0
        and records[(question_id, second_arm)]["majority_exact"] == 1
        for question_id in question_ids
    )
    return {
        "bootstrap": _bootstrap(pairs),
        "direction": f"{first_arm} minus {second_arm}",
        "em_item_mean_difference": sum(pair[0] - pair[2] for pair in pairs) / len(pairs),
        "f1_item_mean_difference": sum(pair[1] - pair[3] for pair in pairs) / len(pairs),
        "first_arm": first_arm,
        "first_em_item_mean": sum(pair[0] for pair in pairs) / len(pairs),
        "first_f1_item_mean": sum(pair[1] for pair in pairs) / len(pairs),
        "item_count": len(question_ids),
        "majority_of_three_mcnemar": {
            "discordant": first_only + second_only,
            "first_only": first_only,
            "p_value_two_sided_exact": _exact_two_sided_sign_test(first_only, second_only),
            "second_only": second_only,
        },
        "name": name,
        "second_arm": second_arm,
        "second_em_item_mean": sum(pair[2] for pair in pairs) / len(pairs),
        "second_f1_item_mean": sum(pair[3] for pair in pairs) / len(pairs),
    }


def _intact_sham_f1_sensitivity(records: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    question_ids = sorted(question_id for question_id, arm in records if arm == "sham_drop" and (question_id, "intact") in records)
    require(len(question_ids) == EXPECTED_SHAM, "intact-sham population changed")
    complete_ids = [
        question_id
        for question_id in question_ids
        if records[(question_id, "intact")]["valid_count"] == 3
        and records[(question_id, "sham_drop")]["valid_count"] == 3
    ]
    require(len(complete_ids) == 44, "intact-sham complete-case population changed")
    complete_first = [records[(question_id, "intact")]["f1_item_mean"] for question_id in complete_ids]
    complete_second = [records[(question_id, "sham_drop")]["f1_item_mean"] for question_id in complete_ids]
    valid_first: list[float] = []
    valid_second: list[float] = []
    for question_id in question_ids:
        first = records[(question_id, "intact")]["valid_f1_repeats"]
        second = records[(question_id, "sham_drop")]["valid_f1_repeats"]
        require(bool(first) and bool(second), f"{question_id}: valid-repeat arm is empty")
        valid_first.append(sum(first) / len(first))
        valid_second.append(sum(second) / len(second))
    first_count = sum(records[(question_id, "intact")]["valid_count"] for question_id in question_ids)
    second_count = sum(records[(question_id, "sham_drop")]["valid_count"] for question_id in question_ids)
    require((first_count, second_count) == (164, 174), "intact-sham valid-repeat counts changed")
    return {
        "complete_case": {
            "definition": "sham-eligible items with all three intact and all three sham_drop repeats succeeded_valid; paired item-arm F1 means",
            "f1_item_mean_difference": sum(first - second for first, second in zip(complete_first, complete_second)) / len(complete_ids),
            "first_arm": "intact",
            "first_f1_item_mean": sum(complete_first) / len(complete_first),
            "item_count": len(complete_ids),
            "second_arm": "sham_drop",
            "second_f1_item_mean": sum(complete_second) / len(complete_second),
        },
        "direction": "intact minus sham_drop",
        "valid_repeat": {
            "definition": "for every sham-eligible item and arm, average F1 over only succeeded_valid repeats; then average paired item differences",
            "f1_item_mean_difference": sum(first - second for first, second in zip(valid_first, valid_second)) / len(question_ids),
            "first_arm": "intact",
            "first_f1_item_mean": sum(valid_first) / len(valid_first),
            "first_valid_repeat_count": first_count,
            "item_count": len(question_ids),
            "second_arm": "sham_drop",
            "second_f1_item_mean": sum(valid_second) / len(valid_second),
            "second_valid_repeat_count": second_count,
        },
    }


def _adversarial_failure_score_em_lower_bound(
    scored: list[dict[str, Any]],
    *,
    first_arm: str,
    second_arm: str,
) -> dict[str, Any]:
    question_ids = sorted(
        {row["question_id"] for row in scored if row["arm"] == first_arm}
        & {row["question_id"] for row in scored if row["arm"] == second_arm}
    )
    first_rows = [row for row in scored if row["question_id"] in question_ids and row["arm"] == first_arm]
    second_rows = [row for row in scored if row["question_id"] in question_ids and row["arm"] == second_arm]
    repeat_count = len(question_ids) * len(REPEATS)
    require(len(first_rows) == repeat_count and len(second_rows) == repeat_count, "failure-score contrast coverage changed")
    first_failures = sum(row["status"] != "succeeded_valid" for row in first_rows)
    second_failures = sum(row["status"] != "succeeded_valid" for row in second_rows)
    first_fixed_exact = sum(int(row["exact"]) for row in first_rows)
    second_fixed_exact = sum(int(row["exact"]) for row in second_rows)
    return {
        "attaining_failure_score_assignment": {"first_arm": 0.0, "second_arm": 1.0},
        "direction": f"{first_arm} minus {second_arm}",
        "em_difference_lower_bound": first_fixed_exact / repeat_count - (second_fixed_exact + second_failures) / repeat_count,
        "first_arm": first_arm,
        "first_arm_fixed_exact_repeat_count": first_fixed_exact,
        "first_arm_invalid_or_failed_repeat_count": first_failures,
        "item_count": len(question_ids),
        "repeat_count_per_arm": repeat_count,
        "second_arm": second_arm,
        "second_arm_fixed_exact_repeat_count": second_fixed_exact,
        "second_arm_invalid_or_failed_repeat_count": second_failures,
    }


def _stability(records: dict[tuple[str, str], dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [record for (_, record_arm), record in records.items() if record_arm == arm]
    distribution = Counter(record["exact_count"] for record in rows)
    return {
        "all_three_exact": distribution[3],
        "all_three_same_valid_normalized_prediction": sum(record["all_three_same_valid_normalized_prediction"] for record in rows),
        "all_three_valid": sum(record["valid_count"] == 3 for record in rows),
        "exact_count_distribution": {str(index): distribution[index] for index in range(4)},
        "item_count": len(rows),
        "items_with_any_invalid_or_failed_repeat": sum(record["valid_count"] < 3 for record in rows),
        "majority_exact": sum(record["majority_exact"] for record in rows),
        "mixed_exact": distribution[1] + distribution[2],
        "none_exact": distribution[0],
        "unique_valid_normalized_prediction_count_distribution": {
            str(key): value
            for key, value in sorted(Counter(record["unique_valid_normalized_predictions"] for record in rows).items())
        },
    }


def verify_analysis(root: Path, main: dict[str, Any], tasks_path: Path) -> dict[str, Any]:
    expected = load_json(root / "analysis_summary.json")
    records = _item_arm_records(main["scored"])
    primary = _contrast(records, name="primary_intact_minus_trace_drop", first_arm="intact", second_arm="trace_drop")
    secondary = {
        "intact_minus_sham": _contrast(records, name="secondary_intact_minus_sham_drop", first_arm="intact", second_arm="sham_drop"),
        "sham_minus_trace": _contrast(records, name="secondary_sham_drop_minus_trace_drop", first_arm="sham_drop", second_arm="trace_drop"),
    }
    adjusted = _holm((name, result["majority_of_three_mcnemar"]["p_value_two_sided_exact"]) for name, result in secondary.items())
    for name, value in adjusted.items():
        secondary[name]["majority_of_three_mcnemar"]["p_value_holm_secondary_family"] = value
    route_gate_counts: dict[str, int] = {}
    for gate in ("billing_valid", "byok_valid", "complete", "response_identity_valid", "response_model_valid", "route_valid", "usage_valid"):
        route_gate_counts[f"{gate}:False"] = 24
        route_gate_counts[f"{gate}:True"] = 537
    route_gate_counts["http_success:True"] = 561
    recomputed = {
        "accounting": {
            "economic_cost_usd": main["economic_cost_usd"],
            "input_tokens": main["input_tokens"],
            "logical_task_count": EXPECTED_MAIN_TASKS,
            "openrouter_cost_usd": main["openrouter_cost_usd"],
            "output_tokens": main["output_tokens"],
            "route_gate_counts": route_gate_counts,
            "source_tasks_sha256": sha256_file(tasks_path),
            "successful_route_contract": {
                "is_byok": True,
                "model": MODEL,
                "openrouter_cost_must_equal_zero": True,
                "routed_provider": ROUTED_PROVIDER,
            },
            "task_status_counts": main["status_counts"],
            "wire_attempt_count": main["wire_attempt_count"],
        },
        "execution_id": RELEASE_ID,
        "invalid_or_failed_count": sum(row["status"] != "succeeded_valid" for row in main["scored"]),
        "preflight_excluded_from_analysis": True,
        "primary": primary,
        "schema_version": 1,
        "scoring": {
            "fresh_outputs_only": True,
            "invalid_or_failed": "retained and scored zero EM and zero F1",
            "normalization": "HybridQA official-compatible lowercase/punctuation/article/whitespace normalization",
        },
        "secondary": secondary,
        "sensitivity": {
            "adversarial_failure_score_em_gap_lower_bounds": {
                "definition": "minimum first-arm minus second-arm EM when each invalid or failed repeat is independently assigned any score in [0,1]; succeeded-repeat EM remains fixed",
                "intact_minus_trace": _adversarial_failure_score_em_lower_bound(main["scored"], first_arm="intact", second_arm="trace_drop"),
                "score_interval": [0.0, 1.0],
                "sham_minus_trace": _adversarial_failure_score_em_lower_bound(main["scored"], first_arm="sham_drop", second_arm="trace_drop"),
            },
            "intact_minus_sham_f1": _intact_sham_f1_sensitivity(records),
        },
        "stability": {arm: _stability(records, arm) for arm in ("intact", "sham_drop", "trace_drop")},
        "status": "complete",
    }
    _equal(recomputed, expected, "analysis summary")
    return recomputed


def verify_preflight_and_batch(root: Path, tasks: list[dict[str, Any]], preflight: dict[str, Any]) -> None:
    marker = load_json(root / "preflight_marker.json")
    claims = load_json(root / "batch_claims.json")
    task_ids = [
        "8032bcea03d462a947e628ee9af68c16b92a8f6b2bf7891082b70c6b6d11e64d",
        "24c3e7d7526d74a1e522bb9b483b2bd423a6b5ddd104cf77a436e6b29b827d18",
        "976b57e020b22982a4fd2a4ab4d689dc11949c217da244da83c2299ff09737b5",
    ]
    require(marker["task_ids"] == task_ids and marker["status"] == "passed", "preflight marker changed")
    require(
        marker["excluded_from_analysis"] is True
        and marker["is_byok"] is True
        and marker["model"] == MODEL
        and marker["routed_provider"] == ROUTED_PROVIDER,
        "preflight route fields changed",
    )
    require(
        marker["accounting"]
        == {
            "economic_cost_usd": preflight["economic_cost_usd"],
            "input_tokens": preflight["input_tokens"],
            "logical_calls": 3,
            "openrouter_cost_usd": 0.0,
            "output_tokens": preflight["output_tokens"],
            "wire_attempts": 3,
        },
        "preflight accounting marker changed",
    )
    for mode, expected_ids in (("main", [task["task_id"] for task in tasks]), ("preflight", task_ids)):
        claim = claims[mode]
        require(
            claim["all_calls_released_by_one_barrier"] is True
            and claim["logical_call_count"] == len(expected_ids)
            and claim["task_ids_sha256"] == sha256_bytes(canonical_json(expected_ids).encode()),
            f"{mode}: batch binding changed",
        )


def verify(root: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    require(root.is_dir(), "Gemini evidence directory is missing")
    verify_inventory(root)
    tasks, _ = verify_protocol_and_tasks(root)
    main_rows = load_jsonl(root / "main_results.jsonl")
    main = verify_results(main_rows, tasks, mode="main")
    preflight_ids = [task["task_id"] for task in tasks[:3]]
    preflight_rows = load_jsonl(root / "preflight_results.jsonl")
    preflight = verify_results(preflight_rows, tasks, mode="preflight", selected_task_ids=preflight_ids)
    analysis = verify_analysis(root, main, root / "tasks.jsonl")
    verify_preflight_and_batch(root, tasks, preflight)
    return {
        "barrier_claim_records_verified": True,
        "credentials_used": False,
        "failed_outcomes_retained_and_scored_zero": 24,
        "main_failure_arm_counts": main["failure_arm_counts"],
        "main_failure_kind_counts": main["failure_kind_counts"],
        "main_logical_calls": len(main_rows),
        "main_openrouter_cost_usd": main["openrouter_cost_usd"],
        "main_status_counts": main["status_counts"],
        "main_upstream_economic_cost_usd": main["economic_cost_usd"],
        "main_wire_attempts": main["wire_attempt_count"],
        "network_used": False,
        "preflight_logical_calls_excluded_from_analysis": len(preflight_rows),
        "preflight_status_counts": preflight["status_counts"],
        "preflight_upstream_economic_cost_usd": preflight["economic_cost_usd"],
        "primary": analysis["primary"],
        "projection_schema": "gemini_passage_ablation_public_result_v2",
        "route": {
            "allow_fallbacks": False,
            "direct_google_api_usage": False,
            "model": MODEL,
            "routed_provider": ROUTED_PROVIDER,
            "transport": "OpenRouter BYOK",
        },
        "sensitivity": analysis["sensitivity"],
        "setting_provenance_verified": True,
        "status": "verified",
        "total_upstream_economic_cost_usd": main["economic_cost_usd"] + preflight["economic_cost_usd"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = verify(args.evidence)
    except VerificationError as error:
        print(json.dumps({"status": "error", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
