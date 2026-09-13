#!/usr/bin/env python3
"""Independently verify the paper-scoped primary hosted-call archive.

This verifier uses only the released records.  It rehashes every ZIP
member, reconstructs prompt and task hashes from each model-visible request,
reparses the exact response content under the released strict rules, binds the
result to every released compact row, and checks the eight explicitly
unissued answer cells against their invalid upstream calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import string
from typing import Any, Iterable, Mapping
import zipfile


SCHEMA_VERSION = 1
ARCHIVE_TYPE = "rop_primary_hosted_call_archive_v1"
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
CONFIG_SHA256 = "108137317c5a5027f51d92453a9f348195ed3f089ac8c59465b27cfb051eae5a"
EXPECTED_ISSUED = 10_192
EXPECTED_UNISSUED = 8
EXPECTED_ATTEMPT_DISTRIBUTION = {1: 10_190, 2: 2}
EXPECTED_TRANSPORT_ATTEMPTS = 10_194
EXPECTED_MEMBERS = {
    "CALLS.jsonl",
    "MANIFEST.json",
    "README.txt",
    "UNISSUED_ANSWER_CELLS.jsonl",
}
MODEL_IDS = {
    "claude": "anthropic/claude-opus-4.7",
    "gemini": "gemini-3.1-pro-preview",
    "openai": "openai/gpt-5.5",
}
OPENROUTER_GENERATION_CONTROLS = {
    "max_tokens": 4096,
    "reasoning": {"effort": "medium", "exclude": False},
}
VERTEX_GENERATION_CONTROLS = {
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
    "thinking_config": {"thinking_level": "MEDIUM"},
}
RUN_IDS = {
    "bridge": "hybridqa_natural_v2_bridge_confirmatory",
    "rewrite": "hybridqa_natural_v3_query_rewrite_confirmatory",
    "answer": "hybridqa_natural_v3_answer_confirmatory",
}
FIXED_CONDITIONS = {
    "bridge": "predicted_bridge_v2",
    "rewrite": "query_rewrite_v1",
    "answer": None,
}
PRIMARY_ANSWER_CONDITIONS = (
    "question_only",
    "predicted_fact",
    "query_rewrite",
)
CONDITIONS_BY_MODEL = {
    "claude": list(PRIMARY_ANSWER_CONDITIONS),
    "openai": list(PRIMARY_ANSWER_CONDITIONS),
    "gemini": [
        *PRIMARY_ANSWER_CONDITIONS,
        "no_kb",
        "oracle_passage",
    ],
}
EXPECTED_ISSUED_BY_EXPERIMENT = {
    "answer": 6_592,
    "bridge": 1_800,
    "rewrite": 1_800,
}
EXPECTED_MODEL_COUNTS = {
    ("answer", "claude"): 1_800,
    ("answer", "gemini"): 3_000,
    ("answer", "openai"): 1_792,
    ("bridge", "claude"): 600,
    ("bridge", "gemini"): 600,
    ("bridge", "openai"): 600,
    ("rewrite", "claude"): 600,
    ("rewrite", "gemini"): 600,
    ("rewrite", "openai"): 600,
}
COMPACT_LOCKS = {
    "bridge": {
        "bytes": 4_991_925,
        "filename": "bridge/details.jsonl",
        "rows": 1_800,
        "sha256": "abf0027a3f61aa338eae4a259b8751f7adecc5ea7f79a08f25ad11b179a79fc5",
    },
    "rewrite": {
        "bytes": 1_783_100,
        "filename": "rewrite/details.jsonl",
        "rows": 1_800,
        "sha256": "ec1810ea0fb10373547d5e52dcc75a325be5810b951edbad672fc492e54c42bc",
    },
    "answer": {
        "bytes": 11_600_206,
        "filename": "answer/details.jsonl",
        "rows": 6_600,
        "sha256": "0e4877be9bb7983f407c632d0c8bc28cf73b4787f53cc330945db2b9d0cc4a72",
    },
}
RECORDED_PARSER_PREFIX = "code/src/ccr/"
CURRENT_PARSER_PREFIX = "code/ccr/"
PARSER_LOCKS = {
    "bridge": {
        "path": "code/src/ccr/retrieval/hybridqa_bridge.py",
        "sha256": "5e0f0fa578a741e29e8f91dd5f683ba6ba1afa96f8f3beaf230ec0729e4c879b",
    },
    "rewrite": {
        "path": "code/src/ccr/retrieval/hybridqa_query_rewrite.py",
        "sha256": "d7dda8a26ad434a73ae676ae9fa602ac0a2db5f3a97b916c634b5af2edd8fe9a",
    },
    "answer": {
        "path": "code/src/ccr/runtime/hybridqa_answers.py",
        "sha256": "ca16bb6b73d5d96d15f5ca2da6839f4e8ce647c0a6b8fbfd3ee854fbc36679c6",
    },
}
RESPONSE_SCHEMAS = {
    "bridge": {
        "type": "object",
        "properties": {
            "selected_link": {"type": "string"},
            "retrieval_fact": {"type": "string"},
        },
        "required": ["selected_link", "retrieval_fact"],
        "additionalProperties": False,
    },
    "rewrite": {
        "type": "object",
        "properties": {"search_query": {"type": "string"}},
        "required": ["search_query"],
        "additionalProperties": False,
    },
    "answer": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}
SCHEMA_NAMES = {
    "bridge": "hybridqa_bridge_selection",
    "rewrite": "hybridqa_query_rewrite",
    "answer": "hybridqa_answer",
}
LOCAL_PATH_PATTERN = re.compile(
    r"(?ix)(?:"
    r"file:(?://+|\\\\+)"
    r"|(?:~|\$HOME|\$\{HOME\}|%USERPROFILE%)[/\\]"
    r"|\\\\[^\\/\s]+[\\/][^\\/\s]+"
    r"|(?<![A-Za-z0-9])[A-Za-z]:[/\\]"
    r"|/(?:users|home|root|content|private|tmp|var|etc|opt|srv|mnt|media|volumes|proc|sys|dev)(?:/|\b)"
    r")"
)
EMAIL_PATTERN = re.compile(
    r"(?i)[a-z0-9._%+\-]{1,64}@[a-z0-9.\-]{1,255}\.[a-z]{2,24}"
)
FORBIDDEN_RECORD_KEYS = {
    "authorization",
    "api_key",
    "access_token",
    "secret",
    "signature",
    "thought_signature",
    "response_id",
    "created",
    "created_at",
    "updated_at",
    "create_time",
    "timestamp",
    "cost",
    "cost_usd",
    "price",
    "latency_seconds",
    "served_provider",
    "is_byok",
    "service_tier",
    "sdk_http_response",
    "system_fingerprint",
    "traffic_type",
}
EXPECTED_PRIVACY_SUMMARY = {
    "credentials_retained": 0,
    "email_addresses_retained": 0,
    "machine_local_paths_retained": 0,
    "provider_transport_metadata_retained": 0,
    "response_material": "strict-parser-visible text",
}
EXPECTED_PAPER_SCOPE = {
    "answer_conditions_by_model": CONDITIONS_BY_MODEL,
}
EXPECTED_COVERAGE = {
    "issued_by_experiment": EXPECTED_ISSUED_BY_EXPERIMENT,
    "issued_calls": EXPECTED_ISSUED,
    "logical_task_rows": EXPECTED_ISSUED,
    "max_attempt": 2,
    "planned_answer_cells": 6_600,
    "response_bearing_terminal_rows": EXPECTED_ISSUED,
    "retried_logical_tasks": 2,
    "retryable_failed_attempts": 2,
    "terminal_attempt_distribution": {
        str(attempt): count
        for attempt, count in EXPECTED_ATTEMPT_DISTRIBUTION.items()
    },
    "transport_attempts_started": EXPECTED_TRANSPORT_ATTEMPTS,
    "unissued_answer_cells": EXPECTED_UNISSUED,
}


class VerificationError(RuntimeError):
    """Raised when any archive, parser, privacy, or compact binding fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _read_compact(path: Path, experiment: str) -> dict[tuple[str, str, str], tuple[dict[str, Any], str]]:
    lock = COMPACT_LOCKS[experiment]
    require(path.is_file(), f"{experiment}: compact details missing")
    require(path.stat().st_size == lock["bytes"], f"{experiment}: compact byte count changed")
    require(sha256_file(path) == lock["sha256"], f"{experiment}: compact hash changed")
    result: dict[tuple[str, str, str], tuple[dict[str, Any], str]] = {}
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            require(line.endswith(b"\n") and line.strip(), f"{experiment}: malformed compact line {line_number}")
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VerificationError(f"{experiment}: invalid compact JSON line {line_number}") from error
            condition = (
                FIXED_CONDITIONS[experiment]
                if experiment != "answer"
                else row.get("condition")
            )
            key = (row.get("model"), row.get("question_id"), condition)
            require(all(isinstance(value, str) and value for value in key), "invalid compact identity")
            require(key not in result, f"{experiment}: duplicate compact identity {key}")
            result[key] = (row, sha256_bytes(line))
    require(len(result) == lock["rows"], f"{experiment}: compact row count changed")
    return result


def _strict_parse(experiment: str, text: str) -> tuple[bool, dict[str, str] | None, str | None]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False, None, "json_decode"
    expected_fields = {
        "bridge": ("selected_link", "retrieval_fact"),
        "rewrite": ("search_query",),
        "answer": ("answer",),
    }[experiment]
    if not isinstance(value, dict) or set(value) != set(expected_fields):
        return False, None, "unexpected_fields"
    if not all(isinstance(value[field], str) and value[field].strip() for field in expected_fields):
        return False, None, "invalid_or_blank_field"
    return True, {field: value[field].strip() for field in expected_fields}, None


def _messages_from_request(transport: str, request: Mapping[str, Any]) -> list[dict[str, str]]:
    if transport == "openrouter":
        expected = {"model", "messages", "max_tokens", "reasoning", "response_format"}
        require(set(request) == expected, "OpenRouter visible request fields changed")
        _validate_generation_controls(transport, request)
        messages = request.get("messages")
        require(isinstance(messages, list) and len(messages) == 2, "OpenRouter messages changed")
        system, user = messages
        require(
            isinstance(system, dict)
            and set(system) == {"content", "role"}
            and system.get("role") == "system"
            and isinstance(system.get("content"), str),
            "OpenRouter system message changed",
        )
        require(
            isinstance(user, dict)
            and set(user) == {"content", "role"}
            and user.get("role") == "user"
            and isinstance(user.get("content"), str)
            and user["content"].startswith("[USER]\n"),
            "OpenRouter user message changed",
        )
        return [
            {"role": "system", "content": system["content"]},
            {"role": "user", "content": user["content"][7:]},
        ]
    require(set(request) == {"config", "contents", "model", "system_instruction"}, "Vertex visible request fields changed")
    _validate_generation_controls(transport, request)
    system = request.get("system_instruction")
    contents = request.get("contents")
    require(isinstance(system, str) and system, "Vertex system instruction changed")
    require(isinstance(contents, str) and contents.startswith("[USER]\n"), "Vertex contents changed")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": contents[7:]},
    ]


def _validate_generation_controls(transport: str, request: Mapping[str, Any]) -> None:
    if transport == "openrouter":
        require(
            request.get("max_tokens") == OPENROUTER_GENERATION_CONTROLS["max_tokens"],
            "OpenRouter max_tokens changed",
        )
        require(
            request.get("reasoning") == OPENROUTER_GENERATION_CONTROLS["reasoning"],
            "OpenRouter reasoning controls changed",
        )
        require("temperature" not in request, "OpenRouter temperature override retained")
        return
    config = request.get("config")
    require(isinstance(config, dict), "Vertex config changed")
    require(
        set(config) == {*VERTEX_GENERATION_CONTROLS, "response_schema"},
        "Vertex config fields changed",
    )
    for key, expected in VERTEX_GENERATION_CONTROLS.items():
        require(config.get(key) == expected, f"Vertex {key} changed")
    require("temperature" not in config, "Vertex temperature override retained")


def _validate_schema(experiment: str, transport: str, request: Mapping[str, Any]) -> None:
    if transport == "openrouter":
        response_format = request.get("response_format")
        require(
            isinstance(response_format, dict)
            and set(response_format) == {"json_schema", "type"}
            and response_format.get("type") == "json_schema",
            "OpenRouter response format changed",
        )
        value = response_format.get("json_schema")
        require(
            isinstance(value, dict) and set(value) == {"name", "schema", "strict"},
            "OpenRouter JSON schema changed",
        )
        require(value.get("name") == SCHEMA_NAMES[experiment], "OpenRouter schema name changed")
        require(value.get("strict") is True, "OpenRouter strict flag changed")
        require(value.get("schema") == RESPONSE_SCHEMAS[experiment], "OpenRouter response schema changed")
    else:
        config = request.get("config")
        require(isinstance(config, dict), "Vertex config changed")
        _validate_generation_controls(transport, request)
        require(config.get("response_schema") == RESPONSE_SCHEMAS[experiment], "Vertex response schema changed")


def _sanitize_usage(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_usage(child)
            for key, child in sorted(value.items())
            if "cost" not in str(key).casefold()
            and "price" not in str(key).casefold()
            and str(key).casefold() not in {"is_byok", "traffic_type"}
        }
    if isinstance(value, list):
        return [_sanitize_usage(child) for child in value]
    return value


def _walk_text(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_text(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_text(child)
    elif isinstance(value, str):
        yield value


def _walk_key_paths(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, ...]]:
    if isinstance(value, dict):
        for key, child in value.items():
            current = (*path, str(key))
            yield current
            yield from _walk_key_paths(child, current)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_paths(child, (*path, "[]"))


def _privacy_check_text(text: str, label: str) -> None:
    require(LOCAL_PATH_PATTERN.search(text) is None, f"{label}: machine-local path retained")
    require(EMAIL_PATTERN.search(text) is None, f"{label}: email retained")


def _privacy_check(
    record: Any,
    label: str,
    *,
    allow_forbidden_inventory: bool = False,
) -> None:
    for text in _walk_text(record):
        _privacy_check_text(text, label)
    forbidden: list[str] = []
    for path in _walk_key_paths(record):
        lowered = path[-1].casefold()
        if allow_forbidden_inventory and len(path) >= 3 and path[:2] in {
            ("privacy", "response_metadata_field_occurrences_removed"),
            ("privacy", "sanitized_usage_field_occurrences_removed"),
        }:
            continue
        if (
            lowered in FORBIDDEN_RECORD_KEYS
            or any(
                fragment in lowered
                for fragment in ("authorization", "api_key", "access_token", "secret", "signature")
            )
            or "cost" in lowered
            or "price" in lowered
        ):
            forbidden.append(".".join(path))
    require(
        not forbidden,
        f"{label}: forbidden metadata keys retained: {sorted(set(forbidden))}",
    )


def _privacy_check_member_payload(name: str, payload: bytes) -> None:
    if name == "MANIFEST.json":
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError("MANIFEST.json: invalid JSON") from error
        _privacy_check(
            value,
            "MANIFEST.json",
            allow_forbidden_inventory=True,
        )
        return
    if name == "README.txt":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise VerificationError("README.txt: non-UTF-8 content") from error
        _privacy_check_text(text, "README.txt")


def _privacy_check_zip_metadata(bundle: zipfile.ZipFile) -> None:
    require(bundle.comment == b"", "ZIP archive comment must be empty")
    for info in bundle.infolist():
        _privacy_check_text(info.filename, f"ZIP member name {info.filename}")
        require(info.comment == b"", f"ZIP member comment must be empty: {info.filename}")
        require(info.extra == b"", f"ZIP member extra field must be empty: {info.filename}")


def _verify_trusted_archive_sha256(path: Path, expected_archive_sha256: str) -> str:
    require(path.is_file(), "hosted-call archive is missing")
    require(
        _is_sha256(expected_archive_sha256)
        and expected_archive_sha256 == expected_archive_sha256.casefold(),
        "trusted expected archive SHA-256 must be a lowercase 64-character hex digest",
    )
    actual = sha256_file(path)
    require(
        actual == expected_archive_sha256,
        "archive SHA-256 differs from the externally supplied trusted digest",
    )
    return actual


def _verify_zip(
    path: Path,
    *,
    expected_archive_sha256: str,
) -> tuple[zipfile.ZipFile, dict[str, Any], dict[str, bytes], str]:
    archive_sha256 = _verify_trusted_archive_sha256(path, expected_archive_sha256)
    try:
        bundle = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as error:
        raise VerificationError("hosted-call archive is not a ZIP") from error
    try:
        _privacy_check_zip_metadata(bundle)
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        require(set(names) == EXPECTED_MEMBERS and len(names) == len(set(names)), "archive member set changed")
        for info in infos:
            require(_safe_name(info.filename), f"unsafe archive member {info.filename}")
            require(info.date_time == FIXED_TIMESTAMP, f"archive timestamp changed: {info.filename}")
            require(info.compress_type == zipfile.ZIP_DEFLATED, f"archive compression changed: {info.filename}")
            require(info.create_system == 3, f"archive creator changed: {info.filename}")
            require(
                info.external_attr >> 16 == stat.S_IFREG | 0o644,
                f"archive mode changed: {info.filename}",
            )
            require(not (info.flag_bits & 0x1), f"encrypted archive member: {info.filename}")
        raw_manifest = bundle.read("MANIFEST.json")
        try:
            manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as error:
            raise VerificationError("archive manifest is invalid JSON") from error
        require(
            raw_manifest == pretty_json_bytes(manifest),
            "archive manifest serialization is not canonical",
        )
        _privacy_check_member_payload("MANIFEST.json", raw_manifest)
        require(
            set(manifest)
            == {
                "archive_type",
                "compact_sources",
                "coverage",
                "members",
                "paper_scope",
                "parser_sources",
                "privacy",
                "schema_version",
                "status",
            },
            "archive manifest fields changed",
        )
        require(manifest.get("archive_type") == ARCHIVE_TYPE, "archive type changed")
        require(manifest.get("schema_version") == SCHEMA_VERSION, "archive schema changed")
        require(manifest.get("status") == "verified_export", "archive export status changed")
        require(manifest.get("compact_sources") == COMPACT_LOCKS, "compact source locks changed")
        require(manifest.get("parser_sources") == PARSER_LOCKS, "parser locks changed")
        require(manifest.get("paper_scope") == EXPECTED_PAPER_SCOPE, "paper scope changed")
        require(
            manifest.get("privacy") == EXPECTED_PRIVACY_SUMMARY,
            "manifest privacy inventory changed",
        )
        declared = manifest.get("members")
        require(isinstance(declared, list) and len(declared) == 3, "archive member manifest changed")
        payloads: dict[str, bytes] = {}
        for row in declared:
            require(isinstance(row, dict) and set(row) == {"bytes", "path", "rows", "sha256"}, "invalid member manifest row")
            name = row.get("path")
            require(name in EXPECTED_MEMBERS - {"MANIFEST.json"}, "invalid declared member")
            require(_is_sha256(row.get("sha256")), f"invalid member hash: {name}")
            payload = bundle.read(name)
            require(len(payload) == row.get("bytes"), f"member byte count changed: {name}")
            require(sha256_bytes(payload) == row.get("sha256"), f"member hash changed: {name}")
            if name == "CALLS.jsonl":
                require(row.get("rows") == EXPECTED_ISSUED, "declared issued count changed")
            elif name == "UNISSUED_ANSWER_CELLS.jsonl":
                require(row.get("rows") == EXPECTED_UNISSUED, "declared unissued count changed")
            else:
                require(row.get("rows") is None, "README row count changed")
            _privacy_check_member_payload(name, payload)
            payloads[name] = payload
        require(set(payloads) == EXPECTED_MEMBERS - {"MANIFEST.json"}, "declared member set changed")
        require(
            manifest.get("coverage") == EXPECTED_COVERAGE,
            "manifest coverage changed",
        )
    except Exception:
        bundle.close()
        raise
    return bundle, manifest, payloads, archive_sha256


def _jsonl(payload: bytes, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        require(line.endswith(b"\n") and line.strip(), f"{label}: malformed line {line_number}")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError(f"{label}: invalid JSON line {line_number}") from error
        require(line == (canonical_json(row) + "\n").encode("utf-8"), f"{label}: noncanonical line {line_number}")
        yield line_number, row


def verify_archive(
    *,
    archive: Path,
    expected_archive_sha256: str,
    compact_paths: Mapping[str, Path],
    parser_root: Path,
) -> dict[str, Any]:
    bundle, manifest, payloads, archive_sha256 = _verify_zip(
        archive,
        expected_archive_sha256=expected_archive_sha256,
    )
    bundle.close()
    for experiment, lock in PARSER_LOCKS.items():
        parser_path = parser_root / lock["path"].replace(
            RECORDED_PARSER_PREFIX, CURRENT_PARSER_PREFIX, 1
        )
        require(parser_path.is_file(), f"{experiment}: released parser source missing")
        require(sha256_file(parser_path) == lock["sha256"], f"{experiment}: released parser source changed")
    compact = {
        experiment: _read_compact(compact_paths[experiment], experiment)
        for experiment in ("bridge", "rewrite", "answer")
    }
    issued_counts: Counter[str] = Counter()
    model_counts: Counter[tuple[str, str]] = Counter()
    seen_task_keys: set[str] = set()
    seen_identities: set[tuple[str, str, str, str]] = set()
    invalid_upstream: dict[tuple[str, str, str], dict[str, Any]] = {}
    parse_valid: Counter[str] = Counter()
    attempt_distribution: Counter[int] = Counter()
    answer_cells: Counter[tuple[str, str]] = Counter()
    for line_number, record in _jsonl(payloads["CALLS.jsonl"], "CALLS.jsonl"):
        label = f"CALLS.jsonl:{line_number}"
        require(
            set(record) == {
                "compact_binding",
                "identity",
                "model_id",
                "model_visible_request",
                "model_visible_request_sha256",
                "parser",
                "provenance",
                "record_type",
                "response_content",
                "response_content_sha256",
                "schema_version",
                "transport",
                "usage",
                "usage_sha256",
            },
            f"{label}: issued record schema changed",
        )
        require(record.get("record_type") == "issued_call" and record.get("schema_version") == 1, f"{label}: record type changed")
        _privacy_check(record, label)
        identity = record.get("identity")
        require(
            isinstance(identity, dict)
            and set(identity) == {"condition", "experiment", "item_id", "model", "run_id", "task_key"},
            f"{label}: identity schema changed",
        )
        experiment = identity.get("experiment")
        model = identity.get("model")
        item_id = identity.get("item_id")
        condition = identity.get("condition")
        task_key = identity.get("task_key")
        require(experiment in RUN_IDS, f"{label}: unknown experiment")
        require(model in MODEL_IDS and record.get("model_id") == MODEL_IDS[model], f"{label}: model binding changed")
        require(all(isinstance(value, str) and value for value in (item_id, condition, task_key)), f"{label}: incomplete identity")
        require(identity.get("run_id") == RUN_IDS[experiment], f"{label}: run ID changed")
        if FIXED_CONDITIONS[experiment] is not None:
            require(condition == FIXED_CONDITIONS[experiment], f"{label}: condition changed")
        else:
            require(
                condition in CONDITIONS_BY_MODEL[model],
                f"{label}: answer condition is outside the paper scope",
            )
            answer_cells[(model, condition)] += 1
        require(task_key not in seen_task_keys, f"{label}: duplicate task key")
        full_identity = (experiment, model, item_id, condition)
        require(full_identity not in seen_identities, f"{label}: duplicate identity")
        seen_task_keys.add(task_key)
        seen_identities.add(full_identity)
        transport = record.get("transport")
        require(transport == ("vertex" if model == "gemini" else "openrouter"), f"{label}: transport changed")
        request = record.get("model_visible_request")
        require(isinstance(request, dict), f"{label}: visible request missing")
        require(request.get("model") == MODEL_IDS[model], f"{label}: request model changed")
        require(
            record.get("model_visible_request_sha256") == sha256_bytes(canonical_json(request).encode("utf-8")),
            f"{label}: visible request hash changed",
        )
        messages = _messages_from_request(transport, request)
        _validate_schema(experiment, transport, request)
        provenance = record.get("provenance")
        require(
            isinstance(provenance, dict)
            and set(provenance) == {
                "config_sha256",
                "prompt_sha256",
                "source_request_json_sha256",
                "source_response_json_sha256",
                "source_usage_json_sha256",
            },
            f"{label}: provenance schema changed",
        )
        require(provenance.get("config_sha256") == CONFIG_SHA256, f"{label}: config hash changed")
        require(all(_is_sha256(provenance.get(key)) for key in provenance), f"{label}: invalid provenance hash")
        prompt_sha = sha256_bytes(canonical_json(messages).encode("utf-8"))
        require(prompt_sha == provenance.get("prompt_sha256"), f"{label}: prompt hash mismatch")
        task_document = {
            "run_id": identity["run_id"],
            "model": model,
            "item_id": item_id,
            "condition": condition,
            "messages": messages,
            "max_output_tokens": None,
            "response_schema": RESPONSE_SCHEMAS[experiment],
            "response_schema_name": SCHEMA_NAMES[experiment],
            "config": CONFIG_SHA256,
        }
        require(sha256_bytes(canonical_json(task_document).encode("utf-8")) == task_key, f"{label}: task key mismatch")
        response_content = record.get("response_content")
        require(isinstance(response_content, str), f"{label}: response content is not text")
        require(
            record.get("response_content_sha256") == sha256_bytes(response_content.encode("utf-8")),
            f"{label}: response-content hash mismatch",
        )
        valid, parsed, error_code = _strict_parse(experiment, response_content)
        parser_record = record.get("parser")
        require(
            parser_record == {
                "error_code": error_code,
                "name": f"{experiment}_strict_json_v1",
                "result": parsed,
                "valid": valid,
            },
            f"{label}: stored parser result differs from independent reparse",
        )
        compact_key = (model, item_id, condition)
        require(compact_key in compact[experiment], f"{label}: compact binding target missing")
        compact_row, compact_sha = compact[experiment].pop(compact_key)
        # Bridge and rewrite compact exports carry no task_key, so the binding
        # is the locked row bytes plus the model/item/condition identity.
        require(compact_row.get("model") == model, f"{label}: compact model changed")
        require(compact_row.get("question_id") == item_id, f"{label}: compact item changed")
        if experiment == "answer":
            require(compact_row.get("condition") == condition, f"{label}: compact condition changed")
        binding = record.get("compact_binding")
        require(
            isinstance(binding, dict)
            and set(binding) == {"projection", "row_sha256"}
            and binding.get("row_sha256") == compact_sha,
            f"{label}: compact row hash mismatch",
        )
        if experiment == "bridge":
            prediction = None if not valid else {
                "selected_link": compact_row.get("selected_link"),
                "retrieval_fact": compact_row.get("retrieval_fact"),
            }
        elif experiment == "rewrite":
            prediction = None if not valid else {"search_query": compact_row.get("search_query")}
        else:
            require(compact_row.get("task_key") == task_key, f"{label}: answer task-key binding changed")
            prediction = None if not valid else {"answer": compact_row.get("prediction")}
            require(_sanitize_usage(compact_row.get("usage")) == record.get("usage"), f"{label}: sanitized usage differs from compact result")
        require(prediction == parsed, f"{label}: compact prediction differs from reparse")
        expected_projection = {
            "attempt": compact_row.get("attempt"),
            "prediction": prediction,
            "status": compact_row.get("status"),
            "valid_json": valid,
        }
        require(binding.get("projection") == expected_projection, f"{label}: compact projection changed")
        attempt = expected_projection["attempt"]
        require(type(attempt) is int and attempt >= 1, f"{label}: invalid terminal attempt")
        attempt_distribution[attempt] += 1
        require(compact_row.get("status") == "succeeded", f"{label}: compact status changed")
        require(compact_row.get("error") is None, f"{label}: compact error changed")
        require(compact_row.get("valid_json") is valid, f"{label}: compact validity changed")
        usage = record.get("usage")
        require(isinstance(usage, dict), f"{label}: usage missing")
        require(record.get("usage_sha256") == sha256_bytes(canonical_json(usage).encode("utf-8")), f"{label}: usage hash changed")
        issued_counts[experiment] += 1
        model_counts[(experiment, model)] += 1
        parse_valid[f"{experiment}/{str(valid).lower()}"] += 1
        if experiment in {"bridge", "rewrite"} and not valid:
            invalid_upstream[(experiment, model, item_id)] = {
                "experiment": experiment,
                "response_content_sha256": record["response_content_sha256"],
                "task_key": task_key,
            }
    require(sum(issued_counts.values()) == EXPECTED_ISSUED, "issued-call total changed")
    require(
        dict(sorted(attempt_distribution.items())) == EXPECTED_ATTEMPT_DISTRIBUTION,
        "terminal attempt distribution changed",
    )
    require(
        dict(sorted(issued_counts.items())) == EXPECTED_ISSUED_BY_EXPERIMENT,
        "issued experiment counts changed",
    )
    require(
        dict(sorted(model_counts.items())) == EXPECTED_MODEL_COUNTS,
        "issued model counts changed",
    )
    require(not compact["bridge"] and not compact["rewrite"], "issued bridge/rewrite compact rows remain unbound")

    seen_unissued: set[tuple[str, str, str]] = set()
    unissued_counts: Counter[str] = Counter()
    for line_number, record in _jsonl(payloads["UNISSUED_ANSWER_CELLS.jsonl"], "UNISSUED_ANSWER_CELLS.jsonl"):
        label = f"UNISSUED_ANSWER_CELLS.jsonl:{line_number}"
        require(
            set(record) == {
                "compact_binding",
                "identity",
                "record_type",
                "schema_version",
                "source_call_issued",
                "upstream_binding",
            },
            f"{label}: unissued record schema changed",
        )
        require(record.get("record_type") == "unissued_answer_cell" and record.get("schema_version") == 1, f"{label}: record type changed")
        require(record.get("source_call_issued") is False, f"{label}: call-issued flag changed")
        _privacy_check(record, label)
        identity = record.get("identity")
        require(
            isinstance(identity, dict)
            and set(identity) == {"condition", "experiment", "item_id", "model", "run_id"}
            and identity.get("experiment") == "answer"
            and identity.get("run_id") == RUN_IDS["answer"],
            f"{label}: identity changed",
        )
        key = (identity.get("model"), identity.get("item_id"), identity.get("condition"))
        require(all(isinstance(value, str) and value for value in key), f"{label}: incomplete identity")
        require(identity["model"] in MODEL_IDS, f"{label}: unknown model")
        require(
            identity["condition"] in CONDITIONS_BY_MODEL[identity["model"]],
            f"{label}: answer condition is outside the paper scope",
        )
        answer_cells[(identity["model"], identity["condition"])] += 1
        require(key not in seen_unissued, f"{label}: duplicate unissued identity")
        seen_unissued.add(key)
        require(key in compact["answer"], f"{label}: compact unissued row missing")
        compact_row, compact_sha = compact["answer"].pop(key)
        require(
            compact_row.get("model") == identity["model"]
            and compact_row.get("question_id") == identity["item_id"]
            and compact_row.get("condition") == identity["condition"],
            f"{label}: compact identity changed",
        )
        require(
            compact_row.get("status") == "upstream_invalid"
            and compact_row.get("task_key") is None
            and compact_row.get("attempt") == 0
            and compact_row.get("valid_json") is False
            and compact_row.get("prediction") is None,
            f"{label}: compact unissued projection changed",
        )
        binding = record.get("compact_binding")
        require(
            binding
            == {
                "projection": {
                    "attempt": 0,
                    "prediction": None,
                    "status": "upstream_invalid",
                    "valid_json": False,
                },
                "row_sha256": compact_sha,
            },
            f"{label}: compact binding changed",
        )
        upstream = record.get("upstream_binding")
        source_experiment = (
            "bridge"
            if identity["condition"] == "predicted_fact"
            else "rewrite" if identity["condition"] == "query_rewrite" else None
        )
        require(source_experiment is not None, f"{label}: unissued condition changed")
        source = invalid_upstream.get((source_experiment, identity["model"], identity["item_id"]))
        require(upstream == source, f"{label}: invalid upstream call binding changed")
        expected_condition = (
            "predicted_fact" if source["experiment"] == "bridge" else "query_rewrite"
        )
        require(identity["condition"] == expected_condition, f"{label}: dependency mapping changed")
        unissued_counts[identity["condition"]] += 1
    require(len(seen_unissued) == EXPECTED_UNISSUED, "unissued answer count changed")
    require(
        dict(sorted(unissued_counts.items()))
        == {"predicted_fact": 6, "query_rewrite": 2},
        "unissued condition counts changed",
    )
    expected_answer_cells = Counter(
        {
            (model, condition): 600
            for model, conditions in CONDITIONS_BY_MODEL.items()
            for condition in conditions
        }
    )
    require(answer_cells == expected_answer_cells, "planned answer inventory changed")
    require(not compact["answer"], "answer compact rows remain unbound")
    require(len(invalid_upstream) == 8, "invalid upstream-call count changed")
    return {
        "archive": {
            "bytes": archive.stat().st_size,
            "filename": archive.name,
            "sha256": archive_sha256,
            "trusted_expected_sha256": expected_archive_sha256,
        },
        "archive_manifest_sha256": sha256_bytes(pretty_json_bytes(manifest)),
        "coverage": {
            "compact_rows_bound_one_to_one": sum(lock["rows"] for lock in COMPACT_LOCKS.values()),
            "issued_by_experiment": dict(sorted(issued_counts.items())),
            "issued_calls": sum(issued_counts.values()),
            "issued_model_counts": {
                f"{experiment}/{model}": count
                for (experiment, model), count in sorted(model_counts.items())
            },
            "logical_task_rows": sum(issued_counts.values()),
            "max_attempt": max(attempt_distribution),
            "planned_answer_cells": sum(answer_cells.values()),
            "response_bearing_terminal_rows": sum(issued_counts.values()),
            "retried_logical_tasks": sum(
                count for attempt, count in attempt_distribution.items() if attempt > 1
            ),
            "retryable_failed_attempts": (
                sum(attempt * count for attempt, count in attempt_distribution.items())
                - sum(issued_counts.values())
            ),
            "terminal_attempt_distribution": {
                str(attempt): count
                for attempt, count in sorted(attempt_distribution.items())
            },
            "transport_attempts_started": sum(
                attempt * count for attempt, count in attempt_distribution.items()
            ),
            "unissued_answer_cells": len(seen_unissued),
            "unissued_by_condition": dict(sorted(unissued_counts.items())),
        },
        "network_used": False,
        "paper_scope": manifest["paper_scope"],
        "parse_validity": dict(sorted(parse_valid.items())),
        "privacy": manifest["privacy"],
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument(
        "--expected-archive-sha256",
        required=True,
        help="trusted SHA-256 supplied outside the archive (lowercase hex)",
    )
    result.add_argument("--bridge-details", type=Path, required=True)
    result.add_argument("--rewrite-details", type=Path, required=True)
    result.add_argument("--answer-details", type=Path, required=True)
    result.add_argument("--parser-root", type=Path, required=True)
    result.add_argument("--report", type=Path)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = verify_archive(
            archive=args.archive,
            expected_archive_sha256=args.expected_archive_sha256,
            compact_paths={
                "bridge": args.bridge_details,
                "rewrite": args.rewrite_details,
                "answer": args.answer_details,
            },
            parser_root=args.parser_root,
        )
    except (VerificationError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"hosted-call archive verification failed: {error}", file=__import__("sys").stderr)
        return 1
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(pretty_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
