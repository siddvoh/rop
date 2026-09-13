#!/usr/bin/env python3
"""Replay routines behind verify.py: no credentials, no network."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, BinaryIO
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/ARTIFACT_MANIFEST.json"
AUTHORITY = ROOT / "data/generated/ccr_reviewer_authority.zip"
PRIMARY_HOSTED_CALL_ARCHIVE = (
    ROOT / "data/generated/ccr_primary_hosted_call_archive.zip"
)
RETRIEVER_CONFIG = (
    ROOT / "configs/hybridqa_retriever_sensitivity_v1.yaml"
)
PRIMARY_HOSTED_LEDGER_AUDIT = ROOT / "data/evidence/primary_hosted_call_ledger_audit.json"
PRIMARY_HOSTED_CALL_ARCHIVE_REPORT = (
    ROOT / "data/evidence/primary_hosted_call_archive.json"
)
PRIMARY_ANSWER_DETAILS = (
    ROOT
    / "data/results/released/hybridqa_natural_v3/answer_confirmatory/details.jsonl"
)
PRIMARY_BRIDGE_DETAILS = (
    ROOT
    / "data/results/released/hybridqa_natural_v2/bridge_confirmatory/details.jsonl"
)
PRIMARY_REWRITE_DETAILS = (
    ROOT
    / "data/results/released/hybridqa_natural_v3/query_rewrite_confirmatory/details.jsonl"
)
RETURNED_OBJECT_PRESERVATION_SCHEMA_MEMBER = (
    "code/contracts/returned_object_preservation.schema.json"
)
RETURNED_OBJECT_PRESERVATION_HARD_EXAMPLE_MEMBER = (
    "code/contracts/examples/returned_object_preservation.hard.json"
)
RETURNED_OBJECT_PRESERVATION_EVALUATOR_MEMBER = (
    "code/contracts/evaluate_returned_object_preservation.py"
)
RETURNED_OBJECT_PRESERVATION_KERNEL_MEMBER = (
    "code/contracts/returned_object_preservation_kernel.py"
)
RETURNED_OBJECT_PRESERVATION_FIXTURE_PREFIX = (
    "code/contracts/fixtures/prospective_v1"
)
RETURNED_OBJECT_PRESERVATION_CASES_MEMBER = (
    RETURNED_OBJECT_PRESERVATION_FIXTURE_PREFIX + "/cases.json"
)
RETURNED_OBJECT_PRESERVATION_EXPECTED_LEDGER_MEMBER = (
    RETURNED_OBJECT_PRESERVATION_FIXTURE_PREFIX + "/expected_ledger.json"
)
RETURNED_OBJECT_PRESERVATION_KERNEL_CASES_MEMBER = (
    RETURNED_OBJECT_PRESERVATION_FIXTURE_PREFIX + "/kernel_cases.json"
)
RETURNED_OBJECT_PRESERVATION_KERNEL_EXPECTED_LEDGER_MEMBER = (
    RETURNED_OBJECT_PRESERVATION_FIXTURE_PREFIX
    + "/kernel_expected_ledger.json"
)
RETURNED_OBJECT_PRESERVATION_FIXTURE_MEMBERS = (
    "all_manifest.json",
    "all_run_envelope.json",
    "cases.json",
    "config.json",
    "dotted_id_manifest.json",
    "dotted_id_run_envelope.json",
    "expected_ledger.json",
    "hard_manifest.json",
    "hard_run_envelope.json",
    "identity_witness.json",
    "invalid_manifest.json",
    "invalid_run_envelope.json",
    "kernel_cases.json",
    "kernel_expected_ledger.json",
    "policy_hard.json",
    "policy_soft.json",
    "population.json",
    "population_dotted.json",
    "population_invalid_manifest.json",
    "population_invalid_run_envelope.json",
    "prompt.txt",
    "resolver.json",
    "returned_component.py",
    "run_pass.json",
    "set_all_manifest.json",
    "set_all_run_envelope.json",
    "set_manifest.json",
    "set_run_envelope.json",
    "soft_manifest.json",
    "soft_run_envelope.json",
    "witness_manifest.json",
    "witness_run_envelope.json",
)
RETURNED_OBJECT_PRESERVATION_SCHEMA_SHA256 = (
    "1787f662673ef768265848e2e61b7b445387020a10c162e383c79218cf0b7d62"
)
RETURNED_OBJECT_PRESERVATION_HARD_EXAMPLE_SHA256 = (
    "74259add4804927f0f5b6f9f4fae82678b7638cd355b74fd71733fac2ff7bb78"
)
RETURNED_OBJECT_PRESERVATION_EVALUATOR_SHA256 = (
    "1b5fd5a9581a72677bff9f67af036cc81c63492ff716c880e2e80065a1ddf33f"
)
RETURNED_OBJECT_PRESERVATION_KERNEL_SHA256 = (
    "1fc5584f070735173d6954a534e2a784c5aad4d9cb0383e3b95198b6cd186c17"
)
RETURNED_OBJECT_PRESERVATION_CASES_SHA256 = (
    "4460b19eef288951d4f4ee1fceb5608b12a2ec619bb8bb508a6214d52ff586bc"
)
RETURNED_OBJECT_PRESERVATION_EXPECTED_LEDGER_SHA256 = (
    "5f1434663c986aea0494e595595c364537f38d5ed09b11552618cca061ac5bc1"
)
RETURNED_OBJECT_PRESERVATION_KERNEL_CASES_SHA256 = (
    "108c39e4e34341da91201ee533a74ca2dfcf029d84105d9bcc40a4a594527488"
)
RETURNED_OBJECT_PRESERVATION_KERNEL_EXPECTED_LEDGER_SHA256 = (
    "2d4769b53cb88d9c92f21c5f593745e7b03ad8d4a65afe43378fe4c5d3c846ed"
)
RETURNED_OBJECT_PRESERVATION_FIXTURE_TREE_SHA256 = (
    "ba878e8262599242dce160f9737440d3d4cc4a82be0251bf8c6a1834fb5e7be7"
)
EXCLUDED_DIRECTORIES = frozenset(
    {".git", ".github", ".pytest_cache", ".venv", "__pycache__"}
)
EXCLUDED_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
RECORDED_PATH_PREFIXES = (
    ("code/configs/experiments/", "configs/"),
    ("code/src/ccr/", "code/ccr/"),
    ("scripts/", "code/analysis/"),
)
RETRIEVER_CONFIG_SHA256 = (
    "5673d2cca5fef9a9d49515ff40175c8b7464b93ab948d5b346afc2a106574cb7"
)
FORBIDDEN_CONTENT = (
    ("work" + "ing/").encode(),
    ("rebut" + "tal/").encode(),
    ("iter" + "ations/").encode(),
)
UNIX_ABSOLUTE_PATH_PATTERN = re.compile(
    rb"(?ix)(?:^|[\s\"'=:(])"
    rb"/(?:users|home|root|content|private|tmp|var|etc|opt|srv|mnt|media|volumes)"
    rb"(?:/|\b)"
)
FILE_URI_PREFIX = ("file" + "://").encode()
HOME_PATH_PREFIX = ("~" + "/").encode()
WINDOWS_DRIVE_PATTERN = re.compile(rb"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[/\\]")
WORKSPACE_METADATA_PATTERN = re.compile(
    rb"(?i)(?:^|[\s\"'=:(])\.[a-z0-9._-]{1,128}[/\\]"
    rb"(?:attachments|worktrees|projects|sessions)(?:[/\\]|\b)"
)
EMAIL_PATTERN = re.compile(
    rb"(?i)[a-z0-9._%+\-]{1,64}@[a-z0-9.\-]{1,255}\.[a-z]{2,24}"
)
PRIVATE_SCAN_CHUNK_BYTES = 1024 * 1024
PRIVATE_SCAN_OVERLAP_BYTES = max(
    max(len(token) for token in FORBIDDEN_CONTENT),
    256,
    64 + 1 + 255 + 1 + 24,
) - 1
REQUIRED_ADDED_AUDIT_REPLAYS = frozenset(
    {
        "bm25_omission_taxonomy",
        "answer_associations",
        "mismatch_overlap_analysis",
        "mismatch_stack_cutoffs",
        "lossy_query_target_divergence",
        "full_cohort_stack_decisions",
        "mismatch_ordering_robustness",
        "mismatch_preservation_gap_uncertainty",
        "mismatch_metric_inversion_uncertainty",
    }
)
AUTHORITY_REQUIRED_ADDED_AUDIT_REPLAYS = frozenset(
    {"qwen_decoded_id_replication", "transfer_scope_audit"}
)
REQUIRED_BOUND_AUDIT_REPORTS = frozenset(
    {
        "identifier_aware_executor_report_binding",
        "hybridqa_bm25_reconstruction_contract_binding",
        "returned_object_preservation_contract_validation",
        "returned_object_preservation_executable_fixture",
    }
)


class VerificationError(RuntimeError):
    """Raised when a reviewer-artifact invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _returned_object_fixture_tree_sha256(root: Path) -> str:
    """Hash the exact path/size/digest inventory of every shipped fixture byte."""

    rows: list[dict[str, Any]] = []
    for name in RETURNED_OBJECT_PRESERVATION_FIXTURE_MEMBERS:
        path = root / name
        if not path.is_file():
            raise VerificationError(
                f"returned-object preservation fixture member missing: {name}"
            )
        rows.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = (
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def contains_private_material(payload: bytes) -> bool:
    lowered = payload.lower()
    return (
        any(token in lowered for token in FORBIDDEN_CONTENT)
        or UNIX_ABSOLUTE_PATH_PATTERN.search(payload) is not None
        or FILE_URI_PREFIX in lowered
        or HOME_PATH_PREFIX in payload
        or WINDOWS_DRIVE_PATTERN.search(payload) is not None
        or WORKSPACE_METADATA_PATTERN.search(payload) is not None
        or EMAIL_PATTERN.search(payload) is not None
    )


def stream_contains_private_material(
    handle: BinaryIO,
    *,
    chunk_bytes: int = PRIVATE_SCAN_CHUNK_BYTES,
) -> bool:
    if chunk_bytes < 1:
        raise ValueError("privacy-scan chunk size must be positive")
    overlap = b""
    while True:
        chunk = handle.read(chunk_bytes)
        if not chunk:
            break
        window = overlap + chunk
        lowered = window.lower()
        if (
            any(token in lowered for token in FORBIDDEN_CONTENT)
            or UNIX_ABSOLUTE_PATH_PATTERN.search(window) is not None
            or FILE_URI_PREFIX in lowered
            or HOME_PATH_PREFIX in window
            or WINDOWS_DRIVE_PATTERN.search(window) is not None
            or WORKSPACE_METADATA_PATTERN.search(window) is not None
            or EMAIL_PATTERN.search(window) is not None
        ):
            return True
        overlap = window[-PRIVATE_SCAN_OVERLAP_BYTES:]
    return False


def safe_relative(path: str) -> bool:
    value = PurePosixPath(path)
    return bool(path) and not value.is_absolute() and ".." not in value.parts


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path.relative_to(ROOT)}") from exc


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON member names instead of silently taking the last."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        try:
            label = path.relative_to(ROOT)
        except ValueError:
            label = path
        raise VerificationError(f"cannot read strict JSON {label}") from exc


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality that does not conflate booleans and integers."""

    return type(left) is type(right) and left == right


def _json_schema_errors(
    instance: Any, schema: Any, *, path: str = "$"
) -> list[str]:
    """Validate the dependency-free JSON Schema subset used by the contract.

    The public artifact intentionally has no third-party runtime dependency.
    The tracked schema uses only the keywords implemented here; its exact hash
    is independently locked below.
    """

    if not isinstance(schema, dict):
        return [f"{path}: schema node is not an object"]
    errors: list[str] = []

    expected_type = schema.get("type")
    if expected_type is not None:
        type_checks = {
            "object": lambda value: isinstance(value, dict),
            "string": lambda value: isinstance(value, str),
            "integer": lambda value: type(value) is int,
            "boolean": lambda value: type(value) is bool,
        }
        check = type_checks.get(expected_type)
        if check is None:
            errors.append(f"{path}: unsupported schema type {expected_type!r}")
        elif not check(instance):
            errors.append(f"{path}: expected {expected_type}")

    if "const" in schema and not _json_equal(instance, schema["const"]):
        errors.append(f"{path}: value differs from const")
    if "enum" in schema and not any(
        _json_equal(instance, candidate) for candidate in schema["enum"]
    ):
        errors.append(f"{path}: value is not in enum")

    if isinstance(instance, str):
        pattern = schema.get("pattern")
        if pattern is not None:
            try:
                matched = re.search(pattern, instance) is not None
            except (re.error, TypeError) as exc:
                return [f"{path}: invalid schema pattern: {exc}"]
            if not matched:
                errors.append(f"{path}: string does not match pattern")

    if type(instance) is int and "minimum" in schema:
        if instance < schema["minimum"]:
            errors.append(f"{path}: integer is below minimum")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(key, str) for key in required
        ):
            errors.append(f"{path}: schema required list is invalid")
        else:
            for key in required:
                if key not in instance:
                    errors.append(f"{path}: missing required member {key!r}")

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{path}: schema properties is invalid")
        else:
            for key, subschema in properties.items():
                if key in instance:
                    errors.extend(
                        _json_schema_errors(
                            instance[key], subschema, path=f"{path}.{key}"
                        )
                    )
            if schema.get("additionalProperties") is False:
                for key in sorted(set(instance) - set(properties)):
                    errors.append(f"{path}: unexpected member {key!r}")

    for index, subschema in enumerate(schema.get("allOf", [])):
        errors.extend(
            _json_schema_errors(instance, subschema, path=f"{path}.allOf[{index}]")
        )

    if "anyOf" in schema:
        matches = [
            not _json_schema_errors(instance, subschema, path=path)
            for subschema in schema["anyOf"]
        ]
        if not any(matches):
            errors.append(f"{path}: no anyOf branch matched")

    if "oneOf" in schema:
        matches = sum(
            not _json_schema_errors(instance, subschema, path=path)
            for subschema in schema["oneOf"]
        )
        if matches != 1:
            errors.append(f"{path}: expected one oneOf match, observed {matches}")

    if "not" in schema and not _json_schema_errors(instance, schema["not"], path=path):
        errors.append(f"{path}: forbidden schema matched")

    if "if" in schema:
        condition_matches = not _json_schema_errors(instance, schema["if"], path=path)
        branch = "then" if condition_matches else "else"
        if branch in schema:
            errors.extend(_json_schema_errors(instance, schema[branch], path=path))
    return errors


def validate_returned_object_preservation_manifest(
    document: Any, schema: Any | None = None
) -> None:
    """Apply the tracked contract schema without a package or network lookup."""

    if schema is None:
        schema = _load_strict_json(
            ROOT / RETURNED_OBJECT_PRESERVATION_SCHEMA_MEMBER
        )
    errors = _json_schema_errors(document, schema)
    if errors:
        raise VerificationError(
            "returned-object preservation manifest is invalid: " + errors[0]
        )


def _require_contract_rejection(
    document: Any, schema: Any, *, label: str
) -> None:
    try:
        validate_returned_object_preservation_manifest(document, schema)
    except VerificationError:
        return
    raise VerificationError(f"returned-object contract accepted {label}")


def verify_returned_object_preservation_contract() -> str:
    """Validate the versioned schema, hard example, and activation boundary."""

    schema_path = ROOT / RETURNED_OBJECT_PRESERVATION_SCHEMA_MEMBER
    example_path = ROOT / RETURNED_OBJECT_PRESERVATION_HARD_EXAMPLE_MEMBER
    if (
        not schema_path.is_file()
        or not example_path.is_file()
    ):
        raise VerificationError(
            "returned-object preservation schema or hard example is missing"
        )
    if (
        sha256_file(schema_path)
        != RETURNED_OBJECT_PRESERVATION_SCHEMA_SHA256
        or sha256_file(example_path)
        != RETURNED_OBJECT_PRESERVATION_HARD_EXAMPLE_SHA256
    ):
        raise VerificationError("returned-object preservation contract hash changed")

    schema = _load_strict_json(schema_path)
    example = _load_strict_json(example_path)
    if (
        not isinstance(schema, dict)
        or schema.get("$schema")
        != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("$id")
        != "urn:ccr:returned-object-preservation-contract:v1"
        or "$ref" in json.dumps(schema, ensure_ascii=False)
    ):
        raise VerificationError("returned-object preservation schema identity changed")
    validate_returned_object_preservation_manifest(example, schema)

    for field in ("population", "resolver", "returned_set", "retention"):
        mutated = copy.deepcopy(example)
        del mutated[field]
        _require_contract_rejection(
            mutated, schema, label=f"hard declaration without {field}"
        )
    for field in ("source_field", "type", "namespace"):
        mutated = copy.deepcopy(example)
        del mutated["selection"][field]
        _require_contract_rejection(
            mutated, schema, label=f"hard declaration without selection.{field}"
        )

    for mode in ("soft", "open"):
        nonapplicable = copy.deepcopy(example)
        nonapplicable["selection"] = {"mode": mode}
        for field in ("population", "resolver", "returned_set", "retention"):
            del nonapplicable[field]
        validate_returned_object_preservation_manifest(nonapplicable, schema)
        conflicting = copy.deepcopy(nonapplicable)
        conflicting["resolver"] = copy.deepcopy(example["resolver"])
        _require_contract_rejection(
            conflicting, schema, label=f"{mode} declaration with hard resolver"
        )

    witness = copy.deepcopy(example)
    del witness["selection"]["source_field"]
    witness["selection"]["identity_witness"] = {
        "witness_version": "rop-identity-witness/v1",
        "source_field": "selector.identity_witness",
        "artifact_locator": "resolver/identity-witness.json",
        "artifact_sha256": "1" * 64,
        "decoding": "bound_exact_inverse_map",
        "record_binding": "sha256_canonical_json",
        "authentication_basis": "trusted_producer_record_binding",
    }
    validate_returned_object_preservation_manifest(witness, schema)
    conflicting_identity_source = copy.deepcopy(witness)
    conflicting_identity_source["selection"]["source_field"] = (
        "selector.selected_link"
    )
    _require_contract_rejection(
        conflicting_identity_source,
        schema,
        label="hard declaration with raw key and identity witness",
    )

    for field in ("can_activate", "can_override_manifest"):
        mutated = copy.deepcopy(example)
        mutated["authority"]["natural_language_prompt"][field] = True
        _require_contract_rejection(
            mutated, schema, label=f"natural-language prompt {field}"
        )

    for field, value in (
        ("chronology_basis", "self_asserted_untrusted"),
        ("log_provenance_basis", "untrusted_log"),
        ("hash_scope", "proves_chronology"),
        ("adversarial_producer_resistance", True),
    ):
        mutated = copy.deepcopy(example)
        mutated["authority"]["manifest_authority"][field] = value
        _require_contract_rejection(
            mutated, schema, label=f"overstated trust field {field}"
        )

    all_retention = copy.deepcopy(example)
    all_retention["retention"]["quantifier"] = "all"
    validate_returned_object_preservation_manifest(all_retention, schema)
    expected_verdicts = {
        "total_verdicts": True,
        "declaration_invalid": "DECLARATION_INVALID",
        "soft_or_open": "NOT_APPLICABLE",
        "selected_key_invalid": "SELECTOR_ERROR",
        "selected_key_unresolvable": "RESOLUTION_ERROR",
        "returned_id_log_missing_or_malformed": "UNVERIFIABLE",
        "retention_satisfied": "PASS",
        "retention_violated": "FAIL",
        "ledger_policy": "retain_every_record",
        "metric_denominator": "hard_resolvable_verifiable_only",
    }
    if example.get("verdict_policy") != expected_verdicts:
        raise VerificationError("returned-object total-verdict policy changed")
    expected_run_policy = {
        "profile_version": "rop-run-profile/v1",
        "invalid_declaration_or_population": "RUN_INVALID",
        "expected_hard_record_failed": "RUN_FAIL",
        "expected_hard_record_missing_or_unverifiable": "RUN_UNVERIFIABLE",
        "all_expected_hard_records_passed": "RUN_PASS",
        "soft_or_open": "NOT_APPLICABLE",
        "status_histogram": "always",
        "eligible_expected_coverage": "eligible_over_expected",
        "diagnostic_pass_fraction": "PASS/(PASS+FAIL)",
        "diagnostic_is_conformance_verdict": False,
    }
    if example.get("run_policy") != expected_run_policy:
        raise VerificationError("returned-object strict run policy changed")
    return "returned_object_preservation_contract_validation"


def verify_returned_object_preservation_evaluator() -> str:
    """Replay the prospective fixture through the public pure evaluator."""

    evaluator_path = ROOT / RETURNED_OBJECT_PRESERVATION_EVALUATOR_MEMBER
    kernel_path = ROOT / RETURNED_OBJECT_PRESERVATION_KERNEL_MEMBER
    cases_path = ROOT / RETURNED_OBJECT_PRESERVATION_CASES_MEMBER
    expected_path = ROOT / RETURNED_OBJECT_PRESERVATION_EXPECTED_LEDGER_MEMBER
    kernel_cases_path = ROOT / RETURNED_OBJECT_PRESERVATION_KERNEL_CASES_MEMBER
    kernel_expected_path = (
        ROOT / RETURNED_OBJECT_PRESERVATION_KERNEL_EXPECTED_LEDGER_MEMBER
    )
    exact_hashes = {
        evaluator_path: RETURNED_OBJECT_PRESERVATION_EVALUATOR_SHA256,
        kernel_path: RETURNED_OBJECT_PRESERVATION_KERNEL_SHA256,
        cases_path: RETURNED_OBJECT_PRESERVATION_CASES_SHA256,
        expected_path: RETURNED_OBJECT_PRESERVATION_EXPECTED_LEDGER_SHA256,
        kernel_cases_path: RETURNED_OBJECT_PRESERVATION_KERNEL_CASES_SHA256,
        kernel_expected_path: (
            RETURNED_OBJECT_PRESERVATION_KERNEL_EXPECTED_LEDGER_SHA256
        ),
    }
    for path, expected in exact_hashes.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise VerificationError(
                "returned-object preservation executable fixture hash changed"
            )
    fixture_root = ROOT / RETURNED_OBJECT_PRESERVATION_FIXTURE_PREFIX
    if (
        _returned_object_fixture_tree_sha256(fixture_root)
        != RETURNED_OBJECT_PRESERVATION_FIXTURE_TREE_SHA256
    ):
        raise VerificationError(
            "returned-object preservation transitive fixture tree changed"
        )

    spec = importlib.util.spec_from_file_location(
        "_ccr_returned_object_preservation_evaluator", evaluator_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load returned-object preservation evaluator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.evaluate_bundle(cases_path)
        replayed = module.canonical_json_bytes(report)
        kernel_report = module.evaluate_kernel_bundle(kernel_cases_path)
        kernel_replayed = module.canonical_json_bytes(kernel_report)
    except Exception as exc:
        raise VerificationError(
            f"returned-object preservation fixture replay failed: {exc}"
        ) from exc
    if replayed != expected_path.read_bytes():
        raise VerificationError("returned-object preservation ledger replay changed")
    if kernel_replayed != kernel_expected_path.read_bytes():
        raise VerificationError("returned-object preservation kernel replay changed")
    if (
        kernel_report.get("status") != "complete"
        or kernel_report.get("kernel_version")
        != "rop-preservation-kernel/v1"
        or kernel_report.get("kernel_sha256")
        != RETURNED_OBJECT_PRESERVATION_KERNEL_SHA256
        or kernel_report.get("is_conformance_verdict") is not False
        or kernel_report.get("case_count") != 6
        or kernel_report.get("verdict_counts")
        != {
            "SELECTOR_ERROR": 1,
            "RESOLUTION_ERROR": 1,
            "UNVERIFIABLE": 1,
            "PASS": 2,
            "FAIL": 1,
        }
        or any(
            row.get("is_conformance_verdict") is not False
            for row in kernel_report.get("cases", [])
        )
    ):
        raise VerificationError("returned-object preservation kernel boundary changed")
    if (
        report.get("status") != "complete"
        or report.get("record_count") != 15
        or report.get("run_count") != 12
        or report.get("verdict_counts")
        != {
            "DECLARATION_INVALID": 1,
            "NOT_APPLICABLE": 1,
            "SELECTOR_ERROR": 2,
            "RESOLUTION_ERROR": 2,
            "UNVERIFIABLE": 1,
            "PASS": 6,
            "FAIL": 2,
        }
    ):
        raise VerificationError("returned-object preservation verdict coverage changed")
    rows = report.get("records")
    if not isinstance(rows, list) or sum(
        row.get("eligible_for_metric") is True for row in rows
    ) != 8:
        raise VerificationError("returned-object preservation denominator changed")

    schema = _load_strict_json(
        ROOT / RETURNED_OBJECT_PRESERVATION_SCHEMA_MEMBER
    )
    for name in (
        "hard_manifest.json",
        "all_manifest.json",
        "set_manifest.json",
        "set_all_manifest.json",
        "dotted_id_manifest.json",
        "soft_manifest.json",
        "population_invalid_manifest.json",
        "witness_manifest.json",
    ):
        validate_returned_object_preservation_manifest(
            _load_strict_json(fixture_root / name), schema
        )
    _require_contract_rejection(
        _load_strict_json(fixture_root / "invalid_manifest.json"),
        schema,
        label="tracked invalid prospective fixture",
    )

    by_case = {row.get("case_id"): row for row in rows}
    if (
        by_case.get("hard_any_alias_pass", {}).get("verdict") != "PASS"
        or by_case.get("prompt_cannot_activate_soft", {}).get("verdict")
        != "NOT_APPLICABLE"
        or by_case.get("hard_all_members_pass", {}).get("verdict") != "PASS"
        or by_case.get("hard_all_members_fail", {}).get("verdict") != "FAIL"
        or by_case.get("hard_key_set_any_accepts_one_object", {}).get("verdict")
        != "PASS"
        or by_case.get(
            "hard_key_set_all_accepts_one_complete_object", {}
        ).get("verdict")
        != "PASS"
        or by_case.get(
            "hard_key_set_all_accepts_singleton_alternative", {}
        ).get("verdict")
        != "PASS"
        or by_case.get(
            "hard_lossless_witness_pass_without_raw_key", {}
        ).get("verdict")
        != "PASS"
        or by_case.get(
            "hard_lossless_witness_binding_mismatch", {}
        ).get("verdict")
        != "SELECTOR_ERROR"
        or by_case.get(
            "hard_lossless_witness_unknown_value", {}
        ).get("verdict")
        != "RESOLUTION_ERROR"
    ):
        raise VerificationError(
            "returned-object preservation authority or retention semantics changed"
        )
    if report.get("run_status_counts") != {
        "RUN_INVALID": 4,
        "RUN_FAIL": 1,
        "RUN_UNVERIFIABLE": 3,
        "RUN_PASS": 3,
        "NOT_APPLICABLE": 1,
    }:
        raise VerificationError("returned-object strict run aggregation changed")
    runs = report.get("runs")
    if not isinstance(runs, list) or any(
        not {
            "status_histogram",
            "eligible_expected_coverage",
            "diagnostic_pass_fraction",
        }
        <= set(row)
        or row.get("diagnostic_pass_fraction", {}).get(
            "is_conformance_verdict"
        )
        is not False
        for row in runs
    ):
        raise VerificationError("returned-object run diagnostics changed")
    by_run = {row.get("run_id"): row for row in runs}
    dotted_pass = by_run.get("dotted_record_id_run_pass", {})
    dotted_duplicate = by_run.get("dotted_record_id_run_duplicate", {})
    dotted_missing = by_run.get("dotted_record_id_run_missing", {})
    witness_pass = by_run.get("lossless_witness_run_pass", {})
    invalid_precedence = by_run.get(
        "strict_run_invalid_precedes_record_fail", {}
    )
    if (
        dotted_pass.get("run_status") != "RUN_PASS"
        or dotted_pass.get("population_binding", {}).get("record_id_field")
        != "identity.record_key"
        or [
            row.get("record_id") for row in dotted_pass.get("records", [])
        ]
        != ["run_alpha", "run_beta"]
        or dotted_duplicate.get("run_status") != "RUN_INVALID"
        or dotted_duplicate.get("duplicate_record_ids") != ["run_alpha"]
        or dotted_duplicate.get("missing_record_ids") != ["run_beta"]
        or dotted_missing.get("run_status") != "RUN_UNVERIFIABLE"
        or dotted_missing.get("missing_record_ids") != ["run_beta"]
        or witness_pass.get("run_status") != "RUN_PASS"
        or invalid_precedence.get("run_status") != "RUN_INVALID"
        or invalid_precedence.get("status_histogram", {}).get("FAIL") != 1
    ):
        raise VerificationError(
            "returned-object population record-ID field semantics changed"
        )
    return "returned_object_preservation_executable_fixture"


def verify_manifest() -> dict[str, Any]:
    document = load_json(MANIFEST)
    if not isinstance(document, dict) or document.get("status") != "complete":
        raise VerificationError("artifact manifest status is not complete")
    rows = document.get("members")
    if not isinstance(rows, list) or not rows:
        raise VerificationError("artifact manifest has no members")
    declared: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"bytes", "path", "sha256"}:
            raise VerificationError("artifact manifest row schema changed")
        path = row["path"]
        if not isinstance(path, str) or not safe_relative(path) or path in declared:
            raise VerificationError(f"invalid or duplicate manifest path: {path!r}")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise VerificationError(f"invalid byte count for {path}")
        digest = row["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise VerificationError(f"invalid SHA-256 for {path}")
        declared[path] = row

    actual: set[str] = set()
    for path in ROOT.rglob("*"):
        relative_parts = path.relative_to(ROOT).parts
        if EXCLUDED_DIRECTORIES.intersection(relative_parts):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        relative = path.relative_to(ROOT).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise VerificationError(f"artifact contains symlink: {relative}")
        if path.is_file() and relative != "data/manifests/ARTIFACT_MANIFEST.json":
            actual.add(relative)
    if actual != set(declared):
        missing = sorted(set(declared) - actual)
        extra = sorted(actual - set(declared))
        raise VerificationError(
            f"artifact member set differs; missing={missing}, extra={extra}"
        )
    for name, row in declared.items():
        path = ROOT / name
        if path.stat().st_size != row["bytes"]:
            raise VerificationError(f"byte count changed: {name}")
        if sha256_file(path) != row["sha256"]:
            raise VerificationError(f"SHA-256 changed: {name}")
    return {"member_count": len(declared), "status": "verified"}


def verify_no_private_material() -> dict[str, int]:
    files_checked = 0
    nested_members_checked = 0
    for path in release_files(ROOT):
        relative = path.relative_to(ROOT).as_posix()
        if contains_private_material(relative.encode()):
            raise VerificationError(f"private token in artifact path: {relative}")
        if path in {
            AUTHORITY,
            PRIMARY_HOSTED_CALL_ARCHIVE,
        }:
            with zipfile.ZipFile(path) as nested_archive:
                for info in nested_archive.infolist():
                    if contains_private_material(info.filename.encode()):
                        raise VerificationError(
                            f"private token in authority path: {info.filename}"
                        )
                    with nested_archive.open(info) as member:
                        if stream_contains_private_material(member):
                            raise VerificationError(
                                f"private token in authority member: {info.filename}"
                            )
                    nested_members_checked += 1
        else:
            with path.open("rb") as handle:
                if stream_contains_private_material(handle):
                    raise VerificationError(
                        f"private token in artifact member: {relative}"
                    )
        files_checked += 1
    return {
        "files_checked": files_checked,
        "nested_archive_members_checked": nested_members_checked,
    }


def verify_primary_hosted_ledger_audit() -> dict[str, Any]:
    report = load_json(PRIMARY_HOSTED_LEDGER_AUDIT)
    required = {
        "status": "verified",
        "scope": "primary_hosted_call_public_projection",
        "planned_answer_cells": 6_600,
        "issued_answer_logical_tasks": 6_592,
        "issued_logical_tasks": 10_192,
        "response_bearing_terminal_rows": 10_192,
        "retried_logical_tasks": 2,
        "retryable_failed_attempts": 2,
        "transport_attempts_started": 10_194,
        "terminal_attempt_distribution": {"1": 10_190, "2": 2},
        "issued_by_experiment": {
            "answer": 6_592,
            "bridge": 1_800,
            "rewrite": 1_800,
        },
        "no_missing_terminal_records_for_issued_logical_tasks": True,
        "every_planned_answer_cell_accounted_for": True,
    }
    if not isinstance(report, dict) or any(
        report.get(key) != value for key, value in required.items()
    ):
        raise VerificationError("primary hosted-call ledger audit changed")
    upstream = report.get("upstream_invalid_cells")
    if not isinstance(upstream, dict) or upstream.get("by_condition") != {
        "predicted_fact": 6,
        "query_rewrite": 2,
    }:
        raise VerificationError("primary upstream-invalid accounting changed")
    if upstream.get("count") != 8:
        raise VerificationError("primary upstream-invalid count changed")

    answer_export = report.get("answer_export")
    if (
        not isinstance(answer_export, dict)
        or answer_export.get("row_count") != 6_600
        or answer_export.get("bytes") != PRIMARY_ANSWER_DETAILS.stat().st_size
        or answer_export.get("sha256") != sha256_file(PRIMARY_ANSWER_DETAILS)
    ):
        raise VerificationError("primary answer export identity changed")

    expected_model_conditions = {
        "claude/predicted_fact": 600,
        "claude/query_rewrite": 600,
        "claude/question_only": 600,
        "gemini/no_kb": 600,
        "gemini/oracle_passage": 600,
        "gemini/predicted_fact": 600,
        "gemini/query_rewrite": 600,
        "gemini/question_only": 600,
        "openai/predicted_fact": 600,
        "openai/query_rewrite": 600,
        "openai/question_only": 600,
    }
    export_upstream: set[tuple[str, str, str]] = set()
    export_rows = 0
    status_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    model_condition_counts: Counter[str] = Counter()
    with PRIMARY_ANSWER_DETAILS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            export_rows += 1
            row = json.loads(line)
            status_counts[str(row.get("status"))] += 1
            valid_counts[str(row.get("valid_json")).lower()] += 1
            model_condition_counts[
                f"{row.get('model')}/{row.get('condition')}"
            ] += 1
            if row.get("status") == "upstream_invalid":
                export_upstream.add(
                    (row.get("model"), row.get("question_id"), row.get("condition"))
                )
    if export_rows != 6_600 or len(export_upstream) != 8:
        raise VerificationError("primary answer export row accounting changed")
    if dict(status_counts) != {"succeeded": 6_592, "upstream_invalid": 8}:
        raise VerificationError("primary answer status accounting changed")
    if dict(valid_counts) != {"false": 28, "true": 6_572}:
        raise VerificationError("primary answer parse accounting changed")
    if dict(model_condition_counts) != expected_model_conditions:
        raise VerificationError("primary answer condition inventory changed")
    if report.get("answer_export_status") != dict(status_counts):
        raise VerificationError("primary answer status report changed")
    if report.get("answer_export_valid_json") != dict(valid_counts):
        raise VerificationError("primary answer parse report changed")
    if report.get("answer_rows_by_model_and_condition") != expected_model_conditions:
        raise VerificationError("primary answer condition report changed")

    reported_upstream_rows = upstream.get("keys")
    if not isinstance(reported_upstream_rows, list):
        raise VerificationError("primary upstream-invalid keys are omitted")
    reported_upstream = {
        (row.get("model"), row.get("item_id"), row.get("condition"))
        for row in reported_upstream_rows
        if isinstance(row, dict)
    }
    if reported_upstream != export_upstream:
        raise VerificationError("primary upstream-invalid item crosslink changed")

    hosted_archive = report.get("hosted_archive")
    if (
        not isinstance(hosted_archive, dict)
        or hosted_archive.get("bytes") != PRIMARY_HOSTED_CALL_ARCHIVE.stat().st_size
        or hosted_archive.get("sha256") != sha256_file(PRIMARY_HOSTED_CALL_ARCHIVE)
    ):
        raise VerificationError("primary hosted-call archive identity changed")
    return {
        "actual_calls_with_complete_raw_records": 10_192,
        "answer_provider_calls": 6_592,
        "planned_answer_cells": 6_600,
        "upstream_invalid_unissued_cells": 8,
        "status": "verified",
    }


def compare_bytes(observed: Path, expected: Path, label: str) -> None:
    if observed.read_bytes() != expected.read_bytes():
        raise VerificationError(f"{label} replay differs from frozen evidence")


def pretty_json_bytes(document: Any) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _subprocess_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    if env is not None:
        environment.update(env)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_subprocess_environment(env),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise VerificationError(f"command failed: {' '.join(command)}\n{detail}")



def release_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in EXCLUDED_NAMES
            and not EXCLUDED_DIRECTORIES.intersection(path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )

def _current_path_for(member: str) -> str:
    for recorded, current in RECORDED_PATH_PREFIXES:
        if member.startswith(recorded):
            return current + member[len(recorded):]
    return member


def _read_artifact_member(member: str) -> bytes:
    """Read one manifest-verified member without permitting path escape."""

    if not isinstance(member, str) or not safe_relative(member):
        raise VerificationError(f"unsafe artifact member path: {member!r}")
    root = ROOT.resolve()
    path = ROOT.joinpath(*PurePosixPath(_current_path_for(member)).parts)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise VerificationError(f"cannot read artifact member: {member}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise VerificationError(f"invalid artifact member: {member}")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read artifact member: {member}") from exc


def replay_delivery_stack_audit() -> str:
    """Replay selection-conditioned delivery under all five frozen stacks."""

    script_path = ROOT / "code/analysis/recompute_delivery_stack_audit.py"
    if not script_path.is_file():
        raise VerificationError("delivery-stack audit script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_delivery_stack_audit", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load delivery-stack audit script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(f"delivery-stack replay failed: {exc}") from exc

    expected = (ROOT / "data/evidence/delivery_stack_audit.json").read_bytes()
    if observed != expected:
        raise VerificationError("delivery-stack audit bytes changed")
    if (
        report.get("status") != "verified_no_call_replay"
        or report.get("api_calls") != 0
        or report.get("join_verification", {}).get(
            "correct_selection_denominators"
        )
        != {"claude": 456, "openai": 509, "gemini": 498}
    ):
        raise VerificationError("delivery-stack audit invariants changed")
    final = report.get("results", {}).get(
        "hybrid_rrf_bge_reranker_v2_m3", {}
    )
    expected_final = {
        "claude": (456, 452, 4),
        "openai": (509, 504, 5),
        "gemini": (498, 493, 5),
    }
    observed_final = {
        model: (
            values.get("correct_selection_denominator"),
            values.get("delivered_at_5"),
            values.get("selection_without_delivery"),
        )
        for model, values in final.get("models", {}).items()
    }
    if observed_final != expected_final:
        raise VerificationError("final-reranker delivery counts changed")
    return "delivery_stack_audit"


def replay_bm25_omission_taxonomy() -> str:
    """Replay the supported rank-and-identifier taxonomy of BM25 omissions."""

    script_path = ROOT / "code/analysis/recompute_bm25_omission_taxonomy.py"
    if not script_path.is_file():
        raise VerificationError("BM25 omission replay script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_bm25_omission_taxonomy", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load BM25 omission replay script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(f"BM25 omission replay failed: {exc}") from exc

    expected = (ROOT / "data/evidence/bm25_omission_taxonomy.json").read_bytes()
    if observed != expected:
        raise VerificationError("BM25 omission evidence bytes changed")
    if (
        report.get("status")
        != "verified_no_call_bm25_omission_rank_and_identifier_taxonomy"
        or report.get("api_calls") != 0
        or report.get("network_used") is not False
    ):
        raise VerificationError("BM25 omission replay status changed")

    expected_model_counts = {
        "claude": (599, 434, 69, 5, 91),
        "openai": (594, 443, 59, 5, 87),
        "gemini": (599, 449, 63, 4, 83),
    }
    observed_model_counts = {
        model: (
            values.get("corpus_resolvable_selected_ids"),
            values.get("top5_preserved"),
            values.get("mutually_exclusive_mechanism_categories", {})
            .get("cutoff_competition_selected_id_rank_6_to_20", {})
            .get("count"),
            values.get("mutually_exclusive_mechanism_categories", {})
            .get("query_document_nonmatch_exhaustive_result_set_below_20", {})
            .get("count"),
            values.get("mutually_exclusive_mechanism_categories", {})
            .get("full_top20_selected_id_absent_rank_or_nonmatch_unresolved", {})
            .get("count"),
        )
        for model, values in report.get("models", {}).items()
    }
    if observed_model_counts != expected_model_counts:
        raise VerificationError("BM25 omission model counts changed")
    if report.get("pooled_model_records") != {
        "corpus_resolvable_selected_ids": 1_792,
        "top5_preserved": 1_326,
        "top5_omitted": 466,
        "mechanism_categories": {
            "cutoff_competition_selected_id_rank_6_to_20": 191,
            "query_document_nonmatch_exhaustive_result_set_below_20": 14,
            "full_top20_selected_id_absent_rank_or_nonmatch_unresolved": 261,
        },
        "secondary_noncausal_identifier_title_categories": {
            "selected_id_rank_6_to_20": 191,
            "absent_top20_same_query_term_set_in_top5_identifier": 9,
            "absent_top20_all_query_terms_in_one_top5_identifier": 129,
            "absent_top20_partial_query_term_overlap_in_top5_identifiers": 96,
            "absent_top20_no_query_term_overlap_in_top5_identifiers": 41,
        },
        "partition_verified": True,
    }:
        raise VerificationError("BM25 omission pooled partition changed")
    boundary = report.get("definition_and_support_boundary", {})
    if (
        boundary.get("deterministic_mechanism_partition_supported") is not True
        or boundary.get("complete_causal_bm25_scoring_attribution_supported")
        is not False
    ):
        raise VerificationError("BM25 omission support boundary changed")
    return "bm25_omission_taxonomy"


def replay_mismatch_overlap_analysis() -> str:
    """Replay P-by-H cells and question overlap for differing identifiers."""

    script_path = ROOT / "code/analysis/recompute_mismatch_overlap.py"
    if not script_path.is_file():
        raise VerificationError("mismatch-overlap replay script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_mismatch_overlap", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load mismatch-overlap replay script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.identity.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(f"mismatch-overlap replay failed: {exc}") from exc

    expected = (ROOT / "data/evidence/mismatch_overlap_analysis.json").read_bytes()
    if observed != expected:
        raise VerificationError("mismatch-overlap evidence bytes changed")
    if (
        report.get("status")
        != "verified_no_call_mismatch_question_overlap_replay"
        or report.get("api_calls") != 0
        or report.get("network_used") is not False
    ):
        raise VerificationError("mismatch-overlap replay status changed")

    expected_counts = {
        "visible_label_bm25": {
            "cells": (329, 42, 210, 3, 74),
            "traced_unique": 19,
            "traced_multiplicity": (4, 4, 11),
            "preserved_unique": 17,
            "preserved_multiplicity": (3, 3, 11),
        },
        "visible_label_hybrid_reranker": {
            "cells": (329, 48, 278, 0, 3),
            "traced_unique": 22,
            "traced_multiplicity": (7, 4, 11),
            "preserved_unique": 22,
            "preserved_multiplicity": (7, 4, 11),
        },
    }
    observed_counts: dict[str, dict[str, Any]] = {}
    for stack, values in report.get("stacks", {}).items():
        cells = values.get("pooled_model_question_records", {}).get("p_x_h_at_5", {})
        traced = values.get("traced_hit_question_overlap", {})
        preserved = values.get("preserved_and_traced_hit_question_overlap", {})
        observed_counts[stack] = {
            "cells": (
                cells.get("records"),
                cells.get("selected_id_preserved_traced_passage_hit"),
                cells.get("selected_id_preserved_traced_passage_miss"),
                cells.get("selected_id_not_preserved_traced_passage_hit"),
                cells.get("selected_id_not_preserved_traced_passage_miss"),
            ),
            "traced_unique": traced.get("unique_question_ids"),
            "traced_multiplicity": tuple(
                traced.get("model_family_multiplicity", {}).get(f"exactly_{value}")
                for value in (1, 2, 3)
            ),
            "preserved_unique": preserved.get("unique_question_ids"),
            "preserved_multiplicity": tuple(
                preserved.get("model_family_multiplicity", {}).get(
                    f"exactly_{value}"
                )
                for value in (1, 2, 3)
            ),
        }
    if observed_counts != expected_counts:
        raise VerificationError("mismatch-overlap cells or multiplicities changed")
    return "mismatch_overlap_analysis"


def replay_mismatch_stack_cutoffs() -> str:
    """Replay all-stack mismatch P-by-H cells and protocol-binding boundaries."""

    script_path = ROOT / "code/analysis/recompute_mismatch_stack_cutoffs.py"
    if not script_path.is_file():
        raise VerificationError("all-stack mismatch replay script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_mismatch_stack_cutoffs", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load all-stack mismatch replay script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.identity.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(
            f"all-stack mismatch replay failed: {exc}"
        ) from exc

    expected = (ROOT / "data/evidence/mismatch_stack_cutoffs.json").read_bytes()
    if observed != expected:
        raise VerificationError("all-stack mismatch evidence bytes changed")
    if (
        report.get("status")
        != "verified_no_call_all_stack_mismatch_p_h_replay"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("retrieval_calls") != 0
        or report.get("network_used") is not False
        or report.get("credentials_used") is not False
        or report.get("random_used") is not False
        or report.get("clock_used") is not False
    ):
        raise VerificationError("all-stack mismatch replay status changed")

    canonical = report.get("canonical_identity_verification", {})
    if (
        canonical.get("raw_identifier_differences") != 331
        or canonical.get("raw_differences_with_resolvable_selected_id") != 329
        or canonical.get("raw_differences_with_unresolvable_selected_id") != 2
        or canonical.get(
            "gold_unresolvable_among_selected_resolvable_differences"
        )
        != 0
        or canonical.get("canonical_object_distinct_records") != 329
        or canonical.get("canonical_object_distinct_records_by_model")
        != {"claude": 143, "openai": 85, "gemini": 101}
        or canonical.get("raw_differences_collapsing_to_same_canonical_pid") != 0
        or canonical.get("canonical_pair_mapping_sha256")
        != "ca2c78d3a92fb23dc7974cf43565c675c4a2094495569cb4e508c120c24ee496"
        or canonical.get("all_returned_ids_at_audited_cutoffs_resolved") is not True
        or canonical.get("returned_id_occurrences_checked_through_max_cutoff")
        != 32_675
    ):
        raise VerificationError("all-stack canonical identity verification changed")

    dependence = report.get("denominators_and_dependence", {})
    if (
        dependence.get("paired_model_question_records") != 329
        or dependence.get("unique_question_clusters") != 160
        or dependence.get("records_by_model")
        != {"claude": 143, "openai": 85, "gemini": 101}
        or dependence.get("question_model_multiplicity")
        != {"exactly_1": 55, "exactly_2": 41, "exactly_3": 64}
    ):
        raise VerificationError("all-stack mismatch dependence metadata changed")

    stacks = report.get("stacks", {})
    if tuple(stacks) != module.STACKS:
        raise VerificationError("all-stack mismatch executor roster changed")
    for stack in module.STACKS:
        cutoffs = stacks.get(stack, {}).get("cutoffs", {})
        if set(cutoffs) != {"1", "5", "20"}:
            raise VerificationError(
                f"all-stack mismatch cutoffs changed for {stack}"
            )
        for cutoff in module.CUTOFFS:
            section = cutoffs[str(cutoff)]
            pooled = module._cell_tuple(
                section["pooled_model_question_records"]
            )
            if pooled != module.EXPECTED_CELLS[stack][cutoff]["pooled"]:
                raise VerificationError(
                    f"all-stack mismatch pooled cells changed for {stack}@{cutoff}"
                )
            for model in module.identity.MODELS:
                per_model = module._cell_tuple(section["per_model"][model])
                if per_model != module.EXPECTED_CELLS[stack][cutoff][model]:
                    raise VerificationError(
                        "all-stack mismatch model cells changed for "
                        f"{stack}@{cutoff}/{model}"
                    )

    ordering = report.get("executor_order_and_pairwise_comparisons", {})
    expected_pairs = [
        (left, right)
        for index, left in enumerate(module.STACKS)
        for right in module.STACKS[index + 1 :]
    ]
    for cutoff in module.CUTOFFS:
        section = ordering.get("cutoffs", {}).get(str(cutoff), {})
        comparisons = section.get("pairwise_comparisons", [])
        observed_pairs = [
            (comparison.get("left_stack"), comparison.get("right_stack"))
            for comparison in comparisons
        ]
        if observed_pairs != expected_pairs:
            raise VerificationError(
                f"all-stack mismatch pair grid changed at cutoff {cutoff}"
            )
        summary = section.get("pairwise_sign_summary", {})
        expected = module.EXPECTED_PAIRWISE_SUMMARY[cutoff]
        for field in (
            "sign_agreement",
            "sign_disagreement",
            "tie_involved",
            "both_metrics_tied",
            "preservation_tie_only",
            "trace_delivery_tie_only",
        ):
            if summary.get(field) != expected[field]:
                raise VerificationError(
                    "all-stack mismatch pairwise sign summary changed at "
                    f"cutoff {cutoff}: {field}"
                )
        headline = section.get("headline_bm25_to_final_reranker", {})
        if (
            headline.get("preservation_gap_count")
            != expected["headline_preservation_gap_count"]
            or headline.get("trace_delivery_gap_count")
            != expected["headline_trace_delivery_gap_count"]
        ):
            raise VerificationError(
                f"all-stack mismatch headline gap changed at cutoff {cutoff}"
            )
    if ordering.get("all_cutoffs_pairwise_sign_summary") != {
        "comparisons": 30,
        "sign_agreement": 13,
        "sign_disagreement": 14,
        "tie_involved": 3,
        "both_metrics_tied": 1,
        "preservation_tie_only": 0,
        "trace_delivery_tie_only": 2,
    }:
        raise VerificationError(
            "all-stack mismatch aggregate pairwise summary changed"
        )

    protocol_binding = report.get("protocol_binding", {})
    result_scope = protocol_binding.get("result_scope", {})
    bridge_binding = protocol_binding.get("verified_inputs", {}).get(
        "bridge_protocol", {}
    )
    if (
        result_scope.get("bridge_primary_cutoff_declared") != 5
        or result_scope.get("bridge_report_cutoffs_declared") != [1, 5, 20]
        or result_scope.get("sensitivity_report_cutoffs_declared") != [1, 5, 20]
        or result_scope.get("rankings_precede_this_join_by_data_dependency")
        is not True
        or result_scope.get("selected_link_semantics_declared_before_selector_calls")
        is not True
        or result_scope.get(
            "selected_link_declared_as_single_exact_shown_wikipedia_key"
        )
        is not True
        or result_scope.get(
            "selected_link_declared_as_page_whose_passage_is_needed"
        )
        is not True
        or result_scope.get(
            "decoded_selected_link_declared_as_primary_predicted_query"
        )
        is not True
        or result_scope.get(
            "selected_link_top_k_obligation_in_frozen_protocol"
        )
        is not False
        or result_scope.get("mismatch_p_h_analysis_in_frozen_protocol")
        is not False
        or result_scope.get("profile_application")
        != "retrospective_application_of_prospective_profile"
        or bridge_binding.get("selector_implementation_member")
        != "code/src/ccr/retrieval/hybridqa_bridge.py"
        or bridge_binding.get("selector_implementation_sha256")
        != "5e0f0fa578a741e29e8f91dd5f683ba6ba1afa96f8f3beaf230ec0729e4c879b"
        or "independently authenticated wall-clock timestamp"
        not in protocol_binding.get("temporal_evidence_boundary", "")
    ):
        raise VerificationError("all-stack mismatch protocol-binding boundary changed")
    return "mismatch_stack_cutoffs"


def replay_lossy_query_target_divergence() -> str:
    """Replay P/H cells under generated, rewritten, and question queries."""

    script_path = ROOT / "code/analysis/recompute_lossy_query_target_divergence.py"
    if not script_path.is_file():
        raise VerificationError("lossy-query target-divergence script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_lossy_query_target_divergence", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load lossy-query target-divergence script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(
            _read_artifact_member,
            script_path=script_path,
        )
        observed = module.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(
            f"lossy-query target-divergence replay failed: {exc}"
        ) from exc

    expected = (
        ROOT / "data/evidence/lossy_query_target_divergence.json"
    ).read_bytes()
    if observed != expected:
        raise VerificationError("lossy-query target-divergence evidence bytes changed")
    if (
        report.get("status")
        != "verified_zero_call_lossy_query_target_divergence"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("retrieval_calls") != 0
        or report.get("network_used") is not False
        or report.get("credentials_used") is not False
        or report.get("random_used") is not False
        or report.get("clock_used") is not False
    ):
        raise VerificationError("lossy-query target-divergence scope changed")

    population = report.get("population_verification", {})
    if (
        population.get("canonical_object_distinct_records") != 329
        or population.get("records_by_model")
        != {"claude": 143, "openai": 85, "gemini": 101}
        or population.get("unique_question_clusters") != 160
        or population.get("row_level_target_and_cell_digest_sha256")
        != "c5e869a4ec0538ec3e74e23e3fec7ae928beaa8e8d4bb9e89e3cb86e6789f7bf"
        or population.get("all_returned_ids_resolve") is not True
    ):
        raise VerificationError("lossy-query target population changed")

    forms = report.get("query_forms", {})
    if tuple(forms) != module.QUERY_FORMS:
        raise VerificationError("lossy-query form roster changed")
    for form in module.QUERY_FORMS:
        for cutoff in module.CUTOFFS:
            section = forms.get(form, {}).get("cutoffs", {}).get(str(cutoff), {})
            if module._cell_tuple(
                section.get("pooled_model_question_records", {})
            ) != module.EXPECTED_CELLS[form][cutoff]["pooled"]:
                raise VerificationError(
                    f"lossy-query pooled cells changed for {form}@{cutoff}"
                )
            for model in module.MODELS:
                if module._cell_tuple(
                    section.get("per_model", {}).get(model, {})
                ) != module.EXPECTED_CELLS[form][cutoff][model]:
                    raise VerificationError(
                        f"lossy-query model cells changed for {form}@{cutoff}/{model}"
                    )

    rewrite = forms.get("direct_query_rewrite_bm25", {})
    rewrite_semantics = rewrite.get("query_semantics", {})
    question_semantics = forms.get("question_only_bm25", {}).get(
        "query_semantics", {}
    )
    fact_semantics = forms.get("model_generated_retrieval_fact_bm25", {}).get(
        "query_semantics", {}
    )
    if (
        rewrite_semantics.get("selected_id_is_query_input") is not False
        or rewrite_semantics.get("intent_to_treat_status_counts")
        != {"succeeded": 328, "upstream_invalid": 1}
        or module._cell_tuple(rewrite_semantics.get("complete_case_at_5", {}))
        != (328, 48, 140, 35, 105)
        or question_semantics.get("selected_id_is_query_input") is not False
        or question_semantics.get("queries_mapping_to_multiple_selected_objects")
        != 40
        or question_semantics.get("records_in_those_many_to_one_queries") != 105
        or fact_semantics.get("selected_id_is_query_input") is not False
        or fact_semantics.get("c0_equivalence_to_decoded_title", {})
        .get("pooled", {})
        .get("executed_term_query_equals_decoded_title_terms")
        != 291
    ):
        raise VerificationError("lossy-query construction audit changed")
    limitations = report.get("limitations", {})
    if (
        limitations.get("retrospective") is not True
        or limitations.get("no_cross_corpus_or_deployment_claim") is not True
        or "not whether a downstream component violated"
        not in limitations.get("query_rewrite_is_not_a_handoff_conformance_test", "")
    ):
        raise VerificationError("lossy-query limitation boundary changed")
    return "lossy_query_target_divergence"


def replay_full_cohort_stack_decisions() -> str:
    """Replay P-versus-H stack decisions on all planned hosted rows."""

    script_path = ROOT / "code/analysis/recompute_full_cohort_stack_decisions.py"
    if not script_path.is_file():
        raise VerificationError("full-cohort stack-decision script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_full_cohort_stack_decisions", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load full-cohort stack-decision script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.identity.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(
            f"full-cohort stack-decision replay failed: {exc}"
        ) from exc

    expected = (ROOT / "data/evidence/full_cohort_stack_decisions.json").read_bytes()
    if observed != expected:
        raise VerificationError("full-cohort stack-decision evidence bytes changed")
    if (
        report.get("status")
        != "verified_no_call_full_cohort_p_h_stack_decision_replay"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("retrieval_calls") != 0
        or report.get("network_used") is not False
        or report.get("credentials_used") is not False
        or report.get("random_used") is not False
        or report.get("clock_used") is not False
    ):
        raise VerificationError("full-cohort stack-decision replay status changed")

    accounting = report.get("cohort_accounting", {})
    if (
        accounting.get("planned_records") != 1_800
        or accounting.get("unique_questions") != 600
        or accounting.get("records_by_model")
        != {"claude": 600, "openai": 600, "gemini": 600}
        or accounting.get("selector_valid_records") != 1_794
        or accounting.get("selector_invalid_records") != 6
        or accounting.get("selector_invalid_by_model") != {"openai": 6}
        or accounting.get("selected_id_resolvable_records") != 1_792
        or accounting.get("selected_id_unresolvable_valid_records") != 2
        or accounting.get("selected_id_unresolvable_valid_by_model")
        != {"claude": 1, "gemini": 1}
        or accounting.get("canonical_selected_traced_match_records") != 1_463
        or accounting.get("canonical_selected_traced_mismatch_records") != 329
        or accounting.get("all_traced_ids_resolve") is not True
        or accounting.get("all_returned_ids_through_rank_20_resolve") is not True
        or accounting.get("returned_id_occurrences_checked_through_rank_20")
        != 176_997
        or accounting.get(
            "invalid_rows_are_p_and_h_misses_for_every_stack_cutoff"
        )
        is not True
    ):
        raise VerificationError("full-cohort population accounting changed")

    population = report.get("full_cohort_all_planned", {})
    if population.get("aggregate_weightings_identical") is not True:
        raise VerificationError("balanced full-cohort weightings no longer coincide")
    expected_summary = {
        "cutoffs": 3,
        "stack_selection_decision_changes": 0,
        "complete_ordering_changes": 0,
        "opposite_pairwise_preferences": 0,
    }
    for weighting in (
        "pooled_record_weighted",
        "model_equal_weighted",
        "question_equal_weighted",
    ):
        section = population.get(weighting, {})
        if section.get("summary") != expected_summary:
            raise VerificationError(
                f"full-cohort decision summary changed for {weighting}"
            )
        for cutoff, winner in module.EXPECTED_PRIMARY_WINNERS.items():
            decision = section.get("cutoffs", {}).get(str(cutoff), {})
            if (
                decision.get("preservation_maximizing_stacks") != winner
                or decision.get("trace_delivery_maximizing_stacks") != winner
                or decision.get("stack_selection_decision_changes") is not False
                or decision.get("complete_ordering_changes") is not False
                or decision.get("pairwise_preference_summary", {}).get(
                    "opposite_preference"
                )
                != 0
            ):
                raise VerificationError(
                    f"full-cohort decision changed for {weighting}@{cutoff}"
                )
    for model in module.identity.MODELS:
        if (
            population.get("per_model_family", {})
            .get(model, {})
            .get("summary")
            != expected_summary
        ):
            raise VerificationError(
                f"full-cohort per-model decision changed for {model}"
            )

    mismatch_reference = report.get("mismatch_only_reference", {}).get(
        "pooled_record_weighted", {}
    )
    if mismatch_reference.get("summary") != {
        "cutoffs": 3,
        "stack_selection_decision_changes": 1,
        "complete_ordering_changes": 3,
        "opposite_pairwise_preferences": 14,
    }:
        raise VerificationError("mismatch-only decision reference changed")
    mismatch_at_one = mismatch_reference.get("cutoffs", {}).get("1", {})
    if (
        mismatch_at_one.get("preservation_maximizing_stacks")
        != ["visible_label_colbert_v2", "visible_label_hybrid_reranker"]
        or mismatch_at_one.get("trace_delivery_maximizing_stacks")
        != ["visible_label_bm25"]
    ):
        raise VerificationError("mismatch-only rank-one decision changed")
    return "full_cohort_stack_decisions"


def replay_mismatch_ordering_robustness() -> str:
    """Replay complete-grid robustness across models and question weights."""

    script_path = ROOT / "code/analysis/recompute_mismatch_ordering_robustness.py"
    if not script_path.is_file():
        raise VerificationError("mismatch ordering robustness script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_mismatch_ordering_robustness", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load mismatch ordering robustness script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.identity.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(
            f"mismatch ordering robustness replay failed: {exc}"
        ) from exc

    expected = (ROOT / "data/evidence/mismatch_ordering_robustness.json").read_bytes()
    if observed != expected:
        raise VerificationError("mismatch ordering robustness evidence bytes changed")
    if (
        report.get("status") != "verified_no_call_mismatch_ordering_robustness"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("retrieval_calls") != 0
        or report.get("network_used") is not False
        or report.get("credentials_used") is not False
        or report.get("random_used") is not False
        or report.get("clock_used") is not False
    ):
        raise VerificationError("mismatch ordering robustness status changed")

    expected_summary = {
        "sign_agreement": 13,
        "sign_disagreement": 14,
        "trace_delivery_tie_only": 3,
    }
    if (
        report.get("question_equal_weighted", {}).get(
            "all_cutoffs_pairwise_sign_summary"
        )
        != expected_summary
        or report.get("model_equal_weighted", {}).get(
            "all_cutoffs_pairwise_sign_summary"
        )
        != expected_summary
    ):
        raise VerificationError("weighted mismatch ordering summary changed")
    per_model = report.get("per_model_family", {})
    if {
        model: section.get("all_cutoffs_pairwise_sign_summary", {}).get(
            "sign_disagreement"
        )
        for model, section in per_model.items()
    } != {"claude": 15, "openai": 9, "gemini": 11}:
        raise VerificationError("per-model mismatch ordering summary changed")
    common = report.get("common_question_intersection", {})
    if common.get("unique_questions") != 64 or {
        model: section.get("all_cutoffs_pairwise_sign_summary", {}).get(
            "sign_disagreement"
        )
        for model, section in common.get("per_model_family", {}).items()
    } != {"claude": 12, "openai": 9, "gemini": 12}:
        raise VerificationError("common-question mismatch ordering summary changed")
    if "binomial interval" not in report.get("uncertainty_boundary", ""):
        raise VerificationError("mismatch ordering uncertainty boundary changed")
    return "mismatch_ordering_robustness"


def replay_mismatch_preservation_gap_uncertainty() -> str:
    """Replay the paired mismatch gap with question-cluster resampling."""

    script_path = (
        ROOT / "code/analysis/recompute_mismatch_preservation_gap_uncertainty.py"
    )
    if not script_path.is_file():
        raise VerificationError(
            "mismatch preservation-gap uncertainty script is missing"
        )
    spec = importlib.util.spec_from_file_location(
        "_ccr_mismatch_preservation_gap_uncertainty", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError(
            "cannot load mismatch preservation-gap uncertainty script"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.identity.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(
            f"mismatch preservation-gap uncertainty replay failed: {exc}"
        ) from exc

    expected = (
        ROOT / "data/evidence/mismatch_preservation_gap_uncertainty.json"
    ).read_bytes()
    if observed != expected:
        raise VerificationError(
            "mismatch preservation-gap uncertainty evidence bytes changed"
        )
    if (
        report.get("status")
        != "verified_no_call_mismatch_preservation_gap_uncertainty"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("retrieval_calls") != 0
        or report.get("network_used") is not False
        or report.get("credentials_used") is not False
        or report.get("clock_used") is not False
    ):
        raise VerificationError(
            "mismatch preservation-gap uncertainty replay status changed"
        )

    dependence = report.get("denominators_and_dependence", {})
    if (
        dependence.get("paired_model_question_records") != 329
        or dependence.get("unique_question_clusters") != 160
        or dependence.get("records_by_model")
        != {"claude": 143, "openai": 85, "gemini": 101}
        or dependence.get("question_model_multiplicity")
        != {"exactly_1": 55, "exactly_2": 41, "exactly_3": 64}
    ):
        raise VerificationError(
            "mismatch preservation-gap denominators changed"
        )

    record = report.get("paired_record_weighted", {})
    outcomes = record.get("paired_outcomes", {})
    if (
        record.get("body_only_bm25_preservation", {}).get("numerator") != 252
        or record.get("rrf_rerank_preservation", {}).get("numerator") != 326
        or abs(record.get("paired_gap_percentage_points", 0.0) - 100 * 74 / 329)
        > 1e-12
        or outcomes
        != {
            "rrf_rerank_only_preserves": 74,
            "body_only_bm25_only_preserves": 0,
            "same_preservation_state": 255,
        }
    ):
        raise VerificationError(
            "mismatch preservation-gap paired counts changed"
        )

    question = report.get("question_equal_weighted_macro", {})
    resampling = report.get("question_cluster_resampling", {})
    if (
        question.get("unique_questions") != 160
        or abs(question.get("paired_gap_percentage_points", 0.0) - 23.854166666666664)
        > 1e-12
        or question.get("question_gap_direction")
        != {"positive": 43, "zero": 117, "negative": 0}
        or resampling.get("samples") != 10_000
        or resampling.get("derived_seed") != 16_771_578_087_979_229_916
        or resampling.get("question_equal_weighted", {})
        .get("paired_preservation_gap", {})
        .get("paper_rounded_one_decimal")
        != [17.8, 30.4]
        or resampling.get("complementary_record_weighted_gap", {}).get(
            "paper_rounded_one_decimal"
        )
        != [16.3, 29.3]
    ):
        raise VerificationError(
            "mismatch preservation-gap question-cluster results changed"
        )
    return "mismatch_preservation_gap_uncertainty"


def replay_mismatch_metric_inversion_uncertainty() -> str:
    """Replay question-cluster stability for the retrospective P/H inversion."""

    script_path = (
        ROOT / "code/analysis/recompute_mismatch_metric_inversion_uncertainty.py"
    )
    if not script_path.is_file():
        raise VerificationError(
            "mismatch metric-inversion uncertainty script is missing"
        )
    spec = importlib.util.spec_from_file_location(
        "_ccr_mismatch_metric_inversion_uncertainty", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError(
            "cannot load mismatch metric-inversion uncertainty script"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.identity.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(
            f"mismatch metric-inversion uncertainty replay failed: {exc}"
        ) from exc

    expected = (
        ROOT / "data/evidence/mismatch_metric_inversion_uncertainty.json"
    ).read_bytes()
    if observed != expected:
        raise VerificationError(
            "mismatch metric-inversion uncertainty evidence bytes changed"
        )
    if (
        report.get("status")
        != "verified_no_call_mismatch_metric_inversion_uncertainty"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("retrieval_calls") != 0
        or report.get("network_used") is not False
        or report.get("credentials_used") is not False
        or report.get("clock_used") is not False
        or report.get("random_used") is not True
        or "fixed SHA-256-derived seeds" not in report.get("randomness", "")
        or "no system entropy" not in report.get("randomness", "")
    ):
        raise VerificationError(
            "mismatch metric-inversion uncertainty replay status changed"
        )

    dependence = report.get("denominators_and_dependence", {})
    if (
        dependence.get("model_question_records") != 329
        or dependence.get("unique_question_clusters") != 160
        or dependence.get("records_by_model")
        != {"claude": 143, "openai": 85, "gemini": 101}
        or dependence.get("question_model_multiplicity")
        != {"exactly_1": 55, "exactly_2": 41, "exactly_3": 64}
        or dependence.get("records_from_questions_with_multiple_model_rows")
        != {"numerator": 274, "denominator": 329, "rate": 274 / 329}
    ):
        raise VerificationError(
            "mismatch metric-inversion uncertainty denominators changed"
        )

    contrast = report.get("bm25_to_bge_m3_contrast", {})
    equal_question = contrast.get("equal_question", {})
    exact_contrast = {
        "k5_P": (199, 960, [14.5, 27.2]),
        "k5_H": (-1, 32, [-7.5, 0.8]),
        "k20_P": (131, 960, [8.8, 19.0]),
        "k20_H": (-17, 480, [-7.7, 0.4]),
    }
    for key, (numerator, denominator, interval) in exact_contrast.items():
        value = equal_question.get(key, {})
        if (
            value.get("point", {}).get("numerator") != numerator
            or value.get("point", {}).get("denominator") != denominator
            or value.get("paper_rounded_one_decimal") != interval
        ):
            raise VerificationError(
                "mismatch metric-inversion contrast result changed"
            )
    if (
        contrast.get("samples") != 10_000
        or contrast.get("derived_seed") != 16_709_038_045_988_039_055
        or contrast.get("draw_projection_sha256")
        != "e43ac836b42baa6b7b81b218a273a32f21976486796ff912cc153afbc2eb9551"
        or contrast.get("opposite_direction_draws", {}).get("5")
        != {"numerator": 9267, "denominator": 10_000, "rate": 0.9267}
        or contrast.get("opposite_direction_draws", {}).get("20")
        != {"numerator": 9547, "denominator": 10_000, "rate": 0.9547}
    ):
        raise VerificationError(
            "mismatch metric-inversion contrast stability changed"
        )

    grid = report.get("joint_grid_reversal_stability", {})
    stability = grid.get("reversal_count_stability", {})
    original_cells = grid.get("original_reversal_cell_stability", {})
    if (
        grid.get("samples") != 10_000
        or grid.get("derived_seed") != 14_507_245_148_007_068_347
        or grid.get("draw_projection_sha256")
        != "f14a9f6aca5b16b6df77298e9070ece2c6e693c330740bb0d9adcc48552c4f39"
        or grid.get("point_grid")
        != {"same_direction": 13, "opposite_direction": 14, "tie_involved": 3}
        or stability.get("point") != 14
        or stability.get("percentile_95_stability_interval") != [7.0, 16.0]
        or stability.get("draws_equal_to_point", {}).get("numerator") != 1897
        or stability.get("draws_at_least_point", {}).get("numerator") != 3729
        or sum(stability.get("frequency", {}).values()) != 10_000
        or len(original_cells.get("cells", [])) != 14
        or original_cells.get("minimum_retention_rate") != 0.632
        or original_cells.get("median_retention_rate") != 0.81065
        or original_cells.get("maximum_retention_rate") != 0.9546
    ):
        raise VerificationError(
            "mismatch metric-inversion grid stability changed"
        )

    boundary = " ".join(report.get("interpretation_boundaries", []))
    for required in (
        "post-selection stability intervals",
        "binomial interval",
        "paired permutation p-value",
        "deployment prevalence",
    ):
        if required not in boundary:
            raise VerificationError(
                "mismatch metric-inversion interpretation boundary changed"
            )
    return "mismatch_metric_inversion_uncertainty"


def replay_answer_associations() -> str:
    """Replay the paper's three equal-five-passage answer associations."""

    script_path = ROOT / "code/analysis/recompute_answer_associations.py"
    if not script_path.is_file():
        raise VerificationError("answer-association replay script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_answer_associations", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load answer-association replay script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.identity.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(f"answer-association replay failed: {exc}") from exc

    expected = (ROOT / "data/evidence/answer_associations.json").read_bytes()
    if observed != expected:
        raise VerificationError("answer-association evidence bytes changed")
    if (
        report.get("status")
        != "verified_no_call_answer_associations"
        or report.get("api_calls") != 0
        or report.get("network_used") is not False
    ):
        raise VerificationError("answer-association replay status changed")
    joins = report.get("join_verification", {})
    if (
        joins.get("bridge_records") != 1_800
        or joins.get("answer_records") != 6_600
        or joins.get("selected_id_equals_traced_id_records") != 1_463
        or joins.get("same_question_table_model_reader_join") is not True
    ):
        raise VerificationError("answer-association join coverage changed")
    contract = report.get("prompt_and_budget_contract", {})
    if (
        contract.get("evidence_passages_per_strict_arm") != 5
        or contract.get("total_input_tokens_matched") is not False
    ):
        raise VerificationError("answer-association context contract changed")
    expected_paper = {
        "decoded_id_bm25__vs__question": (832, 79.2, 50.6, [24.5, 32.8]),
        "decoded_id_bm25__vs__rewrite": (200, 76.0, 45.0, [23.5, 38.9]),
        "rewrite__vs__question": (903, 81.9, 52.6, [25.4, 33.4]),
    }
    observed_paper = {}
    for name, values in report.get("pairwise_matched_five", {}).items():
        pooled = values.get("pooled_delivery_aligned", {})
        observed_paper[name] = (
            pooled.get("records"),
            round(100 * pooled.get("hit_context_em", {}).get("rate", -1), 1),
            round(100 * pooled.get("miss_context_em", {}).get("rate", -1), 1),
            pooled.get("question_cluster_resampling", {})
            .get("pooled_record_weighted", {})
            .get("paper_rounded_one_decimal"),
        )
    if observed_paper != expected_paper:
        raise VerificationError("paper answer-association values changed")
    return "answer_associations"


def verify_identifier_aware_executor_report_binding() -> str:
    """Bind the DB-backed executor report without claiming a DB replay.

    The 418 MB source SQLite database is deliberately not redistributed.  This
    check independently replays the embedded identifier/cohort projections,
    binds the omitted database identity and executor specification, and checks
    every frozen count target.  A fresh title/body FTS execution requires the
    separately hash-locked database and the script's explicit ``--source-db``.
    """

    script_path = ROOT / "code/analysis/recompute_identifier_aware_executors.py"
    report_path = ROOT / "data/evidence/identifier_aware_executor_audit.json"
    if not script_path.is_file() or not report_path.is_file():
        raise VerificationError("identifier-aware script or report is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_identifier_aware_executors", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load identifier-aware audit script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = load_json(report_path)
        payloads = {
            member: _read_artifact_member(member)
            for member in module.EXPECTED_MEMBERS
        }
        member_metadata = {
            member: module._validate_member(
                payloads[member], member, module.EXPECTED_MEMBERS[member]
            )
            for member in module.EXPECTED_MEMBERS
        }
        links, titles, link_to_pid, _, inventory = module._load_inventory(
            payloads[module.LINKS_MEMBER]
        )
        _, cohort = module._load_primary_rows(
            payloads[module.BRIDGE_MEMBER], link_to_pid
        )
    except Exception as exc:
        raise VerificationError(
            f"identifier-aware report binding failed: {exc}"
        ) from exc

    if (
        report.get("status")
        != "verified_zero_call_identifier_aware_executor_audit"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("answer_calls") != 0
        or report.get("network_used") is not False
    ):
        raise VerificationError("identifier-aware report status changed")
    if report.get("inputs", {}).get("artifact_members") != member_metadata:
        raise VerificationError("identifier-aware artifact-member binding changed")
    if report.get("inventory") != inventory or report.get("primary_cohort") != cohort:
        raise VerificationError("identifier-aware embedded projection changed")
    if len(links) != len(titles) or len(links) != 286_270:
        raise VerificationError("identifier-aware inventory count changed")

    database = report.get("inputs", {}).get("source_bm25_database", {})
    for key, value in module.EXPECTED_SOURCE_DB.items():
        if database.get(key) != value:
            raise VerificationError(
                f"identifier-aware source-database binding changed: {key}"
            )
    if database.get("body_stream_sha256") != module.EXPECTED_BODY_STREAM_SHA256:
        raise VerificationError("identifier-aware body-stream binding changed")
    replay = report.get("replay", {})
    if (
        replay.get("script_sha256") != sha256_file(script_path)
        or replay.get("scratch_database_is_scientific_input") is not False
        or replay.get("scratch_database_retained") is not False
        or replay.get("outcome_targets_enforced") is not True
        or replay.get("corpus_title_only_target_enforced") is not True
    ):
        raise VerificationError("identifier-aware replay declaration changed")

    specifications = report.get("executor_specifications", {})
    exact_spec = specifications.get("exact_decoded_equality", {})
    title_spec = specifications.get("title_only_bm25", {})
    combined_spec = specifications.get("uniform_title_body_bm25", {})
    if (
        exact_spec.get("query") != "exact case-sensitive Unicode string equality"
        or exact_spec.get("ties")
        != "impossible in the verified frozen inventory"
        or title_spec.get("index")
        != "ephemeral SQLite FTS5 over decoded title only"
        or title_spec.get("bm25_column_weights") != [1.0]
        or combined_spec.get("bm25_column_weights")
        != {"title": 1.0, "body": 1.0}
        or combined_spec.get("weight_selection")
        != "fixed uniform weights; no outcome tuning"
    ):
        raise VerificationError("identifier-aware executor specification changed")

    executors = report.get("primary_selected_id_preservation", {})
    for model, by_executor in module.EXPECTED_PRIMARY_PRESERVATION.items():
        for executor, expected in by_executor.items():
            record = executors.get(executor, {}).get("per_model", {}).get(model, {})
            observed = (
                record.get("resolvable_selected_ids"),
                *(
                    record.get("selected_id_preservation_on_resolvable", {})
                    .get(str(cutoff), {})
                    .get("numerator")
                    for cutoff in module.CUTOFFS
                ),
            )
            if observed != expected:
                raise VerificationError(
                    f"identifier-aware preservation counts changed: {model}/{executor}"
                )
    for executor, expected in module.EXPECTED_POOLED_CH5.items():
        pooled = executors.get(executor, {}).get("pooled_descriptive", {})
        cells = pooled.get("selection_delivery_joint_all_planned", {}).get("5", {})
        observed = tuple(
            cells.get(key) for key in ("c1_h1", "c1_h0", "c0_h1", "c0_h0")
        )
        if observed != expected:
            raise VerificationError(
                f"identifier-aware joint counts changed: {executor}"
            )
        if (
            executors.get(executor, {}).get("ranking_identity_sha256")
            != module.EXPECTED_PRIMARY_RANKING_SHA256[executor]
        ):
            raise VerificationError(
                f"identifier-aware ranking binding changed: {executor}"
            )

    corpus = report.get("corpus_wide_round_trip", {})
    exact = corpus.get("exact_decoded_equality", {})
    title = corpus.get("title_only_bm25", {})
    observed_corpus = (
        title.get("queries"),
        title.get("nonempty_lexical_queries"),
        title.get("preservation", {}).get("1", {}).get("numerator"),
        title.get("preservation", {}).get("5", {}).get("numerator"),
        title.get("preservation", {}).get("20", {}).get("numerator"),
    )
    if (
        exact.get("queries") != 286_270
        or any(
            value.get("numerator") != 286_270
            for value in exact.get("preservation", {}).values()
        )
        or observed_corpus != module.EXPECTED_CORPUS_TITLE_ONLY
        or title.get("ranking_identity_sha256")
        != module.EXPECTED_CORPUS_TITLE_RANKING_SHA256
    ):
        raise VerificationError("identifier-aware corpus counts changed")
    return "identifier_aware_executor_report_binding"


def verify_hybridqa_bm25_reconstruction_contract() -> str:
    """Bind the public-source reconstruction witness without rebuilding 418 MB."""

    script_path = ROOT / "code/analysis/reconstruct_hybridqa_bm25.py"
    evidence_path = ROOT / "data/evidence/hybridqa_bm25_reconstruction.json"
    if not script_path.is_file() or not evidence_path.is_file():
        raise VerificationError("BM25 reconstruction script or evidence is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_hybridqa_bm25_reconstruction", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load BM25 reconstruction script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = load_json(evidence_path)
        if not isinstance(report, dict):
            raise VerificationError("BM25 reconstruction evidence schema changed")
        expected_report = module.reconstruction_report(
            report.get("reconstructed_output"), script_path
        )
        if report != expected_report:
            raise VerificationError("BM25 reconstruction evidence contract changed")
        if evidence_path.read_bytes() != module.pretty_bytes(expected_report):
            raise VerificationError("BM25 reconstruction evidence bytes changed")
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(
            f"BM25 reconstruction contract binding failed: {exc}"
        ) from exc

    source = report.get("source_contract", {})
    frozen = report.get("frozen_output", {})
    reconstructed = report.get("reconstructed_output", {})
    known_good_runtime = report.get("known_good_rebuild_runtime")
    runtime_acceptance = report.get("rebuild_runtime_acceptance")
    if (
        report.get("status")
        != "verified_exact_public_source_reconstruction"
        or report.get("schema_version") != 2
        or report.get("path_free") is not True
        or report.get("model_calls") != 0
        or report.get("api_calls") != 0
        or report.get("byte_identical_to_frozen_database") is not True
        or report.get("byte_identical_to_frozen_manifest") is not True
        or reconstructed != frozen
        or source.get("repository") != "wenhuchen/WikiTables-WithLinks"
        or source.get("commit")
        != "dc066e1a6d5281511d8b73a6107d5ad2824cc2b2"
        or source.get("raw_json_files") != 15_316
        or source.get("curated_json_files") != 15_314
        or source.get("excluded_files")
        != [
            "2014_Malaysia_Premier_League_0.json",
            "2014_Malaysia_Premier_League_1.json",
        ]
        or source.get("case_variant_filenames_retained_distinctly") is not True
        or known_good_runtime
        != {
            "python": {"implementation": "CPython", "version": "3.14.5"},
            "sqlite": {
                "version": "3.53.1",
                "database_page_size_bytes": 4_096,
                "compile_options": {
                    "count": 53,
                    "canonicalization": (
                        "lexicographically sorted option strings encoded as "
                        "UTF-8 canonical JSON"
                    ),
                    "canonical_sha256": (
                        "99d80fee03818112b412ae76c1b334602ab6b6de6899610155a90a043ed5bbbc"
                    ),
                },
            },
            "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
        }
        or runtime_acceptance
        != {
            "frozen_output_exact_bytes_observed_in_known_good_runtime": True,
            "cross_environment_portability_claimed": False,
            "other_runtime_pass_condition": (
                "all frozen output byte counts and SHA-256 identities match "
                "exactly"
            ),
        }
    ):
        raise VerificationError("BM25 reconstruction witness declaration changed")

    identifier_report = load_json(
        ROOT / "data/evidence/identifier_aware_executor_audit.json"
    )
    if not isinstance(identifier_report, dict):
        raise VerificationError("identifier-aware evidence schema changed")
    source_database = identifier_report.get("inputs", {}).get(
        "source_bm25_database", {}
    )
    cross_binding = {
        "database_bytes": "bytes",
        "database_sha256": "sha256",
        "document_count": "document_count",
        "source_file_count": "source_file_count",
        "source_inventory_sha256": "source_inventory_sha256",
        "internal_manifest_bytes": "manifest_bytes",
        "internal_manifest_sha256": "manifest_sha256",
    }
    for reconstruction_key, identifier_key in cross_binding.items():
        if frozen.get(reconstruction_key) != source_database.get(identifier_key):
            raise VerificationError(
                "BM25 reconstruction and identifier-aware database bindings differ: "
                f"{reconstruction_key}"
            )

    manifest_member = (
        "data/results/released/hybridqa_natural_v1/"
        "bm25_v1/index_manifest.json"
    )
    manifest_payload = _read_artifact_member(manifest_member)
    if (
        len(manifest_payload) != frozen.get("manifest_bytes")
        or hashlib.sha256(manifest_payload).hexdigest()
        != frozen.get("manifest_sha256")
    ):
        raise VerificationError("released BM25 manifest identity changed")
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as exc:
        raise VerificationError("released BM25 manifest is invalid JSON") from exc
    released_manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    if (
        manifest_payload != released_manifest_bytes
        or len(module.canonical_bytes(manifest))
        != frozen.get("internal_manifest_bytes")
        or hashlib.sha256(module.canonical_bytes(manifest)).hexdigest()
        != frozen.get("internal_manifest_sha256")
        or manifest.get("index_version") != "hybridqa-sqlite-fts5-v1"
        or manifest.get("status") != "complete"
        or manifest.get("source_file_count") != frozen.get("source_file_count")
        or manifest.get("document_count") != frozen.get("document_count")
        or manifest.get("duplicate_link_count")
        != frozen.get("duplicate_link_count")
        or manifest.get("source_inventory_sha256")
        != frozen.get("source_inventory_sha256")
    ):
        raise VerificationError("released BM25 manifest contract changed")
    return "hybridqa_bm25_reconstruction_contract_binding"


def replay_qwen_decoded_id_replication() -> str:
    """Replay exact-query-equivalent Qwen decoded-ID BM25 evidence."""

    script_path = ROOT / "code/analysis/recompute_qwen_decoded_id_replication.py"
    if not script_path.is_file():
        raise VerificationError("Qwen decoded-ID replay script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_qwen_decoded_id_replication", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load Qwen decoded-ID replay script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(
            _read_artifact_member, script_path=script_path
        )
        observed = module.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(f"Qwen decoded-ID replay failed: {exc}") from exc

    expected = (ROOT / "data/evidence/qwen_decoded_id_replication.json").read_bytes()
    if observed != expected:
        raise VerificationError("Qwen decoded-ID evidence bytes changed")
    scope = report.get("execution_scope", {})
    if (
        report.get("status")
        != "verified_zero_call_qwen_decoded_id_replication"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("network_used") is not False
        or scope.get("fresh_sqlite_execution") is not False
        or scope.get("exact_query_coverage", {}).get("numerator") != 335
    ):
        raise VerificationError("Qwen decoded-ID replay scope changed")
    cohort = report.get("cohort", {})
    if (
        cohort.get("all_rows") != 335
        or cohort.get("c1_selected_equals_traced") != 229
        or cohort.get("c0_selected_differs_from_traced") != 106
        or cohort.get("corpus_resolvable_selected_ids") != 334
    ):
        raise VerificationError("Qwen decoded-ID cohort changed")
    body = report.get("body_only_bm25", {}).get("all", {}).get("by_cutoff", {})
    exact = report.get("exact_field_baseline", {}).get("all", {}).get(
        "by_cutoff", {}
    )
    if (
        body.get("5", {})
        .get("p_selected_id_preserved_on_resolvable_rows", {})
        .get("numerator")
        != 231
        or exact.get("1", {})
        .get("p_selected_id_preserved_on_resolvable_rows", {})
        .get("numerator")
        != 334
    ):
        raise VerificationError("Qwen decoded-ID preservation counts changed")
    return "qwen_decoded_id_replication"


def replay_transfer_scope_audit() -> str:
    """Replay supplied-ID, selected-ID, and title-collision scope checks."""

    script_path = ROOT / "code/analysis/recompute_transfer_scope_audit.py"
    if not script_path.is_file():
        raise VerificationError("transfer-scope replay script is missing")
    spec = importlib.util.spec_from_file_location(
        "_ccr_transfer_scope_audit", script_path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load transfer-scope replay script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report = module.build_report_from_members(_read_artifact_member)
        observed = module.canonical_json_bytes(report)
    except Exception as exc:
        raise VerificationError(f"transfer-scope replay failed: {exc}") from exc

    expected = (ROOT / "data/evidence/transfer_scope_audit.json").read_bytes()
    if observed != expected:
        raise VerificationError("transfer-scope evidence bytes changed")
    if (
        report.get("status") != "verified_zero_call_transfer_scope_audit"
        or report.get("api_calls") != 0
        or report.get("model_calls") != 0
        or report.get("network_used") is not False
    ):
        raise VerificationError("transfer-scope replay status changed")

    expected_development = {
        "unique_hidden_no_selection": {
            "bm25": (1_182, 890),
            "hybrid_rrf_bge_reranker_v2_m3": (1_182, 1_174),
        },
        "unique_hidden_answer_in_table": {
            "bm25": (215, 178),
            "hybrid_rrf_bge_reranker_v2_m3": (215, 213),
        },
    }
    observed_development = {
        bucket: {
            retriever: (
                values.get("records"),
                values.get("preservation_by_cutoff", {})
                .get("5", {})
                .get("numerator"),
            )
            for retriever, values in bucket_report.get("retrievers", {}).items()
        }
        for bucket, bucket_report in report.get(
            "supplied_traced_id_projection_preservation", {}
        )
        .get("buckets", {})
        .items()
    }
    if observed_development != expected_development:
        raise VerificationError("transfer-scope development counts changed")

    expected_qwen = {
        "all": (335, 334, 330),
        "c1_selected_id_matches_traced_id": (229, 229, 226),
        "c0_selected_id_differs_from_traced_id": (106, 105, 104),
    }
    observed_qwen = {
        name: (
            values.get("all_rows"),
            values.get("corpus_resolvable_selected_ids"),
            values.get("preservation_by_cutoff_on_resolvable_selected_ids", {})
            .get("5", {})
            .get("numerator"),
        )
        for name, values in report.get("qwen_selected_id_preservation", {})
        .get("groups", {})
        .items()
    }
    if observed_qwen != expected_qwen:
        raise VerificationError("transfer-scope Qwen counts changed")
    expected_qwen_joint = {
        "all": (334, 232, 98, 0, 4),
        "c1_selected_id_matches_traced_id": (229, 226, 0, 0, 3),
        "c0_selected_id_differs_from_traced_id": (105, 6, 98, 0, 1),
    }
    observed_qwen_joint = {
        name: (
            values.get("denominator"),
            values.get("p1_h1"),
            values.get("p1_h0"),
            values.get("p0_h1"),
            values.get("p0_h0"),
        )
        for name, group in report.get("qwen_selected_id_preservation", {})
        .get("groups", {})
        .items()
        for values in [
            group.get(
                "selected_vs_traced_return_joint_at_5_on_resolvable_selected_ids",
                {},
            )
        ]
    }
    if observed_qwen_joint != expected_qwen_joint:
        raise VerificationError("transfer-scope Qwen joint P/H cells changed")
    c0_excluded = (
        report.get("qwen_selected_id_preservation", {})
        .get("groups", {})
        .get("c0_selected_id_differs_from_traced_id", {})
        .get("excluded_unresolvable_selected_ids")
    )
    if c0_excluded != {"count": 1, "traced_id_return_at_5": 0}:
        raise VerificationError("transfer-scope Qwen exclusion accounting changed")
    titles = report.get("identifier_title_uniqueness", {})
    if (
        titles.get("identifier_count") != 286_270
        or titles.get("distinct_title_count") != 286_270
        or titles.get("collision_title_count") != 0
        or titles.get("all_title_strings_unique") is not True
    ):
        raise VerificationError("transfer-scope title uniqueness changed")
    return "transfer_scope_audit"


def replay_primary_hosted_call_archive(directory: Path) -> dict[str, Any]:
    report_path = directory / "primary_hosted_call_archive_verification.json"
    archive_export = load_json(PRIMARY_HOSTED_CALL_ARCHIVE_REPORT)
    expected_archive_sha256 = (
        archive_export.get("archive", {}).get("sha256")
        if isinstance(archive_export, dict)
        else None
    )
    if (
        not isinstance(expected_archive_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_archive_sha256)
    ):
        raise VerificationError(
            "external primary hosted-call archive digest is missing or invalid"
        )
    run(
        [
            sys.executable,
            "code/analysis/verify_primary_hosted_call_archive.py",
            "--archive",
            str(PRIMARY_HOSTED_CALL_ARCHIVE),
            "--expected-archive-sha256",
            expected_archive_sha256,
            "--bridge-details",
            str(PRIMARY_BRIDGE_DETAILS),
            "--rewrite-details",
            str(PRIMARY_REWRITE_DETAILS),
            "--answer-details",
            str(PRIMARY_ANSWER_DETAILS),
            "--parser-root",
            str(ROOT),
            "--report",
            str(report_path),
        ]
    )
    compare_bytes(
        report_path,
        ROOT / "data/evidence/primary_hosted_call_archive_verification.json",
        "primary hosted-call archive verification",
    )
    report = load_json(report_path)
    if (
        not isinstance(report, dict)
        or report.get("status") != "verified"
        or report.get("network_used") is not False
        or report.get("coverage", {}).get("issued_calls") != 10_192
        or report.get("coverage", {}).get("logical_task_rows") != 10_192
        or report.get("coverage", {}).get("transport_attempts_started") != 10_194
        or report.get("coverage", {}).get("retried_logical_tasks") != 2
        or report.get("coverage", {}).get("terminal_attempt_distribution")
        != {"1": 10_190, "2": 2}
        or report.get("coverage", {}).get("unissued_answer_cells") != 8
        or report.get("coverage", {}).get("compact_rows_bound_one_to_one")
        != 10_200
    ):
        raise VerificationError("primary hosted-call archive coverage changed")
    return report


def replay_sensitivity() -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "code")]
    )
    program = (
        "import json; from experiments.analyze_retriever_sensitivity import build; "
        "print(json.dumps(build(), ensure_ascii=False, sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=_subprocess_environment(environment),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise VerificationError(
            "retriever-sensitivity replay failed\n" + completed.stderr[-4000:]
        )
    observed = json.loads(completed.stdout)
    expected = load_json(
        ROOT
        / "data/results/released/hybridqa_natural_v3_retriever_sensitivity_v1/analysis.json"
    )
    if observed != expected:
        raise VerificationError("retriever-sensitivity replay differs from release")
    return "retriever_sensitivity"


def replay_authority(directory: Path) -> list[str]:
    if not AUTHORITY.is_file():
        return []
    qwen = directory / "qwen_core_promotion"
    run(
        [
            sys.executable,
            "code/analysis/promote_qwen_core.py",
            "--archive",
            str(AUTHORITY),
            "--output-dir",
            str(qwen),
        ]
    )
    for name in ("items.jsonl", "manifest.json", "summary.json"):
        compare_bytes(
            qwen / name,
            ROOT / "data/evidence/qwen_core_promotion" / name,
            f"Qwen {name}",
        )
    broad = directory / "broad_dev_audit.json"
    run(
        [
            sys.executable,
            "code/analysis/recompute_broad_dev_audit.py",
            "--archive",
            str(AUTHORITY),
            "--output",
            str(broad),
        ]
    )
    compare_bytes(broad, ROOT / "data/evidence/broad_dev_audit.json", "broad audit")
    return ["qwen_core_promotion", "broad_dev_audit"]


def verify_paper(pdf: Path | None, source: Path | None) -> dict[str, object]:
    binding = load_json(ROOT / "data/manifests/PAPER_BINDING.json")
    if not isinstance(binding, dict) or binding.get("status") != "bound":
        raise VerificationError("paper binding status changed")
    result: dict[str, object] = {
        "bound_pdf_sha256": binding["paper_pdf"]["sha256"],
        "bound_source_sha256": binding["paper_source"]["sha256"],
        "provided_pdf_verified": False,
        "provided_source_verified": False,
    }
    if pdf is not None:
        if sha256_file(pdf) != binding["paper_pdf"]["sha256"]:
            raise VerificationError("provided paper PDF does not match PAPER_BINDING.json")
        result["provided_pdf_verified"] = True
    if source is not None:
        if sha256_file(source) != binding["paper_source"]["sha256"]:
            raise VerificationError("provided paper source does not match PAPER_BINDING.json")
        result["provided_source_verified"] = True
    return result
