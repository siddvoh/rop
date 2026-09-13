#!/usr/bin/env python3
"""Evaluate the returned-object-preservation contract without dependencies.

The evaluator is deliberately small and returns a total verdict. A declaration is read
through a pre-outcome envelope, all referenced assets are hash checked, and one
terminal verdict is emitted for every record.  Hard declarations also bind the
complete expected record population before returned IDs exist; strict run
profile v1 aggregates that population without dropping missing or errored
records.  ``corpus_key_set`` uses outer-any acceptable-object semantics and
then applies the declared ``any`` or ``all`` rule within each object, including
singleton member sets.  Record identity is read from the population-bound
field rather than from a fixed key.  A hard declaration may read the raw
selected key or a hash-bound, injective lossless identity witness.  Witness
records are bound to the declaration, population record, encoded value,
decoded selection, and inverse-map bytes.  The v1 trust tier assumes a trusted
manifest authority and producer-supplied logs.  Hashes bind bytes for
internal consistency and replay; they do not prove chronology or resist a
malicious producer.  Natural-language prompt bytes are retained and
hash-bound as provenance; they never activate or override the run-bound
declaration.

``evaluate_preservation_kernel`` exposes the minimal identity-resolution and
membership calculation separately.  Its result is explicitly not a
conformance verdict.  Record and run conformance still require the complete
prospective declaration, population, authority, and asset checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

_KERNEL_PATH = Path(__file__).with_name("returned_object_preservation_kernel.py")
_KERNEL_SPEC = importlib.util.spec_from_file_location(
    "_rop_exact_sibling_kernel", _KERNEL_PATH
)
if _KERNEL_SPEC is None or _KERNEL_SPEC.loader is None:
    raise RuntimeError("cannot load exact sibling preservation kernel")
_KERNEL_MODULE = importlib.util.module_from_spec(_KERNEL_SPEC)
_KERNEL_SPEC.loader.exec_module(_KERNEL_MODULE)
KERNEL_VERSION = _KERNEL_MODULE.KERNEL_VERSION
KernelConfigurationError = _KERNEL_MODULE.KernelConfigurationError
compile_resolver = _KERNEL_MODULE.compile_resolver
resolve_selected_objects = _KERNEL_MODULE.resolve_selected_objects
evaluate_preservation_kernel = _KERNEL_MODULE.evaluate_preservation_kernel


DECLARATION_INVALID = "DECLARATION_INVALID"
NOT_APPLICABLE = "NOT_APPLICABLE"
SELECTOR_ERROR = "SELECTOR_ERROR"
RESOLUTION_ERROR = "RESOLUTION_ERROR"
UNVERIFIABLE = "UNVERIFIABLE"
PASS = "PASS"
FAIL = "FAIL"
MISSING = "MISSING"
RUN_INVALID = "RUN_INVALID"
RUN_FAIL = "RUN_FAIL"
RUN_UNVERIFIABLE = "RUN_UNVERIFIABLE"
RUN_PASS = "RUN_PASS"
VERDICTS = (
    DECLARATION_INVALID,
    NOT_APPLICABLE,
    SELECTOR_ERROR,
    RESOLUTION_ERROR,
    UNVERIFIABLE,
    PASS,
    FAIL,
)
RUN_STATUSES = (
    RUN_INVALID,
    RUN_FAIL,
    RUN_UNVERIFIABLE,
    RUN_PASS,
    NOT_APPLICABLE,
)
HISTOGRAM_STATUSES = VERDICTS + (MISSING,)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$")
FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
WITNESS_VERSION = "rop-identity-witness/v1"
WITNESS_BINDING_PROFILE = "rop-witness-record-binding/v1"


class ContractError(RuntimeError):
    """A declaration, envelope, or bound asset is not auditable."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot read strict JSON: {path.name}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot read bound asset: {path.name}") from exc
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{label} members differ from the contract")
    return value


def _string(value: Any, label: str, pattern: re.Pattern[str] = NAME_RE) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ContractError(f"{label} is invalid")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} is not a lowercase SHA-256")
    return value


def _locator(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{label} is not a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{label} is not a safe relative locator")
    return value


EXPECTED_POLICY = {
    "total_verdicts": True,
    "declaration_invalid": DECLARATION_INVALID,
    "soft_or_open": NOT_APPLICABLE,
    "selected_key_invalid": SELECTOR_ERROR,
    "selected_key_unresolvable": RESOLUTION_ERROR,
    "returned_id_log_missing_or_malformed": UNVERIFIABLE,
    "retention_satisfied": PASS,
    "retention_violated": FAIL,
    "ledger_policy": "retain_every_record",
    "metric_denominator": "hard_resolvable_verifiable_only",
}

EXPECTED_RUN_POLICY = {
    "profile_version": "rop-run-profile/v1",
    "invalid_declaration_or_population": RUN_INVALID,
    "expected_hard_record_failed": RUN_FAIL,
    "expected_hard_record_missing_or_unverifiable": RUN_UNVERIFIABLE,
    "all_expected_hard_records_passed": RUN_PASS,
    "soft_or_open": NOT_APPLICABLE,
    "status_histogram": "always",
    "eligible_expected_coverage": "eligible_over_expected",
    "diagnostic_pass_fraction": "PASS/(PASS+FAIL)",
    "diagnostic_is_conformance_verdict": False,
}


def validate_manifest(document: Any) -> str:
    """Validate the normative semantics needed by the executable evaluator."""

    hard_mode = (
        isinstance(document, dict)
        and isinstance(document.get("selection"), dict)
        and document["selection"].get("mode") == "hard"
    )
    manifest = _exact_object(
        document,
        {
            "contract_version",
            "binding",
            "authority",
            "selection",
            "verdict_policy",
            "run_policy",
        }
        | (
            {"population", "resolver", "returned_set", "retention"}
            if hard_mode
            else set()
        ),
        "manifest",
    )
    if manifest["contract_version"] != "rop/v1":
        raise ContractError("unsupported contract_version")

    binding = _exact_object(
        manifest["binding"],
        {"declaration_id", "declared_phase", "config_locator", "config_sha256"},
        "binding",
    )
    _string(binding["declaration_id"], "binding.declaration_id")
    if binding["declared_phase"] != "pre_outcome":
        raise ContractError("declaration was not sealed pre-outcome")
    _locator(binding["config_locator"], "binding.config_locator")
    _digest(binding["config_sha256"], "binding.config_sha256")

    authority = _exact_object(
        manifest["authority"],
        {"normative_source", "manifest_authority", "natural_language_prompt"},
        "authority",
    )
    if authority["normative_source"] != "run_bound_manifest":
        raise ContractError("only the run-bound manifest may be normative")
    manifest_authority = _exact_object(
        authority["manifest_authority"],
        {
            "authority_id",
            "policy_authority_id",
            "producer_id",
            "policy_basis",
            "target_interface_id",
            "interface_version",
            "policy_revision",
            "chronology_basis",
            "log_provenance_basis",
            "hash_scope",
            "adversarial_producer_resistance",
        },
        "authority.manifest_authority",
    )
    _string(
        manifest_authority["authority_id"],
        "authority.manifest_authority.authority_id",
    )
    _string(
        manifest_authority["policy_authority_id"],
        "authority.manifest_authority.policy_authority_id",
    )
    _string(
        manifest_authority["producer_id"],
        "authority.manifest_authority.producer_id",
    )
    policy_basis = _exact_object(
        manifest_authority["policy_basis"],
        {"locator", "sha256"},
        "authority.manifest_authority.policy_basis",
    )
    _locator(
        policy_basis["locator"],
        "authority.manifest_authority.policy_basis.locator",
    )
    _digest(
        policy_basis["sha256"],
        "authority.manifest_authority.policy_basis.sha256",
    )
    _string(
        manifest_authority["target_interface_id"],
        "authority.manifest_authority.target_interface_id",
    )
    _string(
        manifest_authority["interface_version"],
        "authority.manifest_authority.interface_version",
    )
    _string(
        manifest_authority["policy_revision"],
        "authority.manifest_authority.policy_revision",
    )
    if (
        manifest_authority["chronology_basis"] != "trusted_manifest_authority"
        or manifest_authority["log_provenance_basis"] != "trusted_producer"
        or manifest_authority["hash_scope"] != "byte_binding_and_replay_only"
        or manifest_authority["adversarial_producer_resistance"] is not False
    ):
        raise ContractError("manifest authority overstates the v1 trust tier")
    prompt = _exact_object(
        authority["natural_language_prompt"],
        {"role", "locator", "sha256", "can_activate", "can_override_manifest"},
        "authority.natural_language_prompt",
    )
    if (
        prompt["role"] != "provenance_only"
        or prompt["can_activate"] is not False
        or prompt["can_override_manifest"] is not False
    ):
        raise ContractError("natural-language prompt gained normative authority")
    _locator(prompt["locator"], "authority.natural_language_prompt.locator")
    _digest(prompt["sha256"], "authority.natural_language_prompt.sha256")

    selection = manifest["selection"]
    if not isinstance(selection, dict) or "mode" not in selection:
        raise ContractError("selection.mode is missing")
    mode = selection["mode"]
    if mode in {"soft", "open"}:
        _exact_object(selection, {"mode"}, "selection")
    elif mode == "hard":
        if set(selection) == {"mode", "source_field", "type", "namespace"}:
            _string(selection["source_field"], "selection.source_field", FIELD_RE)
        elif set(selection) == {
            "mode",
            "identity_witness",
            "type",
            "namespace",
        }:
            witness = _exact_object(
                selection["identity_witness"],
                {
                    "witness_version",
                    "source_field",
                    "artifact_locator",
                    "artifact_sha256",
                    "decoding",
                    "record_binding",
                    "authentication_basis",
                },
                "selection.identity_witness",
            )
            if witness["witness_version"] != WITNESS_VERSION:
                raise ContractError("identity witness version is unsupported")
            _string(
                witness["source_field"],
                "selection.identity_witness.source_field",
                FIELD_RE,
            )
            _locator(
                witness["artifact_locator"],
                "selection.identity_witness.artifact_locator",
            )
            _digest(
                witness["artifact_sha256"],
                "selection.identity_witness.artifact_sha256",
            )
            if (
                witness["decoding"] != "bound_exact_inverse_map"
                or witness["record_binding"] != "sha256_canonical_json"
                or witness["authentication_basis"]
                != "trusted_producer_record_binding"
            ):
                raise ContractError("identity witness semantics are unsupported")
        else:
            raise ContractError(
                "hard selection must declare exactly one raw key or identity witness source"
            )
        if selection["type"] not in {
            "corpus_key",
            "corpus_uri",
            "composite_corpus_key",
            "corpus_key_set",
        }:
            raise ContractError("selection.type is unsupported")
        _string(selection["namespace"], "selection.namespace")

        population = _exact_object(
            manifest["population"],
            {
                "population_version",
                "declared_phase",
                "sealed_before",
                "artifact_locator",
                "artifact_sha256",
                "record_id_field",
                "expected_count",
            },
            "population",
        )
        if (
            population["population_version"] != "rop-population/v1"
            or population["declared_phase"] != "pre_outcome"
            or population["sealed_before"] != "returned_ids"
        ):
            raise ContractError("population was not bound before returned IDs")
        _locator(population["artifact_locator"], "population.artifact_locator")
        _digest(population["artifact_sha256"], "population.artifact_sha256")
        _string(population["record_id_field"], "population.record_id_field", FIELD_RE)
        if type(population["expected_count"]) is not int or population["expected_count"] < 1:
            raise ContractError("population.expected_count is invalid")

        resolver = _exact_object(
            manifest["resolver"],
            {"name", "version", "artifact_locator", "artifact_sha256", "returns"},
            "resolver",
        )
        _string(resolver["name"], "resolver.name")
        _string(resolver["version"], "resolver.version")
        _locator(resolver["artifact_locator"], "resolver.artifact_locator")
        _digest(resolver["artifact_sha256"], "resolver.artifact_sha256")
        if resolver["returns"] not in {"alias_set", "chunk_set", "alias_or_chunk_set"}:
            raise ContractError("resolver.returns is unsupported")

        returned = _exact_object(
            manifest["returned_set"],
            {
                "component",
                "revision",
                "implementation_locator",
                "implementation_sha256",
                "id_field",
                "cutoff",
            },
            "returned_set",
        )
        _string(returned["component"], "returned_set.component")
        _string(returned["revision"], "returned_set.revision")
        _locator(returned["implementation_locator"], "returned_set.implementation_locator")
        _digest(returned["implementation_sha256"], "returned_set.implementation_sha256")
        _string(returned["id_field"], "returned_set.id_field", FIELD_RE)
        if type(returned["cutoff"]) is not int or returned["cutoff"] < 1:
            raise ContractError("returned_set.cutoff is invalid")
        if manifest_authority["target_interface_id"] != returned["component"]:
            raise ContractError(
                "policy target_interface_id differs from returned component"
            )

        retention = _exact_object(
            manifest["retention"], {"quantifier", "over"}, "retention"
        )
        if retention["quantifier"] not in {"any", "all"}:
            raise ContractError("retention.quantifier is unsupported")
        if retention["over"] != "resolver_returned_ids":
            raise ContractError("retention.over is unsupported")
        if retention["quantifier"] == "all" and resolver["returns"] not in {
            "chunk_set",
            "alias_or_chunk_set",
        }:
            raise ContractError("all retention requires a chunk/member-set resolver")
    else:
        raise ContractError("selection.mode is unsupported")

    if not manifest_authority["policy_revision"].startswith(f"{mode}/"):
        raise ContractError("policy_revision is not specific to selection.mode")
    if manifest["verdict_policy"] != EXPECTED_POLICY:
        raise ContractError("verdict_policy differs from the total-verdict profile")
    if manifest["run_policy"] != EXPECTED_RUN_POLICY:
        raise ContractError("run_policy differs from strict profile v1")
    return mode


def validate_envelope(
    envelope: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    mode: str,
) -> None:
    expected_members = {
        "envelope_version",
        "declaration_locator",
        "declaration_sha256",
        "declared_phase",
        "sealed_before",
        "authority_id",
        "policy_authority_id",
        "producer_id",
        "target_interface_id",
        "interface_version",
        "policy_revision",
        "chronology_basis",
    }
    if mode == "hard":
        expected_members.update({"population_locator", "population_sha256"})
    envelope = _exact_object(
        envelope,
        expected_members,
        "envelope",
    )
    if envelope["envelope_version"] != "rop-envelope/v1":
        raise ContractError("unsupported envelope_version")
    if envelope["declared_phase"] != "pre_outcome" or envelope["sealed_before"] != "returned_ids":
        raise ContractError("declaration envelope was not sealed before outcomes")
    _string(envelope["authority_id"], "envelope.authority_id")
    _string(envelope["policy_authority_id"], "envelope.policy_authority_id")
    _string(envelope["producer_id"], "envelope.producer_id")
    _string(envelope["target_interface_id"], "envelope.target_interface_id")
    _string(envelope["interface_version"], "envelope.interface_version")
    _string(envelope["policy_revision"], "envelope.policy_revision")
    if envelope["chronology_basis"] != "trusted_manifest_authority":
        raise ContractError("unsupported chronology basis")
    if _locator(envelope["declaration_locator"], "envelope.declaration_locator") != manifest_path.name:
        raise ContractError("envelope declaration locator differs from selected manifest")
    if _digest(envelope["declaration_sha256"], "envelope.declaration_sha256") != sha256_file(manifest_path):
        raise ContractError("envelope declaration digest mismatch")
    manifest_authority = manifest["authority"]["manifest_authority"]
    for field in (
        "authority_id",
        "policy_authority_id",
        "producer_id",
        "target_interface_id",
        "interface_version",
        "policy_revision",
        "chronology_basis",
    ):
        if envelope[field] != manifest_authority[field]:
            raise ContractError(f"envelope and manifest {field} values differ")
    if mode == "hard":
        population = manifest["population"]
        if (
            _locator(envelope["population_locator"], "envelope.population_locator")
            != population["artifact_locator"]
            or _digest(
                envelope["population_sha256"], "envelope.population_sha256"
            )
            != population["artifact_sha256"]
        ):
            raise ContractError("envelope and manifest population bindings differ")


def _bound_path(root: Path, locator: str, expected_digest: str, label: str) -> Path:
    path = root / _locator(locator, f"{label}.locator")
    if not path.is_file() or sha256_file(path) != expected_digest:
        raise ContractError(f"{label} binding mismatch")
    return path


def _load_resolver(path: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    resolver = _exact_object(
        load_json(path), {"resolver_version", "namespace", "aliases", "members"}, "resolver artifact"
    )
    if resolver["resolver_version"] != manifest["resolver"]["version"]:
        raise ContractError("resolver version mismatch")
    if resolver["namespace"] != manifest["selection"]["namespace"]:
        raise ContractError("resolver namespace mismatch")
    try:
        compiled = compile_resolver(resolver)
    except KernelConfigurationError as exc:
        raise ContractError(str(exc)) from exc
    return resolver, compiled


def _load_identity_witness(
    path: Path,
    manifest: dict[str, Any],
    compiled_resolver: Any,
) -> dict[str, Any]:
    """Load an injective exact inverse map for a declared witness source."""

    witness = _exact_object(
        load_json(path),
        {
            "witness_version",
            "namespace",
            "selection_type",
            "encoding",
            "witness_to_selected",
        },
        "identity witness artifact",
    )
    selection = manifest["selection"]
    if (
        witness["witness_version"] != WITNESS_VERSION
        or witness["namespace"] != selection["namespace"]
        or witness["selection_type"] != selection["type"]
        or witness["encoding"] != "exact_inverse_map"
    ):
        raise ContractError("identity witness artifact differs from its declaration")
    mapping = witness["witness_to_selected"]
    if not isinstance(mapping, dict) or not mapping:
        raise ContractError("identity witness inverse map is empty or malformed")
    decoded_encodings: set[bytes] = set()
    for encoded, decoded in mapping.items():
        if not isinstance(encoded, str) or not encoded:
            raise ContractError("identity witness value is malformed")
        if selection["type"] == "corpus_key_set":
            if (
                not isinstance(decoded, list)
                or not decoded
                or any(not isinstance(item, str) or not item for item in decoded)
                or decoded != sorted(decoded)
                or len(decoded) != len(set(decoded))
            ):
                raise ContractError("identity witness decoded key set is malformed")
        elif not isinstance(decoded, str) or not decoded:
            raise ContractError("identity witness decoded key is malformed")
        encoded_decoded = canonical_json_bytes(decoded)
        if encoded_decoded in decoded_encodings:
            raise ContractError("identity witness inverse map is not injective")
        decoded_encodings.add(encoded_decoded)
        try:
            resolve_selected_objects(
                decoded,
                selection["type"],
                compiled_resolver,
            )
        except (ValueError, LookupError) as exc:
            raise ContractError(
                "identity witness decodes outside the bound resolver"
            ) from exc
    return witness


def _load_policy_basis(path: Path, manifest: dict[str, Any]) -> None:
    authority = manifest["authority"]["manifest_authority"]
    policy = _exact_object(
        load_json(path),
        {
            "policy_version",
            "policy_authority_id",
            "producer_id",
            "target_interface_id",
            "target_interface_version",
            "selection_mode",
            "policy_revision",
            "run_profile",
        },
        "policy basis",
    )
    if policy != {
        "policy_version": "rop-policy/v1",
        "policy_authority_id": authority["policy_authority_id"],
        "producer_id": authority["producer_id"],
        "target_interface_id": authority["target_interface_id"],
        "target_interface_version": authority["interface_version"],
        "selection_mode": manifest["selection"]["mode"],
        "policy_revision": authority["policy_revision"],
        "run_profile": manifest["run_policy"]["profile_version"],
    }:
        raise ContractError("policy basis attribution differs from the declaration")


def _load_population(path: Path, manifest: dict[str, Any]) -> list[str]:
    population_binding = manifest["population"]
    population = _exact_object(
        load_json(path),
        {"population_version", "record_id_field", "record_ids"},
        "population artifact",
    )
    if (
        population["population_version"] != population_binding["population_version"]
        or population["record_id_field"] != population_binding["record_id_field"]
    ):
        raise ContractError("population artifact identity differs from its binding")
    record_ids = population["record_ids"]
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or any(
            not isinstance(record_id, str)
            or NAME_RE.fullmatch(record_id) is None
            for record_id in record_ids
        )
        or len(set(record_ids)) != len(record_ids)
        or record_ids != sorted(record_ids)
    ):
        raise ContractError("population record IDs are malformed, duplicated, or unsorted")
    if len(record_ids) != population_binding["expected_count"]:
        raise ContractError("population expected_count differs from bound record IDs")
    return record_ids


def _load_profile(manifest_path: Path, envelope_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    mode = validate_manifest(manifest)
    envelope = load_json(envelope_path)
    validate_envelope(envelope, manifest_path, manifest, mode)
    root = manifest_path.parent
    authority = manifest["authority"]["manifest_authority"]

    binding = manifest["binding"]
    _bound_path(root, binding["config_locator"], binding["config_sha256"], "config")
    prompt = manifest["authority"]["natural_language_prompt"]
    _bound_path(root, prompt["locator"], prompt["sha256"], "prompt provenance")
    policy_basis = authority["policy_basis"]
    policy_path = _bound_path(
        root,
        policy_basis["locator"],
        policy_basis["sha256"],
        "policy basis",
    )
    _load_policy_basis(policy_path, manifest)

    profile: dict[str, Any] = {
        "manifest": manifest,
        "declaration_sha256": envelope["declaration_sha256"],
        "mode": mode,
        "resolver": None,
        "compiled_resolver": None,
        "identity_witness": None,
        "population_record_ids": None,
        "authority_attribution": {
            "policy_authority_id": authority["policy_authority_id"],
            "producer_id": authority["producer_id"],
            "policy_basis": dict(authority["policy_basis"]),
            "target_interface_id": authority["target_interface_id"],
            "interface_version": authority["interface_version"],
            "policy_revision": authority["policy_revision"],
        },
    }
    if mode in {"soft", "open"}:
        return profile

    resolver_path = _bound_path(
        root,
        manifest["resolver"]["artifact_locator"],
        manifest["resolver"]["artifact_sha256"],
        "resolver",
    )
    _bound_path(
        root,
        manifest["returned_set"]["implementation_locator"],
        manifest["returned_set"]["implementation_sha256"],
        "returned component implementation",
    )
    population = manifest["population"]
    population_path = _bound_path(
        root,
        population["artifact_locator"],
        population["artifact_sha256"],
        "population",
    )
    profile["resolver"], profile["compiled_resolver"] = _load_resolver(
        resolver_path,
        manifest,
    )
    if "identity_witness" in manifest["selection"]:
        witness_binding = manifest["selection"]["identity_witness"]
        witness_path = _bound_path(
            root,
            witness_binding["artifact_locator"],
            witness_binding["artifact_sha256"],
            "identity witness",
        )
        profile["identity_witness"] = _load_identity_witness(
            witness_path,
            manifest,
            profile["compiled_resolver"],
        )
    profile["population_record_ids"] = _load_population(population_path, manifest)
    return profile


def _get_dotted(record: Any, path: str) -> Any:
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


def _render_record_identifier(record: Any, field: str) -> str:
    """Return a normalized display value for one configured record-ID field."""

    if not isinstance(record, dict):
        return "<missing>"
    try:
        value = _get_dotted(record, field)
    except KeyError:
        return "<missing>"
    if not isinstance(value, str) or NAME_RE.fullmatch(value) is None:
        return "<invalid>"
    return value


def _observed_record_identifiers(
    records: list[Any], field: str
) -> tuple[list[str], list[str], int, list[str]]:
    """Collect configured identities while retaining malformed/duplicate evidence."""

    rendered = [_render_record_identifier(record, field) for record in records]
    valid = [value for value in rendered if NAME_RE.fullmatch(value) is not None]
    malformed_count = len(rendered) - len(valid)
    duplicates = sorted(
        value for value, count in Counter(valid).items() if count > 1
    )
    return rendered, valid, malformed_count, duplicates


def identity_witness_binding_sha256(
    *,
    declaration_id: str,
    declaration_sha256: str,
    population_artifact_sha256: str,
    record_id: str,
    witness_value: str,
    decoded_selection: Any,
    witness_artifact_sha256: str,
) -> str:
    """Return the deterministic v1 record binding for one identity witness."""

    payload = {
        "binding_profile": WITNESS_BINDING_PROFILE,
        "declaration_id": declaration_id,
        "declaration_sha256": declaration_sha256,
        "population_artifact_sha256": population_artifact_sha256,
        "record_id": record_id,
        "witness_value": witness_value,
        "decoded_selection": decoded_selection,
        "witness_artifact_sha256": witness_artifact_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _selected_from_record(
    record: Any,
    record_id: str,
    manifest: dict[str, Any],
    declaration_sha256: str,
    identity_witness: dict[str, Any] | None,
) -> Any:
    selection = manifest["selection"]
    if "source_field" in selection:
        return _get_dotted(record, selection["source_field"])
    witness_binding = selection["identity_witness"]
    if identity_witness is None:
        raise ContractError("identity witness declaration has no verified inverse map")
    try:
        witnessed_identity = _get_dotted(record, witness_binding["source_field"])
    except KeyError as exc:
        raise ValueError("identity witness is missing") from exc
    if not isinstance(witnessed_identity, dict) or set(witnessed_identity) != {
        "value",
        "binding_sha256",
    }:
        raise ValueError("identity witness record is malformed")
    witness_value = witnessed_identity["value"]
    observed_binding = witnessed_identity["binding_sha256"]
    if not isinstance(witness_value, str) or not witness_value:
        raise ValueError("identity witness value is malformed")
    if not isinstance(observed_binding, str) or SHA256_RE.fullmatch(observed_binding) is None:
        raise ValueError("identity witness record binding is malformed")
    mapping = identity_witness["witness_to_selected"]
    if witness_value not in mapping:
        raise LookupError(witness_value)
    decoded = mapping[witness_value]
    expected_binding = identity_witness_binding_sha256(
        declaration_id=manifest["binding"]["declaration_id"],
        declaration_sha256=declaration_sha256,
        population_artifact_sha256=manifest["population"]["artifact_sha256"],
        record_id=record_id,
        witness_value=witness_value,
        decoded_selection=decoded,
        witness_artifact_sha256=witness_binding["artifact_sha256"],
    )
    if observed_binding != expected_binding:
        raise ValueError("identity witness record binding mismatch")
    return decoded


def _result(record_id: str, verdict: str, reason: str, *, eligible: bool = False) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "verdict": verdict,
        "eligible_for_metric": eligible,
        "reason": reason,
    }


def _evaluate_record_with_profile(
    profile: dict[str, Any],
    record: Any,
) -> dict[str, Any]:
    """Evaluate one record against a profile already loaded and verified."""

    # A valid hard profile replaces this deterministic fallback with the
    # population-bound field. Soft/open and invalid declarations have no
    # usable hard-population field, so their legacy display identity remains
    # stable and cannot affect a conformance decision.
    record_id = _render_record_identifier(record, "record_id")
    manifest = profile["manifest"]
    mode = profile["mode"]
    if mode in {"soft", "open"}:
        return _result(record_id, NOT_APPLICABLE, f"explicit_{mode}_selection")
    record_id = _render_record_identifier(
        record, manifest["population"]["record_id_field"]
    )
    compiled_resolver = profile["compiled_resolver"]
    if compiled_resolver is None:
        return _result(
            record_id,
            DECLARATION_INVALID,
            "hard declaration has no verified resolver",
        )

    try:
        selected = _selected_from_record(
            record,
            record_id,
            manifest,
            profile["declaration_sha256"],
            profile["identity_witness"],
        )
    except (KeyError, TypeError, ValueError):
        return _result(record_id, SELECTOR_ERROR, "selected key is missing or malformed")
    except LookupError:
        return _result(record_id, RESOLUTION_ERROR, "selected key is not resolved by the bound resolver")
    try:
        returned = _get_dotted(
            record["returned"], manifest["returned_set"]["id_field"]
        )
    except (KeyError, TypeError):
        returned = None
        returned_missing = True
    else:
        returned_missing = False
    kernel = evaluate_preservation_kernel(
        selected=selected,
        selection_type=manifest["selection"]["type"],
        resolver=compiled_resolver,
        returned_ids=returned,
        cutoff=manifest["returned_set"]["cutoff"],
        quantifier=manifest["retention"]["quantifier"],
    )
    return _result(
        record_id,
        kernel["verdict"],
        (
            (
                "returned-ID log is missing"
                if returned_missing
                else "returned-ID log is malformed"
            )
            if kernel["verdict"] == UNVERIFIABLE
            else (
                "selected key is not resolved by the bound resolver"
                if kernel["verdict"] == RESOLUTION_ERROR
                else kernel["reason"]
            )
        ),
        eligible=kernel["verdict"] in {PASS, FAIL},
    )


def evaluate_record(
    manifest_path: Path,
    envelope_path: Path,
    record: Any,
) -> dict[str, Any]:
    """Evaluate one record using the fixed terminal precedence."""

    try:
        profile = _load_profile(manifest_path, envelope_path)
    except ContractError as exc:
        record_id = _render_record_identifier(record, "record_id")
        return _result(record_id, DECLARATION_INVALID, str(exc))
    return _evaluate_record_with_profile(profile, record)


def _ratio(numerator: int, denominator: int | None) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _empty_authority_attribution() -> dict[str, Any]:
    return {
        "policy_authority_id": None,
        "producer_id": None,
        "policy_basis": None,
        "target_interface_id": None,
        "interface_version": None,
        "policy_revision": None,
    }


def evaluate_run(
    manifest_path: Path,
    envelope_path: Path,
    run_id: str,
    records: Any,
) -> dict[str, Any]:
    """Aggregate one population-bound run under strict profile v1."""

    _string(run_id, "run_id")
    if not isinstance(records, list):
        raise ContractError("run records must be a list")
    try:
        profile = _load_profile(manifest_path, envelope_path)
    except ContractError as exc:
        record_results = [
            _result(
                _render_record_identifier(record, "record_id"),
                DECLARATION_INVALID,
                str(exc),
            )
            for record in records
        ]
        histogram = {
            status: sum(row["verdict"] == status for row in record_results)
            for status in HISTOGRAM_STATUSES
        }
        (
            _observed_record_ids,
            _valid_actual_ids,
            malformed_record_id_count,
            duplicate_record_ids,
        ) = _observed_record_identifiers(records, "record_id")
        return {
            "run_id": run_id,
            "profile_version": "rop-run-profile/v1",
            "run_status": RUN_INVALID,
            "reason": f"invalid declaration or population binding: {exc}",
            "selection_mode": None,
            "authority_attribution": _empty_authority_attribution(),
            "population_binding": {
                "status": "invalid",
                "expected_record_count": None,
                "reason": str(exc),
            },
            "status_histogram": histogram,
            "eligible_expected_coverage": {
                "eligible_record_count": 0,
                "expected_record_count": None,
                "fraction": None,
            },
            "diagnostic_pass_fraction": {
                "formula": "PASS/(PASS+FAIL)",
                "pass_count": 0,
                "pass_plus_fail_count": 0,
                "value": None,
                "is_conformance_verdict": False,
            },
            "observed_record_count": len(records),
            "observed_expected_record_count": 0,
            "missing_record_ids": [],
            "unexpected_record_ids": [],
            "duplicate_record_ids": duplicate_record_ids,
            "malformed_record_id_count": malformed_record_id_count,
            "records": record_results,
        }

    record_results = [
        _evaluate_record_with_profile(profile, record) for record in records
    ]
    histogram = {
        status: sum(row["verdict"] == status for row in record_results)
        for status in HISTOGRAM_STATUSES
    }
    (
        observed_record_ids,
        valid_actual_ids,
        malformed_record_id_count,
        duplicate_record_ids,
    ) = _observed_record_identifiers(records, "record_id")

    mode = profile["mode"]
    authority_attribution = profile["authority_attribution"]
    if mode in {"soft", "open"}:
        return {
            "run_id": run_id,
            "profile_version": "rop-run-profile/v1",
            "run_status": NOT_APPLICABLE,
            "reason": f"explicit_{mode}_selection",
            "selection_mode": mode,
            "authority_attribution": authority_attribution,
            "population_binding": {
                "status": "not_applicable",
                "expected_record_count": 0,
                "reason": "soft/open declarations carry no hard population",
            },
            "status_histogram": histogram,
            "eligible_expected_coverage": {
                "eligible_record_count": 0,
                "expected_record_count": 0,
                "fraction": None,
            },
            "diagnostic_pass_fraction": {
                "formula": "PASS/(PASS+FAIL)",
                "pass_count": 0,
                "pass_plus_fail_count": 0,
                "value": None,
                "is_conformance_verdict": False,
            },
            "observed_record_count": len(records),
            "observed_expected_record_count": 0,
            "missing_record_ids": [],
            "unexpected_record_ids": [],
            "duplicate_record_ids": duplicate_record_ids,
            "malformed_record_id_count": malformed_record_id_count,
            "records": record_results,
        }

    manifest = profile["manifest"]
    record_id_field = manifest["population"]["record_id_field"]
    (
        observed_record_ids,
        valid_actual_ids,
        malformed_record_id_count,
        duplicate_record_ids,
    ) = _observed_record_identifiers(records, record_id_field)
    expected_record_ids = profile["population_record_ids"]
    if not isinstance(expected_record_ids, list):
        raise ContractError("verified hard profile has no population")
    expected_set = set(expected_record_ids)
    actual_set = set(valid_actual_ids)
    missing_record_ids = sorted(expected_set - actual_set)
    unexpected_record_ids = sorted(actual_set - expected_set)
    histogram[MISSING] = len(missing_record_ids)

    result_by_record_id: dict[str, dict[str, Any]] = {}
    for record_id, result in zip(
        observed_record_ids, record_results, strict=True
    ):
        if (
            record_id in expected_set
            and record_id not in result_by_record_id
        ):
            result_by_record_id[record_id] = result

    expected_results = [
        result_by_record_id[record_id]
        for record_id in expected_record_ids
        if record_id in result_by_record_id
    ]
    eligible_count = sum(
        result["eligible_for_metric"] is True for result in expected_results
    )
    pass_count = sum(result["verdict"] == PASS for result in expected_results)
    fail_count = sum(result["verdict"] == FAIL for result in expected_results)

    population_invalid = bool(
        malformed_record_id_count
        or duplicate_record_ids
        or unexpected_record_ids
    )
    expected_verdicts = {result["verdict"] for result in expected_results}
    if population_invalid or DECLARATION_INVALID in expected_verdicts or NOT_APPLICABLE in expected_verdicts:
        run_status = RUN_INVALID
        reason = "observed records violate the bound population or hard declaration"
    elif FAIL in expected_verdicts:
        run_status = RUN_FAIL
        reason = "one or more expected hard records failed"
    elif missing_record_ids or expected_verdicts.intersection(
        {SELECTOR_ERROR, RESOLUTION_ERROR, UNVERIFIABLE}
    ):
        run_status = RUN_UNVERIFIABLE
        reason = "an expected hard record is missing or not verifiable"
    elif len(expected_results) == len(expected_record_ids) and expected_verdicts == {PASS}:
        run_status = RUN_PASS
        reason = "every expected hard record passed"
    else:
        run_status = RUN_INVALID
        reason = "run status did not match strict profile v1"

    population = manifest["population"]
    return {
        "run_id": run_id,
        "profile_version": manifest["run_policy"]["profile_version"],
        "run_status": run_status,
        "reason": reason,
        "selection_mode": mode,
        "authority_attribution": authority_attribution,
        "population_binding": {
            "status": "verified",
            "population_version": population["population_version"],
            "declared_phase": population["declared_phase"],
            "sealed_before": population["sealed_before"],
            "artifact_locator": population["artifact_locator"],
            "artifact_sha256": population["artifact_sha256"],
            "record_id_field": population["record_id_field"],
            "expected_record_count": len(expected_record_ids),
        },
        "status_histogram": histogram,
        "eligible_expected_coverage": {
            "eligible_record_count": eligible_count,
            "expected_record_count": len(expected_record_ids),
            "fraction": _ratio(eligible_count, len(expected_record_ids)),
        },
        "diagnostic_pass_fraction": {
            "formula": "PASS/(PASS+FAIL)",
            "pass_count": pass_count,
            "pass_plus_fail_count": pass_count + fail_count,
            "value": _ratio(pass_count, pass_count + fail_count),
            "is_conformance_verdict": False,
        },
        "observed_record_count": len(records),
        "observed_expected_record_count": len(expected_set.intersection(actual_set)),
        "missing_record_ids": missing_record_ids,
        "unexpected_record_ids": unexpected_record_ids,
        "duplicate_record_ids": duplicate_record_ids,
        "malformed_record_id_count": malformed_record_id_count,
        "records": record_results,
    }


def evaluate_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = _exact_object(
        load_json(bundle_path),
        {"bundle_version", "cases", "run_cases"},
        "case bundle",
    )
    if (
        bundle["bundle_version"] != "rop-fixture/v1"
        or not isinstance(bundle["cases"], list)
        or not isinstance(bundle["run_cases"], list)
    ):
        raise ContractError("case bundle is invalid")
    root = bundle_path.parent
    results: list[dict[str, Any]] = []
    for case in bundle["cases"]:
        case = _exact_object(
            case,
            {"case_id", "manifest_locator", "envelope_locator", "expected_verdict", "record"},
            "case",
        )
        if case["expected_verdict"] not in VERDICTS:
            raise ContractError("case expected_verdict is invalid")
        manifest_path = root / _locator(case["manifest_locator"], "case.manifest_locator")
        envelope_path = root / _locator(case["envelope_locator"], "case.envelope_locator")
        result = evaluate_record(manifest_path, envelope_path, case["record"])
        result["case_id"] = case["case_id"]
        result["expected_verdict"] = case["expected_verdict"]
        if result["verdict"] != case["expected_verdict"]:
            raise ContractError(
                f"case {case['case_id']} expected {case['expected_verdict']} but observed {result['verdict']}"
            )
        results.append(result)
    counts = {verdict: sum(row["verdict"] == verdict for row in results) for verdict in VERDICTS}
    run_results: list[dict[str, Any]] = []
    for run_case in bundle["run_cases"]:
        run_case = _exact_object(
            run_case,
            {
                "run_case_id",
                "manifest_locator",
                "envelope_locator",
                "expected_run_status",
                "records",
            },
            "run case",
        )
        if run_case["expected_run_status"] not in RUN_STATUSES:
            raise ContractError("run case expected_run_status is invalid")
        manifest_path = root / _locator(
            run_case["manifest_locator"], "run_case.manifest_locator"
        )
        envelope_path = root / _locator(
            run_case["envelope_locator"], "run_case.envelope_locator"
        )
        result = evaluate_run(
            manifest_path,
            envelope_path,
            run_case["run_case_id"],
            run_case["records"],
        )
        result["expected_run_status"] = run_case["expected_run_status"]
        if result["run_status"] != run_case["expected_run_status"]:
            raise ContractError(
                f"run case {run_case['run_case_id']} expected "
                f"{run_case['expected_run_status']} but observed {result['run_status']}"
            )
        run_results.append(result)
    run_counts = {
        status: sum(row["run_status"] == status for row in run_results)
        for status in RUN_STATUSES
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "evaluator_sha256": sha256_file(Path(__file__)),
        "case_bundle_sha256": sha256_file(bundle_path),
        "record_count": len(results),
        "verdict_counts": counts,
        "records": results,
        "run_count": len(run_results),
        "run_status_counts": run_counts,
        "runs": run_results,
    }


def evaluate_kernel_bundle(bundle_path: Path) -> dict[str, Any]:
    """Replay a standalone non-conformance kernel fixture."""

    bundle = _exact_object(
        load_json(bundle_path),
        {
            "bundle_version",
            "resolver_locator",
            "resolver_version",
            "namespace",
            "cases",
        },
        "kernel case bundle",
    )
    if (
        bundle["bundle_version"] != "rop-kernel-fixture/v1"
        or not isinstance(bundle["cases"], list)
    ):
        raise ContractError("kernel case bundle is invalid")
    _string(bundle["resolver_version"], "kernel bundle resolver_version")
    _string(bundle["namespace"], "kernel bundle namespace")
    resolver_path = bundle_path.parent / _locator(
        bundle["resolver_locator"], "kernel bundle resolver_locator"
    )
    _resolver, compiled_resolver = _load_resolver(
        resolver_path,
        {
            "resolver": {"version": bundle["resolver_version"]},
            "selection": {"namespace": bundle["namespace"]},
        },
    )
    rows: list[dict[str, Any]] = []
    for case in bundle["cases"]:
        case = _exact_object(
            case,
            {
                "case_id",
                "selection_type",
                "selected",
                "returned_ids",
                "cutoff",
                "quantifier",
                "expected_verdict",
            },
            "kernel case",
        )
        _string(case["case_id"], "kernel case_id")
        if case["expected_verdict"] not in {
            SELECTOR_ERROR,
            RESOLUTION_ERROR,
            UNVERIFIABLE,
            PASS,
            FAIL,
        }:
            raise ContractError("kernel expected_verdict is invalid")
        row = evaluate_preservation_kernel(
            selected=case["selected"],
            selection_type=case["selection_type"],
            resolver=compiled_resolver,
            returned_ids=case["returned_ids"],
            cutoff=case["cutoff"],
            quantifier=case["quantifier"],
        )
        row["case_id"] = case["case_id"]
        row["expected_verdict"] = case["expected_verdict"]
        if row["verdict"] != case["expected_verdict"]:
            raise ContractError(
                f"kernel case {case['case_id']} expected "
                f"{case['expected_verdict']} but observed {row['verdict']}"
            )
        rows.append(row)
    counts = {
        verdict: sum(row["verdict"] == verdict for row in rows)
        for verdict in (
            SELECTOR_ERROR,
            RESOLUTION_ERROR,
            UNVERIFIABLE,
            PASS,
            FAIL,
        )
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "kernel_version": KERNEL_VERSION,
        "is_conformance_verdict": False,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "kernel_sha256": sha256_file(
            Path(__file__).with_name("returned_object_preservation_kernel.py")
        ),
        "case_bundle_sha256": sha256_file(bundle_path),
        "case_count": len(rows),
        "verdict_counts": counts,
        "cases": rows,
    }
def evaluate_run_document(run_path: Path) -> dict[str, Any]:
    document = _exact_object(
        load_json(run_path),
        {
            "run_version",
            "run_id",
            "manifest_locator",
            "envelope_locator",
            "records",
        },
        "run document",
    )
    if document["run_version"] != "rop-run/v1":
        raise ContractError("unsupported run_version")
    root = run_path.parent
    result = evaluate_run(
        root / _locator(document["manifest_locator"], "run.manifest_locator"),
        root / _locator(document["envelope_locator"], "run.envelope_locator"),
        document["run_id"],
        document["records"],
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "run_input_sha256": sha256_file(run_path),
        "run": result,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--kernel-bundle", type=Path)
    source.add_argument("--run", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.bundle is not None:
        report = evaluate_bundle(args.bundle)
    elif args.kernel_bundle is not None:
        report = evaluate_kernel_bundle(args.kernel_bundle)
    else:
        report = evaluate_run_document(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
