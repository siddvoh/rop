#!/usr/bin/env python3
"""Minimal returned-object preservation calculation.

This module resolves an already supplied selected identity and checks whether
the corresponding object occurs in returned IDs at a cutoff.  It does not
authenticate declarations, bind populations, establish chronology, or verify
asset bytes.  Every result therefore carries ``is_conformance_verdict=false``.
Strict record/run conformance is implemented by
``evaluate_returned_object_preservation.py``, which calls this kernel only
after its full prospective checks pass.
"""

from __future__ import annotations

from typing import Any


KERNEL_VERSION = "rop-preservation-kernel/v1"
SELECTOR_ERROR = "SELECTOR_ERROR"
RESOLUTION_ERROR = "RESOLUTION_ERROR"
UNVERIFIABLE = "UNVERIFIABLE"
PASS = "PASS"
FAIL = "FAIL"
SELECTION_TYPES = frozenset(
    {
        "corpus_key",
        "corpus_uri",
        "composite_corpus_key",
        "corpus_key_set",
    }
)


class KernelConfigurationError(RuntimeError):
    """The caller supplied an invalid kernel configuration."""


class CompiledResolver:
    """Validated resolver indexes shared across record evaluations.

    The source resolver remains available for version/namespace reporting,
    while every accepted identity and each canonical member set is indexed
    once.  Callers should treat the contained mappings as read-only.
    """

    __slots__ = ("source", "identity_owner", "member_sets")

    def __init__(
        self,
        *,
        source: dict[str, Any],
        identity_owner: dict[str, str],
        member_sets: dict[str, frozenset[str]],
    ) -> None:
        self.source = source
        self.identity_owner = identity_owner
        self.member_sets = member_sets


def compile_resolver(resolver: Any) -> CompiledResolver:
    """Validate and index one resolver in time linear in its identities."""

    if isinstance(resolver, CompiledResolver):
        return resolver
    if (
        not isinstance(resolver, dict)
        or set(resolver) != {"resolver_version", "namespace", "aliases", "members"}
        or not isinstance(resolver.get("aliases"), dict)
        or not resolver["aliases"]
        or not isinstance(resolver.get("members"), dict)
        or not resolver["members"]
    ):
        raise KernelConfigurationError("kernel resolver is malformed")
    seen_members: set[str] = set()
    identity_owner = {
        object_id: object_id for object_id in resolver["members"]
    }
    for object_id, object_members in resolver["members"].items():
        if (
            not isinstance(object_id, str)
            or not object_id
            or not isinstance(object_members, list)
            or not object_members
            or any(not isinstance(member, str) or not member for member in object_members)
            or len(object_members) != len(set(object_members))
            or seen_members.intersection(object_members)
        ):
            raise KernelConfigurationError("kernel resolver member sets are malformed")
        for member in object_members:
            if member in identity_owner and identity_owner[member] != object_id:
                raise KernelConfigurationError(
                    "kernel resolver identities have conflicting owners"
                )
            identity_owner[member] = object_id
        seen_members.update(object_members)
    for alias, object_id in resolver["aliases"].items():
        if (
            not isinstance(alias, str)
            or not alias
            or object_id not in resolver["members"]
            or (
                alias in identity_owner
                and identity_owner[alias] != object_id
            )
        ):
            raise KernelConfigurationError("kernel resolver aliases are malformed")
        identity_owner[alias] = object_id
    return CompiledResolver(
        source=resolver,
        identity_owner=identity_owner,
        member_sets={
            object_id: frozenset(object_members)
            for object_id, object_members in resolver["members"].items()
        },
    )


def _resolve_selected(
    value: Any, selection_type: str, resolver: CompiledResolver
) -> list[str]:
    values = value if selection_type == "corpus_key_set" else [value]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
    ):
        raise ValueError("selected key has the wrong type")
    resolved: list[str] = []
    seen: set[str] = set()
    for item in values:
        object_id = resolver.identity_owner.get(item)
        if object_id is None:
            raise LookupError(item)
        if object_id not in seen:
            resolved.append(object_id)
            seen.add(object_id)
    return resolved


def resolve_selected_objects(
    value: Any,
    selection_type: str,
    resolver: CompiledResolver,
) -> list[str]:
    """Resolve selected identities against an already compiled resolver."""

    if selection_type not in SELECTION_TYPES:
        raise KernelConfigurationError("kernel selection_type is unsupported")
    return _resolve_selected(value, selection_type, compile_resolver(resolver))


def evaluate_preservation_kernel(
    *,
    selected: Any,
    selection_type: str,
    resolver: dict[str, Any] | CompiledResolver,
    returned_ids: Any,
    cutoff: int,
    quantifier: str,
) -> dict[str, Any]:
    """Resolve one selection and test returned-set membership.

    Data errors receive the same record-level labels used by the strict
    wrapper. Configuration errors raise ``KernelConfigurationError``. Neither
    kind of output is a conformance verdict without the prospective wrapper.
    """

    if selection_type not in SELECTION_TYPES:
        raise KernelConfigurationError("kernel selection_type is unsupported")
    if quantifier not in {"any", "all"}:
        raise KernelConfigurationError("kernel quantifier is unsupported")
    if type(cutoff) is not int or cutoff < 1:
        raise KernelConfigurationError("kernel cutoff is invalid")
    compiled = compile_resolver(resolver)

    try:
        selected_objects = _resolve_selected(selected, selection_type, compiled)
    except (KeyError, TypeError, ValueError):
        verdict = SELECTOR_ERROR
        reason = "selected key is missing or malformed"
    except LookupError:
        verdict = RESOLUTION_ERROR
        reason = "selected key is not resolved by the supplied resolver"
    else:
        if (
            not isinstance(returned_ids, list)
            or any(not isinstance(item, str) or not item for item in returned_ids)
        ):
            verdict = UNVERIFIABLE
            reason = "returned-ID input is missing or malformed"
        else:
            returned_at_k = returned_ids[:cutoff]
            if quantifier == "any":
                returned_objects = {
                    object_id
                    for item in returned_at_k
                    if (object_id := compiled.identity_owner.get(item)) is not None
                }
                satisfied = bool(set(selected_objects).intersection(returned_objects))
                reason = (
                    "selected canonical object present"
                    if satisfied
                    else "selected canonical object absent"
                )
            else:
                expected_member_sets = [
                    compiled.member_sets[object_id]
                    for object_id in selected_objects
                ]
                returned_set = set(returned_at_k)
                satisfied = any(
                    expected_members.issubset(returned_set)
                    for expected_members in expected_member_sets
                )
                reason = (
                    "all resolver members present for an acceptable object"
                    if satisfied
                    else "no acceptable object has all resolver members present"
                )
            verdict = PASS if satisfied else FAIL
    return {
        "kernel_version": KERNEL_VERSION,
        "is_conformance_verdict": False,
        "verdict": verdict,
        "reason": reason,
    }
