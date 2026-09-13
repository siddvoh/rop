#!/usr/bin/env python3
"""Tiny deterministic component used by the prospective contract fixture."""

from __future__ import annotations


RANKINGS = {
    "pass_alias": ["noise:0", "alpha-alias", "doc:beta"],
    "fail_other_object": ["doc:beta", "noise:1", "noise:2"],
    "all_members_pass": ["chunk:gamma:0", "noise:0", "chunk:gamma:1"],
    "all_members_fail": ["chunk:gamma:0", "doc:beta", "noise:0"],
}


def rank(case_id: str, cutoff: int) -> list[str]:
    """Return the fixed ranked identifiers for a named fixture case."""

    if type(cutoff) is not int or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    return list(RANKINGS[case_id][:cutoff])
