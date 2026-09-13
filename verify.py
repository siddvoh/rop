#!/usr/bin/env python3
"""Verify the paper-scoped ROP artifact without network or credentials."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import replay


def _load_release_verifier() -> Any:
    code_root = ROOT / "code"
    for path in (str(code_root),):
        if path not in sys.path:
            sys.path.insert(0, path)
    importlib.invalidate_caches()
    return importlib.import_module("experiments.verify_released_artifacts")


def verify_hybridqa_releases() -> dict[str, int]:
    """Verify only the released HybridQA families retained in this projection."""

    module = _load_release_verifier()
    released = ROOT / "data/results/released"
    parsed: dict[Path, Any] = {}
    for path in sorted(item for item in released.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT)
        if path.suffix == ".json":
            parsed[relative] = module._read_json(path)
        elif path.suffix == ".jsonl":
            parsed[relative] = module._read_jsonl(path)
    bundles = module._verify_result_bundles(ROOT, parsed)
    sensitivity = module._verify_retriever_sensitivity(ROOT)
    return {
        "result_bundles": bundles,
        "retriever_conditions": sensitivity["retrievers"],
        "retriever_records": sensitivity["records"],
    }


def verify_gemini_passage_ablation() -> dict[str, Any]:
    """Replay the fresh OpenRouter-BYOK passage intervention offline."""

    path = ROOT / "code/analysis/recompute_gemini_passage_ablation.py"
    specification = importlib.util.spec_from_file_location(
        "gemini_passage_ablation_replay", path
    )
    if specification is None or specification.loader is None:
        raise replay.VerificationError(
            "cannot load Gemini passage-ablation replay"
        )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    try:
        return module.verify(ROOT / "data/evidence" / module.RELEASE_ID)
    except module.VerificationError as error:
        raise replay.VerificationError(
            f"Gemini passage-ablation replay failed: {error}"
        ) from error


def require_complete_coverage(replayed: list[str], bound_reports: list[str]) -> None:
    seen = set(replayed)
    if len(seen) != len(replayed):
        raise replay.VerificationError("a replay was reported more than once")
    required = replay.REQUIRED_ADDED_AUDIT_REPLAYS | replay.AUTHORITY_REQUIRED_ADDED_AUDIT_REPLAYS
    absent = sorted(required - seen)
    if absent:
        raise replay.VerificationError(f"replays did not run: {absent}")
    if set(bound_reports) != replay.REQUIRED_BOUND_AUDIT_REPORTS:
        raise replay.VerificationError("bound report set changed")


def verify_paper_binding_agrees() -> None:
    binding = replay.load_json(ROOT / "data/manifests/PAPER_BINDING.json")
    provenance = replay.load_json(ROOT / "data/manifests/ARTIFACT_PROVENANCE.json")
    recorded = provenance.get("paper_binding")
    for field in ("paper_pdf", "paper_source"):
        if not isinstance(recorded, dict) or recorded.get(field) != binding.get(field):
            raise replay.VerificationError(
                f"ARTIFACT_PROVENANCE.json disagrees with PAPER_BINDING.json on {field}"
            )


def verify_scope() -> dict[str, Any]:
    scope = replay.load_json(ROOT / "data/manifests/FOCUSED_SCOPE.json")
    required = {
        "status": "rop_paper_scoped",
        "evidence_collection_used_model_or_api_calls": True,
    }
    if not isinstance(scope, dict) or any(
        scope.get(key) != value for key, value in required.items()
    ):
        raise replay.VerificationError("paper-scope declaration changed")
    forbidden_prefixes = (
        "data/released/cryoet/",
        "data/released/finqa_",
        "data/released/tbd",
        "data/results/released/controlled/",
        "data/released/hybridqa_pilot/",
    )
    present = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(ROOT).as_posix().startswith(forbidden_prefixes)
    ]
    if present:
        raise replay.VerificationError(f"out-of-scope data is present: {present}")
    return {
        "status": scope["status"],
        "evidence_collection_used_model_or_api_calls": True,
        "verification_uses_model_or_api_calls": False,
        "paper_included": scope.get("paper_included"),
        "paper_binding": scope.get("paper_binding"),
        "paper_title": scope.get("paper_title"),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--paper-pdf", type=Path)
    result.add_argument("--paper-source", type=Path)
    result.add_argument(
        "--manifest-only",
        action="store_true",
        help="verify inventory, privacy, scope, provenance, and paper binding only",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = replay.verify_manifest()
        privacy = replay.verify_no_private_material()
        verify_paper_binding_agrees()
        scope = verify_scope()
        primary_hosted_ledger = replay.verify_primary_hosted_ledger_audit()
        if replay.sha256_file(replay.RETRIEVER_CONFIG) != replay.RETRIEVER_CONFIG_SHA256:
            raise replay.VerificationError("executed retriever config SHA-256 changed")
        paper = replay.verify_paper(args.paper_pdf, args.paper_source)
        replayed: list[str] = []
        bound_reports: list[str] = []
        hosted_call_archive: dict[str, Any] | None = None
        releases: dict[str, int] | None = None
        gemini_passage_ablation: dict[str, Any] | None = None
        if not args.manifest_only:
            with tempfile.TemporaryDirectory(prefix="rop-artifact-verify-") as name:
                directory = Path(name)
                hosted_call_archive = replay.replay_primary_hosted_call_archive(directory)
                replayed.append("primary_hosted_call_archive")
                replayed.append(replay.replay_delivery_stack_audit())
                replayed.append(replay.replay_bm25_omission_taxonomy())
                replayed.append(replay.replay_mismatch_overlap_analysis())
                replayed.append(replay.replay_mismatch_stack_cutoffs())
                replayed.append(replay.replay_lossy_query_target_divergence())
                replayed.append(replay.replay_full_cohort_stack_decisions())
                replayed.append(replay.replay_mismatch_ordering_robustness())
                replayed.append(replay.replay_mismatch_preservation_gap_uncertainty())
                replayed.append(replay.replay_mismatch_metric_inversion_uncertainty())
                replayed.append(replay.replay_answer_associations())
                bound_reports.append(replay.verify_identifier_aware_executor_report_binding())
                bound_reports.append(replay.verify_hybridqa_bm25_reconstruction_contract())
                bound_reports.append(replay.verify_returned_object_preservation_contract())
                bound_reports.append(replay.verify_returned_object_preservation_evaluator())
                replayed.append(replay.replay_transfer_scope_audit())
                replayed.append(replay.replay_qwen_decoded_id_replication())
                replayed.append(replay.replay_sensitivity())
                replayed.extend(replay.replay_authority(directory))
                releases = verify_hybridqa_releases()
                replayed.append("hybridqa_result_bundles")
                gemini_passage_ablation = verify_gemini_passage_ablation()
                replayed.append("gemini_passage_ablation_openrouter_byok")
            require_complete_coverage(replayed, bound_reports)
        report = {
            "schema_version": 1,
            "status": "verified",
            "scope": scope,
            "manifest": manifest,
            "privacy": privacy,
            "paper": paper,
            "primary_hosted_call_ledger": primary_hosted_ledger,
            "primary_hosted_call_archive": hosted_call_archive,
            "hybridqa_releases": releases,
            "gemini_passage_ablation": gemini_passage_ablation,
            "replayed": replayed,
            "bound_reports": bound_reports,
            "verification_used_model_or_api_calls": False,
            "credentials_used": False,
            "network_used": False,
        }
    except (
        replay.VerificationError,
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ROP artifact verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
