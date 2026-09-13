"""Offline verification of the released HybridQA result bundles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from ccr.paths import repository_root
from ccr.retrieval.hybridqa_sensitivity import (
    EXPECTED_RANKING_LENGTH,
    RETRIEVERS,
    apply_full_family_holm,
    build_queries,
    compute_retriever_summary,
)


ROOT = repository_root()
SENSITIVITY_ROOT = Path(
    "data/results/released/hybridqa_natural_v3_retriever_sensitivity_v1"
)


class VerificationError(RuntimeError):
    """A released result is missing, malformed, or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise VerificationError(f"invalid UTF-8 JSONL: {path}: {exc}") from exc
    if not lines:
        raise VerificationError(f"empty JSONL artifact: {path}")
    rows: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise VerificationError(f"blank JSONL record: {path}:{line_number}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"invalid JSONL: {path}:{line_number}: {exc}"
            ) from exc
    return rows


def _verify_result_bundles(root: Path, parsed: dict[Path, Any]) -> int:
    released = root / "data/results/released"
    bundle_dirs = sorted(path.parent for path in released.rglob("verification.json"))
    top_level_dirs = {
        path
        for experiment in released.iterdir()
        if experiment.is_dir()
        for path in experiment.iterdir()
        if path.is_dir()
    }
    if set(bundle_dirs) != top_level_dirs:
        absent = sorted(
            str(path.relative_to(root)) for path in top_level_dirs - set(bundle_dirs)
        )
        raise VerificationError(
            f"released result bundle lacks verification.json: {absent}"
        )

    for bundle in bundle_dirs:
        relative_bundle = bundle.relative_to(root)
        required = ("summary.json", "run_manifest.json", "verification.json")
        if any(not (bundle / name).is_file() for name in required):
            raise VerificationError(
                f"incomplete released result bundle: {relative_bundle}"
            )
        verification = parsed[relative_bundle / "verification.json"]
        run_manifest = parsed[relative_bundle / "run_manifest.json"]
        summary = parsed[relative_bundle / "summary.json"]
        if (
            not isinstance(verification, dict)
            or verification.get("status") != "independently_verified"
        ):
            raise VerificationError(
                f"bundle is not independently verified: {relative_bundle}"
            )
        if not isinstance(run_manifest, dict) or not isinstance(summary, dict):
            raise VerificationError(
                f"bundle metadata must be JSON objects: {relative_bundle}"
            )

        hash_values = [
            value
            for document in (verification, run_manifest, summary)
            if (value := document.get("result_set_sha256")) is not None
        ]
        if any(not isinstance(value, str) for value in hash_values):
            raise VerificationError(
                f"result-set hash is not a string: {relative_bundle}"
            )
        hashes = set(hash_values)
        if len(hashes) != 1:
            raise VerificationError(
                f"result-set hashes disagree or are absent: {relative_bundle}"
            )
        expected_hash = next(iter(hashes))
        detail_paths = [
            path
            for path in (bundle / "items.jsonl", bundle / "details.jsonl")
            if path.is_file()
        ]
        if len(detail_paths) != 1:
            raise VerificationError(
                f"bundle must have exactly one primary JSONL result: {relative_bundle}"
            )
        detail_relative = detail_paths[0].relative_to(root)
        rows = parsed[detail_relative]
        if not isinstance(expected_hash, str) or canonical_hash(rows) != expected_hash:
            raise VerificationError(
                f"canonical result-set hash mismatch: {relative_bundle}"
            )

        cohort_hash = run_manifest.get("cohort_file_sha256")
        experiment_id = run_manifest.get("experiment_id")
        if cohort_hash is not None and isinstance(experiment_id, str):
            cohort = root / "data/released" / experiment_id / "confirmatory_cohort.json"
            if not cohort.is_file() or sha256_file(cohort) != cohort_hash:
                raise VerificationError(f"cohort file hash mismatch: {relative_bundle}")

        bridge_hash = run_manifest.get("bridge_details_sha256")
        if bridge_hash is not None:
            bridge = (
                root
                / "data/results/released/hybridqa_natural_v2/"
                "bridge_confirmatory/details.jsonl"
            )
            if not bridge.is_file() or sha256_file(bridge) != bridge_hash:
                raise VerificationError(
                    f"bridge dependency hash mismatch: {relative_bundle}"
                )
    return len(bundle_dirs)


def _verify_retriever_sensitivity(root: Path) -> dict[str, int]:
    base = root / SENSITIVITY_ROOT
    prepare = _read_json(base / "prepare_manifest.json")
    queries = _read_jsonl(base / "queries.jsonl")
    links = _read_jsonl(base / "links.jsonl")
    expected_queries, metadata = build_queries(
        root / "data/released/hybridqa_natural_v3/confirmatory_cohort.json",
        root
        / "data/results/released/hybridqa_natural_v2/"
        "bridge_confirmatory/details.jsonl",
    )
    if queries != expected_queries:
        raise VerificationError("retriever sensitivity queries differ from frozen inputs")
    for key, expected in metadata.items():
        if prepare.get(key) != expected:
            raise VerificationError(
                f"retriever sensitivity query metadata mismatch: {key}"
            )
    if (
        sha256_file(base / "queries.jsonl") != prepare.get("queries_sha256")
        or sha256_file(base / "links.jsonl") != prepare.get("links_sha256")
        or len(links) != prepare.get("document_count")
    ):
        raise VerificationError("retriever sensitivity prepared asset binding failed")

    pid_to_link: dict[int, str] = {}
    for expected_pid, item in enumerate(links):
        if set(item) != {"pid", "link"} or item.get("pid") != expected_pid:
            raise VerificationError("retriever sensitivity passage map is not contiguous")
        link = item.get("link")
        if not isinstance(link, str):
            raise VerificationError(
                "retriever sensitivity passage link is not a string"
            )
        pid_to_link[expected_pid] = link

    recomputed: dict[str, dict[str, Any]] = {}
    observed: dict[str, dict[str, Any]] = {}
    for retriever in RETRIEVERS:
        directory = base / retriever
        rows = _read_jsonl(directory / "details.jsonl")
        if len(rows) != len(queries):
            raise VerificationError(f"{retriever}: sensitivity row count mismatch")
        expected_length = EXPECTED_RANKING_LENGTH[retriever]
        for index, (query, row) in enumerate(zip(queries, rows)):
            allowed_keys = set(query) | {"ranking", "status"}
            if set(row) != allowed_keys or {key: row[key] for key in query} != query:
                raise VerificationError(f"{retriever}: frozen query mismatch at row {index}")
            expected_status = (
                "upstream_invalid" if query.get("text") is None else "succeeded"
            )
            ranking = row.get("ranking")
            if row.get("status") != expected_status or not isinstance(ranking, list):
                raise VerificationError(f"{retriever}: status/ranking mismatch at row {index}")
            if expected_status == "upstream_invalid":
                if ranking:
                    raise VerificationError(f"{retriever}: invalid query has rankings")
                continue
            if len(ranking) != expected_length:
                raise VerificationError(f"{retriever}: ranking length mismatch at row {index}")
            seen: set[int] = set()
            prior_score = math.inf
            for expected_rank, item in enumerate(ranking, start=1):
                required = {"rank", "pid", "link", "score"}
                optional = (
                    {"rrf_score"}
                    if retriever.endswith("reranker_v2_m3")
                    else set()
                )
                pid = item.get("pid")
                score = item.get("score")
                if (
                    set(item) != required | optional
                    or type(pid) is not int
                    or pid not in pid_to_link
                    or pid in seen
                    or item.get("rank") != expected_rank
                    or item.get("link") != pid_to_link[pid]
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or float(score) > prior_score + 1e-12
                ):
                    raise VerificationError(
                        f"{retriever}: invalid ranking at row {index}"
                    )
                if "rrf_score" in item and (
                    not isinstance(item["rrf_score"], (int, float))
                    or not math.isfinite(float(item["rrf_score"]))
                ):
                    raise VerificationError(
                        f"{retriever}: invalid RRF score at row {index}"
                    )
                seen.add(pid)
                prior_score = float(score)
        calculated = compute_retriever_summary(rows, retriever)
        calculated["details_sha256"] = sha256_file(directory / "details.jsonl")
        recomputed[retriever] = calculated
        observed[retriever] = _read_json(directory / "summary.json")
    apply_full_family_holm(recomputed)
    if any(observed[name] != recomputed[name] for name in RETRIEVERS):
        raise VerificationError("retriever sensitivity statistics do not recompute")

    reconstructed = {
        retriever: {
            **recomputed[retriever],
            "summary_sha256": sha256_file(base / retriever / "summary.json"),
        }
        for retriever in RETRIEVERS
    }
    expected_aggregate = {
        "schema_version": 1,
        "status": "complete",
        "assets_manifest_sha256": sha256_file(base / "prepare_manifest.json"),
        "retrievers": reconstructed,
    }
    aggregate_path = base / "aggregate_summary.json"
    if _read_json(aggregate_path) != expected_aggregate:
        raise VerificationError("retriever sensitivity aggregate summary mismatch")
    expected_verification = {
        "status": "verified",
        "aggregate_summary_sha256": sha256_file(aggregate_path),
        "retrievers": {
            retriever: {
                "record_count": recomputed[retriever]["record_count"],
                "details_sha256": recomputed[retriever]["details_sha256"],
                "summary_sha256": sha256_file(base / retriever / "summary.json"),
            }
            for retriever in RETRIEVERS
        },
    }
    if _read_json(base / "aggregate_verification.json") != expected_verification:
        raise VerificationError(
            "retriever sensitivity aggregate verification mismatch"
        )
    return {"retrievers": len(RETRIEVERS), "records": len(RETRIEVERS) * len(queries)}


def verify(root: Path = ROOT) -> dict[str, int | bool]:
    """Verify the retained released-result bundles and retriever sweep."""

    root = root.resolve()
    released = root / "data/results/released"
    if not released.is_dir() or released.is_symlink():
        raise VerificationError("released-result directory is missing or unsafe")
    parsed: dict[Path, Any] = {}
    for path in sorted(item for item in released.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if path.suffix == ".json":
            parsed[relative] = _read_json(path)
        elif path.suffix == ".jsonl":
            parsed[relative] = _read_jsonl(path)
    bundles = _verify_result_bundles(root, parsed)
    sensitivity = _verify_retriever_sensitivity(root)
    return {
        "ok": True,
        "released_result_bundles": bundles,
        "retriever_sensitivity_conditions": sensitivity["retrievers"],
        "retriever_sensitivity_records": sensitivity["records"],
    }


def main() -> int:
    try:
        report = verify()
    except VerificationError as exc:
        print(json.dumps({"ok": False, "failure": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
