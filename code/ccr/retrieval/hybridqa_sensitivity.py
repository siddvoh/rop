"""Auditable helpers for the frozen HybridQA retriever-sensitivity sweep.

This module deliberately contains no model imports.  The Colab-facing runner
imports GPU packages lazily, while preparation, checkpointing, export, and
verification remain testable on a CPU-only machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


RETRIEVERS = (
    "bge_m3",
    "colbert_v2",
    "hybrid_rrf",
    "hybrid_rrf_bge_reranker_v2_m3",
)
MODELS = ("claude", "openai", "gemini")
REPORT_K = (1, 5, 20)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED_SALT = "ccr-hybridqa-retriever-sensitivity-v1"
HOLM_FAMILY_LABEL = "all_4_retrievers_x_3_models_x_3_recall_cutoffs"
EXPECTED_RANKING_LENGTH = {
    "bge_m3": 100,
    "colbert_v2": 100,
    "hybrid_rrf": 20,
    "hybrid_rrf_bge_reranker_v2_m3": 20,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number}: expected a JSON object")
                yield value


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_queries(
    cohort_path: Path,
    bridge_details_path: Path,
    *,
    expected_size: int = 600,
    expected_models: tuple[str, ...] = MODELS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate frozen identities and return de-duplicated retrieval queries."""
    cohort = json.loads(cohort_path.read_text())
    items = cohort.get("items")
    if not isinstance(items, list) or len(items) != expected_size:
        raise RuntimeError(f"expected {expected_size} cohort items, found {len(items or [])}")
    by_id = {str(item["question_id"]): item for item in items}
    if len(by_id) != expected_size:
        raise RuntimeError("cohort question IDs are not unique")

    bridge: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(bridge_details_path):
        key = (str(row.get("model")), str(row.get("question_id")))
        if key in bridge:
            raise RuntimeError(f"duplicate bridge identity: {key}")
        bridge[key] = row
    expected = {(model, question_id) for model in expected_models for question_id in by_id}
    if set(bridge) != expected:
        missing = sorted(expected - set(bridge))[:3]
        extra = sorted(set(bridge) - expected)[:3]
        raise RuntimeError(f"bridge identity mismatch; missing={missing}, extra={extra}")

    queries: list[dict[str, Any]] = []
    invalid_by_model: dict[str, int] = defaultdict(int)
    for question_id in sorted(by_id):
        item = by_id[question_id]
        queries.append(
            {
                "query_id": f"question::{question_id}",
                "condition": "question_only",
                "question_id": question_id,
                "model": None,
                "text": str(item["question"]),
                "gold_link": str(item["link"]),
            }
        )
        for model in expected_models:
            row = bridge[(model, question_id)]
            valid = row.get("status") == "succeeded" and bool(row.get("primary_predicted_query"))
            if not valid:
                invalid_by_model[model] += 1
            queries.append(
                {
                    "query_id": f"predicted::{model}::{question_id}",
                    "condition": "predicted_fact",
                    "question_id": question_id,
                    "model": model,
                    "text": str(row["primary_predicted_query"]) if valid else None,
                    "gold_link": str(item["link"]),
                    "upstream_status": row.get("status"),
                }
            )
    metadata = {
        "cohort_sha256": sha256_file(cohort_path),
        "bridge_details_sha256": sha256_file(bridge_details_path),
        "cohort_size": expected_size,
        "models": list(expected_models),
        "query_record_count": len(queries),
        "callable_query_count": sum(row["text"] is not None for row in queries),
        "upstream_invalid_by_model": dict(sorted(invalid_by_model.items())),
    }
    return queries, metadata


def prepare_assets(
    cohort_path: Path,
    bridge_details_path: Path,
    bm25_db_path: Path,
    bm25_manifest_path: Path,
    output_dir: Path,
    *,
    expected_size: int = 600,
) -> dict[str, Any]:
    """Materialize immutable queries and a ColBERT collection from the BM25 corpus."""
    for path in (cohort_path, bridge_details_path, bm25_db_path, bm25_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"required frozen input is missing: {path}. See docs/COLAB_RETRIEVER_SWEEP.md."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = output_dir / "prepare_manifest.json"
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text())
        input_checks = {
            "cohort_sha256": cohort_path,
            "bridge_details_sha256": bridge_details_path,
            "bm25_db_sha256": bm25_db_path,
            "bm25_manifest_sha256": bm25_manifest_path,
        }
        output_checks = {
            "queries_sha256": output_dir / "queries.jsonl",
            "collection_sha256": output_dir / "collection.tsv",
            "links_sha256": output_dir / "links.jsonl",
        }
        checks = {**input_checks, **output_checks}
        if all(
            path.is_file() and sha256_file(path) == prior.get(key)
            for key, path in checks.items()
        ):
            return prior
    queries, query_metadata = build_queries(
        cohort_path, bridge_details_path, expected_size=expected_size
    )
    queries_path = output_dir / "queries.jsonl"
    collection_path = output_dir / "collection.tsv"
    links_path = output_dir / "links.jsonl"
    write_jsonl_atomic(queries_path, queries)

    connection = sqlite3.connect(f"file:{bm25_db_path.resolve()}?mode=ro", uri=True)
    rows = connection.execute("SELECT rowid, link, text FROM passage_fts ORDER BY rowid").fetchall()
    connection.close()
    collection_tmp = collection_path.with_name(collection_path.name + ".tmp")
    links_tmp = links_path.with_name(links_path.name + ".tmp")
    with collection_tmp.open("w") as collection, links_tmp.open("w") as links:
        for pid, (_, link, text) in enumerate(rows):
            safe_text = " ".join(str(text).split())
            collection.write(f"{pid}\t{safe_text}\n")
            links.write(json.dumps({"pid": pid, "link": link}, ensure_ascii=False) + "\n")
        collection.flush()
        links.flush()
        os.fsync(collection.fileno())
        os.fsync(links.fileno())
    os.replace(collection_tmp, collection_path)
    os.replace(links_tmp, links_path)

    index_manifest = json.loads(bm25_manifest_path.read_text())
    document_count = int(index_manifest.get("document_count", -1))
    if document_count != len(rows):
        raise RuntimeError(
            f"BM25 manifest/database document mismatch: {document_count} != {len(rows)}"
        )
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        **query_metadata,
        "bm25_db_sha256": sha256_file(bm25_db_path),
        "bm25_manifest_sha256": sha256_file(bm25_manifest_path),
        "bm25_source_inventory_sha256": index_manifest.get("source_inventory_sha256"),
        "document_count": document_count,
        "queries_sha256": sha256_file(queries_path),
        "collection_sha256": sha256_file(collection_path),
        "links_sha256": sha256_file(links_path),
    }
    atomic_json(output_dir / "prepare_manifest.json", manifest)
    return manifest


def reciprocal_rank_fusion(
    rankings: Iterable[list[dict[str, Any]]], *, rrf_k: int = 60, limit: int = 100
) -> list[dict[str, Any]]:
    scores: dict[int, float] = defaultdict(float)
    links: dict[int, str] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for fallback_rank, row in enumerate(ranking, start=1):
            pid = int(row["pid"])
            rank = int(row.get("rank", fallback_rank))
            scores[pid] += 1.0 / (rrf_k + rank)
            links[pid] = str(row["link"])
            best_rank[pid] = min(best_rank.get(pid, rank), rank)
    ordered = sorted(scores, key=lambda pid: (-scores[pid], best_rank[pid], pid))[:limit]
    return [
        {"rank": rank, "pid": pid, "link": links[pid], "score": scores[pid]}
        for rank, pid in enumerate(ordered, start=1)
    ]


def exact_mcnemar_p(question_miss_predicted_hit: int, question_hit_predicted_miss: int) -> float:
    """Two-sided exact McNemar/binomial test for paired binary outcomes."""
    b = int(question_miss_predicted_hit)
    c = int(question_hit_predicted_miss)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return deterministic Holm step-down adjusted p-values."""
    if not p_values:
        return {}
    for key, value in p_values.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid p-value for {key}: {value}")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (family_size - index) * p_values[key]))
        adjusted[key] = running
    return adjusted


def apply_full_family_holm(summaries: dict[str, dict[str, Any]]) -> None:
    """Add adjusted p-values for the single frozen 36-test sensitivity family."""
    raw: dict[str, float] = {}
    for retriever in RETRIEVERS:
        for model in MODELS:
            for k in REPORT_K:
                key = f"{retriever}|{model}|@{k}"
                raw[key] = summaries[retriever]["recall"][model][f"@{k}"][
                    "exact_mcnemar_p_two_sided"
                ]
    expected_size = len(RETRIEVERS) * len(MODELS) * len(REPORT_K)
    if len(raw) != expected_size:
        raise RuntimeError(f"Holm family size mismatch: {len(raw)} != {expected_size}")
    adjusted = holm_adjust(raw)
    for retriever in RETRIEVERS:
        summaries[retriever]["statistics"].update(
            {
                "multiplicity_method": "holm_step_down",
                "multiplicity_family": HOLM_FAMILY_LABEL,
                "multiplicity_family_size": expected_size,
            }
        )
        for model in MODELS:
            for k in REPORT_K:
                key = f"{retriever}|{model}|@{k}"
                metric = summaries[retriever]["recall"][model][f"@{k}"]
                metric["exact_mcnemar_p_holm_adjusted"] = adjusted[key]
                metric["holm_family"] = HOLM_FAMILY_LABEL


def paired_bootstrap_interval(
    differences: list[int], *, samples: int = BOOTSTRAP_SAMPLES, seed: int
) -> list[float]:
    """Deterministic percentile CI for the mean paired recall difference."""
    if not differences:
        raise ValueError("paired bootstrap requires at least one item")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    import numpy as np

    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(samples, dtype=np.float64)
    batch_size = 1_000
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indexes = generator.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indexes].mean(axis=1)
    return [
        float(np.quantile(means, 0.025, method="linear")),
        float(np.quantile(means, 0.975, method="linear")),
    ]


def _bootstrap_seed(retriever: str, model: str, k: int) -> int:
    payload = f"{BOOTSTRAP_SEED_SALT}:{retriever}:{model}:{k}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def compute_retriever_summary(
    rows: list[dict[str, Any]], retriever: str, *, bootstrap_samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    question_rows = {
        row["question_id"]: row for row in rows if row["condition"] == "question_only"
    }
    summary: dict[str, Any] = {
        "retriever": retriever,
        "record_count": len(rows),
        "upstream_invalid_count": sum(row["status"] == "upstream_invalid" for row in rows),
        "upstream_invalid_by_model": {
            model: sum(
                row["status"] == "upstream_invalid" and row.get("model") == model
                for row in rows
            )
            for model in MODELS
        },
        "recall": {},
        "statistics": {
            "paired_bootstrap_samples": bootstrap_samples,
            "confidence_level": 0.95,
            "bootstrap_rng": "numpy.PCG64",
            "bootstrap_seed_salt": BOOTSTRAP_SEED_SALT,
            "paired_test": "two_sided_exact_mcnemar",
        },
    }
    for model in MODELS:
        model_rows = [row for row in rows if row.get("model") == model]
        if len(model_rows) != len(question_rows):
            raise RuntimeError(f"{retriever}/{model}: paired row count mismatch")
        values: dict[str, Any] = {}
        for k in REPORT_K:
            q_hits: list[bool] = []
            p_hits: list[bool] = []
            for predicted in model_rows:
                question = question_rows[predicted["question_id"]]
                q_links = [row["link"] for row in question["ranking"][:k]]
                p_links = [row["link"] for row in predicted["ranking"][:k]]
                q_hits.append(question["gold_link"] in q_links)
                p_hits.append(predicted["gold_link"] in p_links)
            b = sum(not q and p for q, p in zip(q_hits, p_hits))
            c = sum(q and not p for q, p in zip(q_hits, p_hits))
            differences = [int(p) - int(q) for q, p in zip(q_hits, p_hits)]
            seed = _bootstrap_seed(retriever, model, k)
            values[f"@{k}"] = {
                "question_only": sum(q_hits) / len(q_hits),
                "predicted_fact": sum(p_hits) / len(p_hits),
                "paired_difference": sum(differences) / len(differences),
                "paired_bootstrap_95_ci": paired_bootstrap_interval(
                    differences, samples=bootstrap_samples, seed=seed
                ),
                "bootstrap_seed": seed,
                "question_miss_predicted_hit": b,
                "question_hit_predicted_miss": c,
                "exact_mcnemar_p_two_sided": exact_mcnemar_p(b, c),
            }
        summary["recall"][model] = values
    return summary


class ResultStore:
    """A durable, exactly-resumable ranking ledger (one transaction per query)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS rankings(
            retriever TEXT NOT NULL, query_id TEXT NOT NULL, query_sha256 TEXT NOT NULL,
            ranking_json TEXT NOT NULL, PRIMARY KEY(retriever, query_id))"""
        )
        self.connection.commit()

    def get(self, retriever: str, query: dict[str, Any]) -> list[dict[str, Any]] | None:
        digest = hashlib.sha256(canonical_bytes(query)).hexdigest()
        row = self.connection.execute(
            "SELECT query_sha256, ranking_json FROM rankings WHERE retriever=? AND query_id=?",
            (retriever, query["query_id"]),
        ).fetchone()
        if not row:
            return None
        if row[0] != digest:
            raise RuntimeError(f"query changed after checkpoint: {query['query_id']}")
        return json.loads(row[1])

    def put(self, retriever: str, query: dict[str, Any], ranking: list[dict[str, Any]]) -> None:
        digest = hashlib.sha256(canonical_bytes(query)).hexdigest()
        self.connection.execute(
            "INSERT INTO rankings VALUES(?,?,?,?)",
            (retriever, query["query_id"], digest, json.dumps(ranking, sort_keys=True)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def run_resumable(
    store: ResultStore,
    retriever: str,
    queries: Iterable[dict[str, Any]],
    retrieve_one: Callable[[str], list[dict[str, Any]]],
) -> dict[str, int]:
    completed = skipped = upstream_invalid = 0
    for query in queries:
        if query.get("text") is None:
            upstream_invalid += 1
            continue
        if store.get(retriever, query) is not None:
            skipped += 1
            continue
        store.put(retriever, query, retrieve_one(str(query["text"])))
        completed += 1
    return {"completed": completed, "resumed": skipped, "upstream_invalid": upstream_invalid}


def export_results(assets_dir: Path, store_path: Path, output_dir: Path) -> dict[str, Any]:
    queries = list(iter_jsonl(assets_dir / "queries.jsonl"))
    store = ResultStore(store_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for retriever in RETRIEVERS:
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for query in queries:
            ranking = None if query.get("text") is None else store.get(retriever, query)
            if query.get("text") is not None and ranking is None:
                missing.append(str(query["query_id"]))
            status = (
                "upstream_invalid"
                if query.get("text") is None
                else ("succeeded" if ranking is not None else "missing")
            )
            rows.append({**query, "ranking": ranking or [], "status": status})
        if missing:
            raise RuntimeError(
                f"{retriever} has {len(missing)} missing rankings; first={missing[:3]}"
            )
        retrieval_dir = output_dir / retriever
        details_path = retrieval_dir / "details.jsonl"
        write_jsonl_atomic(details_path, rows)
        summary = compute_retriever_summary(rows, retriever)
        summary["details_sha256"] = sha256_file(details_path)
        summaries[retriever] = summary
    apply_full_family_holm(summaries)
    all_summary: dict[str, Any] = {}
    for retriever in RETRIEVERS:
        retrieval_dir = output_dir / retriever
        summary = summaries[retriever]
        summary_path = retrieval_dir / "summary.json"
        atomic_json(summary_path, summary)
        all_summary[retriever] = {
            **summary,
            "summary_sha256": sha256_file(summary_path),
        }
    store.close()
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "assets_manifest_sha256": sha256_file(assets_dir / "prepare_manifest.json"),
        "retrievers": all_summary,
    }
    atomic_json(output_dir / "summary.json", aggregate)
    return aggregate


def verify_export(assets_dir: Path, output_dir: Path) -> dict[str, Any]:
    prepare = json.loads((assets_dir / "prepare_manifest.json").read_text())
    for key, filename in (
        ("queries_sha256", "queries.jsonl"),
        ("collection_sha256", "collection.tsv"),
        ("links_sha256", "links.jsonl"),
    ):
        if sha256_file(assets_dir / filename) != prepare[key]:
            raise RuntimeError(f"prepared {filename} hash mismatch")
    expected_queries = list(iter_jsonl(assets_dir / "queries.jsonl"))
    link_rows = list(iter_jsonl(assets_dir / "links.jsonl"))
    pid_to_link = {int(row["pid"]): str(row["link"]) for row in link_rows}
    if len(pid_to_link) != len(link_rows) or set(pid_to_link) != set(range(len(link_rows))):
        raise RuntimeError("prepared PID/link map is not contiguous and unique")
    expected_records = int(prepare["query_record_count"])
    if len(expected_queries) != expected_records:
        raise RuntimeError("prepared query count mismatch")
    verified: dict[str, Any] = {}
    recomputed_summaries: dict[str, dict[str, Any]] = {}
    observed_summaries: dict[str, dict[str, Any]] = {}
    for retriever in RETRIEVERS:
        directory = output_dir / retriever
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        details = list(iter_jsonl(directory / "details.jsonl"))
        if len(details) != expected_records:
            raise RuntimeError(f"{retriever}: record count mismatch")
        expected_length = min(EXPECTED_RANKING_LENGTH[retriever], len(pid_to_link))
        for index, (query, row) in enumerate(zip(expected_queries, details)):
            allowed_keys = set(query) | {"ranking", "status"}
            if set(row) != allowed_keys:
                raise RuntimeError(f"{retriever}: unexpected detail fields at row {index}")
            if {key: row[key] for key in query} != query:
                raise RuntimeError(f"{retriever}: frozen query content mismatch at row {index}")
            expected_status = "upstream_invalid" if query.get("text") is None else "succeeded"
            if row["status"] != expected_status:
                raise RuntimeError(f"{retriever}: status mismatch at row {index}")
            ranking = row["ranking"]
            if not isinstance(ranking, list):
                raise RuntimeError(f"{retriever}: ranking is not a list at row {index}")
            if expected_status == "upstream_invalid":
                if ranking:
                    raise RuntimeError(f"{retriever}: invalid upstream row has a ranking")
                continue
            if len(ranking) != expected_length:
                raise RuntimeError(f"{retriever}: ranking length mismatch at row {index}")
            seen_pids = set()
            prior_score = math.inf
            for expected_rank, item in enumerate(ranking, start=1):
                required = {"rank", "pid", "link", "score"}
                reranker_fields = {"rrf_score"} if retriever.endswith("reranker_v2_m3") else set()
                allowed = required | reranker_fields
                if set(item) != allowed:
                    raise RuntimeError(f"{retriever}: ranking fields mismatch at row {index}")
                pid = item["pid"]
                score = item["score"]
                if type(pid) is not int or pid not in pid_to_link or pid in seen_pids:
                    raise RuntimeError(f"{retriever}: invalid or duplicate PID at row {index}")
                if item["rank"] != expected_rank or item["link"] != pid_to_link[pid]:
                    raise RuntimeError(f"{retriever}: rank/link mismatch at row {index}")
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise RuntimeError(f"{retriever}: non-finite score at row {index}")
                if float(score) > prior_score + 1e-12:
                    raise RuntimeError(f"{retriever}: ranking scores are not descending")
                if "rrf_score" in item and (
                    not isinstance(item["rrf_score"], (int, float))
                    or not math.isfinite(item["rrf_score"])
                ):
                    raise RuntimeError(f"{retriever}: invalid RRF score at row {index}")
                prior_score = float(score)
                seen_pids.add(pid)
        if sha256_file(directory / "details.jsonl") != summary["details_sha256"]:
            raise RuntimeError(f"{retriever}: details hash mismatch")
        recomputed = compute_retriever_summary(details, retriever)
        recomputed["details_sha256"] = sha256_file(directory / "details.jsonl")
        recomputed_summaries[retriever] = recomputed
        observed_summaries[retriever] = summary
    apply_full_family_holm(recomputed_summaries)
    reconstructed: dict[str, Any] = {}
    for retriever in RETRIEVERS:
        summary = observed_summaries[retriever]
        recomputed = recomputed_summaries[retriever]
        if summary != recomputed:
            raise RuntimeError(f"{retriever}: summary does not match independent recomputation")
        summary_path = output_dir / retriever / "summary.json"
        summary_hash = sha256_file(summary_path)
        reconstructed[retriever] = {**recomputed, "summary_sha256": summary_hash}
        verified[retriever] = {
            "record_count": recomputed["record_count"],
            "details_sha256": recomputed["details_sha256"],
            "summary_sha256": summary_hash,
        }
    expected_aggregate = {
        "schema_version": 1,
        "status": "complete",
        "assets_manifest_sha256": sha256_file(assets_dir / "prepare_manifest.json"),
        "retrievers": reconstructed,
    }
    aggregate_path = output_dir / "summary.json"
    if json.loads(aggregate_path.read_text()) != expected_aggregate:
        raise RuntimeError("aggregate summary does not match independent reconstruction")
    result = {
        "status": "verified",
        "aggregate_summary_sha256": sha256_file(aggregate_path),
        "retrievers": verified,
    }
    atomic_json(output_dir / "verification.json", result)
    return result
