"""Run and resume the frozen question-only versus gold-fact BM25 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from ccr.retrieval.bm25 import INDEX_VERSION, canonical_bytes, open_index, retrieve

RUN_VERSION = "hybridqa-bm25-baseline-v1"
CONDITIONS = {"question_only": "question", "gold_fact": "F"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_or_create_run_manifest(
    run_dir: Path,
    cohort_path: Path,
    index_manifest_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text())
    index_manifest = json.loads(index_manifest_path.read_text())
    if index_manifest.get("index_version") != INDEX_VERSION:
        raise RuntimeError("the index manifest does not match this runner")
    manifest = {
        "run_version": RUN_VERSION,
        "experiment_id": cohort["experiment_id"],
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort_file_sha256": sha256_file(cohort_path),
        "config_file_sha256": sha256_file(config_path),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "index_version": INDEX_VERSION,
        "conditions": list(CONDITIONS),
        "report_k": [1, 5, 20],
        "item_count": len(cohort["items"]),
    }
    path = run_dir / "run_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("existing run manifest differs; use a new run directory")
    _write_json_atomic(path, manifest)
    return cohort


def _gold_exists(connection: sqlite3.Connection, link: str) -> bool:
    return connection.execute("SELECT 1 FROM links WHERE link=?", (link,)).fetchone() is not None


def run(
    cohort_path: Path,
    db_path: Path,
    index_manifest_path: Path,
    config_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    cohort = _validate_or_create_run_manifest(
        run_dir, cohort_path, index_manifest_path, config_path
    )
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    connection = open_index(db_path)
    for item in cohort["items"]:
        output_path = items_dir / f"{item['question_id']}.json"
        if output_path.exists():
            existing = json.loads(output_path.read_text())
            if existing.get("question_id") != item["question_id"]:
                raise RuntimeError(f"corrupt item output: {output_path}")
            continue
        result: dict[str, Any] = {
            "question_id": item["question_id"],
            "table_id": item["table_id"],
            "gold_link": item["link"],
            "gold_link_in_corpus": _gold_exists(connection, item["link"]),
            "conditions": {},
        }
        for condition, source_field in CONDITIONS.items():
            query = item[source_field]
            result["conditions"][condition] = {
                "query": query,
                "ranking": retrieve(connection, query, 20),
            }
        _write_json_atomic(output_path, result)
    connection.close()

    outputs = [
        json.loads((items_dir / f"{item['question_id']}.json").read_text())
        for item in cohort["items"]
    ]
    recall = {condition: {str(k): 0 for k in (1, 5, 20)} for condition in CONDITIONS}
    paired = {
        "question_miss_gold_fact_hit_at_5": 0,
        "question_hit_gold_fact_miss_at_5": 0,
        "both_hit_at_5": 0,
        "both_miss_at_5": 0,
    }
    for output in outputs:
        hits: dict[str, bool] = {}
        for condition in CONDITIONS:
            links = [row["link"] for row in output["conditions"][condition]["ranking"]]
            for k in (1, 5, 20):
                recall[condition][str(k)] += int(output["gold_link"] in links[:k])
            hits[condition] = output["gold_link"] in links[:5]
        q_hit, f_hit = hits["question_only"], hits["gold_fact"]
        if q_hit and f_hit:
            paired["both_hit_at_5"] += 1
        elif q_hit:
            paired["question_hit_gold_fact_miss_at_5"] += 1
        elif f_hit:
            paired["question_miss_gold_fact_hit_at_5"] += 1
        else:
            paired["both_miss_at_5"] += 1
    n = len(outputs)
    summary = {
        "run_version": RUN_VERSION,
        "status": "complete",
        "item_count": n,
        "gold_links_missing_from_corpus": sum(
            not output["gold_link_in_corpus"] for output in outputs
        ),
        "recall": {
            condition: {f"@{k}": count / n for k, count in counts.items()}
            for condition, counts in recall.items()
        },
        "paired_counts": paired,
        "result_set_sha256": hashlib.sha256(canonical_bytes(outputs)).hexdigest(),
    }
    _write_json_atomic(run_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.cohort, args.db, args.index_manifest, args.config, args.run_dir
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
