"""Freeze the natural HybridQA confirmatory cohort deterministically."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

from ccr.paths import repository_root

import yaml

ROOT = repository_root()
CONFIG = ROOT / "configs" / "hybridqa_natural_v1.yaml"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def freeze(config_path: Path = CONFIG, *, write: bool = True) -> dict:
    raw_config = config_path.read_bytes()
    config = yaml.safe_load(raw_config)
    dataset = config["dataset"]
    source_path = ROOT / dataset["source_candidates"]
    candidates = json.loads(source_path.read_text())
    source_hash = sha256_bytes(source_path.read_bytes())
    salt = dataset["selection_salt"]
    ranked = sorted(
        candidates,
        key=lambda item: sha256_bytes(f"{salt}:{item['question_id']}".encode()),
    )

    selected = []
    per_table: dict[str, int] = {}
    for item in ranked:
        table_id = item["table_id"]
        if per_table.get(table_id, 0) >= int(dataset["max_questions_per_table"]):
            continue
        selection_hash = sha256_bytes(f"{salt}:{item['question_id']}".encode())
        selected.append({**item, "selection_hash": selection_hash})
        per_table[table_id] = per_table.get(table_id, 0) + 1
        if len(selected) == int(dataset["confirmatory_size"]):
            break

    if len(selected) != int(dataset["confirmatory_size"]):
        raise RuntimeError(f"only {len(selected)} eligible questions after table cap")

    output = {
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "source_path": dataset["source_candidates"],
        "source_sha256": source_hash,
        "config_sha256": sha256_bytes(raw_config),
        "selection_salt": salt,
        "selection_rule": "ascending SHA256(salt:question_id), max one question per table",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "unique_table_count": len(per_table),
        "items": selected,
    }
    output["cohort_sha256"] = sha256_bytes(canonical_bytes(output["items"]))
    if write:
        destination = ROOT / dataset["frozen_output"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: output[key] for key in (
        "source_sha256", "config_sha256", "cohort_sha256", "selected_count",
        "unique_table_count")}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    freeze(parser.parse_args().config)
