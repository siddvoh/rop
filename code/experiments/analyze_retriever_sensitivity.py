"""Build the paper-facing BM25-versus-neural-retriever comparison artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccr.paths import repository_root
from ccr.retrieval.hybridqa_sensitivity import MODELS, RETRIEVERS, atomic_json, sha256_file


ROOT = repository_root()
BASELINE = ROOT / "data/results/released/hybridqa_natural_v2/bridge_confirmatory/summary.json"
SENSITIVITY = ROOT / "data/results/released/hybridqa_natural_v3_retriever_sensitivity_v1"
OUTPUT = SENSITIVITY / "analysis.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    baseline = load(BASELINE)
    aggregate = load(SENSITIVITY / "aggregate_summary.json")
    rows: dict[str, dict[str, Any]] = {}
    effects: list[float] = []
    all_adjusted_significant = True
    all_stronger_than_bm25 = True
    strongest: dict[str, dict[str, Any]] = {}
    for retriever in RETRIEVERS:
        rows[retriever] = {}
        for model in MODELS:
            metric = aggregate["retrievers"][retriever]["recall"][model]["@5"]
            old = baseline["models"][model]
            effect = float(metric["paired_difference"])
            effects.append(effect)
            adjusted_p = float(metric["exact_mcnemar_p_holm_adjusted"])
            all_adjusted_significant &= adjusted_p < 0.05
            all_stronger_than_bm25 &= (
                float(metric["predicted_fact"]) > float(old["predicted_fact_recall_at_5"])
            )
            rows[retriever][model] = {
                "question_only_recall_at_5": metric["question_only"],
                "predicted_fact_recall_at_5": metric["predicted_fact"],
                "paired_difference": metric["paired_difference"],
                "paired_bootstrap_95_ci": metric["paired_bootstrap_95_ci"],
                "holm_adjusted_p": adjusted_p,
                "delta_question_only_vs_bm25": (
                    float(metric["question_only"]) - float(old["question_recall_at_5"])
                ),
                "delta_predicted_fact_vs_bm25": (
                    float(metric["predicted_fact"])
                    - float(old["predicted_fact_recall_at_5"])
                ),
            }
            if (
                model not in strongest
                or metric["predicted_fact"] > strongest[model]["predicted_fact_recall_at_5"]
            ):
                strongest[model] = {
                    "retriever": retriever,
                    "question_only_recall_at_5": metric["question_only"],
                    "predicted_fact_recall_at_5": metric["predicted_fact"],
                    "paired_difference": metric["paired_difference"],
                    "paired_bootstrap_95_ci": metric["paired_bootstrap_95_ci"],
                    "holm_adjusted_p": adjusted_p,
                }
    return {
        "schema_version": 1,
        "baseline_summary_sha256": sha256_file(BASELINE),
        "sensitivity_aggregate_sha256": sha256_file(SENSITIVITY / "aggregate_summary.json"),
        "metric": "gold_passage_recall_at_5",
        "bm25": {
            model: {
                "question_only": baseline["models"][model]["question_recall_at_5"],
                "predicted_fact": baseline["models"][model]["predicted_fact_recall_at_5"],
            }
            for model in MODELS
        },
        "neural_retrievers": rows,
        "strongest_absolute_predicted_fact": strongest,
        "robustness_checks": {
            "all_four_retrievers_improve_predicted_fact_over_question_only": all(
                effect > 0 for effect in effects
            ),
            "all_36_family_adjusted_tests_significant_at_0_05": all_adjusted_significant,
            "all_neural_retrievers_improve_predicted_fact_over_bm25_for_all_models": (
                all_stronger_than_bm25
            ),
            "minimum_within_retriever_paired_improvement": min(effects),
            "maximum_within_retriever_paired_improvement": max(effects),
        },
    }


def main() -> None:
    report = build()
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
