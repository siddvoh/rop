from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code/analysis/recompute_gemini_passage_ablation.py"
EVIDENCE = ROOT / "data/evidence/gemini_passage_ablation_openrouter_byok_v1"
BRIDGE_ROWS = (
    ROOT
    / "data/results/released/hybridqa_natural_v2/bridge_confirmatory/details.jsonl"
)
ANSWER_ROWS = (
    ROOT
    / "data/results/released/hybridqa_natural_v3/answer_confirmatory/details.jsonl"
)


def _jsonl_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_replay():
    specification = importlib.util.spec_from_file_location(
        "gemini_passage_ablation_replay", SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load Gemini passage-ablation replay")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class GeminiPassageAblationReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _load_replay().verify(EVIDENCE)

    def test_route_accounting_and_failure_retention(self) -> None:
        self.assertEqual(self.report["status"], "verified")
        self.assertEqual(self.report["main_logical_calls"], 561)
        self.assertEqual(self.report["main_wire_attempts"], 570)
        self.assertEqual(
            self.report["main_status_counts"],
            {"failed_terminal": 24, "succeeded_valid": 537},
        )
        self.assertEqual(
            self.report["main_failure_arm_counts"],
            {"intact": 13, "sham_drop": 3, "trace_drop": 8},
        )
        self.assertEqual(self.report["failed_outcomes_retained_and_scored_zero"], 24)
        self.assertEqual(self.report["main_openrouter_cost_usd"], 0.0)
        self.assertEqual(self.report["route"]["transport"], "OpenRouter BYOK")
        self.assertEqual(
            self.report["route"]["routed_provider"], "Google AI Studio"
        )

    def test_preflight_is_separate_from_the_analysis_population(self) -> None:
        self.assertEqual(
            self.report["preflight_logical_calls_excluded_from_analysis"], 3
        )
        self.assertEqual(
            self.report["preflight_status_counts"], {"succeeded_valid": 3}
        )
        self.assertTrue(self.report["barrier_claim_records_verified"])

    def test_analysis_and_execution_protocol_are_explicit(self) -> None:
        primary = self.report["primary"]
        self.assertEqual(primary["item_count"], 64)
        self.assertAlmostEqual(primary["first_em_item_mean"], 0.9270833333333334)
        self.assertAlmostEqual(primary["second_em_item_mean"], 0.15104166666666666)
        self.assertAlmostEqual(
            primary["em_item_mean_difference"], 0.7760416666666666
        )
        self.assertEqual(
            self.report["projection_schema"],
            "gemini_passage_ablation_public_result_v2",
        )
        self.assertTrue(self.report["setting_provenance_verified"])
        self.assertFalse(self.report["route"]["direct_google_api_usage"])
        self.assertAlmostEqual(
            self.report["main_upstream_economic_cost_usd"], 4.3857
        )
        self.assertAlmostEqual(
            self.report["preflight_upstream_economic_cost_usd"], 0.024284
        )
        self.assertAlmostEqual(
            self.report["total_upstream_economic_cost_usd"], 4.409984
        )
        self.assertFalse(self.report["network_used"])
        self.assertFalse(self.report["credentials_used"])

    def test_historical_cohort_eligibility_is_rederived(self) -> None:
        cohort = json.loads((EVIDENCE / "cohort.json").read_text(encoding="utf-8"))
        bridge = {
            row["question_id"]: row
            for row in _jsonl_rows(BRIDGE_ROWS)
            if row["model"] == "gemini"
        }
        answers = {
            (row["question_id"], row["condition"]): row
            for row in _jsonl_rows(ANSWER_ROWS)
            if row["model"] == "gemini"
        }
        self.assertEqual(len(bridge), 600)
        self.assertEqual(len(answers), 3_000)

        def exact(question_id: str, condition: str, expected: int) -> bool:
            row = answers[(question_id, condition)]
            return (
                row["status"] == "succeeded"
                and row["valid_json"] is True
                and row["exact"] == expected
            )

        eligible = []
        for question_id, row in bridge.items():
            aligned = (
                row["status"] == "succeeded"
                and row["valid_json"] is True
                and row["selection_correct"] is True
                and row["selected_link"] == row["gold_link"]
            )
            question_miss = row["gold_link"] not in row["question_ranking"][:5]
            predicted_hit = row["gold_link"] in row["predicted_ranking"][:5]
            historical_rescue = exact(question_id, "question_only", 0) and exact(
                question_id, "predicted_fact", 1
            )
            external_evidence_required = exact(
                question_id, "oracle_passage", 1
            ) and exact(question_id, "no_kb", 0)
            if all(
                (
                    aligned,
                    question_miss,
                    predicted_hit,
                    historical_rescue,
                    external_evidence_required,
                )
            ):
                eligible.append(question_id)

        self.assertEqual(len(cohort), 64)
        self.assertEqual(len(set(cohort)), 64)
        self.assertEqual(len(eligible), 64)
        self.assertEqual(sorted(eligible), sorted(cohort))

    def test_paper_sensitivity_claims_are_explicitly_replayed(self) -> None:
        sensitivity = self.report["sensitivity"]
        f1 = sensitivity["intact_minus_sham_f1"]
        complete_case = f1["complete_case"]
        self.assertEqual(complete_case["item_count"], 44)
        self.assertAlmostEqual(
            complete_case["f1_item_mean_difference"],
            0.006944444444444445,
        )
        valid_repeat = f1["valid_repeat"]
        self.assertEqual(valid_repeat["item_count"], 59)
        self.assertEqual(valid_repeat["first_valid_repeat_count"], 164)
        self.assertEqual(valid_repeat["second_valid_repeat_count"], 174)
        self.assertAlmostEqual(
            valid_repeat["f1_item_mean_difference"],
            0.005178907721280603,
        )

        bounds = sensitivity[
            "adversarial_failure_score_em_gap_lower_bounds"
        ]
        self.assertEqual(bounds["score_interval"], [0.0, 1.0])
        intact_trace = bounds["intact_minus_trace"]
        self.assertEqual(intact_trace["repeat_count_per_arm"], 192)
        self.assertEqual(
            intact_trace["second_arm_invalid_or_failed_repeat_count"], 8
        )
        self.assertEqual(
            intact_trace["em_difference_lower_bound"], 0.734375
        )
        sham_trace = bounds["sham_minus_trace"]
        self.assertEqual(sham_trace["repeat_count_per_arm"], 177)
        self.assertEqual(
            sham_trace["second_arm_invalid_or_failed_repeat_count"], 8
        )
        self.assertAlmostEqual(
            sham_trace["em_difference_lower_bound"],
            0.768361581920904,
        )


if __name__ == "__main__":
    unittest.main()
