from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "code/analysis"
sys.path.insert(0, str(SCRIPTS))
import qwen_projection
import promote_qwen_core
import recompute_broad_dev_audit
import recompute_qwen_decoded_id_replication
import recompute_transfer_scope_audit


ARCHIVE = ROOT / "data/generated/ccr_reviewer_authority.zip"
LINKS = (
    ROOT
    / "data/results/released/hybridqa_natural_v3_retriever_sensitivity_v1/links.jsonl"
)


class QwenProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = qwen_projection.load_archive(ARCHIVE)

    def test_archive_identity_and_scope(self) -> None:
        self.assertEqual(self.data["archive"]["bytes"], 3_909_370)
        self.assertEqual(
            self.data["archive"]["sha256"],
            "470dae7dea93cb59bb5badf1568264db6493b3c2780a1d544aebddd7ff99984d",
        )
        self.assertEqual(len(self.data["qwen"]), 335)
        self.assertEqual(len(self.data["broad"]), 1_397)

    def test_qwen_paper_counts(self) -> None:
        rows = self.data["qwen"]
        self.assertEqual(sum(row["selected_id_resolvable"] for row in rows), 334)
        distinct = [row for row in rows if not row["selected_matches_trace"]]
        self.assertEqual(len(distinct), 106)
        self.assertEqual(sum(row["selected_id_resolvable"] for row in distinct), 105)
        final = qwen_projection.qwen_group(distinct, "final_hybrid")
        joint = final["by_cutoff"]["5"][
            "selected_vs_traced_joint_on_resolvable_rows"
        ]
        self.assertEqual(
            (joint["p1_h1"], joint["p1_h0"], joint["p0_h1"], joint["p0_h0"]),
            (6, 98, 0, 1),
        )
        body = qwen_projection.qwen_group(rows, "body_bm25")
        self.assertEqual(
            body["by_cutoff"]["5"][
                "p_selected_id_preserved_on_resolvable_rows"
            ]["numerator"],
            231,
        )

    def test_broad_development_table(self) -> None:
        report = qwen_projection.broad_summary(self.data["broad"])
        observed = {
            name: (
                values["question"]["hits_at_5"],
                values["decoded_trace_id"]["hits_at_5"],
            )
            for name, values in report["retrievers"].items()
        }
        self.assertEqual(observed, qwen_projection.EXPECTED_BROAD_HITS)

    def test_reports_recompute_byte_for_byte(self) -> None:
        broad = recompute_broad_dev_audit.build_report(ARCHIVE)
        self.assertEqual(
            qwen_projection.canonical_json_bytes(broad),
            (ROOT / "data/evidence/broad_dev_audit.json").read_bytes(),
        )
        decoded = recompute_qwen_decoded_id_replication.build_report(self.data)
        self.assertEqual(
            qwen_projection.canonical_json_bytes(decoded),
            (ROOT / "data/evidence/qwen_decoded_id_replication.json").read_bytes(),
        )
        transfer = recompute_transfer_scope_audit.build_report(
            self.data, LINKS.read_bytes()
        )
        self.assertEqual(
            qwen_projection.canonical_json_bytes(transfer),
            (ROOT / "data/evidence/transfer_scope_audit.json").read_bytes(),
        )

    def test_promoted_files_recompute_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            promote_qwen_core.promote(ARCHIVE, output)
            for filename in ("items.jsonl", "manifest.json", "summary.json"):
                self.assertEqual(
                    (output / filename).read_bytes(),
                    (ROOT / "data/evidence/qwen_core_promotion" / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
