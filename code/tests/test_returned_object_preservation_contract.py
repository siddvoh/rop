from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "code/contracts/returned_object_preservation.schema.json"
EXAMPLE_PATH = (
    ROOT / "code/contracts/examples/returned_object_preservation.hard.json"
)
SPEC = importlib.util.spec_from_file_location(
    "hybridqa_replay_contract_tests",
    ROOT / "code/replay.py",
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class ReturnedObjectPreservationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load(SCHEMA_PATH)
        self.example = load(EXAMPLE_PATH)

    def assert_invalid(self, document: object) -> None:
        with self.assertRaises(verifier.VerificationError):
            verifier.validate_returned_object_preservation_manifest(
                document, self.schema
            )

    def test_tracked_schema_and_valid_hard_example_are_hash_bound(self) -> None:
        verifier.validate_returned_object_preservation_manifest(
            self.example, self.schema
        )
        self.assertEqual(
            hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
            verifier.RETURNED_OBJECT_PRESERVATION_SCHEMA_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(EXAMPLE_PATH.read_bytes()).hexdigest(),
            verifier.RETURNED_OBJECT_PRESERVATION_HARD_EXAMPLE_SHA256,
        )
        old_root = verifier.ROOT
        try:
            verifier.ROOT = ROOT
            self.assertEqual(
                verifier.verify_returned_object_preservation_contract(),
                "returned_object_preservation_contract_validation",
            )
            self.assertEqual(
                verifier.verify_returned_object_preservation_evaluator(),
                "returned_object_preservation_executable_fixture",
            )
        finally:
            verifier.ROOT = old_root

    def test_transitive_fixture_tree_locks_every_shipped_dependency(self) -> None:
        fixture = ROOT / verifier.RETURNED_OBJECT_PRESERVATION_FIXTURE_PREFIX
        self.assertEqual(
            verifier._returned_object_fixture_tree_sha256(fixture),
            verifier.RETURNED_OBJECT_PRESERVATION_FIXTURE_TREE_SHA256,
        )
        self.assertEqual(
            len(verifier.RETURNED_OBJECT_PRESERVATION_FIXTURE_MEMBERS), 32
        )
        for name in verifier.RETURNED_OBJECT_PRESERVATION_FIXTURE_MEMBERS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "fixture"
                shutil.copytree(fixture, copied)
                path = copied / name
                path.write_bytes(path.read_bytes() + b"\n")
                self.assertNotEqual(
                    verifier._returned_object_fixture_tree_sha256(copied),
                    verifier.RETURNED_OBJECT_PRESERVATION_FIXTURE_TREE_SHA256,
                )

    def test_hard_mode_requires_every_identity_and_returned_set_field(self) -> None:
        for field in ("population", "resolver", "returned_set", "retention"):
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.example)
                del mutated[field]
                self.assert_invalid(mutated)
        for field in ("source_field", "type", "namespace"):
            with self.subTest(field=f"selection.{field}"):
                mutated = copy.deepcopy(self.example)
                del mutated["selection"][field]
                self.assert_invalid(mutated)
        for section, field in (
            ("binding", "config_sha256"),
            ("binding", "config_locator"),
            ("resolver", "version"),
            ("resolver", "artifact_locator"),
            ("resolver", "artifact_sha256"),
            ("returned_set", "component"),
            ("returned_set", "revision"),
            ("returned_set", "implementation_locator"),
            ("returned_set", "implementation_sha256"),
            ("returned_set", "id_field"),
            ("returned_set", "cutoff"),
            ("population", "artifact_locator"),
            ("population", "artifact_sha256"),
            ("population", "record_id_field"),
            ("population", "expected_count"),
        ):
            with self.subTest(field=f"{section}.{field}"):
                mutated = copy.deepcopy(self.example)
                del mutated[section][field]
                self.assert_invalid(mutated)

    def test_hard_mode_accepts_exactly_one_raw_or_lossless_witness_source(self) -> None:
        witness = copy.deepcopy(self.example)
        witness["selection"] = {
            "mode": "hard",
            "identity_witness": {
                "witness_version": "rop-identity-witness/v1",
                "source_field": "selector.identity_witness",
                "artifact_locator": "resolver/identity-witness.json",
                "artifact_sha256": "1" * 64,
                "decoding": "bound_exact_inverse_map",
                "record_binding": "sha256_canonical_json",
                "authentication_basis": "trusted_producer_record_binding",
            },
            "type": "corpus_key",
            "namespace": "hybridqa.wikipedia.link",
        }
        verifier.validate_returned_object_preservation_manifest(
            witness, self.schema
        )

        both = copy.deepcopy(witness)
        both["selection"]["source_field"] = "selector.selected_link"
        self.assert_invalid(both)

        for field in (
            "witness_version",
            "source_field",
            "artifact_locator",
            "artifact_sha256",
            "decoding",
            "record_binding",
            "authentication_basis",
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(witness)
                del mutated["selection"]["identity_witness"][field]
                self.assert_invalid(mutated)

        raw_without_source = copy.deepcopy(self.example)
        del raw_without_source["selection"]["source_field"]
        self.assert_invalid(raw_without_source)

    def test_explicit_soft_and_open_are_valid_only_without_hard_blocks(self) -> None:
        for mode in ("soft", "open"):
            with self.subTest(mode=mode):
                document = copy.deepcopy(self.example)
                document["selection"] = {"mode": mode}
                for field in ("population", "resolver", "returned_set", "retention"):
                    del document[field]
                verifier.validate_returned_object_preservation_manifest(
                    document, self.schema
                )

                for conflict in ("population", "resolver", "returned_set", "retention"):
                    mutated = copy.deepcopy(document)
                    mutated[conflict] = copy.deepcopy(self.example[conflict])
                    self.assert_invalid(mutated)

                conflicting_selection = copy.deepcopy(document)
                conflicting_selection["selection"]["source_field"] = (
                    "selector.selected_link"
                )
                self.assert_invalid(conflicting_selection)

    def test_any_accepts_aliases_but_all_requires_a_chunk_member_set(self) -> None:
        for quantifier in ("any", "all"):
            for resolver_set in (
                "alias_set",
                "chunk_set",
                "alias_or_chunk_set",
            ):
                with self.subTest(
                    quantifier=quantifier, resolver_set=resolver_set
                ):
                    document = copy.deepcopy(self.example)
                    document["retention"]["quantifier"] = quantifier
                    document["resolver"]["returns"] = resolver_set
                    if quantifier == "all" and resolver_set == "alias_set":
                        self.assert_invalid(document)
                    else:
                        verifier.validate_returned_object_preservation_manifest(
                            document, self.schema
                        )

        key_set = copy.deepcopy(self.example)
        key_set["selection"]["type"] = "corpus_key_set"
        verifier.validate_returned_object_preservation_manifest(
            key_set, self.schema
        )

        dotted_population_id = copy.deepcopy(self.example)
        dotted_population_id["population"][
            "record_id_field"
        ] = "identity.record_key"
        verifier.validate_returned_object_preservation_manifest(
            dotted_population_id, self.schema
        )

    def test_natural_language_prompt_cannot_activate_or_override(self) -> None:
        prompt = self.example["authority"]["natural_language_prompt"]
        self.assertEqual(prompt["role"], "provenance_only")
        self.assertIs(prompt["can_activate"], False)
        self.assertIs(prompt["can_override_manifest"], False)
        for field in ("can_activate", "can_override_manifest"):
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.example)
                mutated["authority"]["natural_language_prompt"][field] = True
                self.assert_invalid(mutated)

        mutated = copy.deepcopy(self.example)
        mutated["authority"]["normative_source"] = "natural_language_prompt"
        self.assert_invalid(mutated)

    def test_trust_tier_is_explicit_and_cannot_overstate_hash_guarantees(self) -> None:
        trust = self.example["authority"]["manifest_authority"]
        self.assertEqual(trust["policy_authority_id"], "illustrative-policy-authority-v1")
        self.assertEqual(trust["producer_id"], "illustrative-producer-v1")
        self.assertEqual(trust["target_interface_id"], self.example["returned_set"]["component"])
        self.assertEqual(trust["interface_version"], "hybridqa-retriever-api/v1")
        self.assertEqual(trust["policy_revision"], "hard/illustrative-r1")
        self.assertEqual(trust["chronology_basis"], "trusted_manifest_authority")
        self.assertEqual(trust["log_provenance_basis"], "trusted_producer")
        self.assertEqual(trust["hash_scope"], "byte_binding_and_replay_only")
        self.assertIs(trust["adversarial_producer_resistance"], False)
        for field, value in (
            ("chronology_basis", "self_asserted_untrusted"),
            ("log_provenance_basis", "untrusted_log"),
            ("hash_scope", "proves_chronology"),
            ("adversarial_producer_resistance", True),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.example)
                mutated["authority"]["manifest_authority"][field] = value
                self.assert_invalid(mutated)

    def test_total_verdicts_and_denominator_are_fixed(self) -> None:
        expected = {
            "total_verdicts": True,
            "declaration_invalid": "DECLARATION_INVALID",
            "soft_or_open": "NOT_APPLICABLE",
            "selected_key_invalid": "SELECTOR_ERROR",
            "selected_key_unresolvable": "RESOLUTION_ERROR",
            "returned_id_log_missing_or_malformed": "UNVERIFIABLE",
            "retention_satisfied": "PASS",
            "retention_violated": "FAIL",
            "ledger_policy": "retain_every_record",
            "metric_denominator": "hard_resolvable_verifiable_only",
        }
        self.assertEqual(self.example["verdict_policy"], expected)
        for field, value in expected.items():
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.example)
                mutated["verdict_policy"][field] = (
                    False if value is True else "changed"
                )
                self.assert_invalid(mutated)

        expected_run = {
            "profile_version": "rop-run-profile/v1",
            "invalid_declaration_or_population": "RUN_INVALID",
            "expected_hard_record_failed": "RUN_FAIL",
            "expected_hard_record_missing_or_unverifiable": "RUN_UNVERIFIABLE",
            "all_expected_hard_records_passed": "RUN_PASS",
            "soft_or_open": "NOT_APPLICABLE",
            "status_histogram": "always",
            "eligible_expected_coverage": "eligible_over_expected",
            "diagnostic_pass_fraction": "PASS/(PASS+FAIL)",
            "diagnostic_is_conformance_verdict": False,
        }
        self.assertEqual(self.example["run_policy"], expected_run)
        for field in expected_run:
            mutated = copy.deepcopy(self.example)
            mutated["run_policy"][field] = "changed"
            self.assert_invalid(mutated)

    def test_bad_digest_cutoff_and_duplicate_members_return_terminal_verdicts(self) -> None:
        for section, field, value in (
            ("binding", "declared_phase", "post_outcome"),
            ("binding", "config_sha256", "0" * 63),
            ("resolver", "artifact_sha256", "not-a-digest"),
            ("returned_set", "cutoff", 0),
            ("returned_set", "cutoff", True),
            ("population", "declared_phase", "post_outcome"),
            ("population", "expected_count", 0),
        ):
            with self.subTest(field=f"{section}.{field}"):
                mutated = copy.deepcopy(self.example)
                mutated[section][field] = value
                self.assert_invalid(mutated)

        with tempfile.TemporaryDirectory() as name:
            duplicate = Path(name) / "duplicate.json"
            duplicate.write_text('{"selection": 1, "selection": 2}\n')
            with self.assertRaises(verifier.VerificationError):
                verifier._load_strict_json(duplicate)


if __name__ == "__main__":
    unittest.main()
