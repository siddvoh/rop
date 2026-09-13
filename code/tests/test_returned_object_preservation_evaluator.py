from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = ROOT / "code/contracts/evaluate_returned_object_preservation.py"
KERNEL_PATH = ROOT / "code/contracts/returned_object_preservation_kernel.py"
FIXTURE = ROOT / "code/contracts/fixtures/prospective_v1"
SPEC = importlib.util.spec_from_file_location("rop_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)
KERNEL_SPEC = importlib.util.spec_from_file_location("rop_kernel", KERNEL_PATH)
assert KERNEL_SPEC is not None and KERNEL_SPEC.loader is not None
kernel = importlib.util.module_from_spec(KERNEL_SPEC)
KERNEL_SPEC.loader.exec_module(kernel)


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def reseal(root: Path, manifest_name: str, envelope_name: str) -> None:
    envelope_path = root / envelope_name
    envelope = load(envelope_path)
    assert isinstance(envelope, dict)
    envelope["declaration_sha256"] = hashlib.sha256(
        (root / manifest_name).read_bytes()
    ).hexdigest()
    write_json(envelope_path, envelope)


class ReturnedObjectPreservationEvaluatorTests(unittest.TestCase):
    def fixture_copy(self, temporary: str) -> Path:
        target = Path(temporary) / "fixture"
        shutil.copytree(
            FIXTURE,
            target,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "all_envelope.json",
                "hard_envelope.json",
                "invalid_envelope.json",
                "soft_envelope.json",
                "expected_ledger.json",
            ),
        )
        return target

    def result(
        self,
        root: Path,
        record: object,
        manifest: str = "hard_manifest.json",
        envelope: str = "hard_run_envelope.json",
    ) -> dict[str, object]:
        return evaluator.evaluate_record(root / manifest, root / envelope, record)

    def test_tracked_bundle_replays_byte_for_byte_and_exercises_every_verdict(self) -> None:
        report = evaluator.evaluate_bundle(FIXTURE / "cases.json")
        self.assertEqual(
            evaluator.canonical_json_bytes(report),
            (FIXTURE / "expected_ledger.json").read_bytes(),
        )
        self.assertEqual(report["record_count"], 15)
        self.assertEqual(
            {row["verdict"] for row in report["records"]},
            set(evaluator.VERDICTS),
        )
        self.assertEqual(
            sum(row["eligible_for_metric"] for row in report["records"]), 8
        )
        self.assertEqual(report["run_count"], 12)
        self.assertEqual(
            report["run_status_counts"],
            {
                "RUN_INVALID": 4,
                "RUN_FAIL": 1,
                "RUN_UNVERIFIABLE": 3,
                "RUN_PASS": 3,
                "NOT_APPLICABLE": 1,
            },
        )

    def test_cli_is_deterministic_and_matches_the_pure_function(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            for output in (first, second):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(EVALUATOR_PATH),
                        "--bundle",
                        str(FIXTURE / "cases.json"),
                        "--output",
                        str(output),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), (FIXTURE / "expected_ledger.json").read_bytes())

    def test_minimal_kernel_replays_separately_and_is_not_conformance(self) -> None:
        resolver = load(FIXTURE / "resolver.json")
        direct = kernel.evaluate_preservation_kernel(
            selected="doc:alpha",
            selection_type="corpus_key",
            resolver=resolver,
            returned_ids=["alpha-alias"],
            cutoff=1,
            quantifier="any",
        )
        self.assertEqual(direct["verdict"], "PASS")
        self.assertIs(direct["is_conformance_verdict"], False)
        compiled = kernel.compile_resolver(resolver)
        kernel_cases = load(FIXTURE / "kernel_cases.json")["cases"]
        for case in kernel_cases:
            arguments = {
                "selected": case["selected"],
                "selection_type": case["selection_type"],
                "returned_ids": case["returned_ids"],
                "cutoff": case["cutoff"],
                "quantifier": case["quantifier"],
            }
            with self.subTest(
                case_id=case["case_id"],
                implementation="compiled_equivalence",
            ):
                self.assertEqual(
                    kernel.evaluate_preservation_kernel(
                        resolver=resolver,
                        **arguments,
                    ),
                    kernel.evaluate_preservation_kernel(
                        resolver=compiled,
                        **arguments,
                    ),
                )
        with self.assertRaises(kernel.KernelConfigurationError):
            kernel.evaluate_preservation_kernel(
                selected="doc:alpha",
                selection_type="corpus_key",
                resolver={"aliases": {}, "members": {}},
                returned_ids=["doc:alpha"],
                cutoff=1,
                quantifier="any",
            )
        canonical_collision = copy.deepcopy(resolver)
        canonical_collision["members"]["obj:alpha"] = ["obj:beta"]
        with self.assertRaises(kernel.KernelConfigurationError):
            kernel.evaluate_preservation_kernel(
                selected="doc:alpha",
                selection_type="corpus_key",
                resolver=canonical_collision,
                returned_ids=["obj:beta"],
                cutoff=1,
                quantifier="all",
            )
        conflicting = copy.deepcopy(resolver)
        conflicting["aliases"]["chunk:alpha:0"] = "obj:beta"
        with self.assertRaises(kernel.KernelConfigurationError):
            kernel.evaluate_preservation_kernel(
                selected="doc:alpha",
                selection_type="corpus_key",
                resolver=conflicting,
                returned_ids=["chunk:alpha:0", "chunk:alpha:1"],
                cutoff=2,
                quantifier="any",
            )
        report = evaluator.evaluate_kernel_bundle(FIXTURE / "kernel_cases.json")
        self.assertEqual(
            evaluator.canonical_json_bytes(report),
            (FIXTURE / "kernel_expected_ledger.json").read_bytes(),
        )
        self.assertEqual(report["kernel_version"], "rop-preservation-kernel/v1")
        self.assertIs(report["is_conformance_verdict"], False)
        self.assertEqual(report["case_count"], 6)
        self.assertTrue(
            all(row["is_conformance_verdict"] is False for row in report["cases"])
        )
        self.assertEqual(
            report["verdict_counts"],
            {
                "SELECTOR_ERROR": 1,
                "RESOLUTION_ERROR": 1,
                "UNVERIFIABLE": 1,
                "PASS": 2,
                "FAIL": 1,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "kernel.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR_PATH),
                    "--kernel-bundle",
                    str(FIXTURE / "kernel_cases.json"),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                output.read_bytes(),
                (FIXTURE / "kernel_expected_ledger.json").read_bytes(),
            )

    def test_standalone_run_cli_emits_strict_run_pass(self) -> None:
        run_document = load(FIXTURE / "run_pass.json")
        with mock.patch.object(
            evaluator,
            "_load_profile",
            wraps=evaluator._load_profile,
        ) as load_profile, mock.patch.object(
            evaluator,
            "compile_resolver",
            wraps=evaluator.compile_resolver,
        ) as compile_profile_resolver:
            direct = evaluator.evaluate_run(
                FIXTURE / run_document["manifest_locator"],
                FIXTURE / run_document["envelope_locator"],
                run_document["run_id"],
                run_document["records"],
            )
        self.assertEqual(load_profile.call_count, 1)
        self.assertEqual(compile_profile_resolver.call_count, 1)
        self.assertEqual(direct["run_status"], "RUN_PASS")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR_PATH),
                    "--run",
                    str(FIXTURE / "run_pass.json"),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = load(output)
            self.assertEqual(report["run"]["run_status"], "RUN_PASS")
            self.assertEqual(
                report["run"]["eligible_expected_coverage"]["fraction"], 1.0
            )
            self.assertIs(
                report["run"]["diagnostic_pass_fraction"][
                    "is_conformance_verdict"
                ],
                False,
            )

    def test_key_set_uses_outer_any_for_any_and_per_object_all(self) -> None:
        report = evaluator.evaluate_bundle(FIXTURE / "cases.json")
        by_case = {row["case_id"]: row for row in report["records"]}
        self.assertEqual(
            by_case["hard_key_set_any_accepts_one_object"]["verdict"], "PASS"
        )
        self.assertEqual(
            by_case["hard_key_set_all_accepts_one_complete_object"]["verdict"],
            "PASS",
        )
        self.assertEqual(
            by_case["hard_key_set_all_accepts_singleton_alternative"]["verdict"],
            "PASS",
        )
        mixed_partial_members = {
            "record_id": "set-all-mixed-partials",
            "selector": {"selected_key": ["doc:alpha", "doc:gamma"]},
            "returned": {
                "object_id": ["chunk:alpha:0", "chunk:gamma:0"]
            },
        }
        self.assertEqual(
            self.result(
                FIXTURE,
                mixed_partial_members,
                "set_all_manifest.json",
                "set_all_run_envelope.json",
            )["verdict"],
            "FAIL",
        )

    def test_bound_lossless_witness_replaces_raw_key_with_total_verdicts(self) -> None:
        report = evaluator.evaluate_bundle(FIXTURE / "cases.json")
        by_case = {row["case_id"]: row for row in report["records"]}
        self.assertEqual(
            by_case["hard_lossless_witness_pass_without_raw_key"]["verdict"],
            "PASS",
        )
        self.assertEqual(
            by_case["hard_lossless_witness_binding_mismatch"]["verdict"],
            "SELECTOR_ERROR",
        )
        self.assertEqual(
            by_case["hard_lossless_witness_unknown_value"]["verdict"],
            "RESOLUTION_ERROR",
        )
        by_run = {row["run_id"]: row for row in report["runs"]}
        self.assertEqual(
            by_run["lossless_witness_run_pass"]["run_status"], "RUN_PASS"
        )
        witness_case = next(
            case
            for case in load(FIXTURE / "cases.json")["cases"]
            if case["case_id"] == "hard_lossless_witness_pass_without_raw_key"
        )
        self.assertNotIn("selected_key", witness_case["record"]["selector"])

        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            witness_path = root / "identity_witness.json"
            witness = load(witness_path)
            witness["witness_to_selected"]["title:Duplicate"] = "doc:alpha"
            write_json(witness_path, witness)
            manifest = load(root / "witness_manifest.json")
            manifest["selection"]["identity_witness"]["artifact_sha256"] = (
                hashlib.sha256(witness_path.read_bytes()).hexdigest()
            )
            write_json(root / "witness_manifest.json", manifest)
            reseal(root, "witness_manifest.json", "witness_run_envelope.json")
            self.assertEqual(
                self.result(
                    root,
                    witness_case["record"],
                    "witness_manifest.json",
                    "witness_run_envelope.json",
                )["verdict"],
                "DECLARATION_INVALID",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            witness_path = root / "identity_witness.json"
            witness = load(witness_path)
            witness["witness_to_selected"]["title:Alpha"] = "doc:outside"
            write_json(witness_path, witness)
            manifest = load(root / "witness_manifest.json")
            manifest["selection"]["identity_witness"]["artifact_sha256"] = (
                hashlib.sha256(witness_path.read_bytes()).hexdigest()
            )
            write_json(root / "witness_manifest.json", manifest)
            reseal(root, "witness_manifest.json", "witness_run_envelope.json")
            self.assertEqual(
                self.result(
                    root,
                    witness_case["record"],
                    "witness_manifest.json",
                    "witness_run_envelope.json",
                )["verdict"],
                "DECLARATION_INVALID",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            manifest = load(root / "witness_manifest.json")
            manifest["returned_set"]["cutoff"] = 2
            write_json(root / "witness_manifest.json", manifest)
            reseal(root, "witness_manifest.json", "witness_run_envelope.json")
            self.assertEqual(
                self.result(
                    root,
                    witness_case["record"],
                    "witness_manifest.json",
                    "witness_run_envelope.json",
                )["verdict"],
                "SELECTOR_ERROR",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            witness_path = root / "identity_witness.json"
            witness_path.write_bytes(witness_path.read_bytes() + b"\n")
            self.assertEqual(
                self.result(
                    root,
                    witness_case["record"],
                    "witness_manifest.json",
                    "witness_run_envelope.json",
                )["verdict"],
                "DECLARATION_INVALID",
            )

    def test_run_uses_bound_dotted_record_id_field(self) -> None:
        report = evaluator.evaluate_bundle(FIXTURE / "cases.json")
        by_run = {row["run_id"]: row for row in report["runs"]}

        passed = by_run["dotted_record_id_run_pass"]
        self.assertEqual(passed["run_status"], "RUN_PASS")
        self.assertEqual(
            passed["population_binding"]["record_id_field"],
            "identity.record_key",
        )
        self.assertEqual(
            [row["record_id"] for row in passed["records"]],
            ["run_alpha", "run_beta"],
        )
        self.assertEqual(passed["unexpected_record_ids"], [])

        duplicate = by_run["dotted_record_id_run_duplicate"]
        self.assertEqual(duplicate["run_status"], "RUN_INVALID")
        self.assertEqual(duplicate["duplicate_record_ids"], ["run_alpha"])
        self.assertEqual(duplicate["missing_record_ids"], ["run_beta"])

        missing = by_run["dotted_record_id_run_missing"]
        self.assertEqual(missing["run_status"], "RUN_UNVERIFIABLE")
        self.assertEqual(missing["missing_record_ids"], ["run_beta"])
        self.assertEqual(missing["status_histogram"]["MISSING"], 1)

    def test_strict_run_precedence_population_and_diagnostics(self) -> None:
        report = evaluator.evaluate_bundle(FIXTURE / "cases.json")
        by_run = {row["run_id"]: row for row in report["runs"]}
        self.assertEqual(by_run["strict_run_pass"]["run_status"], "RUN_PASS")
        self.assertEqual(
            by_run["strict_run_fail_precedes_error"]["run_status"], "RUN_FAIL"
        )
        self.assertEqual(
            by_run["strict_run_unverifiable_missing"]["status_histogram"][
                "MISSING"
            ],
            1,
        )
        self.assertEqual(
            by_run["strict_run_invalid_population_binding"]["run_status"],
            "RUN_INVALID",
        )
        self.assertEqual(
            by_run["strict_run_invalid_precedes_record_fail"]["run_status"],
            "RUN_INVALID",
        )
        self.assertEqual(
            by_run["strict_run_invalid_precedes_record_fail"]["status_histogram"][
                "FAIL"
            ],
            1,
        )
        for row in report["runs"]:
            self.assertIn("status_histogram", row)
            self.assertIn("eligible_expected_coverage", row)
            self.assertIn("diagnostic_pass_fraction", row)
            self.assertIs(
                row["diagnostic_pass_fraction"]["is_conformance_verdict"],
                False,
            )

    def test_primary_declaration_binds_real_assets_and_pre_outcome_envelope(self) -> None:
        manifest = load(FIXTURE / "hard_manifest.json")
        envelope = load(FIXTURE / "hard_run_envelope.json")
        self.assertIsInstance(manifest, dict)
        self.assertIsInstance(envelope, dict)
        self.assertEqual(envelope["declared_phase"], "pre_outcome")
        self.assertEqual(envelope["sealed_before"], "returned_ids")
        self.assertEqual(
            envelope["authority_id"],
            manifest["authority"]["manifest_authority"]["authority_id"],
        )
        self.assertEqual(envelope["chronology_basis"], "trusted_manifest_authority")
        self.assertEqual(
            manifest["authority"]["manifest_authority"],
            {
                "authority_id": "prospective-fixture-authority-v1",
                "policy_authority_id": "prospective-fixture-policy-authority-v1",
                "producer_id": "prospective-fixture-producer-v1",
                "policy_basis": {
                    "locator": "policy_hard.json",
                    "sha256": "4d90150aba96827a9bf6c6b7d92038ede9efcbd154b5f778f65ca6af3a4421c3",
                },
                "target_interface_id": "fixture-ranker",
                "interface_version": "fixture-ranker-api/v1",
                "policy_revision": "hard/fixture-r1",
                "chronology_basis": "trusted_manifest_authority",
                "log_provenance_basis": "trusted_producer",
                "hash_scope": "byte_binding_and_replay_only",
                "adversarial_producer_resistance": False,
            },
        )
        self.assertEqual(
            envelope["declaration_sha256"],
            hashlib.sha256((FIXTURE / "hard_manifest.json").read_bytes()).hexdigest(),
        )
        bindings = (
            (manifest["binding"], "config_locator", "config_sha256"),
            (
                manifest["authority"]["natural_language_prompt"],
                "locator",
                "sha256",
            ),
            (
                manifest["authority"]["manifest_authority"]["policy_basis"],
                "locator",
                "sha256",
            ),
            (manifest["population"], "artifact_locator", "artifact_sha256"),
            (manifest["resolver"], "artifact_locator", "artifact_sha256"),
            (
                manifest["returned_set"],
                "implementation_locator",
                "implementation_sha256",
            ),
        )
        for section, locator, digest in bindings:
            with self.subTest(locator=section[locator]):
                self.assertEqual(
                    section[digest],
                    hashlib.sha256((FIXTURE / section[locator]).read_bytes()).hexdigest(),
                )

    def test_aliases_and_members_canonicalize_to_object_identity_for_any(self) -> None:
        for returned_id in ("alpha-alias", "obj:alpha", "chunk:alpha:1"):
            with self.subTest(returned_id=returned_id):
                row = {
                    "record_id": "canonical",
                    "selector": {"selected_key": "doc:alpha"},
                    "returned": {"object_id": [returned_id]},
                }
                self.assertEqual(self.result(FIXTURE, row)["verdict"], "PASS")
        row = {
            "record_id": "wrong-object",
            "selector": {"selected_key": "doc:alpha"},
            "returned": {"object_id": ["doc:beta", "unknown:id"]},
        }
        self.assertEqual(self.result(FIXTURE, row)["verdict"], "FAIL")

    def test_resolver_rejects_cross_owner_alias_member_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            resolver_path = root / "resolver.json"
            resolver = load(resolver_path)
            resolver["aliases"]["chunk:alpha:0"] = "obj:beta"
            write_json(resolver_path, resolver)
            manifest = load(root / "hard_manifest.json")
            manifest["resolver"]["artifact_sha256"] = hashlib.sha256(
                resolver_path.read_bytes()
            ).hexdigest()
            write_json(root / "hard_manifest.json", manifest)
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            record = {
                "record_id": "resolver-collision",
                "selector": {"selected_key": "doc:alpha"},
                "returned": {
                    "object_id": ["chunk:alpha:0", "chunk:alpha:1"]
                },
            }
            self.assertEqual(self.result(root, record)["verdict"], "DECLARATION_INVALID")

        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            resolver_path = root / "resolver.json"
            resolver = load(resolver_path)
            resolver["members"]["obj:alpha"] = ["obj:beta"]
            write_json(resolver_path, resolver)
            manifest = load(root / "hard_manifest.json")
            manifest["resolver"]["artifact_sha256"] = hashlib.sha256(
                resolver_path.read_bytes()
            ).hexdigest()
            write_json(root / "hard_manifest.json", manifest)
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            record = {
                "record_id": "resolver-canonical-collision",
                "selector": {"selected_key": "doc:alpha"},
                "returned": {"object_id": ["obj:beta"]},
            }
            self.assertEqual(self.result(root, record)["verdict"], "DECLARATION_INVALID")

    def test_cutoff_is_applied_before_canonical_membership(self) -> None:
        row = {
            "record_id": "after-cutoff",
            "selector": {"selected_key": "doc:alpha"},
            "returned": {
                "object_id": ["noise:0", "noise:1", "noise:2", "alpha-alias"]
            },
        }
        self.assertEqual(self.result(FIXTURE, row)["verdict"], "FAIL")

    def test_all_requires_every_member_not_an_object_alias(self) -> None:
        alias_only = {
            "record_id": "alias-is-not-all-members",
            "selector": {"selected_key": "doc:gamma"},
            "returned": {"object_id": ["doc:gamma"]},
        }
        self.assertEqual(
            self.result(
                FIXTURE, alias_only, "all_manifest.json", "all_run_envelope.json"
            )["verdict"],
            "FAIL",
        )
        all_members = copy.deepcopy(alias_only)
        all_members["returned"]["object_id"] = ["chunk:gamma:0", "chunk:gamma:1"]
        self.assertEqual(
            self.result(
                FIXTURE, all_members, "all_manifest.json", "all_run_envelope.json"
            )["verdict"],
            "PASS",
        )

    def test_all_rejects_alias_only_resolver_but_accepts_singleton_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            manifest = load(root / "all_manifest.json")
            manifest["resolver"]["returns"] = "alias_set"
            write_json(root / "all_manifest.json", manifest)
            reseal(root, "all_manifest.json", "all_run_envelope.json")
            row = {
                "record_id": "bad-all-declaration",
                "selector": {"selected_key": "doc:gamma"},
                "returned": {"object_id": ["chunk:gamma:0", "chunk:gamma:1"]},
            }
            self.assertEqual(
                self.result(root, row, "all_manifest.json", "all_run_envelope.json")[
                    "verdict"
                ],
                "DECLARATION_INVALID",
            )

        singleton = {
            "record_id": "singleton-all",
            "selector": {"selected_key": "doc:beta"},
            "returned": {"object_id": ["chunk:beta:0"]},
        }
        self.assertEqual(
            self.result(
                FIXTURE, singleton, "all_manifest.json", "all_run_envelope.json"
            )["verdict"],
            "PASS",
        )

    def test_terminal_precedence_is_total_and_deterministic(self) -> None:
        malformed = {"record_id": "precedence", "returned": {"wrong": "bad"}}
        first = self.result(
            FIXTURE,
            malformed,
            "invalid_manifest.json",
            "invalid_run_envelope.json",
        )
        second = self.result(
            FIXTURE,
            malformed,
            "invalid_manifest.json",
            "invalid_run_envelope.json",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["record_id"], "precedence")
        self.assertEqual(first["verdict"], "DECLARATION_INVALID")
        self.assertEqual(
            self.result(
                FIXTURE, malformed, "soft_manifest.json", "soft_run_envelope.json"
            )["verdict"],
            "NOT_APPLICABLE",
        )
        self.assertEqual(self.result(FIXTURE, malformed)["verdict"], "SELECTOR_ERROR")
        unresolved = {
            "record_id": "precedence-unresolved",
            "selector": {"selected_key": "missing"},
            "returned": {},
        }
        self.assertEqual(self.result(FIXTURE, unresolved)["verdict"], "RESOLUTION_ERROR")
        missing_log = {
            "record_id": "precedence-log",
            "selector": {"selected_key": "doc:alpha"},
            "returned": {},
        }
        self.assertEqual(self.result(FIXTURE, missing_log)["verdict"], "UNVERIFIABLE")

    def test_prompt_is_hash_bound_provenance_but_cannot_activate_or_override(self) -> None:
        prompt = (FIXTURE / "prompt.txt").read_text(encoding="utf-8")
        self.assertIn("should be retained", prompt)
        soft = {
            "record_id": "prompt-has-no-authority",
            "selector": {"selected_key": "doc:alpha"},
            "returned": {"object_id": []},
        }
        self.assertEqual(
            self.result(
                FIXTURE, soft, "soft_manifest.json", "soft_run_envelope.json"
            )["verdict"],
            "NOT_APPLICABLE",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            manifest = load(root / "hard_manifest.json")
            manifest["authority"]["natural_language_prompt"]["can_override_manifest"] = True
            write_json(root / "hard_manifest.json", manifest)
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            self.assertEqual(self.result(root, soft)["verdict"], "DECLARATION_INVALID")

    def test_policy_basis_is_mode_specific_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            policy_path = root / "policy_hard.json"
            policy = load(policy_path)
            policy["selection_mode"] = "soft"
            write_json(policy_path, policy)
            manifest = load(root / "hard_manifest.json")
            manifest["authority"]["manifest_authority"]["policy_basis"][
                "sha256"
            ] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            write_json(root / "hard_manifest.json", manifest)
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            record = {
                "record_id": "policy-mode-mismatch",
                "selector": {"selected_key": "doc:alpha"},
                "returned": {"object_id": ["doc:alpha"]},
            }
            self.assertEqual(
                self.result(root, record)["verdict"], "DECLARATION_INVALID"
            )

    def test_every_asset_and_envelope_digest_drift_fails_before_record_read(self) -> None:
        row = {"record_id": "drift", "selector": 1, "returned": {}}
        for filename in (
            "config.json",
            "prompt.txt",
            "resolver.json",
            "returned_component.py",
            "policy_hard.json",
            "population.json",
            "hard_run_envelope.json",
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = self.fixture_copy(temporary)
                path = root / filename
                if filename == "hard_run_envelope.json":
                    envelope = load(path)
                    envelope["declaration_sha256"] = "0" * 64
                    write_json(path, envelope)
                else:
                    path.write_bytes(path.read_bytes() + b"\n")
                self.assertEqual(self.result(root, row)["verdict"], "DECLARATION_INVALID")

    def test_authority_mismatch_and_overstated_trust_fail_before_record_read(self) -> None:
        row = {"record_id": "trust-drift", "selector": 1, "returned": {}}
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            envelope = load(root / "hard_run_envelope.json")
            envelope["authority_id"] = "other-authority"
            write_json(root / "hard_run_envelope.json", envelope)
            self.assertEqual(self.result(root, row)["verdict"], "DECLARATION_INVALID")
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            manifest = load(root / "hard_manifest.json")
            manifest["authority"]["manifest_authority"][
                "adversarial_producer_resistance"
            ] = True
            write_json(root / "hard_manifest.json", manifest)
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            self.assertEqual(self.result(root, row)["verdict"], "DECLARATION_INVALID")

    def test_malformed_returned_logs_are_unverifiable_but_empty_is_a_verifiable_fail(self) -> None:
        base = {
            "record_id": "returned-log",
            "selector": {"selected_key": "doc:alpha"},
            "returned": {"object_id": None},
        }
        for malformed in (None, "doc:alpha", ["doc:alpha", 3], [""]):
            with self.subTest(malformed=malformed):
                row = copy.deepcopy(base)
                row["returned"]["object_id"] = malformed
                self.assertEqual(self.result(FIXTURE, row)["verdict"], "UNVERIFIABLE")
        row = copy.deepcopy(base)
        row["returned"]["object_id"] = []
        result = self.result(FIXTURE, row)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIs(result["eligible_for_metric"], True)

        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            manifest = load(root / "hard_manifest.json")
            manifest["returned_set"]["id_field"] = "ranking.ids"
            write_json(root / "hard_manifest.json", manifest)
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            dotted = {
                "record_id": "dotted-returned-field",
                "selector": {"selected_key": "doc:alpha"},
                "returned": {"ranking": {"ids": ["alpha-alias"]}},
            }
            self.assertEqual(self.result(root, dotted)["verdict"], "PASS")
            dotted["returned"] = {"ranking.ids": ["alpha-alias"]}
            self.assertEqual(
                self.result(root, dotted)["verdict"], "UNVERIFIABLE"
            )

    def test_duplicate_json_and_unsafe_locator_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            (root / "hard_manifest.json").write_text(
                '{"contract_version":"rop/v1","contract_version":"rop/v1"}\n',
                encoding="utf-8",
            )
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            self.assertEqual(
                self.result(root, {"record_id": "duplicate"})["verdict"],
                "DECLARATION_INVALID",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_copy(temporary)
            manifest = load(root / "hard_manifest.json")
            manifest["binding"]["config_locator"] = "../config.json"
            write_json(root / "hard_manifest.json", manifest)
            reseal(root, "hard_manifest.json", "hard_run_envelope.json")
            self.assertEqual(
                self.result(root, {"record_id": "escape"})["verdict"],
                "DECLARATION_INVALID",
            )

    def test_component_outputs_match_the_audited_fixture_records(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "fixture_returned_component", FIXTURE / "returned_component.py"
        )
        assert spec is not None and spec.loader is not None
        component = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(component)
        cases = load(FIXTURE / "cases.json")["cases"]
        by_record = {case["record"]["record_id"]: case["record"] for case in cases}
        for case_id in component.RANKINGS:
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    component.rank(case_id, 3),
                    by_record[case_id]["returned"]["object_id"],
                )


if __name__ == "__main__":
    unittest.main()
