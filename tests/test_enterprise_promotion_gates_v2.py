"""Isolated synthetic declaration tests: no physical evidence or release claim.

Run directly with Python/unittest to avoid repository conftest/plugin imports.
Only the explicitly named stdlib-only module and temporary synthetic files are
read. None of these assertions verifies real scans, provenance or deployment.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_enterprise_promotion_gates_v2.py"
SPEC = importlib.util.spec_from_file_location("enterprise_promotion_v2_synthetic_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gates)


def synthetic_evidence(scan_count: int = 150, ply_correct: int = 135, cad_correct: int = 144) -> dict:
    """Fabricated input declarations for contract tests; not observed results."""
    scan_ids = [f"SYNTHETIC_SCAN_{index:03}" for index in range(scan_count)]
    candidate = {key: char * 64 for key, char in (
        ("source_sha256", "a"), ("configuration_sha256", "b"),
        ("model_bundle_sha256", "c"), ("semantics_sha256", "d"))}
    scope = {
        "semantic_target": "visible_observed_conventional_bends",
        "declared_families": [f"SYNTHETIC_FAMILY_{index}" for index in range(5)],
        "acquisition_conditions": ["SYNTHETIC_CONDITION_ONLY"],
        "expected_scan_ids": scan_ids,
    }
    binding = gates.build_evidence_binding(candidate, scope, protocol_sha256="e" * 64, evaluator_sha256="f" * 64)
    evidence = {
        "schema_version": 2, "kind": gates.EVIDENCE_KIND,
        "candidate": candidate, "scope": scope, "binding": binding,
        "truth": {"binding": deepcopy(binding), "records": []},
        "ply_only": {"binding": deepcopy(binding), "inference_mode": "blind_ply_only", "results": []},
        "cad_ply": {"binding": deepcopy(binding), "inference_mode": "blind_cad_ply", "results": []},
        "operations": {"binding": deepcopy(binding), "jobs": []},
        "determinism": {"binding": deepcopy(binding), "replays": []},
    }
    for index, scan_id in enumerate(scan_ids):
        instance_ids = [f"{scan_id}_TRUTH_BEND_0", f"{scan_id}_TRUTH_BEND_1"]
        evidence["truth"]["records"].append({
            "scan_id": scan_id, "part_id": f"SYNTHETIC_PART_{index // 2:03}",
            "family": f"SYNTHETIC_FAMILY_{min(index // 30, 4)}",
            "acquisition_condition": "SYNTHETIC_CONDITION_ONLY",
            "release_metric_eligible": True, "reviewed": True,
            "locked_holdout": index // 2 < 30, "instance_geometry_reviewed": True,
            "visible_conventional_count": 2, "instance_ids": instance_ids,
        })
        for lane, exact_limit in (("ply_only", ply_correct), ("cad_ply", cad_correct)):
            predicted = 2 if index < exact_limit else 1
            evidence[lane]["results"].append({
                "scan_id": scan_id, "status": "succeeded", "predicted_count": predicted,
                "reported_range": [0, 3], "render_admissible": True,
                "rendered_instances": [
                    {"instance_id": f"PREDICTED_{bend}", "admissible": True,
                     "matched_truth_instance_id": instance_ids[bend]}
                    for bend in range(predicted)
                ],
                "blind_inference_contract_passed": True,
            })
            evidence["operations"]["jobs"].append({
                "scan_id": scan_id, "lane": lane, "status": "succeeded", "latency_sec": 200.0,
            })
    fingerprint = gates.semantic_results_sha256(evidence["ply_only"]["results"], evidence["cad_ply"]["results"])
    evidence["determinism"]["replays"] = [
        {"replay_id": f"SYNTHETIC_REPLAY_{index}", "accepted": True,
         "semantic_fingerprint_sha256": fingerprint, "scan_ids": list(scan_ids)}
        for index in range(10)
    ]
    return evidence


def rebind(evidence: dict) -> None:
    """Rebind mutated synthetic declarations, never real artifacts."""
    old = evidence["binding"]
    binding = gates.build_evidence_binding(evidence["candidate"], evidence["scope"],
                                          protocol_sha256=old["protocol_sha256"],
                                          evaluator_sha256=old["evaluator_sha256"])
    evidence["binding"] = binding
    for section in ("truth", "ply_only", "cad_ply", "operations", "determinism"):
        evidence[section]["binding"] = deepcopy(binding)


def refingerprint(evidence: dict) -> None:
    fingerprint = gates.semantic_results_sha256(evidence["ply_only"]["results"], evidence["cad_ply"]["results"])
    for replay in evidence["determinism"]["replays"]:
        replay["semantic_fingerprint_sha256"] = fingerprint


def fail_scan(evidence: dict, lane: str, index: int, status: str = "failed") -> None:
    result = evidence[lane]["results"][index]
    result.update(status=status, predicted_count=None, reported_range=None,
                  render_admissible=False, rendered_instances=[])
    for job in evidence["operations"]["jobs"]:
        if job["scan_id"] == result["scan_id"] and job["lane"] == lane:
            job["status"] = status
    refingerprint(evidence)


class PromotionV2Contracts(unittest.TestCase):
    def assert_blocked(self, evidence, gate=None):
        report = gates.evaluate_promotion_gates_v2(evidence)
        self.assertEqual(report["evidence_contract_status"], "blocked")
        self.assertFalse(report["production_use_allowed"])
        self.assertEqual(report["autonomous_disposition_status"], "NOT_AUTHORIZED")
        self.assertTrue(report["requires_independent_verification"])
        if gate is not None:
            self.assertIn(gate, report["blockers"])
        return report

    def test_full_size_synthetic_contract_is_not_production_authorization(self):
        evidence = synthetic_evidence()
        report = gates.evaluate_promotion_gates_v2(evidence)
        self.assertEqual(report["evidence_contract_status"], "eligible")
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["blocked_gate_count"], 0)
        self.assertFalse(report["production_use_allowed"])
        self.assertEqual(report["promotion_status"], "not_authorized")
        self.assertEqual(report["autonomous_disposition_status"], "NOT_AUTHORIZED")
        self.assertEqual(report["supervised_shadow_status"], "REQUIRES_SEPARATE_AUTHORIZATION")
        self.assertTrue(report["requires_independent_verification"])
        self.assertTrue(report["requires_prospective_metric_definition_review"])
        self.assertEqual(report["evidence_declaration_sha256"], gates.canonical_sha256(evidence))
        checks = {item["gate_id"]: item for item in report["checks"]}
        self.assertEqual(checks["ply_only.exact_accuracy"]["observed"],
                         {"numerator": 135, "denominator": 150, "rate": 0.9})
        self.assertEqual(checks["ply_only.render_admissible_instance_precision"]["observed"],
                         {"numerator": 285, "denominator": 285, "rate": 1.0})
        self.assertEqual(checks["operations.job_success_rate"]["observed"]["denominator"], 300)

    def test_fixed_policy_rejects_overrides_and_three_scan_development_receipt(self):
        for override in ({}, {"minimum_truth_scans": 3}, {"unknown": 1}, {"maximum_p95_latency_sec": 999}):
            with self.subTest(override=override):
                evidence = synthetic_evidence()
                evidence["thresholds"] = override
                self.assert_blocked(evidence, "policy.no_overrides")
        self.assert_blocked(synthetic_evidence(3, 2, 3), "truth.release_scan_count")

    def test_bad_json_schema_and_legacy_eligible_cannot_qualify(self):
        for value in (None, [], True, 2, "eligible", {"schema_version": 1, "promotion_status": "eligible"}):
            with self.subTest(value=value):
                self.assert_blocked(value)
        for value in (float("nan"), float("inf"), -float("inf"), (1, 2), {1: "not a JSON key"}):
            with self.subTest(value=value):
                evidence = synthetic_evidence()
                evidence["unexpected"] = value
                self.assert_blocked(evidence, "evidence.strict_json")

    def test_oversized_integer_encodings_fail_closed_in_all_shapes(self):
        oversized = 10 ** 10000
        self.assert_blocked(oversized)
        for field in ("truth_count", "predicted_count", "latency", "candidate_digest", "scope_name"):
            with self.subTest(field=field):
                evidence = synthetic_evidence()
                if field == "truth_count":
                    evidence["truth"]["records"][0]["visible_conventional_count"] = oversized
                elif field == "predicted_count":
                    evidence["ply_only"]["results"][0]["predicted_count"] = oversized
                elif field == "latency":
                    evidence["operations"]["jobs"][0]["latency_sec"] = oversized
                elif field == "candidate_digest":
                    evidence["candidate"]["source_sha256"] = oversized
                else:
                    evidence["scope"]["declared_families"][0] = oversized
                self.assert_blocked(evidence)

    def test_no_empty_duplicate_or_missing_declared_scope(self):
        for key in ("declared_families", "acquisition_conditions", "expected_scan_ids"):
            for value in ([], [" "], ["duplicate", "duplicate"], None):
                with self.subTest(key=key, value=value):
                    evidence = synthetic_evidence()
                    evidence["scope"][key] = value
                    rebind(evidence)
                    self.assert_blocked(evidence, "scope.declaration")
        evidence = synthetic_evidence()
        evidence["scope"]["declared_families"].append("UNOBSERVED_DECLARED_FAMILY")
        rebind(evidence)
        report = self.assert_blocked(evidence, "truth.declared_family_coverage")
        family = next(item for item in report["checks"] if item["gate_id"] == "truth.declared_family_coverage")
        self.assertEqual(family["observed"]["UNOBSERVED_DECLARED_FAMILY"], 0)
        evidence = synthetic_evidence()
        evidence["scope"]["acquisition_conditions"].append("UNOBSERVED_CONDITION")
        rebind(evidence)
        self.assert_blocked(evidence, "truth.declared_condition_coverage")

    def test_identity_component_or_section_mismatch_blocks(self):
        for key in synthetic_evidence()["candidate"]:
            with self.subTest(candidate_key=key):
                evidence = synthetic_evidence()
                evidence["candidate"][key] = "0" * 64
                self.assert_blocked(evidence, "identity.binding")
        for section in ("truth", "ply_only", "cad_ply", "operations", "determinism"):
            for key in ("candidate_sha256", "protocol_sha256", "evaluator_sha256", "scope_sha256"):
                with self.subTest(section=section, key=key):
                    evidence = synthetic_evidence()
                    evidence[section]["binding"][key] = "0" * 64
                    self.assert_blocked(evidence, f"binding.{section}")
        evidence = synthetic_evidence()
        evidence["binding"]["protocol_sha256"] = "not a hash"
        self.assert_blocked(evidence, "identity.binding")

    def test_truth_requires_unique_records_strict_flags_and_part_split(self):
        mutations = [
            ("release_metric_eligible", "true"), ("release_metric_eligible", 1),
            ("release_metric_eligible", False), ("reviewed", 1), ("reviewed", False),
            ("locked_holdout", 1), ("instance_geometry_reviewed", 1),
            ("visible_conventional_count", True), ("visible_conventional_count", 2.0),
            ("visible_conventional_count", -1), ("family", "unknown"),
            ("acquisition_condition", "UNDECLARED"), ("part_id", ""),
            ("instance_ids", ["same", "same"]),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                evidence = synthetic_evidence()
                evidence["truth"]["records"][0][key] = value
                self.assert_blocked(evidence, "truth.unique_reviewed_cohort")
        evidence = synthetic_evidence()
        evidence["truth"]["records"].append(deepcopy(evidence["truth"]["records"][0]))
        self.assert_blocked(evidence, "truth.unique_reviewed_cohort")
        evidence = synthetic_evidence()
        evidence["truth"]["records"][0]["locked_holdout"] = False
        self.assert_blocked(evidence, "truth.unique_reviewed_cohort")

    def test_missing_part_holdout_and_geometry_coverage_do_not_qualify(self):
        evidence = synthetic_evidence()
        for record in evidence["truth"]["records"]:
            record["locked_holdout"] = False
        self.assert_blocked(evidence, "truth.locked_holdout_part_count")
        evidence = synthetic_evidence()
        for index in range(10):
            evidence["truth"]["records"][index].update(instance_geometry_reviewed=False, instance_ids=None)
        self.assert_blocked(evidence, "truth.instance_geometry_coverage")
        evidence = synthetic_evidence()
        for record in evidence["truth"]["records"]:
            record["part_id"] = record["family"]
        self.assert_blocked(evidence, "truth.release_part_count")

    def test_lanes_must_have_exact_paired_cohort_not_equal_sized_aggregates(self):
        for lane in ("ply_only", "cad_ply"):
            for mutation in ("disjoint", "missing", "duplicate", "operator", "unproven", "aggregate"):
                with self.subTest(lane=lane, mutation=mutation):
                    evidence = synthetic_evidence()
                    if mutation == "disjoint":
                        evidence[lane]["results"][0]["scan_id"] = "DIFFERENT_SCAN"
                    elif mutation == "missing":
                        evidence[lane]["results"].pop()
                    elif mutation == "duplicate":
                        evidence[lane]["results"][-1] = deepcopy(evidence[lane]["results"][0])
                    elif mutation == "operator":
                        evidence[lane]["inference_mode"] = "operator_assisted"
                    elif mutation == "unproven":
                        evidence[lane]["results"][0]["blind_inference_contract_passed"] = 1
                    else:
                        evidence[lane].pop("results")
                        evidence[lane]["aggregate"] = {"accuracy": 1.0, "scans_processed": 150}
                    self.assert_blocked(evidence, f"{lane}.paired_blind_cohort")

    def test_failed_and_abstained_jobs_stay_in_denominators(self):
        for status in ("failed", "abstained"):
            with self.subTest(status=status):
                evidence = synthetic_evidence()
                fail_scan(evidence, "ply_only", 0, status)
                fail_scan(evidence, "ply_only", 1, status)
                report = self.assert_blocked(evidence, "operations.job_success_rate")
                checks = {item["gate_id"]: item for item in report["checks"]}
                self.assertEqual(checks["ply_only.exact_accuracy"]["observed"]["denominator"], 150)
                self.assertEqual(checks["ply_only.exact_accuracy"]["observed"]["numerator"], 133)
                self.assertEqual(checks["ply_only.range_containment"]["observed"]["denominator"], 150)
                self.assertEqual(checks["operations.job_success_rate"]["observed"],
                                 {"numerator": 298, "denominator": 300, "rate": 0.993333})
                self.assertEqual(checks["ply_only.render_evaluated_scan_coverage"]["observed"]["render_evaluated_scans"], 148)

    def test_count_range_and_instance_types_are_strict(self):
        for key, value in (("predicted_count", True), ("predicted_count", 2.1),
                           ("predicted_count", -1), ("reported_range", [True, 3]),
                           ("reported_range", [3, 0]), ("reported_range", [0, 1]),
                           ("render_admissible", 1), ("rendered_instances", None),
                           ("status", [])):
            with self.subTest(key=key, value=value):
                evidence = synthetic_evidence()
                evidence["ply_only"]["results"][0][key] = value
                self.assert_blocked(evidence, "ply_only.paired_blind_cohort")

    def test_empty_or_single_easy_owned_instance_cannot_cover_larger_count(self):
        for length in (0, 1):
            evidence = synthetic_evidence()
            for result in evidence["ply_only"]["results"]:
                result["rendered_instances"] = result["rendered_instances"][:length]
            self.assert_blocked(evidence, "ply_only.paired_blind_cohort")

    def test_duplicate_truth_match_gets_no_double_credit(self):
        evidence = synthetic_evidence()
        for result in evidence["ply_only"]["results"]:
            if len(result["rendered_instances"]) == 2:
                result["rendered_instances"][1]["matched_truth_instance_id"] = result["rendered_instances"][0]["matched_truth_instance_id"]
        refingerprint(evidence)
        report = self.assert_blocked(evidence, "ply_only.render_admissible_instance_precision")
        precision = next(item for item in report["checks"] if item["gate_id"] == "ply_only.render_admissible_instance_precision")
        self.assertEqual(precision["observed"]["numerator"], 150)
        self.assertEqual(precision["observed"]["denominator"], 285)

    def test_wrong_or_inadmissible_markers_are_not_count_accuracy(self):
        for field, value in (("admissible", False), ("matched_truth_instance_id", "UNMATCHED")):
            with self.subTest(field=field):
                evidence = synthetic_evidence()
                for result in evidence["ply_only"]["results"][:3]:
                    result["rendered_instances"][0][field] = value
                refingerprint(evidence)
                report = self.assert_blocked(evidence, "ply_only.render_admissible_instance_precision")
                exact = next(item for item in report["checks"] if item["gate_id"] == "ply_only.exact_accuracy")
                self.assertTrue(exact["passed"])

    def test_low_range_and_insufficient_paired_uplift_block(self):
        evidence = synthetic_evidence()
        for result in evidence["ply_only"]["results"][:4]:
            result["reported_range"] = None
        refingerprint(evidence)
        self.assert_blocked(evidence, "ply_only.range_containment")
        self.assert_blocked(synthetic_evidence(ply_correct=138, cad_correct=144), "cad_ply.paired_uplift")

    def test_exact_five_percentage_point_uplift_uses_rational_comparison(self):
        evidence = synthetic_evidence(scan_count=200, ply_correct=180, cad_correct=190)
        report = gates.evaluate_promotion_gates_v2(evidence)
        uplift = next(item for item in report["checks"] if item["gate_id"] == "cad_ply.paired_uplift")
        self.assertEqual(uplift["observed"], {"numerator": 10, "denominator": 200, "rate": 0.05})
        self.assertTrue(uplift["passed"])
        self.assertEqual(report["evidence_contract_status"], "eligible")

    def test_replays_need_ten_accepted_complete_matching_results(self):
        for count in (0, 1, 9):
            with self.subTest(count=count):
                evidence = synthetic_evidence()
                evidence["determinism"]["replays"] = evidence["determinism"]["replays"][:count]
                self.assert_blocked(evidence, "operations.determinism")
        mutations = [
            ("accepted", False), ("accepted", 1), ("replay_id", "SYNTHETIC_REPLAY_1"),
            ("semantic_fingerprint_sha256", "0" * 64), ("scan_ids", []),
            ("scan_ids", ["DIFFERENT_SCAN"]),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                evidence = synthetic_evidence()
                evidence["determinism"]["replays"][0][key] = value
                self.assert_blocked(evidence, "operations.determinism")
        evidence = synthetic_evidence()
        for replay in evidence["determinism"]["replays"]:
            replay["semantic_fingerprint_sha256"] = "0" * 64
        self.assert_blocked(evidence, "operations.determinism")
        evidence = synthetic_evidence()
        evidence["determinism"] = {"binding": evidence["binding"], "deterministic_repeat_rate": 1.0}
        self.assert_blocked(evidence, "operations.determinism")

    def test_semantic_fingerprint_is_order_independent_but_result_sensitive(self):
        evidence = synthetic_evidence()
        original = gates.semantic_results_sha256(evidence["ply_only"]["results"], evidence["cad_ply"]["results"])
        for lane in ("ply_only", "cad_ply"):
            evidence[lane]["results"].reverse()
            for result in evidence[lane]["results"]:
                result["rendered_instances"].reverse()
        self.assertEqual(original, gates.semantic_results_sha256(evidence["ply_only"]["results"], evidence["cad_ply"]["results"]))
        self.assertEqual(gates.evaluate_promotion_gates_v2(evidence)["evidence_contract_status"], "eligible")
        evidence["ply_only"]["results"][0]["render_admissible"] = False
        self.assert_blocked(evidence, "operations.determinism")

    def test_operations_require_all_jobs_matching_lane_outcomes_and_finite_latency(self):
        for mutation in ("missing", "duplicate", "status", "lane"):
            with self.subTest(mutation=mutation):
                evidence = synthetic_evidence()
                if mutation == "missing":
                    evidence["operations"]["jobs"].pop()
                elif mutation == "duplicate":
                    evidence["operations"]["jobs"][-1] = deepcopy(evidence["operations"]["jobs"][0])
                else:
                    evidence["operations"]["jobs"][0][mutation] = "invalid"
                self.assert_blocked(evidence, "operations.complete_job_cohort")
        for duration in (True, "200", -1, 10 ** 400):
            with self.subTest(duration=str(duration)[:20]):
                evidence = synthetic_evidence()
                evidence["operations"]["jobs"][0]["latency_sec"] = duration
                self.assert_blocked(evidence, "operations.complete_job_cohort")
        evidence = synthetic_evidence()
        for job in evidence["operations"]["jobs"][:16]:
            job["latency_sec"] = 301
        self.assert_blocked(evidence, "operations.p95_latency")

    def test_mutated_report_thresholds_cannot_change_future_evaluation(self):
        report = gates.evaluate_promotion_gates_v2(synthetic_evidence())
        report["thresholds"]["minimum_truth_scans"] = 0
        report2 = self.assert_blocked(synthetic_evidence(3, 2, 3), "truth.release_scan_count")
        self.assertEqual(report2["thresholds"]["minimum_truth_scans"], 150)

    def test_cli_uses_only_explicit_paths_and_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix="sherman-promotion-v2-synthetic-") as directory:
            source = Path(directory) / "synthetic-input.json"
            target = Path(directory) / "synthetic-report.json"
            source.write_text(json.dumps(synthetic_evidence()), encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(gates.main(["--evidence-json", str(source), "--output-json", str(target)]), 0)
                original = target.read_bytes()
                self.assertEqual(gates.main(["--evidence-json", str(source), "--output-json", str(target)]), 2)
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(json.loads(original)["production_use_allowed"])
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                             ["synthetic-input.json", "synthetic-report.json"])

    def test_cli_rejects_duplicate_keys_nonfinite_numbers_and_missing_parent(self):
        with tempfile.TemporaryDirectory(prefix="sherman-promotion-v2-synthetic-") as directory:
            source = Path(directory) / "synthetic-input.json"
            target = Path(directory) / "never-created.json"
            for raw in ('{"kind": "first", "kind": "second"}', '{"number": NaN}', '{"number": Infinity}'):
                source.write_text(raw, encoding="utf-8")
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(gates.main(["--evidence-json", str(source), "--output-json", str(target)]), 2)
                self.assertFalse(target.exists())
            source.write_text(json.dumps(synthetic_evidence()), encoding="utf-8")
            missing = Path(directory) / "not-created" / "report.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(gates.main(["--evidence-json", str(source), "--output-json", str(missing)]), 2)
            self.assertFalse(missing.parent.exists())

    def test_pure_evaluator_does_not_read_write_or_discover_paths(self):
        evidence = synthetic_evidence()
        with patch.object(Path, "open", side_effect=AssertionError("filesystem access")), \
             patch.object(Path, "stat", side_effect=AssertionError("filesystem access")), \
             patch.object(Path, "glob", side_effect=AssertionError("filesystem discovery")), \
             patch.object(Path, "rglob", side_effect=AssertionError("filesystem discovery")):
            report = gates.evaluate_promotion_gates_v2(evidence)
        self.assertEqual(report["evidence_contract_status"], "eligible")


if __name__ == "__main__":
    unittest.main()
