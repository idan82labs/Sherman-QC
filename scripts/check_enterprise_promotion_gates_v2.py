#!/usr/bin/env python3
"""Check a v2 evidence contract; never authorize production or physical access.

This module is pure until ``main`` is explicitly called. Hashes bind the supplied
declarations together; they do NOT attest provenance, untouched truth, actual
geometry, authorization, or the deployed candidate. An independent verifier and
separate release decision remain mandatory even when this contract is eligible.
No legacy aggregate report is upgraded or interpreted as v2 evidence.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


POLICY_VERSION = "ENTERPRISE_BEND_PROMOTION_CONTRACT_V2"
EVIDENCE_KIND = "SHERMAN_QC_ENTERPRISE_PROMOTION_EVIDENCE_V2"
REPORT_KIND = "SHERMAN_QC_ENTERPRISE_PROMOTION_GATE_REPORT_V2"
# Versioned release policy, not a calibration/configuration interface.
_THRESHOLDS = (
    ("minimum_truth_scans", 150),
    ("minimum_truth_parts", 75),
    ("minimum_locked_holdout_parts", 30),
    ("minimum_family_scans", 30),
    ("minimum_ply_exact_accuracy", 0.90),
    ("minimum_cad_ply_exact_accuracy", 0.95),
    ("minimum_cad_uplift", 0.05),
    ("minimum_range_containment", 0.98),
    ("minimum_visual_instance_accuracy", 0.99),
    ("minimum_instance_truth_coverage", 0.95),
    ("minimum_determinism_repeats", 10),
    ("minimum_deterministic_repeat_rate", 1.0),
    ("maximum_p95_latency_sec", 300.0),
    ("minimum_job_success_rate", 0.995),
)
_CANDIDATE_KEYS = {
    "source_sha256", "configuration_sha256", "model_bundle_sha256",
    "semantics_sha256",
}
_BINDING_KEYS = {
    "candidate_sha256", "protocol_sha256", "evaluator_sha256", "scope_sha256",
}
_SCOPE_KEYS = {
    "semantic_target", "declared_families", "acquisition_conditions",
    "expected_scan_ids",
}
_TRUTH_KEYS = {
    "scan_id", "part_id", "family", "acquisition_condition",
    "release_metric_eligible", "reviewed", "locked_holdout",
    "instance_geometry_reviewed", "visible_conventional_count", "instance_ids",
}
_RESULT_KEYS = {
    "scan_id", "status", "predicted_count", "reported_range",
    "render_admissible", "rendered_instances",
    "blind_inference_contract_passed",
}
_MODES = {"ply_only": "blind_ply_only", "cad_ply": "blind_cad_ply"}
_STATUSES = {"succeeded", "failed", "abstained"}


def _strict_json(value: Any, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_strict_json(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _strict_json(item, depth + 1)
                   for key, item in value.items())
    return False


def canonical_sha256(value: Any) -> str:
    """Hash a supplied JSON value, without reading files or verifying its claims."""
    if not _strict_json(value):
        raise ValueError("canonical identity requires finite, plain JSON values")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex64(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def _name(value: Any) -> bool:
    return type(value) is str and bool(value.strip()) and value == value.strip()


def _names(value: Any) -> bool:
    return (type(value) is list and bool(value) and all(_name(item) for item in value)
            and len(set(value)) == len(value))


def _shape(value: Any, keys: set[str]) -> bool:
    return type(value) is dict and set(value) == keys


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _duration(value: Any) -> bool:
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def build_evidence_binding(
    candidate: Any, scope: Any, *, protocol_sha256: str, evaluator_sha256: str,
) -> dict[str, str]:
    """Construct declaration bindings, NOT provenance or release attestation."""
    if not _hex64(protocol_sha256) or not _hex64(evaluator_sha256):
        raise ValueError("protocol and evaluator identities must be SHA256 digests")
    return {
        "candidate_sha256": canonical_sha256(candidate),
        "protocol_sha256": protocol_sha256,
        "evaluator_sha256": evaluator_sha256,
        "scope_sha256": canonical_sha256(scope),
    }


def semantic_results_sha256(ply_results: list[dict[str, Any]], cad_results: list[dict[str, Any]]) -> str:
    """Fingerprint validated result declarations, invariant to record ordering."""
    def ordered(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**result, "rendered_instances": sorted(result["rendered_instances"], key=lambda item: item["instance_id"])}
                for result in sorted(results, key=lambda item: item["scan_id"])]

    return canonical_sha256({"ply_only": ordered(ply_results), "cad_ply": ordered(cad_results)})


def _report(checks: list[dict[str, Any]], binding: Any = None, evidence_digest: str | None = None) -> dict[str, Any]:
    blockers = [item["gate_id"] for item in checks if not item["passed"]]
    return {
        "schema_version": 2,
        "kind": REPORT_KIND,
        "policy_version": POLICY_VERSION,
        "evidence_contract_status": "eligible" if not blockers else "blocked",
        "promotion_status": "not_authorized",
        "production_use_allowed": False,
        "autonomous_disposition_status": "NOT_AUTHORIZED",
        "supervised_shadow_status": "REQUIRES_SEPARATE_AUTHORIZATION",
        "requires_independent_verification": True,
        "requires_prospective_metric_definition_review": True,
        "verification_boundary": (
            "Supplied declarations only; no physical truth, evidence artifact, "
            "deployed candidate, review authority or release decision verified."
        ),
        "binding": dict(binding) if binding is not None else None,
        "evidence_declaration_sha256": evidence_digest,
        "thresholds": dict(_THRESHOLDS),
        "gate_count": len(checks),
        "passed_gate_count": sum(item["passed"] for item in checks),
        "blocked_gate_count": len(blockers),
        "blockers": blockers,
        "checks": checks,
    }


def evaluate_promotion_gates_v2(evidence: Any) -> dict[str, Any]:
    """Validate paired, candidate-bound declarations against fixed v2 policy.

    ``truth.records`` has one reviewed record per expected scan. Both lane
    ``results`` and ``operations.jobs`` include failed and abstained jobs.
    Render precision uses distinct matched reviewed truth-instance IDs divided
    by all emitted owned instances, with every predicted count instance owned.
    It is NOT exact-set accuracy or a count/localization recall claim. Failed and
    abstained jobs emit no accepted instances, remain count/range/operational
    failures, and cannot supply positive render scan coverage. Every accepted
    determinism replay covers
    the entire expected scan cohort and supplies its semantic-result digest.
    """
    checks: list[dict[str, Any]] = []

    def check(gate_id: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append({"gate_id": gate_id, "passed": bool(passed),
                       "observed": observed, "required": required})

    keys = {"schema_version", "kind", "candidate", "scope", "binding",
            "truth", "ply_only", "cad_ply", "operations", "determinism"}
    valid_json = _strict_json(evidence)
    check("evidence.strict_json", valid_json, valid_json, True)
    if not valid_json:
        return _report(checks)
    try:
        evidence_digest = canonical_sha256(evidence)
    except (ValueError, OverflowError, RecursionError):
        check("evidence.canonical_identity", False, False, True)
        return _report(checks)
    no_overrides = type(evidence) is dict and "thresholds" not in evidence
    check("policy.no_overrides", no_overrides, no_overrides, True)
    schema_ok = (_shape(evidence, keys)
                 and type(evidence["schema_version"]) is int
                 and evidence["schema_version"] == 2
                 and evidence["kind"] == EVIDENCE_KIND)
    check("evidence.schema", schema_ok, schema_ok, EVIDENCE_KIND)
    if not schema_ok:
        return _report(checks, evidence_digest=evidence_digest)

    candidate, scope, binding = (evidence[key] for key in ("candidate", "scope", "binding"))
    candidate_ok = (_shape(candidate, _CANDIDATE_KEYS)
                    and all(_hex64(item) for item in candidate.values()))
    scope_ok = (_shape(scope, _SCOPE_KEYS)
                and scope["semantic_target"] == "visible_observed_conventional_bends"
                and all(_names(scope[key]) for key in
                        ("declared_families", "acquisition_conditions", "expected_scan_ids")))
    binding_ok = (_shape(binding, _BINDING_KEYS)
                  and all(_hex64(item) for item in binding.values())
                  and binding["candidate_sha256"] == canonical_sha256(candidate)
                  and binding["scope_sha256"] == canonical_sha256(scope))
    check("identity.candidate", candidate_ok, candidate_ok, True)
    check("scope.declaration", scope_ok, scope_ok, True)
    check("identity.binding", binding_ok, binding_ok, True)
    if not (candidate_ok and scope_ok and binding_ok):
        return _report(checks, evidence_digest=evidence_digest)

    expected = set(scope["expected_scan_ids"])
    families = set(scope["declared_families"])
    conditions = set(scope["acquisition_conditions"])
    sections_ok = True
    for section_name in ("truth", "ply_only", "cad_ply", "operations", "determinism"):
        section = evidence[section_name]
        valid = type(section) is dict and section.get("binding") == binding
        check(f"binding.{section_name}", valid, valid, True)
        sections_ok = sections_ok and valid
    if not sections_ok:
        return _report(checks, binding, evidence_digest)

    truth_section = evidence["truth"]
    records = truth_section.get("records")
    truth_ok = _shape(truth_section, {"binding", "records"}) and type(records) is list
    truth_by_id: dict[str, Any] = {}
    parts: dict[str, tuple[str, bool]] = {}
    if truth_ok:
        for record in records:
            valid = (
                _shape(record, _TRUTH_KEYS)
                and _name(record["scan_id"]) and _name(record["part_id"])
                and _name(record["family"]) and record["family"] in families
                and _name(record["acquisition_condition"])
                and record["acquisition_condition"] in conditions
                and record["release_metric_eligible"] is True
                and record["reviewed"] is True
                and type(record["locked_holdout"]) is bool
                and type(record["instance_geometry_reviewed"]) is bool
                and _integer(record["visible_conventional_count"])
            )
            if valid:
                instance_ids = record["instance_ids"]
                valid = (
                    (type(instance_ids) is list
                     and all(_name(item) for item in instance_ids)
                     and len(set(instance_ids)) == len(instance_ids)
                     and len(instance_ids) == record["visible_conventional_count"])
                    if record["instance_geometry_reviewed"] else instance_ids is None
                )
            if not valid:
                truth_ok = False
                continue
            if record["scan_id"] in truth_by_id:
                truth_ok = False
            truth_by_id[record["scan_id"]] = record
            part_identity = (record["family"], record["locked_holdout"])
            if record["part_id"] in parts and parts[record["part_id"]] != part_identity:
                truth_ok = False
            parts[record["part_id"]] = part_identity
        truth_ok = truth_ok and set(truth_by_id) == expected
    check("truth.unique_reviewed_cohort", truth_ok, truth_ok, True)

    lanes: dict[str, dict[str, Any]] = {}
    lanes_ok = True
    for lane_name, mode in _MODES.items():
        lane = evidence[lane_name]
        valid = (_shape(lane, {"binding", "inference_mode", "results"})
                 and lane.get("inference_mode") == mode
                 and type(lane.get("results")) is list)
        by_id: dict[str, Any] = {}
        if valid:
            for result in lane["results"]:
                result_ok = (
                    _shape(result, _RESULT_KEYS) and _name(result["scan_id"])
                    and type(result["status"]) is str and result["status"] in _STATUSES
                    and type(result["render_admissible"]) is bool
                    and type(result["rendered_instances"]) is list
                    and result["blind_inference_contract_passed"] is True
                )
                if not result_ok:
                    valid = False
                    continue
                reported_range = result["reported_range"]
                rendered = result["rendered_instances"]
                marker_ids: set[str] = set()
                instances_ok = True
                for instance in rendered:
                    instance_ok = (
                        _shape(instance, {"instance_id", "admissible", "matched_truth_instance_id"})
                        and _name(instance["instance_id"])
                        and type(instance["admissible"]) is bool
                        and (instance["matched_truth_instance_id"] is None
                             or _name(instance["matched_truth_instance_id"]))
                    )
                    if not instance_ok:
                        instances_ok = False
                        continue
                    if instance["instance_id"] in marker_ids:
                        instances_ok = False
                    marker_ids.add(instance["instance_id"])
                if result["status"] == "succeeded":
                    result_ok = instances_ok and _integer(result["predicted_count"]) and len(rendered) == result["predicted_count"] and (
                        reported_range is None or (
                            type(reported_range) is list and len(reported_range) == 2
                            and all(_integer(item) for item in reported_range)
                            and reported_range[0] <= result["predicted_count"] <= reported_range[1]
                        )
                    )
                else:
                    result_ok = (result["predicted_count"] is None and reported_range is None
                                 and result["render_admissible"] is False
                                 and not rendered)
                if result["scan_id"] in by_id or not result_ok:
                    valid = False
                by_id[result["scan_id"]] = result
            valid = valid and set(by_id) == expected
        check(f"{lane_name}.paired_blind_cohort", valid, valid, True)
        lanes_ok = lanes_ok and valid
        lanes[lane_name] = by_id

    operations = evidence["operations"]
    ops_ok = (_shape(operations, {"binding", "jobs"})
              and type(operations.get("jobs")) is list)
    jobs: dict[tuple[str, str], Any] = {}
    if ops_ok:
        for job in operations["jobs"]:
            valid = (_shape(job, {"scan_id", "lane", "status", "latency_sec"})
                     and _name(job["scan_id"]) and type(job["lane"]) is str
                     and job["lane"] in _MODES and type(job["status"]) is str
                     and job["status"] in _STATUSES and _duration(job["latency_sec"]))
            if not valid:
                ops_ok = False
                continue
            key = (job["lane"], job["scan_id"])
            corresponding = lanes[job["lane"]].get(job["scan_id"])
            if key in jobs or corresponding is None or job["status"] != corresponding["status"]:
                ops_ok = False
            jobs[key] = job
        ops_ok = ops_ok and set(jobs) == {(lane, scan_id) for lane in _MODES for scan_id in expected}
    check("operations.complete_job_cohort", ops_ok, ops_ok, True)

    determinism = evidence["determinism"]
    replays = determinism.get("replays")
    replay_ok = (_shape(determinism, {"binding", "replays"}) and type(replays) is list)
    replay_ids: set[str] = set()
    fingerprints: set[str] = set()
    if replay_ok:
        for replay in replays:
            valid = (_shape(replay, {"replay_id", "accepted", "semantic_fingerprint_sha256", "scan_ids"})
                     and _name(replay["replay_id"]) and replay["accepted"] is True
                     and _hex64(replay["semantic_fingerprint_sha256"])
                     and _names(replay["scan_ids"]) and set(replay["scan_ids"]) == expected)
            if not valid:
                replay_ok = False
                continue
            if replay["replay_id"] in replay_ids:
                replay_ok = False
            replay_ids.add(replay["replay_id"])
            fingerprints.add(replay["semantic_fingerprint_sha256"])
    repeat_min = dict(_THRESHOLDS)["minimum_determinism_repeats"]
    expected_fingerprint = semantic_results_sha256(evidence["ply_only"]["results"], evidence["cad_ply"]["results"]) if lanes_ok else None
    check("operations.determinism", replay_ok and len(replay_ids) >= repeat_min and fingerprints == {expected_fingerprint},
          {"valid_accepted_replays": len(replay_ids), "unique_fingerprints": len(fingerprints),
           "matches_evaluated_result": fingerprints == {expected_fingerprint}},
          {"minimum_accepted_replays": repeat_min, "unique_fingerprints": 1, "matches_evaluated_result": True})
    if not (truth_ok and lanes_ok and ops_ok):
        return _report(checks, binding, evidence_digest)

    cfg = dict(_THRESHOLDS)
    scan_count = len(expected)
    family_counts = {family: sum(record["family"] == family for record in records)
                     for family in sorted(families)}
    condition_counts = {condition: sum(record["acquisition_condition"] == condition for record in records)
                        for condition in sorted(conditions)}
    check("truth.release_scan_count", scan_count >= cfg["minimum_truth_scans"],
          scan_count, cfg["minimum_truth_scans"])
    check("truth.release_part_count", len(parts) >= cfg["minimum_truth_parts"],
          len(parts), cfg["minimum_truth_parts"])
    holdout_count = sum(holdout for _, holdout in parts.values())
    check("truth.locked_holdout_part_count", holdout_count >= cfg["minimum_locked_holdout_parts"],
          holdout_count, cfg["minimum_locked_holdout_parts"])
    check("truth.declared_family_coverage", all(count >= cfg["minimum_family_scans"] for count in family_counts.values()),
          family_counts, cfg["minimum_family_scans"])
    check("truth.declared_condition_coverage", all(condition_counts.values()), condition_counts, ">= 1 each; not a physical qualification")

    def rate_gate(gate_id: str, numerator: int, denominator: int, threshold: str) -> None:
        required = cfg[threshold]
        rate = Fraction(numerator, denominator) if denominator else Fraction(0)
        check(gate_id, denominator > 0 and rate >= Fraction(str(required)),
              {"numerator": numerator, "denominator": denominator, "rate": round(float(rate), 6)}, required)

    rate_gate("truth.instance_geometry_coverage", sum(record["instance_geometry_reviewed"] for record in records),
              scan_count, "minimum_instance_truth_coverage")
    exact: dict[str, int] = {}
    for lane_name, by_id in lanes.items():
        exact[lane_name] = sum(result["status"] == "succeeded" and result["predicted_count"] == truth_by_id[scan_id]["visible_conventional_count"]
                               for scan_id, result in by_id.items())
        rate_gate(f"{lane_name}.exact_accuracy", exact[lane_name], scan_count,
                  "minimum_ply_exact_accuracy" if lane_name == "ply_only" else "minimum_cad_ply_exact_accuracy")
    range_matches = sum(result["status"] == "succeeded" and result["reported_range"] is not None
                        and result["reported_range"][0] <= truth_by_id[scan_id]["visible_conventional_count"] <= result["reported_range"][1]
                        for scan_id, result in lanes["ply_only"].items())
    rate_gate("ply_only.range_containment", range_matches, scan_count, "minimum_range_containment")
    matched_instances = 0
    owned_instances = 0
    render_scan_coverage = 0
    for scan_id, result in lanes["ply_only"].items():
        truth = truth_by_id[scan_id]
        rendered = result["rendered_instances"]
        owned_instances += len(rendered)
        render_checked = (result["status"] == "succeeded" and result["render_admissible"]
                          and truth["instance_geometry_reviewed"])
        if not render_checked:
            continue
        truth_ids = set(truth["instance_ids"])
        # One reviewed truth instance earns at most one credit, independently of
        # result ordering. Incorrect/inadmissible/duplicated ownership remains in
        # the denominator. A correct count cannot manufacture marker matches.
        matched_instances += len({instance["matched_truth_instance_id"] for instance in rendered
                                  if instance["admissible"] and instance["matched_truth_instance_id"] in truth_ids})
        if rendered or truth["visible_conventional_count"] == 0:
            render_scan_coverage += 1
    rate_gate("ply_only.render_admissible_instance_precision", matched_instances, owned_instances,
              "minimum_visual_instance_accuracy")
    check("ply_only.render_evaluated_scan_coverage", render_scan_coverage >= cfg["minimum_truth_scans"],
          {"render_evaluated_scans": render_scan_coverage, "expected_scans": scan_count,
           "owned_instances": owned_instances, "distinct_matched_instances": matched_instances},
          cfg["minimum_truth_scans"])
    rate_gate("cad_ply.paired_uplift", exact["cad_ply"] - exact["ply_only"], scan_count, "minimum_cad_uplift")
    latencies = sorted(float(job["latency_sec"]) for job in jobs.values())
    p95 = latencies[math.ceil(len(latencies) * 0.95) - 1]
    check("operations.p95_latency", p95 <= cfg["maximum_p95_latency_sec"],
          {"nearest_rank_p95_sec": p95, "denominator": len(jobs)}, cfg["maximum_p95_latency_sec"])
    rate_gate("operations.job_success_rate", sum(job["status"] == "succeeded" for job in jobs.values()),
              len(jobs), "minimum_job_success_rate")
    return _report(checks, binding, evidence_digest)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object keys are forbidden")
        result[key] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with args.evidence_json.open(encoding="utf-8") as handle:
            evidence = json.load(handle, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        report = evaluate_promotion_gates_v2(evidence)
        # Exclusive creation: no historical receipt is overwritten and no parent
        # directories/default artifact locations are created or discovered.
        with args.output_json.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, RecursionError) as exc:
        print(f"Evidence contract check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"evidence_contract_status": report["evidence_contract_status"],
                      "production_use_allowed": False, "requires_independent_verification": True}))
    return 0 if report["evidence_contract_status"] == "eligible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
