"""Mandatory fail-closed tamper regressions for compact PWTS evidence.

Every mutation operates on a deep copy of an already-produced evidence artifact.
This suite never creates or modifies scan geometry.  It is suitable for copied
real preflight tables and for real-like compact table fixtures in unit tests.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

from research.pwts_v1.independent_validator import (
    ValidationPolicy,
    ValidationResult,
    sha256_json,
    table_digest,
    validate_evidence,
)


Evidence = MutableMapping[str, Any]
Mutation = Callable[[Evidence], None]


@dataclass(frozen=True)
class TamperRegression:
    regression_id: str
    expected_rejection: str
    mutate: Mutation


@dataclass(frozen=True)
class TamperResult:
    regression_id: str
    expected_rejection: str
    rejected: bool
    observed_errors: tuple[str, ...]


def _first(evidence: Evidence, table: str) -> MutableMapping[str, Any]:
    rows = evidence.get(table)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"baseline lacks nonempty {table}")
    return rows[0]


def _raw_byte_changed(evidence: Evidence) -> None:
    row = _first(evidence, "case_manifest")
    row["observed_raw_ply_sha256"] = "f" * 64 if row["raw_ply_sha256"] != "f" * 64 else "e" * 64


def _exchange_case_source_bindings(evidence: Evidence) -> None:
    rows = evidence["case_manifest"]
    if len(rows) < 2:
        raise ValueError("source-binding exchange requires two cases")
    rows[0]["raw_ply_sha256"], rows[1]["raw_ply_sha256"] = (
        rows[1]["raw_ply_sha256"],
        rows[0]["raw_ply_sha256"],
    )


def _sample_id_substitution(evidence: Evidence) -> None:
    row = _first(evidence, "case_manifest")
    forged = sha256_json(["forged-equal-count-point-id"])
    row["sampled_point_id_digest"] = forged
    row["validator_sampled_point_id_digest"] = forged
    row["sample_count"] = int(row["sample_count"])


def _unauthorized_configuration(evidence: Evidence) -> None:
    evidence["config_sha256"] = sha256_json(
        {"unauthorized_change": True, "claimed_complete_configuration": True}
    )
    for row in evidence.get("case_manifest", []):
        row["config_sha256"] = evidence["config_sha256"]


def _remove_seed_adjust_runner_aggregates(evidence: Evidence) -> None:
    removed = evidence["seed_primitive"].pop(0)
    token = removed["case_token"]
    field = next(row for row in evidence["field_summary"] if row["case_token"] == token)
    field["deterministic_local_maximum_count"] -= 1
    for row in evidence["rejection_census"]:
        if row["case_token"] == token and row["stage"] in {
            "deterministic_local_maxima",
            "seed_primitives",
        }:
            row["denominator"] -= 1
            row["numerator"] = max(0, int(row["numerator"]) - 1)
    # Only the runner side is adjusted; the independent side remains bound.
    for row in evidence["equivalence"]:
        if row["case_token"] != token:
            continue
        counts = row["per_table_row_counts"]["seed_primitive"]
        counts["BASE_A"] -= 1
        remaining = [
            seed for seed in evidence["seed_primitive"]
            if seed["case_token"] == token
        ]
        row["per_table_digests"]["seed_primitive"]["BASE_A"] = table_digest(
            remaining
        )


def _mutate_trace_node_id(evidence: Evidence) -> None:
    _first(evidence, "trace_node")["point_id"] = "f" * 64


def _exchange_membership_sides(evidence: Evidence) -> None:
    rows = evidence["strip_membership"]
    left = next(row for row in rows if str(row["side"]).lower() == "left")
    right = next(row for row in rows if str(row["side"]).lower() == "right")
    left["side"], right["side"] = right["side"], left["side"]
    rows.sort(
        key=lambda row: (
            row["case_token"],
            row["trace_id"],
            0 if str(row["side"]).lower() == "left" else 1,
            row["point_id"],
        )
    )


def _forge_certificate_validity(evidence: Evidence) -> None:
    row = _first(evidence, "certificate")
    row["trace_median_response_rad"] = 0.0
    row["gate_trace_response"] = False
    row["valid"] = True
    row["first_rejection_code"] = "VALID"


def _forge_or_remove_deletion(evidence: Evidence) -> None:
    rows = evidence["destructive_negative"]
    if len(rows) < 2:
        raise ValueError("deletion tamper requires bilateral rows")
    rows.pop(0)


def _noncanonical_duplicate_delete(evidence: Evidence) -> None:
    rows = evidence["strip_membership"]
    if len(rows) < 3:
        raise ValueError("row-order tamper requires at least three memberships")
    rows[1] = copy.deepcopy(rows[0])
    rows.reverse()


def _inject_source_points(evidence: Evidence) -> None:
    _first(evidence, "case_manifest")["source_points"] = [[1.0, 2.0, 3.0]]
    _first(evidence, "case_manifest")["source_points_serialized"] = True


def _forge_resource_cap(evidence: Evidence) -> None:
    row = _first(evidence, "resource")
    row["process_tree_peak_rss_bytes"] = 12_884_901_889
    row["reported_passing"] = True


def _forge_equivalence_boolean(evidence: Evidence) -> None:
    row = _first(evidence, "equivalence")
    table = next(iter(row["per_table_digests"]))
    variants = row["per_table_digests"][table]
    variant = list(variants)[-1]
    variants[variant] = "0" * 64
    row["semantic_equivalence"] = True
    row["first_mismatch"] = None


def _inject_runner_semantic_import(evidence: Evidence) -> None:
    reports = evidence["safety"]["import_reports"]
    reports["independent_validator"].append("research.pwts_v1.semantic_core")


MANDATORY_TAMPER_REGRESSIONS = (
    TamperRegression("T01_RAW_BYTE_CHANGED", "raw source hash mismatch", _raw_byte_changed),
    TamperRegression("T02_CASE_SOURCE_BINDING_EXCHANGE", "raw source hash mismatch", _exchange_case_source_bindings),
    TamperRegression("T03_SAMPLED_POINT_ID_SUBSTITUTION", "BASE_A digest not bound", _sample_id_substitution),
    TamperRegression("T04_UNAUTHORIZED_CONFIGURATION", "unauthorized configuration hash", _unauthorized_configuration),
    TamperRegression("T05_SEED_ROW_REMOVAL", "row-count mismatch", _remove_seed_adjust_runner_aggregates),
    TamperRegression("T06_TRACE_NODE_ID_MUTATION", "trace-node digest mismatch", _mutate_trace_node_id),
    TamperRegression("T07_MEMBERSHIP_SIDE_EXCHANGE", "strip digest mismatch", _exchange_membership_sides),
    TamperRegression("T08_FORGED_CERTIFICATE_VALIDITY", "forged certificate validity", _forge_certificate_validity),
    TamperRegression("T09_DELETION_EVIDENCE_FORGERY", "missing DELETE_LEFT", _forge_or_remove_deletion),
    TamperRegression("T10_ROW_ORDER_DUPLICATE_DELETE", "noncanonical row order", _noncanonical_duplicate_delete),
    TamperRegression("T11_SOURCE_POINT_SERIALIZATION", "forbidden serialized payload", _inject_source_points),
    TamperRegression("T12_RESOURCE_ACCOUNTING_FORGERY", "RSS cap exceeded", _forge_resource_cap),
    TamperRegression("T13_FORGED_EQUIVALENCE", "semantic digest mismatch", _forge_equivalence_boolean),
    TamperRegression("T14_SEMANTIC_IMPORT_BOUNDARY", "forbidden imports", _inject_runner_semantic_import),
)


def run_tamper_suite(
    baseline: Mapping[str, Any],
    *,
    policy: ValidationPolicy | None = None,
    validator: Callable[..., ValidationResult] = validate_evidence,
) -> tuple[TamperResult, ...]:
    """Run all fourteen regressions on independent copies of a valid artifact."""

    effective_policy = policy or ValidationPolicy()
    baseline_result = validator(copy.deepcopy(baseline), policy=effective_policy)
    if not baseline_result.valid:
        raise ValueError(
            "tamper baseline must first validate: " + "; ".join(baseline_result.errors)
        )

    results: list[TamperResult] = []
    for regression in MANDATORY_TAMPER_REGRESSIONS:
        copied: Evidence = copy.deepcopy(dict(baseline))
        regression.mutate(copied)
        observed = validator(copied, policy=effective_policy)
        results.append(
            TamperResult(
                regression_id=regression.regression_id,
                expected_rejection=regression.expected_rejection,
                rejected=not observed.valid,
                observed_errors=observed.errors,
            )
        )
    return tuple(results)


def require_all_tampers_rejected(results: tuple[TamperResult, ...]) -> None:
    accepted = [result.regression_id for result in results if not result.rejected]
    if accepted:
        raise AssertionError(f"tampered artifacts accepted: {accepted}")
    if len(results) != 14:
        raise AssertionError(f"expected 14 tamper results, received {len(results)}")


__all__ = [
    "MANDATORY_TAMPER_REGRESSIONS",
    "TamperRegression",
    "TamperResult",
    "require_all_tampers_rejected",
    "run_tamper_suite",
]
