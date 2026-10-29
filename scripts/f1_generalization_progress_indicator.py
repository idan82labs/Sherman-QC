#!/usr/bin/env python3
"""Grade whether an F1 version changed evidence for physical generalization.

The Generalization Progress Indicator (GPI) is intentionally separate from
implementation success.  A source-free lock or synthetic benchmark can be
excellent engineering work while remaining GPI-C because it provides no clean
unseen-family physical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GRADE_A = "A_VERIFIED_PHYSICAL_PROGRESS"
GRADE_B = "B_POSITIVE_PHYSICAL_INDICATOR"
GRADE_C = "C_NO_GENERALIZATION_EVIDENCE"
GRADE_D = "D_NEGATIVE_PHYSICAL_INDICATOR"
GRADE_F = "F_NON_EVALUABLE"


@dataclass(frozen=True)
class GPIEvidence:
    version: str
    protocol_valid: bool
    leakage_free: bool
    physical_scans: int = 0
    physical_families: int = 0
    untouched_family_holdout: bool = False
    independent_line_truth: bool = False
    matched_frozen_baseline: bool = False
    f1_delta: float | None = None
    exact_count_accuracy_delta: float | None = None
    minimum_family_recall_delta: float | None = None
    visual_gate_passed: bool | None = None
    deterministic_replay_passed: bool | None = None
    resource_gate_passed: bool | None = None
    all_target_gates_passed: bool = False


def grade(evidence: GPIEvidence) -> dict[str, Any]:
    if not evidence.version.strip():
        raise ValueError("version is required")
    if evidence.physical_scans < 0 or evidence.physical_families < 0:
        raise ValueError("physical counts must be non-negative")

    if not evidence.protocol_valid or not evidence.leakage_free:
        reasons = []
        if not evidence.protocol_valid:
            reasons.append("protocol_invalid_or_terminal_prephysical_failure")
        if not evidence.leakage_free:
            reasons.append("identity_truth_or_benchmark_leakage")
        return _report(evidence, GRADE_F, "NON_EVALUABLE", reasons)

    clean_physical_comparison = all(
        (
            evidence.physical_scans > 0,
            evidence.physical_families > 0,
            evidence.untouched_family_holdout,
            evidence.independent_line_truth,
            evidence.matched_frozen_baseline,
            evidence.f1_delta is not None,
            evidence.exact_count_accuracy_delta is not None,
            evidence.minimum_family_recall_delta is not None,
            evidence.visual_gate_passed is not None,
        )
    )
    if not clean_physical_comparison:
        return _report(
            evidence,
            GRADE_C,
            "NO_CLEAN_PHYSICAL_COMPARISON",
            _missing_clean_comparison_fields(evidence),
        )

    assert evidence.f1_delta is not None
    assert evidence.exact_count_accuracy_delta is not None
    assert evidence.minimum_family_recall_delta is not None
    negative = (
        evidence.visual_gate_passed is False
        or evidence.f1_delta < 0.0
        or evidence.exact_count_accuracy_delta < 0.0
        or evidence.minimum_family_recall_delta < 0.0
    )
    if negative:
        reasons = []
        if evidence.visual_gate_passed is False:
            reasons.append("native_resolution_visual_gate_failed")
        if evidence.f1_delta < 0.0:
            reasons.append("f1_regressed")
        if evidence.exact_count_accuracy_delta < 0.0:
            reasons.append("exact_count_accuracy_regressed")
        if evidence.minimum_family_recall_delta < 0.0:
            reasons.append("minimum_family_recall_regressed")
        return _report(evidence, GRADE_D, "NEGATIVE", reasons)

    positive = (
        evidence.visual_gate_passed is True
        and evidence.f1_delta > 0.0
        and evidence.exact_count_accuracy_delta >= 0.0
        and evidence.minimum_family_recall_delta >= 0.0
    )
    if not positive:
        return _report(
            evidence,
            GRADE_C,
            "INCONCLUSIVE_PHYSICAL_COMPARISON",
            ["no_strict_f1_gain_without_count_or_family_regression"],
        )

    verified_scale = evidence.physical_scans >= 30 and evidence.physical_families >= 15
    operationally_valid = all(
        (
            evidence.deterministic_replay_passed is True,
            evidence.resource_gate_passed is True,
        )
    )
    if verified_scale and operationally_valid and evidence.all_target_gates_passed:
        return _report(
            evidence,
            GRADE_A,
            "VERIFIED_POSITIVE",
            ["clean_30_scan_15_family_or_larger_win_with_all_frozen_gates"],
        )
    return _report(
        evidence,
        GRADE_B,
        "POSITIVE_BUT_UNDERPOWERED_OR_NOT_PROMOTION_COMPLETE",
        ["clean_physical_win_but_scale_or_full_gate_requirements_remain"],
    )


def _missing_clean_comparison_fields(evidence: GPIEvidence) -> list[str]:
    checks = {
        "no_physical_scans": evidence.physical_scans <= 0,
        "no_physical_families": evidence.physical_families <= 0,
        "not_untouched_family_holdout": not evidence.untouched_family_holdout,
        "no_independent_line_truth": not evidence.independent_line_truth,
        "no_matched_frozen_baseline": not evidence.matched_frozen_baseline,
        "no_f1_delta": evidence.f1_delta is None,
        "no_exact_count_delta": evidence.exact_count_accuracy_delta is None,
        "no_minimum_family_recall_delta": evidence.minimum_family_recall_delta is None,
        "no_visual_ruling": evidence.visual_gate_passed is None,
    }
    return [name for name, missing in checks.items() if missing]


def _report(
    evidence: GPIEvidence,
    grade_value: str,
    direction: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_generalization_progress_indicator",
        "version": evidence.version,
        "grade": grade_value,
        "direction": direction,
        "reasons": reasons,
        "evidence": {
            "protocol_valid": evidence.protocol_valid,
            "leakage_free": evidence.leakage_free,
            "physical_scans": evidence.physical_scans,
            "physical_families": evidence.physical_families,
            "untouched_family_holdout": evidence.untouched_family_holdout,
            "independent_line_truth": evidence.independent_line_truth,
            "matched_frozen_baseline": evidence.matched_frozen_baseline,
            "f1_delta": evidence.f1_delta,
            "exact_count_accuracy_delta": evidence.exact_count_accuracy_delta,
            "minimum_family_recall_delta": evidence.minimum_family_recall_delta,
            "visual_gate_passed": evidence.visual_gate_passed,
            "deterministic_replay_passed": evidence.deterministic_replay_passed,
            "resource_gate_passed": evidence.resource_gate_passed,
            "all_target_gates_passed": evidence.all_target_gates_passed,
        },
    }
