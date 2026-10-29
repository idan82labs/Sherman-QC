#!/usr/bin/env python3
"""Unambiguous metrics for future blind physical F1 evaluations.

This module does not run inference and must only receive post-inference scoring
and manual visual-adjudication records.  It deliberately separates count,
localization, and marker-rendering evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 2
VISUAL_PASS_RULING = "coherent"
VISUAL_FAILURE_RULINGS = frozenset({"geometric_failure", "render_failure"})
VISUAL_RULINGS = frozenset({VISUAL_PASS_RULING, *VISUAL_FAILURE_RULINGS})


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _required_nonnegative_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def score_localized_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate already-matched cases without overloading count accuracy.

    Each row must contain gold_count, prediction_count, true_positive,
    false_positive, and false_negative.  The caller remains responsible for
    the frozen geometric matcher that produced those integers.
    """

    if not rows:
        raise ValueError("at least one scored case is required")

    exact_count_cases = 0
    exact_localized_cases = 0
    tp_total = fp_total = fn_total = 0
    case_metrics: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in seen_case_ids:
            raise ValueError("case_id must be present and unique")
        seen_case_ids.add(case_id)

        gold_count = _required_nonnegative_int(row, "gold_count")
        prediction_count = _required_nonnegative_int(row, "prediction_count")
        true_positive = _required_nonnegative_int(row, "true_positive")
        false_positive = _required_nonnegative_int(row, "false_positive")
        false_negative = _required_nonnegative_int(row, "false_negative")
        if true_positive + false_negative != gold_count:
            raise ValueError(f"{case_id}: TP + FN must equal gold_count")
        if true_positive + false_positive != prediction_count:
            raise ValueError(f"{case_id}: TP + FP must equal prediction_count")

        exact_count = prediction_count == gold_count
        exact_localized_target_set = (
            true_positive == gold_count == prediction_count
            and false_positive == 0
            and false_negative == 0
        )
        exact_count_cases += int(exact_count)
        exact_localized_cases += int(exact_localized_target_set)
        tp_total += true_positive
        fp_total += false_positive
        fn_total += false_negative
        case_metrics.append(
            {
                "case_id": case_id,
                "gold_count": gold_count,
                "prediction_count": prediction_count,
                "exact_count": exact_count,
                "exact_localized_target_set": exact_localized_target_set,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )

    precision = _ratio(tp_total, tp_total + fp_total)
    recall = _ratio(tp_total, tp_total + fn_total)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    denominator = len(case_metrics)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_count": denominator,
        "exact_count_cases": exact_count_cases,
        "exact_count_accuracy": exact_count_cases / denominator,
        "exact_localized_target_set_cases": exact_localized_cases,
        "exact_localized_target_set_accuracy": exact_localized_cases / denominator,
        "true_positive": tp_total,
        "false_positive": fp_total,
        "false_negative": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cases": case_metrics,
    }


def adjudicate_visual_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_marker_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Aggregate native-resolution visual and marker-level adjudication.

    Every emitted marker must appear exactly once with an explicit boolean
    ``render_admissible`` judgment.  Marker admissibility is therefore a
    marker-level rate; gold bend count never enters this calculation.  Missed
    bends remain the responsibility of recall.
    """

    if not rows:
        raise ValueError("at least one visual case is required")

    expected_cases = set(expected_marker_ids)
    observed_cases: set[str] = set()
    evidence_complete = True
    all_cases_coherent = True
    admissible_markers = 0
    marker_denominator = 0
    cases: list[dict[str, Any]] = []

    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in observed_cases:
            raise ValueError("case_id must be present and unique")
        observed_cases.add(case_id)
        if case_id not in expected_marker_ids:
            raise ValueError(f"unexpected visual case: {case_id}")

        ruling = str(row.get("ruling") or "")
        if ruling not in VISUAL_RULINGS:
            raise ValueError(f"{case_id}: invalid visual ruling")
        reviewer = str(row.get("reviewer") or "").strip()
        artifact_checks = row.get("artifact_checks")
        artifacts_ok = (
            isinstance(artifact_checks, Mapping)
            and bool(artifact_checks)
            and all(value is True for value in artifact_checks.values())
        )

        expected_ids = [str(value) for value in expected_marker_ids[case_id]]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError(f"{case_id}: expected marker IDs are not unique")
        marker_rows = list(row.get("markers") or [])
        observed_ids = [str(marker.get("marker_id") or "") for marker in marker_rows]
        marker_set_complete = (
            len(observed_ids) == len(set(observed_ids))
            and set(observed_ids) == set(expected_ids)
        )
        marker_judgments_valid = all(
            isinstance(marker.get("render_admissible"), bool) for marker in marker_rows
        )
        complete = bool(reviewer) and artifacts_ok and marker_set_complete and marker_judgments_valid
        coherent = ruling == VISUAL_PASS_RULING
        evidence_complete &= complete
        all_cases_coherent &= coherent
        if complete:
            marker_denominator += len(marker_rows)
            admissible_markers += sum(bool(marker["render_admissible"]) for marker in marker_rows)

        cases.append(
            {
                "case_id": case_id,
                "ruling": ruling,
                "reviewer_present": bool(reviewer),
                "artifacts_ok": artifacts_ok,
                "marker_set_complete": marker_set_complete,
                "marker_judgments_valid": marker_judgments_valid,
                "evidence_complete": complete,
                "coherent": coherent,
                "marker_count": len(marker_rows),
                "render_admissible_marker_count": sum(
                    bool(marker.get("render_admissible")) for marker in marker_rows
                ),
            }
        )

    case_set_complete = observed_cases == expected_cases
    evidence_complete &= case_set_complete
    visual_gate_passed = evidence_complete and all_cases_coherent
    return {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(rows),
        "expected_case_count": len(expected_cases),
        "case_set_complete": case_set_complete,
        "visual_evidence_complete": evidence_complete,
        "all_cases_coherent": all_cases_coherent,
        "visual_gate_passed": visual_gate_passed,
        "render_admissible_markers": admissible_markers,
        "marker_denominator": marker_denominator,
        "render_admissible_marker_rate": _ratio(admissible_markers, marker_denominator),
        "cases": cases,
    }
