"""Development-tuned, source-only precision ownership after frozen F1 V51.

This module is deliberately outside the frozen V51 chain.  It reads only the
V51 result and its source-derived candidate pool at inference time.  The gates
were calibrated on the 2026-08-30 AI-adjudicated development cohort, so this is
research-only and must not be presented as promotion or generalization evidence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from scripts.evaluate_f1_against_manual_gold import _endpoints, pair_metrics


PROFILE = "F1_V52_DEVELOPMENT_PRECISION_OWNERSHIP_V1"
CALIBRATION_TRUTH_TIER = "codex_manual_multiview_scan_only"
CALIBRATED_NUMERIC_CUTOFF_COUNT = 23

# These family-specific gates are intentionally explicit.  Missing evidence
# fails closed.  Thresholds were selected on one development cohort only.
PLANE_MAX_LENGTH_MM = 80.0
PLANE_MIN_NORMAL_SECOND_EIGENVALUE = 0.375
PLANE_MIN_TWO_SIDED_TRANSITION_SCORE = 0.84

SCAN_MIN_LENGTH_MM = 65.0
SCAN_MAX_SURFACE_GAP_MM = 0.15
SCAN_MAX_P90_LINE_DISTANCE_MM = 1.25
SCAN_MIN_NORMAL_SECOND_EIGENVALUE = 0.30
SCAN_MIN_TWO_SIDED_TRANSITION_SCORE = 0.75

COMPACT_CYLINDER_MIN_LENGTH_MM = 18.0
COMPACT_CYLINDER_MAX_LENGTH_MM = 21.0
COMPACT_CYLINDER_MAX_NORMAL_SECOND_EIGENVALUE = 0.32
COMPACT_CYLINDER_MAX_P90_LINE_DISTANCE_MM = 2.0
COMPACT_CYLINDER_MIN_TWO_SIDED_TRANSITION_SCORE = 0.80

LONG_CYLINDER_MIN_LENGTH_MM = 150.0
LONG_CYLINDER_MIN_SELECTOR_SCORE = 1.0
LONG_CYLINDER_MIN_NORMAL_SECOND_EIGENVALUE = 0.43
LONG_CYLINDER_MAX_P90_LINE_DISTANCE_MM = 1.25
LONG_CYLINDER_MIN_TWO_SIDED_TRANSITION_SCORE = 0.95

CONSENSUS_MIN_LENGTH_MM = 150.0
CONSENSUS_MIN_SELECTOR_SCORE = 0.85

SIDECAR_MIN_NORMAL_SECOND_EIGENVALUE = 0.30
SIDECAR_MAX_P90_LINE_DISTANCE_MM = 2.0
SIDECAR_MIN_TWO_SIDED_TRANSITION_SCORE = 0.70

DUPLICATE_MAX_AXIS_ANGLE_DEG = 2.0
DUPLICATE_MAX_TRANSVERSE_OFFSET_MM = 4.0
DUPLICATE_MIN_SHORTER_SPAN_OVERLAP = 0.80


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _length(row: dict[str, Any]) -> float:
    start = row.get("line_start")
    end = row.get("line_end")
    if not isinstance(start, list) or not isinstance(end, list) or len(start) != 3 or len(end) != 3:
        raise ValueError("every V51 prediction must have three-dimensional line_start/line_end")
    value = math.dist([float(item) for item in start], [float(item) for item in end])
    if not math.isfinite(value) or value <= 1e-9:
        raise ValueError("every V51 prediction must have a finite nonzero line length")
    return value


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _candidate_map(pool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in pool.get("candidates") or []:
        if not isinstance(value, dict):
            raise ValueError("candidate pool rows must be objects")
        candidate_id = str(value.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate pool row is missing candidate_id")
        if candidate_id in result and _canonical(result[candidate_id]) != _canonical(value):
            raise ValueError(f"candidate pool contains conflicting rows for {candidate_id}")
        result[candidate_id] = deepcopy(value)
    return result


def _quality_snapshot(candidate: dict[str, Any] | None) -> dict[str, float | None]:
    row = candidate or {}
    return {
        "normal_second_eigenvalue": _number(row, "normal_second_eigenvalue"),
        "p90_line_distance_mm": _number(row, "p90_line_distance_mm"),
        "surface_gap_mm": _number(row, "min_surface_gap"),
        "two_sided_transition_score": _number(row, "two_sided_transition_score"),
    }


def _all_present(values: list[float | None]) -> bool:
    return all(value is not None for value in values)


def _gate(
    prediction: dict[str, Any], candidate: dict[str, Any] | None
) -> tuple[bool, str, dict[str, float | None]]:
    kind = str(prediction.get("source_kind") or "")
    length = _length(prediction)
    selector = _number(prediction, "selector_score")
    quality = _quality_snapshot(candidate)
    quality.update({"length_mm": length, "selector_score": selector})
    normal = quality["normal_second_eigenvalue"]
    p90 = quality["p90_line_distance_mm"]
    gap = quality["surface_gap_mm"]
    transition = quality["two_sided_transition_score"]

    if kind == "plane_lattice":
        accepted = _all_present([normal, transition]) and bool(
            length <= PLANE_MAX_LENGTH_MM
            and normal >= PLANE_MIN_NORMAL_SECOND_EIGENVALUE
            and transition >= PLANE_MIN_TWO_SIDED_TRANSITION_SCORE
        )
        return accepted, "plane_lattice_quality_gate", quality

    if kind == "scan_mesh_adjacency":
        accepted = _all_present([gap, p90, normal, transition]) and bool(
            length >= SCAN_MIN_LENGTH_MM
            and gap <= SCAN_MAX_SURFACE_GAP_MM
            and p90 <= SCAN_MAX_P90_LINE_DISTANCE_MM
            and normal >= SCAN_MIN_NORMAL_SECOND_EIGENVALUE
            and transition >= SCAN_MIN_TWO_SIDED_TRANSITION_SCORE
        )
        return accepted, "scan_adjacency_quality_gate", quality

    if kind == "cylindrical_standard":
        compact = _all_present([normal, p90, transition]) and bool(
            COMPACT_CYLINDER_MIN_LENGTH_MM <= length <= COMPACT_CYLINDER_MAX_LENGTH_MM
            and normal <= COMPACT_CYLINDER_MAX_NORMAL_SECOND_EIGENVALUE
            and p90 <= COMPACT_CYLINDER_MAX_P90_LINE_DISTANCE_MM
            and transition >= COMPACT_CYLINDER_MIN_TWO_SIDED_TRANSITION_SCORE
        )
        long = _all_present([selector, normal, p90, transition]) and bool(
            length >= LONG_CYLINDER_MIN_LENGTH_MM
            and selector >= LONG_CYLINDER_MIN_SELECTOR_SCORE
            and normal >= LONG_CYLINDER_MIN_NORMAL_SECOND_EIGENVALUE
            and p90 <= LONG_CYLINDER_MAX_P90_LINE_DISTANCE_MM
            and transition >= LONG_CYLINDER_MIN_TWO_SIDED_TRANSITION_SCORE
        )
        return compact or long, "cylindrical_compact_or_elite_long_gate", quality

    if kind == "source_consensus_midpoint":
        accepted = _all_present([selector]) and bool(
            length >= CONSENSUS_MIN_LENGTH_MM and selector >= CONSENSUS_MIN_SELECTOR_SCORE
        )
        return accepted, "cross_source_consensus_gate", quality

    if kind == "v41_fragment_tolerant_parent_bridge_sidecar":
        accepted = _all_present([normal, p90, transition]) and bool(
            normal >= SIDECAR_MIN_NORMAL_SECOND_EIGENVALUE
            and p90 <= SIDECAR_MAX_P90_LINE_DISTANCE_MM
            and transition >= SIDECAR_MIN_TWO_SIDED_TRANSITION_SCORE
        )
        return accepted, "fragment_sidecar_quality_gate", quality

    return False, "unsupported_or_low_precision_source_family", quality


def _prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("bend_id") or ""),
        str(row.get("source_kind") or ""),
        _canonical([row.get("line_start"), row.get("line_end")]),
    )


def _is_duplicate(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    metrics = pair_metrics(_endpoints(left), _endpoints(right))
    duplicate = bool(
        float(metrics["axis_angle_deg"]) <= DUPLICATE_MAX_AXIS_ANGLE_DEG
        and float(metrics["transverse_offset_mm"]) <= DUPLICATE_MAX_TRANSVERSE_OFFSET_MM
        and float(metrics["shorter_span_overlap"]) >= DUPLICATE_MIN_SHORTER_SPAN_OVERLAP
    )
    return duplicate, metrics


def compose_v52_development(
    v51_result: dict[str, Any], pool: dict[str, Any]
) -> dict[str, Any]:
    """Apply deterministic, development-calibrated precision ownership to V51."""

    if bool(v51_result.get("cad_used")) or bool(v51_result.get("part_identity_used")):
        raise ValueError("V52 development precision ownership requires source-only V51 input")
    if bool(v51_result.get("labels_used")):
        raise ValueError("V52 inference input must not contain labels")

    source = deepcopy(v51_result)
    candidates = _candidate_map(pool)
    gate_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    decisions: list[dict[str, Any]] = []
    for prediction_value in source.get("predictions") or []:
        if not isinstance(prediction_value, dict):
            raise ValueError("V51 predictions must be objects")
        prediction = deepcopy(prediction_value)
        prediction_id = str(prediction.get("bend_id") or "")
        hypothesis_id = str(prediction.get("source_hypothesis_id") or "")
        candidate = candidates.get(hypothesis_id)
        accepted, gate_name, quality = _gate(prediction, candidate)
        decision = {
            "prediction_id": prediction_id,
            "source_hypothesis_id": hypothesis_id,
            "source_kind": str(prediction.get("source_kind") or ""),
            "gate": gate_name,
            "gate_accepted": accepted,
            "final_accepted": accepted,
            "reason": "quality_gate_accepted" if accepted else "quality_gate_vetoed",
            "quality": {
                key: None if value is None else round(float(value), 12)
                for key, value in quality.items()
            },
        }
        decisions.append(decision)
        if accepted:
            gate_rows.append((prediction, decision))

    # Reconcile exact/near-exact cross-family aliases after quality gating.  The
    # highest selector score owns the line; remaining ties use a stable key.
    gate_rows.sort(
        key=lambda item: (
            -float(_number(item[0], "selector_score") or 0.0),
            _prediction_key(item[0]),
        )
    )
    retained: list[dict[str, Any]] = []
    duplicate_audit: list[dict[str, Any]] = []
    for prediction, decision in gate_rows:
        duplicate_owner: dict[str, Any] | None = None
        duplicate_metrics: dict[str, float] | None = None
        for incumbent in retained:
            duplicate, metrics = _is_duplicate(prediction, incumbent)
            if duplicate:
                duplicate_owner = incumbent
                duplicate_metrics = metrics
                break
        if duplicate_owner is None:
            retained.append(prediction)
            continue
        decision["final_accepted"] = False
        decision["reason"] = "near_duplicate_owned_by_higher_ranked_prediction"
        decision["duplicate_owner_prediction_id"] = str(duplicate_owner.get("bend_id") or "")
        duplicate_audit.append(
            {
                "suppressed_prediction_id": str(prediction.get("bend_id") or ""),
                "owner_prediction_id": str(duplicate_owner.get("bend_id") or ""),
                "axis_angle_deg": round(float(duplicate_metrics["axis_angle_deg"]), 12),
                "transverse_offset_mm": round(
                    float(duplicate_metrics["transverse_offset_mm"]), 12
                ),
                "shorter_span_overlap": round(
                    float(duplicate_metrics["shorter_span_overlap"]), 12
                ),
            }
        )

    retained.sort(key=_prediction_key)
    decisions.sort(key=lambda row: (str(row["prediction_id"]), str(row["source_hypothesis_id"])))
    duplicate_audit.sort(
        key=lambda row: (str(row["suppressed_prediction_id"]), str(row["owner_prediction_id"]))
    )
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v52_development_precision_ownership_result",
        "profile": PROFILE,
        "v51_retained_as_input": True,
        "v51_result_sha256": str(v51_result.get("v51_result_sha256") or ""),
        "candidate_count": len(retained),
        "predictions": retained,
        "precision_ownership_decisions": decisions,
        "duplicate_reconciliation": duplicate_audit,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
        "promotion_eligible": False,
        "development_tuning_labels_used": True,
        "development_tuning_truth_tier": CALIBRATION_TRUTH_TIER,
    }
    hash_payload = {
        "profile": PROFILE,
        "predictions": retained,
        "decisions": decisions,
        "duplicate_reconciliation": duplicate_audit,
        "v51_result_sha256": result["v51_result_sha256"],
    }
    result["v52_result_sha256"] = hashlib.sha256(
        _canonical(hash_payload).encode("utf-8")
    ).hexdigest()
    return result
