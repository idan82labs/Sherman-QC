#!/usr/bin/env python3
"""Apply a development-only source-axis/span recovery after V51.

The runtime boundary is deliberately truth-free.  It consumes an immutable
V51 result, an immutable cached V51-recovery result, the scan-only candidate
pool, and a frozen feature tree.  Candidate endpoints are reconstructed from
source-derived anchors/axes and deterministic span variants.  The model was
fitted on the consumed 2026-08-30 AI-adjudicated development cohort; this is
not promotion or generalization evidence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import numpy as np

from backend.f1_post_v51_precision_ownership_v52_development import (
    compose_v52_development,
)
from scripts.apply_f1_v51_source_only_recovery_v1 import candidate_endpoints
from scripts.evaluate_f1_against_manual_gold import _endpoints, pair_metrics


SCHEMA_VERSION = 1
MODEL_KIND = "sherman_qc_f1_v51_today_axis_span_recovery_model_v1"
RESULT_KIND = "sherman_qc_f1_v51_today_axis_span_recovery_result_v1"
PROFILE = "F1_V51_TODAY_AXIS_SPAN_RECOVERY_DEVELOPMENT_V1"

DEFAULT_SPAN_SCALES = (0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00)
DEFAULT_FIXED_SPANS_MM = (12.0, 16.0, 18.0, 20.0, 24.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 240.0)
ANCHOR_MODES = ("center", "low_endpoint", "high_endpoint")

BASE_FEATURE_NAMES = (
    "case_percentile_score",
    "case_percentile_consensus_count",
    "case_percentile_cluster_source_count",
    "case_percentile_support_count",
    "case_percentile_finite_support_fraction",
    "case_percentile_endpoint_support_fraction",
    "case_percentile_normal_second_eigenvalue",
    "case_percentile_corridor_points_per_mm",
    "case_percentile_transverse_angular_span_deg",
    "case_percentile_two_sided_transition_score",
    "finite_support_fraction",
    "endpoint_support_fraction",
    "normal_second_eigenvalue",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
    "cluster_source_count",
    "cluster_member_count",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "corridor_points_per_mm",
    "transverse_angular_span_deg",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_to_wide_normal_second_ratio",
    "angle_deg",
    "radius_mm",
)

VARIANT_FEATURE_NAMES = (
    "base_span_mm",
    "variant_span_mm",
    "span_ratio",
    "abs_log_span_ratio",
    "is_fixed_span",
    "is_center_anchor",
    "is_low_endpoint_anchor",
    "is_high_endpoint_anchor",
    "source_plane_lattice",
    "source_cylindrical_standard",
    "source_cylindrical_wide",
    "source_scan_mesh_adjacency",
    "source_strict_blind_owned",
)

FEATURE_NAMES = BASE_FEATURE_NAMES + VARIANT_FEATURE_NAMES


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite source feature")
    return result


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("candidate axis must be finite 3D")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("candidate axis is degenerate")
    value = value / norm
    pivot = int(np.argmax(np.abs(value)))
    return -value if value[pivot] < 0.0 else value


def _candidate_geometry(
    candidate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    endpoints = candidate_endpoints(candidate)
    axis = _canonical_axis(
        np.asarray(candidate.get("axis_direction", endpoints[1] - endpoints[0]), dtype=np.float64)
    )
    projections = [float(np.dot(point, axis)) for point in endpoints]
    low = np.asarray(endpoints[int(np.argmin(projections))], dtype=np.float64)
    high = np.asarray(endpoints[int(np.argmax(projections))], dtype=np.float64)
    center = np.asarray(candidate.get("anchor", np.mean(endpoints, axis=0)), dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("candidate anchor must be finite 3D")
    base_span = _number(candidate.get("span_mm"), float(np.linalg.norm(endpoints[1] - endpoints[0])))
    if base_span <= 1e-9:
        raise ValueError("candidate span must be positive")
    return center, low, high, axis, base_span


def _span_specs(
    base_span: float,
    span_scales: tuple[float, ...],
    fixed_spans_mm: tuple[float, ...],
) -> list[tuple[float, str, float]]:
    # Fixed spans own exact geometry ties, making the feature vector and ID
    # independent of iteration order.
    values: dict[float, tuple[float, str, float]] = {}
    for span in sorted(fixed_spans_mm):
        if span <= 1e-9 or not math.isfinite(span):
            raise ValueError("fixed spans must be finite and positive")
        key = round(float(span), 9)
        values[key] = (float(span), "fixed", float(span))
    for scale in sorted(span_scales):
        if scale <= 1e-9 or not math.isfinite(scale):
            raise ValueError("span scales must be finite and positive")
        span = float(base_span * scale)
        key = round(span, 9)
        values.setdefault(key, (span, "scale", float(scale)))
    return [values[key] for key in sorted(values)]


def _variant_features(
    candidate: dict[str, Any],
    *,
    base_span: float,
    variant_span: float,
    span_kind: str,
    anchor_mode: str,
) -> dict[str, float]:
    result = {name: _number(candidate.get(name)) for name in BASE_FEATURE_NAMES}
    source = str(candidate.get("representative_source") or candidate.get("candidate_source") or "")
    ratio = variant_span / base_span
    result.update(
        {
            "base_span_mm": base_span,
            "variant_span_mm": variant_span,
            "span_ratio": ratio,
            "abs_log_span_ratio": abs(math.log(max(ratio, 1e-12))),
            "is_fixed_span": float(span_kind == "fixed"),
            "is_center_anchor": float(anchor_mode == "center"),
            "is_low_endpoint_anchor": float(anchor_mode == "low_endpoint"),
            "is_high_endpoint_anchor": float(anchor_mode == "high_endpoint"),
            "source_plane_lattice": float(source == "plane_lattice"),
            "source_cylindrical_standard": float(source == "cylindrical_standard"),
            "source_cylindrical_wide": float(source == "cylindrical_wide"),
            "source_scan_mesh_adjacency": float(source == "scan_mesh_adjacency"),
            "source_strict_blind_owned": float(source == "strict_blind_owned"),
        }
    )
    return result


def generate_axis_span_variants(
    pool: dict[str, Any],
    *,
    span_scales: tuple[float, ...] = DEFAULT_SPAN_SCALES,
    fixed_spans_mm: tuple[float, ...] = DEFAULT_FIXED_SPANS_MM,
) -> list[dict[str, Any]]:
    """Generate stable truth-free variants from cached line objects."""

    if int(pool.get("working_point_budget") or 0) != 200000:
        raise ValueError("candidate pool point-budget contract mismatch")
    line_objects = pool.get("line_objects")
    if not isinstance(line_objects, list):
        raise ValueError("candidate pool line_objects are missing")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in sorted(line_objects, key=lambda row: str(row.get("candidate_id") or "")):
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_ids:
            raise ValueError("line-object candidate IDs must be unique and non-empty")
        seen_ids.add(candidate_id)
        center, low, high, axis, base_span = _candidate_geometry(candidate)
        source = str(
            candidate.get("representative_source")
            or candidate.get("candidate_source")
            or "unknown"
        )
        for span, span_kind, span_parameter in _span_specs(base_span, span_scales, fixed_spans_mm):
            endpoint_by_mode = {
                "center": (center - axis * span / 2.0, center + axis * span / 2.0),
                "low_endpoint": (low, low + axis * span),
                "high_endpoint": (high - axis * span, high),
            }
            for anchor_mode in ANCHOR_MODES:
                start, end = endpoint_by_mode[anchor_mode]
                identity_payload = {
                    "candidate_id": candidate_id,
                    "span_mm": round(span, 9),
                    "span_kind": span_kind,
                    "span_parameter": round(span_parameter, 9),
                    "anchor_mode": anchor_mode,
                    "line_start": start.tolist(),
                    "line_end": end.tolist(),
                }
                token = hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest()[:20]
                rows.append(
                    {
                        "bend_id": f"V51AS1-{token}",
                        "line_start": start.tolist(),
                        "line_end": end.tolist(),
                        "selection_route": "source_candidate_axis_span_variant_v1",
                        "source_hypothesis_id": candidate_id,
                        "source_kind": source,
                        "axis_span_variant": {
                            "anchor_mode": anchor_mode,
                            "span_kind": span_kind,
                            "span_parameter": round(span_parameter, 9),
                            "base_span_mm": round(base_span, 9),
                            "variant_span_mm": round(span, 9),
                        },
                        "features": _variant_features(
                            candidate,
                            base_span=base_span,
                            variant_span=span,
                            span_kind=span_kind,
                            anchor_mode=anchor_mode,
                        ),
                    }
                )
    rows.sort(key=lambda row: str(row["bend_id"]))
    return rows


def validate_model(model: dict[str, Any]) -> dict[str, Any]:
    if model.get("schema_version") != SCHEMA_VERSION or model.get("kind") != MODEL_KIND:
        raise ValueError("axis/span model schema or kind mismatch")
    if tuple(model.get("feature_names") or []) != FEATURE_NAMES:
        raise ValueError("axis/span model feature contract mismatch")
    tree = model.get("tree")
    if not isinstance(tree, dict):
        raise ValueError("axis/span model tree is missing")
    arrays = [tree.get(key) for key in ("children_left", "children_right", "feature_index", "threshold", "positive_probability")]
    if any(not isinstance(value, list) for value in arrays):
        raise ValueError("axis/span tree arrays are missing")
    if len({len(value) for value in arrays}) != 1 or not arrays[0]:
        raise ValueError("axis/span tree arrays are inconsistent")
    profile = model.get("profile")
    required = {
        "probability_threshold",
        "recovery_cap",
        "duplicate_angle_deg",
        "duplicate_offset_mm",
        "duplicate_span_overlap",
        "span_scales",
        "fixed_spans_mm",
    }
    if not isinstance(profile, dict) or not required.issubset(profile):
        raise ValueError("axis/span model profile is incomplete")
    return profile


def tree_probability(features: dict[str, float], model: dict[str, Any]) -> float:
    tree = model["tree"]
    node = 0
    visited = 0
    while True:
        left = int(tree["children_left"][node])
        right = int(tree["children_right"][node])
        if left == right == -1:
            result = _number(tree["positive_probability"][node])
            if not 0.0 <= result <= 1.0:
                raise ValueError("tree probability is outside [0,1]")
            return result
        feature_index = int(tree["feature_index"][node])
        threshold = _number(tree["threshold"][node])
        value = _number(features[FEATURE_NAMES[feature_index]])
        node = left if value <= threshold else right
        visited += 1
        if node < 0 or node >= len(tree["children_left"]) or visited > len(tree["children_left"]):
            raise ValueError("axis/span model tree is invalid")


def _duplicate(left: dict[str, Any], right: dict[str, Any], profile: dict[str, Any]) -> bool:
    metrics = pair_metrics(_endpoints(left), _endpoints(right))
    return bool(
        metrics["axis_angle_deg"] <= _number(profile["duplicate_angle_deg"])
        and metrics["transverse_offset_mm"] <= _number(profile["duplicate_offset_mm"])
        and metrics["shorter_span_overlap"] >= _number(profile["duplicate_span_overlap"])
    )


def compose_axis_span_recovery(
    v51_result: dict[str, Any],
    recall_result: dict[str, Any],
    pool: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    """Compose precision-retained V51, recall additions, and span variants."""

    profile = validate_model(model)
    for name, payload in (("v51", v51_result), ("recall", recall_result)):
        if payload.get("source_only") is not True or payload.get("research_only") is not True:
            raise ValueError(f"{name} input is not source-only/research-only")
        if any(payload.get(flag) is not False for flag in ("labels_used", "cad_used", "part_identity_used")):
            raise ValueError(f"{name} input reports a forbidden inference source")
    precision_result = compose_v52_development(v51_result, pool)
    original_ids = {str(row.get("bend_id")) for row in v51_result.get("predictions") or []}
    recall_additions = [
        deepcopy(row)
        for row in recall_result.get("predictions") or []
        if str(row.get("bend_id")) not in original_ids
    ]
    retained = [deepcopy(row) for row in precision_result["predictions"]] + recall_additions
    if len({str(row.get("bend_id")) for row in retained}) != len(retained):
        raise ValueError("base union contains duplicate bend IDs")

    variants = generate_axis_span_variants(
        pool,
        span_scales=tuple(float(value) for value in profile["span_scales"]),
        fixed_spans_mm=tuple(float(value) for value in profile["fixed_spans_mm"]),
    )
    threshold = _number(profile["probability_threshold"])
    eligible = []
    for row in variants:
        probability = round(tree_probability(row["features"], model), 12)
        if probability >= threshold:
            row["selector_score"] = probability
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            -float(row["selector_score"]),
            abs(math.log(max(float(row["axis_span_variant"]["variant_span_mm"]) / float(row["axis_span_variant"]["base_span_mm"]), 1e-12))),
            str(row["bend_id"]),
        )
    )

    emitted: list[dict[str, Any]] = []
    owned_hypotheses: set[str] = set()
    audit: list[dict[str, Any]] = []
    cap = int(profile["recovery_cap"])
    for row in eligible:
        hypothesis_id = str(row["source_hypothesis_id"])
        reason = "accepted"
        if hypothesis_id in owned_hypotheses:
            reason = "same_source_hypothesis_already_owned"
        elif any(_duplicate(row, incumbent, profile) for incumbent in retained):
            reason = "base_union_owns_duplicate"
        elif any(_duplicate(row, incumbent, profile) for incumbent in emitted):
            reason = "higher_ranked_variant_owns_duplicate"
        if reason == "accepted" and len(emitted) < cap:
            clean = {key: value for key, value in row.items() if key != "features"}
            emitted.append(clean)
            owned_hypotheses.add(hypothesis_id)
        elif reason == "accepted":
            reason = "case_recovery_cap_reached"
        audit.append(
            {
                "bend_id": row["bend_id"],
                "source_hypothesis_id": hypothesis_id,
                "selector_score": row["selector_score"],
                "decision": reason,
            }
        )
    predictions = retained + emitted
    predictions.sort(key=lambda row: (str(row.get("bend_id")), str(row.get("source_hypothesis_id"))))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "profile": PROFILE,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "development_tuned_model": True,
        "axis_span_audit": {
            "precision_retained_count": len(precision_result["predictions"]),
            "recall_addition_count": len(recall_additions),
            "base_union_count": len(retained),
            "generated_variant_count": len(variants),
            "eligible_variant_count": len(eligible),
            "emitted_variant_count": len(emitted),
            "probability_threshold": threshold,
            "recovery_cap": cap,
            "decisions": audit,
        },
        "axis_span_model_sha256": hashlib.sha256(_canonical_bytes(model)).hexdigest(),
    }
