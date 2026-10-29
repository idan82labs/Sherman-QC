"""Feature extraction and geometric suppression for selector V2.

Only source-derived candidate-pool fields are accepted.  Absolute part or
case identity, filenames, CAD, labels, expected counts, and oracle feedback
are deliberately absent from the feature vector.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


PROFILE = "F1_MULTIFAMILY_NEIGHBOR_CONSENSUS_SELECTOR_V2"
RANDOM_STATE = 20260810
SCORE_THRESHOLD = 0.80
NMS_ANGLE_DEG = 5.0
NMS_DISTANCE_MM = 4.0
NMS_OVERLAP = 0.60
SOURCE_CATEGORIES = (
    "bend_hypothesis",
    "control_support_strip",
    "cylindrical_standard",
    "cylindrical_wide",
    "plane_lattice",
    "scan_mesh_adjacency",
    "strict_blind_owned",
)
BASE_FEATURE_NAMES = (
    "span_mm",
    "radius_mm",
    "transverse_offset_mm",
    "vote_count",
    "support_count",
    "normal_spread_deg",
    "score",
    "consensus_count",
    "radius_consensus_count",
    "finite_support_fraction",
    "finite_support_longest_run_fraction",
    "endpoint_support_fraction",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "corridor_points_per_mm",
    "transverse_angular_span_deg",
    "transverse_angular_bin_fraction",
    "median_corridor_radius_mm",
    "normal_second_eigenvalue",
    "normal_spread_deg_scan",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_normal_second_eigenvalue_r6mm",
    "tight_to_wide_normal_second_ratio",
    "cluster_member_count",
    "cluster_source_count",
    "case_percentile_support_count",
    "case_percentile_score",
    "case_percentile_consensus_count",
    "case_percentile_finite_support_fraction",
    "case_percentile_finite_support_longest_run_fraction",
    "case_percentile_endpoint_support_fraction",
    "case_percentile_normal_second_eigenvalue",
    "case_percentile_normal_spread_deg_scan",
    "case_percentile_tight_normal_second_eigenvalue_r2mm",
    "case_percentile_tight_to_wide_normal_second_ratio",
    "case_percentile_corridor_points_per_mm",
    "case_percentile_transverse_angular_span_deg",
    "case_percentile_cluster_member_count",
    "case_percentile_cluster_source_count",
    "case_percentile_two_sided_transition_score",
    "case_percentile_two_sided_normal_angle_deg",
    "case_percentile_two_sided_support_balance",
    "case_percentile_two_sided_min_support_fraction",
    "case_percentile_two_sided_normal_coherence",
)
NEIGHBORHOODS = (
    (2.0, 2.0, 0.80),
    (3.0, 3.0, 0.70),
    (5.0, 4.0, 0.60),
    (8.0, 6.0, 0.50),
)
NEIGHBOR_FEATURE_SUFFIXES = (
    "count",
    "source_count",
    "log1p_count",
    "max_case_percentile_score",
    "mean_case_percentile_score",
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *(
        f"neighbor_{angle:g}_{distance:g}_{overlap:g}_{suffix}"
        for angle, distance, overlap in NEIGHBORHOODS
        for suffix in NEIGHBOR_FEATURE_SUFFIXES
    ),
    *(f"source_is_{value}" for value in SOURCE_CATEGORIES),
)


def _finite(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    result = float(value)
    return result if math.isfinite(result) else 0.0


def endpoints(row: dict[str, Any]) -> np.ndarray:
    values = row.get("span_endpoints")
    if isinstance(values, list) and len(values) >= 2:
        result = np.asarray(values[:2], dtype=float)
    else:
        anchor = np.asarray(row.get("anchor"), dtype=float)
        axis = np.asarray(row.get("axis_direction"), dtype=float)
        span = _finite(row.get("span_mm"))
        if anchor.shape != (3,) or axis.shape != (3,) or span <= 0.0:
            raise ValueError(f"candidate has no usable line geometry: {row.get('candidate_id')}")
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError(f"candidate has a zero axis: {row.get('candidate_id')}")
        axis = axis / norm
        result = np.stack((anchor - 0.5 * span * axis, anchor + 0.5 * span * axis))
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError(f"candidate line geometry is invalid: {row.get('candidate_id')}")
    if float(np.linalg.norm(result[1] - result[0])) <= 0.0:
        raise ValueError(f"candidate line has zero span: {row.get('candidate_id')}")
    return result


def _geometry_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    line_endpoints = np.asarray([endpoints(row) for row in rows], dtype=float)
    axes = line_endpoints[:, 1] - line_endpoints[:, 0]
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes / lengths[:, None]
    midpoints = line_endpoints.mean(axis=1)
    angle = np.degrees(np.arccos(np.clip(np.abs(axes @ axes.T), -1.0, 1.0)))
    delta = midpoints[:, None, :] - midpoints[None, :, :]
    projected = (delta * axes[:, None, :]).sum(axis=2)
    perpendicular = np.linalg.norm(
        delta - projected[:, :, None] * axes[:, None, :], axis=2
    )
    symmetric_perpendicular = np.maximum(perpendicular, perpendicular.T)
    center_distance = np.abs(projected)
    overlap = np.maximum(0.0, (lengths[:, None] + lengths[None, :]) / 2.0 - center_distance)
    overlap_fraction = overlap / np.minimum(lengths[:, None], lengths[None, :])
    return line_endpoints, angle, symmetric_perpendicular, overlap_fraction


def build_feature_matrix(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    values = [dict(row) for row in rows]
    if not values:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    base = np.asarray(
        [[_finite(row.get(name)) for name in BASE_FEATURE_NAMES] for row in values],
        dtype=float,
    )
    _, angle, distance, overlap = _geometry_arrays(values)
    score_column = BASE_FEATURE_NAMES.index("case_percentile_score")
    sources = [str(row.get("representative_source") or "unknown") for row in values]
    neighbor_columns: list[list[float]] = []
    for index in range(len(values)):
        extra: list[float] = []
        for max_angle, max_distance, min_overlap in NEIGHBORHOODS:
            members = np.flatnonzero(
                (angle[index] <= max_angle)
                & (distance[index] <= max_distance)
                & (overlap[index] >= min_overlap)
            )
            scores = base[members, score_column]
            extra.extend(
                (
                    float(len(members)),
                    float(len({sources[member] for member in members})),
                    float(np.log1p(len(members))),
                    float(np.max(scores)),
                    float(np.mean(scores)),
                )
            )
        neighbor_columns.append(extra)
    source_columns = [
        [float(source == category) for category in SOURCE_CATEGORIES]
        for source in sources
    ]
    result = np.hstack(
        (base, np.asarray(neighbor_columns, dtype=float), np.asarray(source_columns, dtype=float))
    )
    if result.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("selector feature schema drifted")
    return result


def _similar(
    first: int,
    second: int,
    angle: np.ndarray,
    distance: np.ndarray,
    overlap: np.ndarray,
) -> bool:
    return bool(
        angle[first, second] <= NMS_ANGLE_DEG
        and distance[first, second] <= NMS_DISTANCE_MM
        and overlap[first, second] >= NMS_OVERLAP
    )


def select_indices(rows: list[dict[str, Any]], scores: np.ndarray) -> list[int]:
    return select_indices_configurable(
        rows,
        scores,
        threshold=SCORE_THRESHOLD,
        max_angle=NMS_ANGLE_DEG,
        max_distance=NMS_DISTANCE_MM,
        min_overlap=NMS_OVERLAP,
    )


def select_indices_configurable(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    threshold: float,
    max_angle: float,
    max_distance: float,
    min_overlap: float,
) -> list[int]:
    if len(rows) != len(scores):
        raise ValueError("row and score counts differ")
    if not rows:
        return []
    _, angle, distance, overlap = _geometry_arrays(rows)
    order = sorted(
        range(len(rows)),
        key=lambda index: (-float(scores[index]), str(rows[index].get("candidate_id") or "")),
    )
    accepted: list[int] = []
    for index in order:
        if float(scores[index]) < threshold:
            break
        if all(
            not (
                angle[index, prior] <= max_angle
                and distance[index, prior] <= max_distance
                and overlap[index, prior] >= min_overlap
            )
            for prior in accepted
        ):
            accepted.append(index)
    return accepted
