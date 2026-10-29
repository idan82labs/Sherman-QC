"""Deterministic support gates for plane-lattice bend recovery.

This module deliberately does not decide how many bends a part should have.
It turns independently discovered planar surface hypotheses into finite bend
segments only when both planes provide local scan support along their
intersection.  The helper is kept separate from the F1 solver while the
geometry is evaluated on the blind corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class _Plane:
    plane_id: str
    normal: np.ndarray
    offset: float
    support_count: int


def extract_deterministic_planes(
    points: np.ndarray,
    *,
    distance_threshold_mm: float = 0.8,
    max_planes: int = 12,
    min_plane_points: int = 75,
    seed: int = 7,
    iterations: int = 500,
) -> list[dict[str, Any]]:
    """Sequential fixed-seed RANSAC with deterministic PCA refinement."""

    remaining = np.asarray(points, dtype=np.float64)
    if remaining.ndim != 2 or remaining.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    remaining = remaining[np.isfinite(remaining).all(axis=1)]
    order = np.lexsort((remaining[:, 2], remaining[:, 1], remaining[:, 0]))
    remaining = remaining[order]
    rng = np.random.default_rng(seed)
    planes: list[dict[str, Any]] = []
    for plane_index in range(max_planes):
        point_count = len(remaining)
        if point_count < max(3, min_plane_points):
            break
        best_mask: np.ndarray | None = None
        best_key: tuple[Any, ...] | None = None
        for _ in range(iterations):
            sample_indices = rng.choice(point_count, size=3, replace=False)
            first, second, third = remaining[sample_indices]
            normal = np.cross(second - first, third - first)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-9:
                continue
            normal /= norm
            offset = -float(np.dot(normal, first))
            pivot = int(np.argmax(np.abs(normal)))
            if normal[pivot] < 0.0:
                normal = -normal
                offset = -offset
            residual = np.abs(remaining @ normal + offset)
            mask = residual <= distance_threshold_mm
            inlier_count = int(np.count_nonzero(mask))
            if inlier_count < min_plane_points:
                continue
            median_residual = float(np.median(residual[mask]))
            key = (
                -inlier_count,
                median_residual,
                *(round(float(value), 12) for value in normal),
                round(offset, 12),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_mask = mask
        if best_mask is None:
            break

        refined_mask = best_mask
        normal = np.zeros(3, dtype=np.float64)
        offset = 0.0
        for _ in range(2):
            inlier_points = remaining[refined_mask]
            center = np.mean(inlier_points, axis=0)
            covariance = (inlier_points - center).T @ (inlier_points - center)
            _values, vectors = np.linalg.eigh(covariance)
            normal = vectors[:, 0]
            pivot = int(np.argmax(np.abs(normal)))
            if normal[pivot] < 0.0:
                normal = -normal
            offset = -float(np.dot(normal, center))
            refined_mask = np.abs(remaining @ normal + offset) <= distance_threshold_mm
        support_count = int(np.count_nonzero(refined_mask))
        if support_count < min_plane_points:
            break
        planes.append(
            {
                "plane_id": f"RP{plane_index + 1}",
                "normal": normal.tolist(),
                "offset": offset,
                "support_count": support_count,
            }
        )
        remaining = remaining[~refined_mask]
    return planes


def _canonical_plane(record: Mapping[str, Any], index: int) -> _Plane:
    normal = np.asarray(record.get("normal"), dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if normal.shape != (3,) or not np.isfinite(normal).all() or norm <= 1e-12:
        raise ValueError("plane normal must be a finite non-zero 3-vector")
    normal = normal / norm
    offset = float(record.get("offset", record.get("d", 0.0))) / norm
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0.0:
        normal = -normal
        offset = -offset
    return _Plane(
        plane_id=str(record.get("plane_id") or f"P{index + 1}"),
        normal=normal,
        offset=offset,
        support_count=int(record.get("support_count") or 0),
    )


def merge_near_duplicate_planes(
    plane_hypotheses: Sequence[Mapping[str, Any]],
    *,
    max_angle_deg: float = 3.0,
    max_offset_mm: float = 2.5,
) -> list[dict[str, Any]]:
    """Collapse thickness/noise replicas of the same geometric plane.

    The highest-support member is retained as the representative.  Plane
    identity is geometry-only; input order is used only as a final stable
    tiebreak.
    """

    planes = [_canonical_plane(record, index) for index, record in enumerate(plane_hypotheses)]
    planes.sort(key=lambda item: (-item.support_count, item.plane_id))
    cosine = float(np.cos(np.deg2rad(max_angle_deg)))
    groups: list[list[_Plane]] = []
    for plane in planes:
        for group in groups:
            representative = group[0]
            if (
                float(np.dot(plane.normal, representative.normal)) >= cosine
                and abs(plane.offset - representative.offset) <= max_offset_mm
            ):
                group.append(plane)
                break
        else:
            groups.append([plane])

    merged: list[dict[str, Any]] = []
    for group in groups:
        representative = group[0]
        merged.append(
            {
                "plane_id": representative.plane_id,
                "normal": representative.normal.tolist(),
                "offset": float(representative.offset),
                "support_count": int(representative.support_count),
                "merged_plane_ids": sorted(item.plane_id for item in group),
            }
        )
    merged.sort(key=lambda item: str(item["plane_id"]))
    return merged


def _plane_intersection(first: _Plane, second: _Plane) -> tuple[np.ndarray, np.ndarray] | None:
    axis = np.cross(first.normal, second.normal)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return None
    axis = axis / axis_norm
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
    matrix = np.vstack((first.normal, second.normal))
    anchor = matrix.T @ np.linalg.solve(
        matrix @ matrix.T,
        -np.asarray((first.offset, second.offset), dtype=np.float64),
    )
    return anchor, axis


def _supported_interval(
    points: np.ndarray,
    plane: _Plane,
    anchor: np.ndarray,
    axis: np.ndarray,
    *,
    plane_tolerance_mm: float,
    corridor_radius_mm: float,
    min_cross_support_width_mm: float,
    quantile_low: float,
    quantile_high: float,
) -> tuple[np.ndarray, int, float] | None:
    plane_mask = np.abs(points @ plane.normal + plane.offset) <= plane_tolerance_mm
    plane_points = points[plane_mask]
    if not len(plane_points):
        return None
    delta = plane_points - anchor
    longitudinal = delta @ axis
    radial = np.linalg.norm(delta - np.outer(longitudinal, axis), axis=1)
    corridor_mask = radial <= corridor_radius_mm
    delta = delta[corridor_mask]
    longitudinal = longitudinal[corridor_mask]
    if not len(longitudinal):
        return None
    cross_axis = np.cross(axis, plane.normal)
    cross_axis /= max(float(np.linalg.norm(cross_axis)), 1e-12)
    cross_coordinate = delta @ cross_axis
    cross_width = float(np.quantile(cross_coordinate, 0.95) - np.quantile(cross_coordinate, 0.05))
    if cross_width < min_cross_support_width_mm:
        return None
    interval = np.quantile(longitudinal, (quantile_low, quantile_high))
    return interval, int(len(longitudinal)), cross_width


def build_supported_plane_intersections(
    points: np.ndarray,
    plane_hypotheses: Sequence[Mapping[str, Any]],
    *,
    plane_tolerance_mm: float = 1.0,
    corridor_radius_mm: float = 14.0,
    min_cross_support_width_mm: float = 4.0,
    min_support_points: int = 20,
    min_span_mm: float = 12.0,
    max_abs_normal_dot: float = 0.45,
    support_quantiles: tuple[float, float] = (0.02, 0.98),
    plane_merge_angle_deg: float = 3.0,
    plane_merge_offset_mm: float = 2.5,
    line_merge_angle_deg: float = 3.0,
    line_merge_distance_mm: float = 3.0,
) -> dict[str, Any]:
    """Return finite, two-plane-supported bend-line candidates.

    No truth count, part identifier, or corpus-specific threshold enters this
    calculation.  Candidate extent is the overlap of the robust longitudinal
    support intervals supplied independently by the two incident planes.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    points = points[np.isfinite(points).all(axis=1)]
    merged_records = merge_near_duplicate_planes(
        plane_hypotheses,
        max_angle_deg=plane_merge_angle_deg,
        max_offset_mm=plane_merge_offset_mm,
    )
    planes = [_canonical_plane(record, index) for index, record in enumerate(merged_records)]
    candidates: list[dict[str, Any]] = []
    low, high = support_quantiles
    for first_index, first in enumerate(planes):
        for second in planes[first_index + 1 :]:
            normal_dot = abs(float(np.dot(first.normal, second.normal)))
            if normal_dot > max_abs_normal_dot:
                continue
            intersection = _plane_intersection(first, second)
            if intersection is None:
                continue
            anchor, axis = intersection
            first_support = _supported_interval(
                points,
                first,
                anchor,
                axis,
                plane_tolerance_mm=plane_tolerance_mm,
                corridor_radius_mm=corridor_radius_mm,
                min_cross_support_width_mm=min_cross_support_width_mm,
                quantile_low=low,
                quantile_high=high,
            )
            second_support = _supported_interval(
                points,
                second,
                anchor,
                axis,
                plane_tolerance_mm=plane_tolerance_mm,
                corridor_radius_mm=corridor_radius_mm,
                min_cross_support_width_mm=min_cross_support_width_mm,
                quantile_low=low,
                quantile_high=high,
            )
            if first_support is None or second_support is None:
                continue
            first_interval, first_count, first_cross_width = first_support
            second_interval, second_count, second_cross_width = second_support
            if min(first_count, second_count) < min_support_points:
                continue
            start_parameter = max(float(first_interval[0]), float(second_interval[0]))
            end_parameter = min(float(first_interval[1]), float(second_interval[1]))
            span_mm = end_parameter - start_parameter
            if span_mm < min_span_mm:
                continue
            midpoint_parameter = 0.5 * (start_parameter + end_parameter)
            candidate_anchor = anchor + midpoint_parameter * axis
            half_span = 0.5 * span_mm
            candidate_id = f"PLI_{first.plane_id}_{second.plane_id}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "incident_plane_ids": [first.plane_id, second.plane_id],
                    "anchor": candidate_anchor.tolist(),
                    "axis_direction": axis.tolist(),
                    "span_mm": float(span_mm),
                    "span_endpoints": [
                        (candidate_anchor - half_span * axis).tolist(),
                        (candidate_anchor + half_span * axis).tolist(),
                    ],
                    "incident_plane_support_counts": [first_count, second_count],
                    "incident_plane_cross_support_widths_mm": [first_cross_width, second_cross_width],
                    "angle_deg": float(np.degrees(np.arccos(np.clip(normal_dot, -1.0, 1.0)))),
                    "source_kind": "scan_supported_plane_intersection",
                }
            )
    raw_candidate_count = len(candidates)
    candidates.sort(
        key=lambda item: (
            -min(item["incident_plane_support_counts"]),
            -float(item["span_mm"]),
            str(item["candidate_id"]),
        )
    )
    line_cosine = float(np.cos(np.deg2rad(line_merge_angle_deg)))
    unique_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_axis = np.asarray(candidate["axis_direction"], dtype=np.float64)
        candidate_anchor = np.asarray(candidate["anchor"], dtype=np.float64)
        for retained in unique_candidates:
            retained_axis = np.asarray(retained["axis_direction"], dtype=np.float64)
            if abs(float(np.dot(candidate_axis, retained_axis))) < line_cosine:
                continue
            retained_anchor = np.asarray(retained["anchor"], dtype=np.float64)
            anchor_delta = candidate_anchor - retained_anchor
            line_distance = float(
                np.linalg.norm(anchor_delta - np.dot(anchor_delta, retained_axis) * retained_axis)
            )
            if line_distance > line_merge_distance_mm:
                continue
            retained_half_span = 0.5 * float(retained["span_mm"])
            candidate_half_span = 0.5 * float(candidate["span_mm"])
            candidate_center = float(np.dot(anchor_delta, retained_axis))
            overlap = min(retained_half_span, candidate_center + candidate_half_span) - max(
                -retained_half_span,
                candidate_center - candidate_half_span,
            )
            if overlap < 0.5 * min(float(retained["span_mm"]), float(candidate["span_mm"])):
                continue
            retained.setdefault("merged_candidate_ids", []).append(candidate["candidate_id"])
            retained.setdefault("merged_incident_plane_pairs", []).append(candidate["incident_plane_ids"])
            break
        else:
            candidate["merged_candidate_ids"] = [candidate["candidate_id"]]
            candidate["merged_incident_plane_pairs"] = [candidate["incident_plane_ids"]]
            unique_candidates.append(candidate)
    candidates = unique_candidates
    candidates.sort(key=lambda item: str(item["candidate_id"]))
    return {
        "semantic": "diagnostic_scan_supported_plane_lattice",
        "merged_planes": merged_records,
        "raw_candidate_count": int(raw_candidate_count),
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
    }
