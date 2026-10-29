"""Deterministic cross-source clustering for finite F1 line hypotheses."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _endpoints(record: Mapping[str, Any]) -> np.ndarray:
    if record.get("span_endpoints") is not None:
        value = np.asarray(record["span_endpoints"], dtype=np.float64)
    else:
        anchor = np.asarray(record["anchor"], dtype=np.float64)
        axis = np.asarray(record["axis_direction"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        half = 0.5 * float(record["span_mm"]) * axis
        value = np.asarray((anchor - half, anchor + half))
    if value.shape != (2, 3) or not np.isfinite(value).all():
        raise ValueError("line endpoints must be a finite 2x3 array")
    return value


def _canonical(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    endpoints = _endpoints(record)
    delta = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("line span must be nonzero")
    axis = delta / length
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
        endpoints = endpoints[::-1]
    return endpoints, axis, length, np.mean(endpoints, axis=0)


def _pair_metrics(
    left: tuple[np.ndarray, np.ndarray, float, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, float, np.ndarray],
) -> tuple[float, float, float]:
    left_endpoints, left_axis, left_length, left_midpoint = left
    right_endpoints, right_axis, right_length, right_midpoint = right
    dot = float(np.clip(abs(np.dot(left_axis, right_axis)), 0.0, 1.0))
    angle = math.degrees(math.acos(dot))
    delta = right_midpoint - left_midpoint
    left_offset = float(np.linalg.norm(delta - left_axis * np.dot(delta, left_axis)))
    right_offset = float(np.linalg.norm(delta - right_axis * np.dot(delta, right_axis)))
    offset = max(left_offset, right_offset)
    right_interval = sorted(float(np.dot(point - left_midpoint, left_axis)) for point in right_endpoints)
    left_interval = (-0.5 * left_length, 0.5 * left_length)
    overlap = max(
        0.0,
        min(left_interval[1], right_interval[1]) - max(left_interval[0], right_interval[0]),
    )
    return angle, offset, overlap / max(min(left_length, right_length), 1e-9)


def _support_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(record.get("finite_support_longest_run_fraction") or 0.0),
        -float(record.get("endpoint_support_fraction") or 0.0),
        -float(record.get("finite_support_fraction") or 0.0),
        float(record.get("median_line_distance_mm") or 1e9),
        -float(record.get("normal_second_eigenvalue") or 0.0),
        -float(record.get("support_count") or 0.0),
        str(record.get("candidate_source") or ""),
        str(record.get("bend_id") or record.get("candidate_id") or ""),
    )


def _aggregate_numeric(members: Sequence[Mapping[str, Any]], name: str) -> tuple[float, float]:
    values = [float(member.get(name) or 0.0) for member in members]
    return (float(np.mean(values)), float(np.max(values))) if values else (0.0, 0.0)


def cluster_line_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    axis_tolerance_deg: float = 4.0,
    offset_tolerance_mm: float = 2.0,
    minimum_shorter_overlap: float = 0.55,
) -> list[dict[str, Any]]:
    """Collapse duplicate hypotheses into deterministic cross-source line objects."""

    valid: list[tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, float, np.ndarray]]] = []
    for candidate in candidates:
        try:
            valid.append((dict(candidate), _canonical(candidate)))
        except (KeyError, TypeError, ValueError):
            continue
    valid.sort(key=lambda item: _support_rank(item[0]))
    groups: list[list[tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, float, np.ndarray]]]] = []
    for item in valid:
        best_index = None
        best_key = None
        for index, group in enumerate(groups):
            metrics = _pair_metrics(item[1], group[0][1])
            angle, offset, overlap = metrics
            if (
                angle > axis_tolerance_deg
                or offset > offset_tolerance_mm
                or overlap < minimum_shorter_overlap
            ):
                continue
            key = (offset, angle, -overlap, index)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        if best_index is None:
            groups.append([item])
        else:
            groups[best_index].append(item)

    feature_names = (
        "finite_support_fraction",
        "finite_support_longest_run_fraction",
        "endpoint_support_fraction",
        "median_line_distance_mm",
        "p90_line_distance_mm",
        "transverse_angular_span_deg",
        "transverse_angular_bin_fraction",
        "normal_second_eigenvalue",
        "normal_spread_deg_scan",
        "tight_normal_second_eigenvalue_r2mm",
        "tight_normal_second_eigenvalue_r6mm",
        "tight_to_wide_normal_second_ratio",
        "corridor_points_per_mm",
        "two_sided_transition_score",
        "two_sided_normal_angle_deg",
        "two_sided_support_balance",
        "two_sided_min_support_fraction",
        "two_sided_normal_coherence",
        "support_count",
        "vote_count",
        "normal_spread_deg",
        "score",
        "confidence",
        "stability_score",
    )
    objects: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        members = [item[0] for item in group]
        representative = dict(group[0][0])
        sources = sorted({str(member.get("candidate_source") or "unknown") for member in members})
        aggregate: dict[str, float] = {}
        for name in feature_names:
            mean_value, max_value = _aggregate_numeric(members, name)
            aggregate[f"cluster_mean_{name}"] = mean_value
            aggregate[f"cluster_max_{name}"] = max_value
        representative.update(aggregate)
        representative.update(
            {
                "bend_id": f"LO{index}",
                "candidate_source": "cross_source_line_object",
                "representative_source": str(group[0][0].get("candidate_source") or "unknown"),
                "cluster_member_count": len(members),
                "cluster_source_count": len(sources),
                "cluster_sources": sources,
                "cluster_member_ids": sorted(
                    str(member.get("bend_id") or member.get("candidate_id") or "")
                    for member in members
                ),
            }
        )
        objects.append(representative)
    return objects
