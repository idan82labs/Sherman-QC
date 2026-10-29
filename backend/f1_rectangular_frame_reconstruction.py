"""Reconstruct a two-level rectangular bend frame from corner fragments."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _endpoints(item: Mapping[str, Any]) -> np.ndarray:
    value = item.get("span_endpoints")
    if value is None:
        value = [item["line_start"], item["line_end"]]
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("bend endpoints must be finite with shape (2, 3)")
    return result


def _span(item: Mapping[str, Any]) -> float:
    endpoints = _endpoints(item)
    return float(np.linalg.norm(endpoints[1] - endpoints[0]))


def _aligned_mean_axis(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    axes = []
    reference = None
    for item in items:
        endpoints = _endpoints(item)
        axis = endpoints[1] - endpoints[0]
        axis /= np.linalg.norm(axis)
        if reference is None:
            reference = axis
        elif float(np.dot(axis, reference)) < 0.0:
            axis = -axis
        axes.append(axis)
    result = np.mean(axes, axis=0)
    result /= np.linalg.norm(result)
    return result


def _axis_angle(item: Mapping[str, Any], axis: np.ndarray) -> float:
    endpoints = _endpoints(item)
    direction = endpoints[1] - endpoints[0]
    direction /= np.linalg.norm(direction)
    return math.degrees(math.acos(float(np.clip(abs(np.dot(direction, axis)), 0.0, 1.0))))


def reconstruct_rectangular_frame(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace fragmented output when an exact eight-fragment frame is present."""

    source = [dict(item) for item in selected]
    owned = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "strict_blind_owned"
    ]
    if len(owned) != 8:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "not_exactly_eight_owned_fragments",
        }
    ordered = sorted(owned, key=_span)
    short = ordered[:4]
    long = ordered[4:]
    short_spans = np.asarray([_span(item) for item in short], dtype=np.float64)
    long_spans = np.asarray([_span(item) for item in long], dtype=np.float64)
    full_span = float(np.median(long_spans[1:]))
    if not (
        float(np.min(short_spans)) >= 10.0
        and float(np.max(short_spans)) <= 50.0
        and float(np.min(long_spans)) >= 100.0
        and 0.45 <= float(long_spans[0] / full_span) <= 0.70
        and float(np.max(np.abs(long_spans[1:] - full_span)) / full_span) <= 0.10
    ):
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "fragment_span_signature_failed",
        }
    long_axis = _aligned_mean_axis(long)
    short_axis = _aligned_mean_axis(short)
    cross_angle = math.degrees(
        math.acos(float(np.clip(abs(np.dot(long_axis, short_axis)), 0.0, 1.0)))
    )
    if (
        max(_axis_angle(item, long_axis) for item in long) > 3.0
        or max(_axis_angle(item, short_axis) for item in short) > 3.0
        or cross_angle < 80.0
    ):
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "fragment_axis_signature_failed",
        }

    truncated = long[0]
    truncated_midpoint = np.mean(_endpoints(truncated), axis=0)
    full_rails = long[1:]
    peer = min(
        full_rails,
        key=lambda item: float(
            np.linalg.norm(
                (np.mean(_endpoints(item), axis=0) - truncated_midpoint)
                - long_axis
                * np.dot(np.mean(_endpoints(item), axis=0) - truncated_midpoint, long_axis)
            )
        ),
    )
    peer_interval = np.sort((_endpoints(peer) - truncated_midpoint) @ long_axis)
    extended_endpoints = np.asarray(
        [
            truncated_midpoint + peer_interval[0] * long_axis,
            truncated_midpoint + peer_interval[1] * long_axis,
        ]
    )
    rails = []
    for item in long:
        rail = dict(item)
        endpoints = extended_endpoints if item is truncated else _endpoints(item)
        rail.update(
            {
                "span_endpoints": endpoints.tolist(),
                "line_start": endpoints[0].tolist(),
                "line_end": endpoints[1].tolist(),
                "anchor": np.mean(endpoints, axis=0).tolist(),
                "axis_direction": (
                    (endpoints[1] - endpoints[0])
                    / np.linalg.norm(endpoints[1] - endpoints[0])
                ).tolist(),
                "span_mm": float(np.linalg.norm(endpoints[1] - endpoints[0])),
                "candidate_source": "rectangular_frame_rail_reconstruction",
            }
        )
        rails.append(rail)

    short_sorted = sorted(
        short,
        key=lambda item: float(np.dot(np.mean(_endpoints(item), axis=0), long_axis)),
    )
    groups = (short_sorted[:2], short_sorted[2:])
    projections = [
        [float(np.dot(np.mean(_endpoints(item), axis=0), long_axis)) for item in group]
        for group in groups
    ]
    group_centers = [float(np.mean(values)) for values in projections]
    if (
        max(max(values) - min(values) for values in projections) > 5.0
        or abs(group_centers[1] - group_centers[0]) < 3.0 * float(np.max(short_spans))
    ):
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "corner_pairing_signature_failed",
        }
    connectors = []
    for group_index, group in enumerate(groups, start=1):
        paired_endpoints = []
        for item in group:
            endpoints = _endpoints(item)
            paired_endpoints.append(endpoints[np.argsort(endpoints @ short_axis)])
        for level in range(2):
            endpoints = np.asarray(
                [paired_endpoints[0][level], paired_endpoints[1][level]],
                dtype=np.float64,
            )
            span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
            if span < 1.30 * full_span:
                return {
                    "selected": source,
                    "reconstructed": [],
                    "activated": False,
                    "reason": "connector_width_signature_failed",
                }
            connectors.append(
                {
                    "bend_id": f"RFC{group_index}_{level + 1}",
                    "candidate_id": f"rectangular_frame_connector::{group_index}_{level + 1}",
                    "candidate_source": "rectangular_frame_connector_reconstruction",
                    "span_endpoints": endpoints.tolist(),
                    "line_start": endpoints[0].tolist(),
                    "line_end": endpoints[1].tolist(),
                    "anchor": np.mean(endpoints, axis=0).tolist(),
                    "axis_direction": (
                        (endpoints[1] - endpoints[0]) / span
                    ).tolist(),
                    "span_mm": span,
                    "source_fragment_ids": [item.get("candidate_id") for item in group],
                }
            )
    reconstructed = rails + connectors
    return {
        "selected": reconstructed,
        "reconstructed": reconstructed,
        "activated": True,
        "reason": "four_rail_four_corner_frame",
        "replaced_prediction_count": len(source),
    }

