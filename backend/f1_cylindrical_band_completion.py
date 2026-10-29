"""Bounded scan-normal completion of finite bend-axis bands.

The patch-graph solver can merge several collinear, spatially separated bend
objects when broad face atoms bridge the gaps.  This module looks for the
underlying finite cylindrical bands directly in the scan normals, but only
inside corridors already proposed by F1.  It is deterministic and deliberately
does not accept part identifiers, expected counts, CAD, or labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CylindricalBandCandidate:
    candidate_id: str
    seed_ids: tuple[str, ...]
    anchor: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    span_mm: float
    radius_mm: float
    transverse_offset_mm: float
    vote_count: int
    support_count: int
    normal_spread_deg: float
    score: float
    consensus_count: int = 1
    radius_consensus_count: int = 1
    # Transient only.  These indices address the bounded source cloud used by
    # the detector and are deliberately omitted from ``to_dict`` so the
    # compact-cache contract cannot serialize source-point identities.
    support_indices: tuple[int, ...] = ()

    @property
    def span_endpoints(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        axis = np.asarray(self.axis_direction, dtype=np.float64)
        anchor = np.asarray(self.anchor, dtype=np.float64)
        half = 0.5 * self.span_mm * axis
        return tuple(anchor - half), tuple(anchor + half)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "seed_ids": list(self.seed_ids),
            "anchor": list(self.anchor),
            "axis_direction": list(self.axis_direction),
            "span_mm": self.span_mm,
            "span_endpoints": [list(item) for item in self.span_endpoints],
            "radius_mm": self.radius_mm,
            "transverse_offset_mm": self.transverse_offset_mm,
            "vote_count": self.vote_count,
            "support_count": self.support_count,
            "normal_spread_deg": self.normal_spread_deg,
            "score": self.score,
            "consensus_count": self.consensus_count,
            "radius_consensus_count": self.radius_consensus_count,
        }


def complete_cylindrical_bands(
    points: np.ndarray,
    normals: np.ndarray,
    seed_hypotheses: Sequence[Any],
    *,
    point_budget: int = 200_000,
    candidate_budget: int = 512,
    radii_mm: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
    center_bin_mm: float = 1.25,
    axial_bin_mm: float = 3.0,
    corridor_radius_mm: float = 12.0,
    corridor_axial_extension_mm: float = 12.0,
    transient_support_sink: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Return finite scan-normal bands inside F1-proposed corridors.

    Input sampling is stable under repeated runs.  Each seed is evaluated in a
    bounded tube; Hough cells vote for a cylinder-axis center and occupied
    axial bins are split into finite runs.  Multiple runs may therefore replace
    one merged seed without using a target object count.
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must be matching Nx3 arrays")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    normal_lengths = np.linalg.norm(normals, axis=1)
    finite &= normal_lengths > 1e-8
    source_indices = np.flatnonzero(finite).astype(np.int64, copy=False)
    points = points[finite]
    normals = normals[finite]
    normal_lengths = normal_lengths[finite]
    normals = normals / normal_lengths[:, None]
    input_count = len(points)
    if len(points) > max(1, int(point_budget)):
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        take = np.linspace(0, len(order) - 1, int(point_budget), dtype=np.int64)
        selected = order[take]
        points = points[selected]
        normals = normals[selected]
        source_indices = source_indices[selected]

    seeds = _dedupe_seeds(seed_hypotheses)
    raw: list[CylindricalBandCandidate] = []
    for seed_index, seed in enumerate(seeds):
        raw.extend(
            _seed_candidates(
                points,
                normals,
                source_indices,
                seed,
                seed_index=seed_index,
                radii_mm=radii_mm,
                center_bin_mm=float(center_bin_mm),
                axial_bin_mm=float(axial_bin_mm),
                corridor_radius_mm=float(corridor_radius_mm),
                corridor_axial_extension_mm=float(corridor_axial_extension_mm),
            )
        )
    selected = _nonmaximum_suppression(raw, budget=max(1, int(candidate_budget)))
    if transient_support_sink is not None:
        transient_support_sink.clear()
        transient_support_sink.update(
            {
                item.candidate_id: tuple(int(value) for value in item.support_indices)
                for item in selected
            }
        )
    return {
        "algorithm": "f1_cylindrical_band_completion_v1",
        "input_point_count": input_count,
        "sampled_point_count": len(points),
        "seed_count": len(seeds),
        "raw_candidate_count": len(raw),
        "selected_candidate_count": len(selected),
        "parameters": {
            "point_budget": int(point_budget),
            "candidate_budget": int(candidate_budget),
            "radii_mm": [float(value) for value in radii_mm],
            "center_bin_mm": float(center_bin_mm),
            "axial_bin_mm": float(axial_bin_mm),
            "corridor_radius_mm": float(corridor_radius_mm),
            "corridor_axial_extension_mm": float(corridor_axial_extension_mm),
        },
        "candidates": [item.to_dict() for item in selected],
    }


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _unit(value: Any) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        return None
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return None
    vector = vector / length
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector *= -1.0
    return vector


def _dedupe_seeds(values: Sequence[Any]) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for item in values:
        axis = _unit(_value(item, "axis_direction"))
        anchor = np.asarray(_value(item, "anchor", ()), dtype=np.float64)
        try:
            span = float(_value(item, "span_mm", 0.0) or 0.0)
        except (TypeError, ValueError):
            span = 0.0
        if axis is None or anchor.shape != (3,) or span < 8.0:
            continue
        seed_id = str(_value(item, "bend_id", _value(item, "evidence_id", f"seed-{len(seeds) + 1}")))
        duplicate = None
        for existing in seeds:
            angle = math.degrees(math.acos(float(np.clip(abs(np.dot(axis, existing["axis"])), 0.0, 1.0))))
            delta = anchor - existing["anchor"]
            offset = float(np.linalg.norm(delta - axis * np.dot(delta, axis)))
            axial = abs(float(np.dot(delta, axis)))
            gap = max(0.0, axial - 0.5 * (span + existing["span"]))
            if angle <= 4.0 and offset <= 4.0 and gap <= 6.0:
                duplicate = existing
                break
        if duplicate is None:
            seeds.append({"ids": [seed_id], "axis": axis, "anchor": anchor, "span": span})
        else:
            duplicate["ids"].append(seed_id)
            if span > duplicate["span"]:
                duplicate.update({"axis": axis, "anchor": anchor, "span": span})
    return seeds


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.8:
        reference = np.asarray([0.0, 1.0, 0.0])
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _seed_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    source_indices: np.ndarray,
    seed: Mapping[str, Any],
    *,
    seed_index: int,
    radii_mm: Sequence[float],
    center_bin_mm: float,
    axial_bin_mm: float,
    corridor_radius_mm: float,
    corridor_axial_extension_mm: float,
) -> list[CylindricalBandCandidate]:
    axis = np.asarray(seed["axis"], dtype=np.float64)
    anchor = np.asarray(seed["anchor"], dtype=np.float64)
    span = float(seed["span"])
    delta = points - anchor
    axial = delta @ axis
    radial = delta - axial[:, None] * axis
    radial_distance = np.linalg.norm(radial, axis=1)
    normal_axial = normals @ axis
    normal_transverse = normals - normal_axial[:, None] * axis
    normal_transverse_length = np.linalg.norm(normal_transverse, axis=1)
    mask = (
        (radial_distance <= corridor_radius_mm)
        & (np.abs(axial) <= 0.5 * span + corridor_axial_extension_mm)
        & (normal_transverse_length >= 0.35)
    )
    if int(np.count_nonzero(mask)) < 24:
        return []
    local_points = points[mask]
    local_normals = normals[mask]
    local_source_indices = source_indices[mask]
    local_axial = axial[mask]
    transverse_normals = normal_transverse[mask] / normal_transverse_length[mask, None]
    u, v = _basis(axis)
    point_u = local_points @ u
    point_v = local_points @ v
    seed_u = float(np.dot(anchor, u))
    seed_v = float(np.dot(anchor, v))
    result: list[CylindricalBandCandidate] = []
    ordinal = 1
    for radius in radii_mm:
        radius = float(radius)
        for sign in (-1.0, 1.0):
            center_u = point_u - sign * radius * (transverse_normals @ u)
            center_v = point_v - sign * radius * (transverse_normals @ v)
            near_seed = np.hypot(center_u - seed_u, center_v - seed_v) <= corridor_radius_mm
            if int(np.count_nonzero(near_seed)) < 12:
                continue
            indices = np.flatnonzero(near_seed)
            bins_u = np.rint(center_u[indices] / center_bin_mm).astype(np.int64)
            bins_v = np.rint(center_v[indices] / center_bin_mm).astype(np.int64)
            keys = np.column_stack((bins_u, bins_v))
            unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            for cell_index in np.flatnonzero(counts >= 10):
                member_local = indices[inverse == cell_index]
                center = unique[cell_index].astype(np.float64) * center_bin_mm
                center_offset = float(np.hypot(center[0] - seed_u, center[1] - seed_v))
                axial_values = local_axial[member_local]
                axial_bins = np.floor(axial_values / axial_bin_mm).astype(np.int64)
                bin_ids, bin_counts = np.unique(axial_bins, return_counts=True)
                occupied = bin_ids[bin_counts >= 2]
                if len(occupied) == 0:
                    continue
                for run in _integer_runs(occupied, maximum_gap=1):
                    run_start = float(run[0] * axial_bin_mm)
                    run_end = float((run[-1] + 1) * axial_bin_mm)
                    in_run = member_local[(axial_values >= run_start) & (axial_values <= run_end)]
                    if len(in_run) < 12:
                        continue
                    actual_axial = local_axial[in_run]
                    low, high = np.quantile(actual_axial, (0.02, 0.98))
                    run_span = float(high - low)
                    if run_span < max(8.0, 3.0 * axial_bin_mm):
                        continue
                    candidate_center = u * center[0] + v * center[1]
                    candidate_anchor = candidate_center + axis * (float(np.dot(anchor, axis)) + 0.5 * (low + high))
                    radial_angles = np.degrees(
                        np.arctan2(local_normals[in_run] @ v, local_normals[in_run] @ u)
                    )
                    spread = _circular_spread_deg(radial_angles)
                    if spread < 8.0 or spread > 175.0:
                        continue
                    votes = int(len(member_local))
                    support = int(len(in_run))
                    # A physical sheet bend is a finite partial cylinder.  The
                    # square-root terms keep dense long borders from drowning
                    # short bends while still rewarding independent support.
                    score = (
                        math.sqrt(support)
                        * math.sqrt(max(run_span, 1.0))
                        * math.sqrt(max(spread, 1.0))
                        * (1.0 + min(votes, 5000) / 5000.0)
                        / (1.0 + 0.04 * center_offset)
                    )
                    result.append(
                        CylindricalBandCandidate(
                            candidate_id=f"CB{seed_index + 1}_{ordinal}",
                            seed_ids=tuple(sorted(str(item) for item in seed["ids"])),
                            anchor=tuple(float(value) for value in candidate_anchor),
                            axis_direction=tuple(float(value) for value in axis),
                            span_mm=run_span,
                            radius_mm=radius,
                            transverse_offset_mm=center_offset,
                            vote_count=votes,
                            support_count=support,
                            normal_spread_deg=spread,
                            score=float(score),
                            support_indices=tuple(
                                sorted(int(value) for value in local_source_indices[in_run])
                            ),
                        )
                    )
                    ordinal += 1
    return result


def _integer_runs(values: np.ndarray, *, maximum_gap: int) -> list[np.ndarray]:
    values = np.unique(np.sort(values))
    if len(values) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(values) > maximum_gap) + 1
    return [item for item in np.split(values, boundaries) if len(item)]


def _circular_spread_deg(values: np.ndarray) -> float:
    values = np.mod(np.asarray(values, dtype=np.float64), 360.0)
    if len(values) < 2:
        return 0.0
    ordered = np.sort(values)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    return float(360.0 - np.max(gaps))


def _line_interval(item: CylindricalBandCandidate, axis: np.ndarray) -> tuple[float, float]:
    center = float(np.dot(np.asarray(item.anchor), axis))
    return center - item.span_mm / 2.0, center + item.span_mm / 2.0


def _same_band(left: CylindricalBandCandidate, right: CylindricalBandCandidate) -> bool:
    left_axis = np.asarray(left.axis_direction)
    right_axis = np.asarray(right.axis_direction)
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(left_axis, right_axis)), 0.0, 1.0))))
    if angle > 4.0:
        return False
    axis = left_axis if np.dot(left_axis, right_axis) >= 0.0 else -left_axis
    delta = np.asarray(right.anchor) - np.asarray(left.anchor)
    offset = float(np.linalg.norm(delta - axis * np.dot(delta, axis)))
    if offset > 2.5:
        return False
    left_interval = _line_interval(left, axis)
    right_interval = _line_interval(right, axis)
    overlap = max(0.0, min(left_interval[1], right_interval[1]) - max(left_interval[0], right_interval[0]))
    return overlap / max(1e-9, min(left.span_mm, right.span_mm)) >= 0.55


def _nonmaximum_suppression(
    values: Sequence[CylindricalBandCandidate], *, budget: int
) -> list[CylindricalBandCandidate]:
    selected: list[CylindricalBandCandidate] = []
    groups: list[list[CylindricalBandCandidate]] = []
    for candidate in sorted(values, key=lambda item: (-item.score, item.candidate_id)):
        duplicate_index = next(
            (index for index, existing in enumerate(selected) if _same_band(candidate, existing)),
            None,
        )
        if duplicate_index is not None:
            groups[duplicate_index].append(candidate)
            continue
        if len(selected) < budget:
            selected.append(candidate)
            groups.append([candidate])
    return [
        replace(
            representative,
            consensus_count=len(group),
            radius_consensus_count=len({item.radius_mm for item in group}),
            support_indices=tuple(
                sorted(
                    {
                        int(index)
                        for item in group
                        for index in item.support_indices
                    }
                )
            ),
        )
        for representative, group in zip(selected, groups)
    ]
