#!/usr/bin/env python3
"""Fixed-frame differential conformance at source-native CAD bend events.

EXP27 measures local scan conformance around the four source-native RoundBend
events of ``50476000_A``.  It consumes only the already frozen, independently
corroborated EXP24/EXP25 scan-to-CAD poses.  It never refines a pose, moves a
bend axis, fits a radius, searches a neighborhood, or reads evaluator roles.

The measurement is intentionally local.  Each native cylindrical face is
paired through analytic CADIR topology with its two tangent planar faces.
Normal-qualified angular coverage prevents planar material inside a radial
annulus from masquerading as a bend.  Two fixed planar guards remove at most
one bounded cross-sectional translation algebraically; no rotation or
per-axial-bin alignment is allowed.

Outputs are source-only screening measurements and native-resolution visual
artifacts.  They are not manufacturing tolerances, defect labels, accuracy, or
bend-line F1.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXP17_PATH = ROOT / "scripts" / "run_exp_cad_0017_native_roundbend_patch_axes.py"
EXP25_PATH = ROOT / "scripts" / "run_exp_cad_0025_analytic_brep_carrier_orientation.py"

SPEC17 = importlib.util.spec_from_file_location("exp17_native_axes", EXP17_PATH)
EXP17 = importlib.util.module_from_spec(SPEC17)
assert SPEC17 and SPEC17.loader
SPEC17.loader.exec_module(EXP17)

SPEC25 = importlib.util.spec_from_file_location("exp25_orientation", EXP25_PATH)
EXP25 = importlib.util.module_from_spec(SPEC25)
assert SPEC25 and SPEC25.loader
SPEC25.loader.exec_module(EXP25)

EXP21 = EXP25.EXP21


EXPECTED_CASE_TOKENS = ("N011", "N012")
VOXEL_THICKNESS_FRACTION = 1.0 / 8.0
CAPTURE_THICKNESS_FRACTION = 0.75
AXIAL_END_EXCLUSION_THICKNESS = 1.0
CYLINDER_ANGLE_EDGE_FRACTION = 0.08
AXIAL_BIN_MIN = 8
AXIAL_BIN_MAX = 24
AXIAL_BIN_LENGTH_THICKNESS = 5.0
ANGULAR_BIN_COUNT = 8
MIN_POINTS_PER_CELL = 3
CYLINDER_NORMAL_MAX_DEG = 20.0
PLANE_NORMAL_MAX_DEG = 15.0
PLANE_GUARD_NEAR_THICKNESS = 2.0
PLANE_GUARD_FAR_THICKNESS = 6.0
SEAM_NEAR_THICKNESS = 0.5
SEAM_FAR_THICKNESS = 1.5
PLANE_COVERAGE_MIN = 0.75
SEAM_COVERAGE_MIN = 0.75
CYLINDER_MEASURABLE_COVERAGE_MIN = 0.60
CYLINDER_NOMINAL_COVERAGE = 0.80
LOCAL_TRANSLATION_MAX_THICKNESS = 0.50
MIN_ANGLE_BIN_FRACTION = 0.75
TOPOLOGY_AXIS_COSINE_MIN = 0.999
TOPOLOGY_PLANE_AXIS_DOT_MAX = 1e-6
TRANSFORM_EQUAL_ATOL = 1e-9


def xyz(value: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray([value[name] for name in "xyz"], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("invalid finite xyz value")
    return result


def normalized(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid nonzero vector")
    return result / norm


def canonical_int64_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def density_cap_fixed_identities(
    points: np.ndarray,
    normals: np.ndarray,
    voxel_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the first raw identity in every CAD-frame voxel."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or normals.shape != points.shape
        or not np.isfinite(points).all()
        or not np.isfinite(normals).all()
        or not math.isfinite(voxel_mm)
        or voxel_mm <= 0.0
    ):
        raise ValueError("invalid density-cap input")
    keys = np.floor(points / voxel_mm).astype(np.int64)
    _unique, first_indices = np.unique(keys, axis=0, return_index=True)
    selected = np.sort(first_indices.astype(np.int64))
    capped_normals = normals[selected]
    lengths = np.linalg.norm(capped_normals, axis=1)
    valid = lengths > 1e-12
    selected = selected[valid]
    capped_normals = capped_normals[valid] / lengths[valid, None]
    return points[selected], capped_normals, selected


def axial_bin_count(span_mm: float, thickness_mm: float) -> int:
    if span_mm <= 0.0 or thickness_mm <= 0.0:
        raise ValueError("span and thickness must be positive")
    return max(
        AXIAL_BIN_MIN,
        min(
            AXIAL_BIN_MAX,
            int(math.ceil(span_mm / (AXIAL_BIN_LENGTH_THICKNESS * thickness_mm))),
        ),
    )


def angle_coordinates(
    radial_units: np.ndarray,
    first_endpoint: np.ndarray,
    second_endpoint: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return minor-arc angles starting at the first tangent endpoint."""

    radial_units = np.asarray(radial_units, dtype=np.float64)
    first = normalized(first_endpoint)
    second = normalized(second_endpoint)
    axis = normalized(axis)
    theta = math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))
    if theta <= math.radians(5.0) or theta >= math.pi - 1e-8:
        raise ValueError("unsupported native bend angular span")
    direction = 1.0 if float(np.dot(axis, np.cross(first, second))) >= 0.0 else -1.0
    dot = np.clip(radial_units @ first, -1.0, 1.0)
    cross = np.cross(np.broadcast_to(first, radial_units.shape), radial_units)
    signed = direction * (cross @ axis)
    angles = np.mod(np.arctan2(signed, dot), 2.0 * math.pi)
    return angles, theta


def _indices(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    names = (
        "faces", "loops", "coedges", "edges", "vertices", "points", "surfaces",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        rows = model.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"missing CADIR collection {name}")
        index = {str(row["id"]): row for row in rows}
        if len(index) != len(rows):
            raise ValueError(f"duplicate CADIR id in {name}")
        result[name] = index
    return result


def _edge_endpoints(edge: Mapping[str, Any], index: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    endpoints = []
    for name in ("start", "end"):
        vertex = index["vertices"].get(str(edge.get(name)))
        if vertex is None:
            raise ValueError("missing edge vertex")
        point = index["points"].get(str(vertex.get("point")))
        if point is None:
            raise ValueError("missing vertex point")
        endpoints.append(xyz(point["position"]))
    return endpoints[0], endpoints[1]


def _face_boundary_points(face_id: str, index: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    face = index["faces"][face_id]
    values = []
    for loop_id in face.get("loops", []):
        loop = index["loops"].get(str(loop_id))
        if loop is None or str(loop.get("face")) != face_id:
            raise ValueError(f"incomplete loop topology for {face_id}")
        for coedge_id in loop.get("coedges", []):
            coedge = index["coedges"].get(str(coedge_id))
            if coedge is None or str(coedge.get("owner_loop")) != str(loop_id):
                raise ValueError(f"incomplete coedge topology for {face_id}")
            edge = index["edges"].get(str(coedge.get("edge")))
            if edge is None:
                raise ValueError(f"missing boundary edge for {face_id}")
            values.extend(_edge_endpoints(edge, index))
    if not values:
        raise ValueError(f"empty boundary for {face_id}")
    return np.asarray(values, dtype=np.float64)


def tangent_guards_for_cylinder(
    model: Mapping[str, Any],
    cylinder_face_id: str,
    event_axis: np.ndarray,
    axis_foot: np.ndarray,
    thickness_mm: float,
) -> list[dict[str, Any]]:
    """Follow radial coedges and return the two axial tangent plane faces."""

    index = _indices(model)
    face = index["faces"].get(cylinder_face_id)
    if face is None:
        raise ValueError(f"missing native cylinder face {cylinder_face_id}")
    axis = normalized(event_axis)
    rows = []
    seen_faces: set[str] = set()
    for loop_id in face.get("loops", []):
        loop = index["loops"].get(str(loop_id))
        if loop is None or str(loop.get("face")) != cylinder_face_id:
            raise ValueError("incomplete cylinder loop topology")
        for coedge_id in loop.get("coedges", []):
            coedge = index["coedges"].get(str(coedge_id))
            if coedge is None:
                raise ValueError("missing cylinder coedge")
            radial = index["coedges"].get(str(coedge.get("radial_next")))
            edge = index["edges"].get(str(coedge.get("edge")))
            if radial is None or edge is None:
                raise ValueError("incomplete radial topology")
            adjacent_loop = index["loops"].get(str(radial.get("owner_loop")))
            if adjacent_loop is None:
                raise ValueError("missing adjacent loop")
            adjacent_face_id = str(adjacent_loop.get("face"))
            adjacent_face = index["faces"].get(adjacent_face_id)
            if adjacent_face is None:
                raise ValueError("missing adjacent face")
            surface = index["surfaces"].get(str(adjacent_face.get("surface")))
            if surface is None or surface.get("geometry", {}).get("kind") != "plane":
                continue
            first, second = _edge_endpoints(edge, index)
            edge_axis = normalized(second - first)
            plane_geometry = surface["geometry"]
            plane_normal = normalized(xyz(plane_geometry["normal"]))
            if abs(float(np.dot(edge_axis, axis))) < TOPOLOGY_AXIS_COSINE_MIN:
                continue
            if abs(float(np.dot(plane_normal, axis))) > TOPOLOGY_PLANE_AXIS_DOT_MAX:
                continue
            if adjacent_face_id in seen_faces:
                raise ValueError("duplicate tangent face")
            seen_faces.add(adjacent_face_id)
            midpoint = (first + second) / 2.0
            radial_unit = midpoint - axis_foot
            radial_unit = radial_unit - float(np.dot(radial_unit, axis)) * axis
            radial_unit = normalized(radial_unit)
            if float(np.dot(plane_normal, radial_unit)) < 0.0:
                plane_normal = -plane_normal
            transverse = normalized(np.cross(axis, plane_normal))
            boundary = _face_boundary_points(adjacent_face_id, index)
            coordinates = (boundary - midpoint) @ transverse
            positive_extent = float(np.max(coordinates))
            negative_extent = float(-np.min(coordinates))
            if positive_extent >= negative_extent:
                interior = transverse
                extent = positive_extent
            else:
                interior = -transverse
                extent = negative_extent
            if extent + 1e-8 < PLANE_GUARD_FAR_THICKNESS * thickness_mm:
                raise ValueError(f"insufficient analytic plane guard clearance for {adjacent_face_id}")
            rows.append(
                {
                    "face_id": adjacent_face_id,
                    "surface_id": str(surface["id"]),
                    "shared_edge_id": str(edge["id"]),
                    "edge_start_mm": first,
                    "edge_end_mm": second,
                    "edge_midpoint_mm": midpoint,
                    "plane_origin_mm": xyz(plane_geometry["origin"]),
                    "plane_normal": plane_normal,
                    "radial_unit": radial_unit,
                    "interior_direction": interior,
                    "interior_extent_mm": extent,
                }
            )
    if len(rows) != 2:
        raise ValueError(
            f"expected two analytic tangent plane guards for {cylinder_face_id}, got {len(rows)}"
        )
    rows.sort(key=lambda row: (row["face_id"], row["shared_edge_id"]))
    return rows


def native_event_bundles(
    cadir: Mapping[str, Any],
    extraction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    model = cadir["model"]
    index = _indices(model)
    thickness = float(extraction["sheet_thickness_mm"])
    bundles = []
    for event in extraction["events"]:
        start = np.asarray(event["line_start_mm"], dtype=np.float64)
        end = np.asarray(event["line_end_mm"], dtype=np.float64)
        axis = normalized(end - start)
        sides = []
        face_rows = []
        for face_id in event["face_ids"]:
            face = index["faces"].get(str(face_id))
            if face is None:
                raise ValueError(f"missing event face {face_id}")
            surface = index["surfaces"].get(str(face.get("surface")))
            if surface is None or surface.get("geometry", {}).get("kind") != "cylinder":
                raise ValueError(f"event face is not analytic cylinder: {face_id}")
            geometry = surface["geometry"]
            radius = float(geometry["radius"])
            face_rows.append((str(face_id), surface, geometry, radius))
        if not np.allclose(
            sorted(row[3] for row in face_rows),
            sorted(float(value) for value in event["radii_mm"]),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"event/cylinder radius inventory mismatch for {event['event_id']}")
        for face_id, surface, geometry, radius in sorted(face_rows, key=lambda row: row[3]):
            cylinder_axis = normalized(xyz(geometry["axis"]))
            if abs(float(np.dot(cylinder_axis, axis))) < 1.0 - 1e-8:
                raise ValueError(f"event/cylinder axis mismatch for {face_id}")
            origin = xyz(geometry["origin"])
            foot = origin - float(np.dot(origin, axis)) * axis
            guards = tangent_guards_for_cylinder(
                model, str(face_id), axis, foot, thickness
            )
            _angles, theta = angle_coordinates(
                np.vstack([guards[0]["radial_unit"], guards[1]["radial_unit"]]),
                guards[0]["radial_unit"],
                guards[1]["radial_unit"],
                axis,
            )
            sides.append(
                {
                    "cylinder_face_id": str(face_id),
                    "cylinder_surface_id": str(surface["id"]),
                    "radius_mm": radius,
                    "axis": axis,
                    "axis_foot_mm": foot,
                    "line_start_mm": start,
                    "line_end_mm": end,
                    "span_mm": float(np.linalg.norm(end - start)),
                    "guards": guards,
                    "theta_cad_rad": theta,
                }
            )
        if len(sides) != 2:
            raise ValueError("native event must expose inner and outer cylinder sides")
        bundles.append(
            {
                "event_id": str(event["event_id"]),
                "sheet_thickness_mm": thickness,
                "line_start_mm": start,
                "line_end_mm": end,
                "axis": axis,
                "span_mm": float(np.linalg.norm(end - start)),
                "sides": sides,
            }
        )
    return bundles


def normal_angles_deg(normals: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(np.asarray(normals) @ normalized(reference)), 0.0, 1.0)
    return np.degrees(np.arccos(dots))


def _bin_indices(values: np.ndarray, low: float, high: float, count: int) -> np.ndarray:
    scaled = np.floor((values - low) / (high - low) * count).astype(np.int64)
    return np.clip(scaled, 0, count - 1)


def _coverage(counts: np.ndarray) -> float:
    return float(np.mean(np.asarray(counts) >= MIN_POINTS_PER_CELL))


def _median_matrix(
    axial_indices: np.ndarray,
    angular_indices: np.ndarray,
    values: np.ndarray,
    axial_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((axial_count, ANGULAR_BIN_COUNT), dtype=np.int64)
    medians = np.full((axial_count, ANGULAR_BIN_COUNT), np.nan, dtype=np.float64)
    for axial in range(axial_count):
        for angular in range(ANGULAR_BIN_COUNT):
            mask = (axial_indices == axial) & (angular_indices == angular)
            counts[axial, angular] = int(np.count_nonzero(mask))
            if counts[axial, angular] >= MIN_POINTS_PER_CELL:
                medians[axial, angular] = float(np.median(values[mask]))
    return counts, medians


def plane_guard_observation(
    points: np.ndarray,
    normals: np.ndarray,
    guard: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    near_thickness: float,
    far_thickness: float,
) -> dict[str, Any]:
    thickness = float(event["sheet_thickness_mm"])
    axis = np.asarray(event["axis"], dtype=np.float64)
    start = np.asarray(event["line_start_mm"], dtype=np.float64)
    span = float(event["span_mm"])
    axial_low = AXIAL_END_EXCLUSION_THICKNESS * thickness
    axial_high = span - axial_low
    plane_normal = np.asarray(guard["plane_normal"], dtype=np.float64)
    plane_origin = np.asarray(guard["plane_origin_mm"], dtype=np.float64)
    midpoint = np.asarray(guard["edge_midpoint_mm"], dtype=np.float64)
    interior = np.asarray(guard["interior_direction"], dtype=np.float64)
    signed_distance = (points - plane_origin) @ plane_normal
    axial = (points - start) @ axis
    inward = (points - midpoint) @ interior
    geometric = (
        (np.abs(signed_distance) <= CAPTURE_THICKNESS_FRACTION * thickness)
        & (axial >= axial_low)
        & (axial <= axial_high)
        & (inward >= near_thickness * thickness)
        & (inward <= far_thickness * thickness)
    )
    errors = normal_angles_deg(normals, plane_normal)
    qualified = geometric & (errors <= PLANE_NORMAL_MAX_DEG)
    bin_count = axial_bin_count(span, thickness)
    axial_indices = _bin_indices(axial[qualified], axial_low, axial_high, bin_count)
    counts = np.bincount(axial_indices, minlength=bin_count)
    coverage = _coverage(counts)
    modes: list[Optional[np.ndarray]] = []
    qualified_indices = np.flatnonzero(qualified)
    for bin_index in range(bin_count):
        chosen = qualified_indices[axial_indices == bin_index]
        if len(chosen) < MIN_POINTS_PER_CELL:
            modes.append(None)
            continue
        vectors = normals[chosen].copy()
        flip = vectors @ plane_normal < 0.0
        vectors[flip] *= -1.0
        vector = np.sum(vectors, axis=0)
        modes.append(normalized(vector))
    return {
        "geometric_mask": geometric,
        "qualified_mask": qualified,
        "signed_distance": signed_distance,
        "normal_error_deg": errors,
        "axial_bin_counts": counts,
        "axial_normal_modes": modes,
        "coverage_fraction": coverage,
        "qualified_point_count": int(np.count_nonzero(qualified)),
        "median_signed_offset_mm": (
            float(np.median(signed_distance[qualified])) if np.any(qualified) else None
        ),
        "median_normal_error_deg": (
            float(np.median(errors[geometric])) if np.any(geometric) else None
        ),
        "actionable": bool(
            coverage >= PLANE_COVERAGE_MIN
            and np.any(geometric)
            and float(np.median(errors[geometric])) <= PLANE_NORMAL_MAX_DEG
        ),
    }


def cylinder_base_arrays(
    points: np.ndarray,
    normals: np.ndarray,
    side: Mapping[str, Any],
    thickness_mm: float,
) -> dict[str, np.ndarray]:
    axis = np.asarray(side["axis"], dtype=np.float64)
    foot = np.asarray(side["axis_foot_mm"], dtype=np.float64)
    start = np.asarray(side["line_start_mm"], dtype=np.float64)
    span = float(side["span_mm"])
    relative = points - foot
    axis_parameter = (points - start) @ axis
    radial = relative - (relative @ axis)[:, None] * axis[None, :]
    radial_distance = np.linalg.norm(radial, axis=1)
    radial_unit = radial / np.maximum(radial_distance[:, None], 1e-12)
    angles, theta = angle_coordinates(
        radial_unit,
        np.asarray(side["guards"][0]["radial_unit"]),
        np.asarray(side["guards"][1]["radial_unit"]),
        axis,
    )
    fractions = angles / theta
    normal_error = np.degrees(
        np.arccos(np.clip(np.abs(np.sum(normals * radial_unit, axis=1)), 0.0, 1.0))
    )
    low = AXIAL_END_EXCLUSION_THICKNESS * thickness_mm
    high = span - low
    radial_residual = radial_distance - float(side["radius_mm"])
    base = (
        (axis_parameter >= low)
        & (axis_parameter <= high)
        & (fractions >= CYLINDER_ANGLE_EDGE_FRACTION)
        & (fractions <= 1.0 - CYLINDER_ANGLE_EDGE_FRACTION)
        & (np.abs(radial_residual) <= CAPTURE_THICKNESS_FRACTION * thickness_mm)
        & (normal_error <= CYLINDER_NORMAL_MAX_DEG)
    )
    return {
        "axis_parameter": axis_parameter,
        "radial_distance": radial_distance,
        "radial_unit": radial_unit,
        "angular_fraction": fractions,
        "normal_error_deg": normal_error,
        "radial_residual_mm": radial_residual,
        "base_mask": base,
    }


def event_conformance(
    points: np.ndarray,
    normals: np.ndarray,
    event: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thickness = float(event["sheet_thickness_mm"])
    span = float(event["span_mm"])
    bin_count = axial_bin_count(span, thickness)
    low = AXIAL_END_EXCLUSION_THICKNESS * thickness
    high = span - low
    bases = [cylinder_base_arrays(points, normals, side, thickness) for side in event["sides"]]

    # A scan identity can belong to only one of the two thickness-separated
    # cylinder faces.  This prevents an intermediate shell point from counting
    # twice.
    costs = np.column_stack(
        [
            np.where(base["base_mask"], np.abs(base["radial_residual_mm"]), np.inf)
            for base in bases
        ]
    )
    owners = np.argmin(costs, axis=1)
    finite_owner = np.isfinite(np.min(costs, axis=1))

    surface_rows = []
    surface_arrays = []
    for surface_index, (side, base) in enumerate(zip(event["sides"], bases)):
        owned = finite_owner & (owners == surface_index) & base["base_mask"]
        guards = [
            plane_guard_observation(
                points, normals, guard, event,
                near_thickness=PLANE_GUARD_NEAR_THICKNESS,
                far_thickness=PLANE_GUARD_FAR_THICKNESS,
            )
            for guard in side["guards"]
        ]
        seams = [
            plane_guard_observation(
                points, normals, guard, event,
                near_thickness=SEAM_NEAR_THICKNESS,
                far_thickness=SEAM_FAR_THICKNESS,
            )
            for guard in side["guards"]
        ]
        axial_indices = _bin_indices(base["axis_parameter"][owned], low, high, bin_count)
        angular_indices = _bin_indices(
            base["angular_fraction"][owned],
            CYLINDER_ANGLE_EDGE_FRACTION,
            1.0 - CYLINDER_ANGLE_EDGE_FRACTION,
            ANGULAR_BIN_COUNT,
        )
        raw_counts, _raw_medians = _median_matrix(
            axial_indices,
            angular_indices,
            base["radial_residual_mm"][owned],
            bin_count,
        )
        cylinder_coverage = _coverage(raw_counts)
        guard_actionable = all(bool(row["actionable"]) for row in guards)
        seam_actionable = all(
            float(row["coverage_fraction"]) >= SEAM_COVERAGE_MIN for row in seams
        )
        offsets = [row["median_signed_offset_mm"] for row in guards]
        delta = np.zeros(3, dtype=np.float64)
        delta_norm = math.inf
        if guard_actionable and all(value is not None for value in offsets):
            directions = np.vstack([row["radial_unit"] for row in side["guards"]])
            delta = np.linalg.pinv(directions) @ np.asarray(offsets, dtype=np.float64)
            delta_norm = float(np.linalg.norm(delta))

        angle_errors = []
        angle_profile = np.full(bin_count, np.nan, dtype=np.float64)
        if guard_actionable:
            for bin_index in range(bin_count):
                first_mode = guards[0]["axial_normal_modes"][bin_index]
                second_mode = guards[1]["axial_normal_modes"][bin_index]
                if first_mode is None or second_mode is None:
                    continue
                observed = math.acos(float(np.clip(np.dot(first_mode, second_mode), -1.0, 1.0)))
                error = abs(observed - float(side["theta_cad_rad"]))
                angle_profile[bin_index] = error
                angle_errors.append(error)
        minimum_angle_bins = int(math.ceil(MIN_ANGLE_BIN_FRACTION * bin_count))

        status = "ABSTAIN"
        reasons = []
        if not guard_actionable:
            reasons.append("ADJACENT_PLANAR_GUARDS_NOT_ACTIONABLE")
        if guard_actionable and delta_norm > LOCAL_TRANSLATION_MAX_THICKNESS * thickness:
            reasons.append("BOUNDED_LOCAL_TRANSLATION_GATE")
        if guard_actionable and len(angle_errors) < minimum_angle_bins:
            reasons.append("ANGLE_BIN_COVERAGE_GATE")
        measurable = cylinder_coverage >= CYLINDER_MEASURABLE_COVERAGE_MIN
        missing_supported = (not measurable) and seam_actionable
        if guard_actionable and delta_norm <= LOCAL_TRANSLATION_MAX_THICKNESS * thickness and len(angle_errors) >= minimum_angle_bins:
            if measurable:
                status = "MEASURED_FIXED_FRAME"
            elif missing_supported:
                status = "MEASURED_MISSING_CYLINDER_SUPPORT"
            else:
                reasons.append("CYLINDER_UNOBSERVED_WITHOUT_TWO_SEAM_WITNESSES")

        corrected = np.asarray([], dtype=np.float64)
        corrected_counts = raw_counts
        corrected_medians = np.full_like(raw_counts, np.nan, dtype=np.float64)
        radius_bias = None
        form_tail = None
        radius_component = 0.0
        form_component = 0.0
        if status == "MEASURED_FIXED_FRAME":
            corrected = (
                base["radial_residual_mm"][owned]
                - base["radial_unit"][owned] @ delta
            )
            corrected_counts, corrected_medians = _median_matrix(
                axial_indices, angular_indices, corrected, bin_count
            )
            radius_bias = float(np.median(corrected))
            axial_medians = np.asarray(
                [
                    float(np.median(corrected[axial_indices == index]))
                    for index in range(bin_count)
                    if np.count_nonzero(axial_indices == index) >= MIN_POINTS_PER_CELL
                ],
                dtype=np.float64,
            )
            if len(axial_medians):
                form_tail = float(np.percentile(np.abs(axial_medians - radius_bias), 90))
            else:
                form_tail = 0.0
            radius_component = abs(radius_bias) / thickness
            form_component = float(form_tail) / thickness

        angle_tail = float(np.percentile(angle_errors, 90)) if angle_errors else None
        angle_component = (
            float(angle_tail) / max(float(side["theta_cad_rad"]), math.radians(30.0))
            if angle_tail is not None
            else 0.0
        )
        missing_component = max(
            0.0,
            (CYLINDER_NOMINAL_COVERAGE - cylinder_coverage) / CYLINDER_NOMINAL_COVERAGE,
        )
        score = None
        if status.startswith("MEASURED"):
            score = max(
                radius_component,
                form_component,
                angle_component,
                missing_component,
            )
        row = {
            "cylinder_face_id": side["cylinder_face_id"],
            "radius_mm": float(side["radius_mm"]),
            "theta_cad_deg": math.degrees(float(side["theta_cad_rad"])),
            "status": status,
            "abstention_reasons": reasons,
            "owned_normal_qualified_point_count": int(np.count_nonzero(owned)),
            "cylinder_coverage_fraction": cylinder_coverage,
            "plane_guard_coverage_fractions": [float(value["coverage_fraction"]) for value in guards],
            "seam_witness_coverage_fractions": [float(value["coverage_fraction"]) for value in seams],
            "plane_guard_median_normal_error_deg": [value["median_normal_error_deg"] for value in guards],
            "plane_guard_median_signed_offset_mm": offsets,
            "bounded_cross_section_translation_mm": delta.tolist() if np.isfinite(delta_norm) else None,
            "bounded_cross_section_translation_norm_mm": delta_norm if np.isfinite(delta_norm) else None,
            "angle_bin_count": len(angle_errors),
            "angle_error_p90_deg": math.degrees(angle_tail) if angle_tail is not None else None,
            "corrected_radius_bias_mm": radius_bias,
            "corrected_axial_form_p90_mm": form_tail,
            "components": {
                "radius_bias_over_thickness": radius_component,
                "axial_form_over_thickness": form_component,
                "angle_error_fraction": angle_component,
                "missing_support_fraction": missing_component,
            },
            "surface_anomaly_score": score,
        }
        surface_rows.append(row)
        surface_arrays.append(
            {
                "owned_mask": owned,
                "guard_masks": [value["qualified_mask"] for value in guards],
                "seam_masks": [value["qualified_mask"] for value in seams],
                "corrected_residuals": corrected,
                "owned_axial_indices": axial_indices,
                "owned_angular_indices": angular_indices,
                "coverage_counts": corrected_counts,
                "residual_matrix": corrected_medians,
                "angle_error_profile_rad": angle_profile,
                "radial_unit": base["radial_unit"],
                "axis_parameter": base["axis_parameter"],
            }
        )

    measured = [
        (index, row)
        for index, row in enumerate(surface_rows)
        if row["surface_anomaly_score"] is not None
    ]
    thickness_component = None
    if len(measured) == 2 and all(row["corrected_radius_bias_mm"] is not None for _index, row in measured):
        observed = sorted(
            float(row["radius_mm"]) + float(row["corrected_radius_bias_mm"])
            for _index, row in measured
        )
        thickness_component = abs((observed[1] - observed[0]) - thickness) / thickness
    candidates = [float(row["surface_anomaly_score"]) for _index, row in measured]
    if thickness_component is not None:
        candidates.append(float(thickness_component))
    event_score = max(candidates) if candidates else None
    result = {
        "event_id": event["event_id"],
        "status": "MEASURED" if event_score is not None else "ABSTAIN",
        "surface_count": len(surface_rows),
        "measured_surface_count": len(measured),
        "thickness_consistency_component": thickness_component,
        "event_anomaly_score": event_score,
        "surfaces": surface_rows,
    }
    arrays = {"surfaces": surface_arrays}
    return result, arrays


def _serializable_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "event_id": bundle["event_id"],
        "sheet_thickness_mm": bundle["sheet_thickness_mm"],
        "line_start_mm": np.asarray(bundle["line_start_mm"]).tolist(),
        "line_end_mm": np.asarray(bundle["line_end_mm"]).tolist(),
        "span_mm": bundle["span_mm"],
        "sides": [],
    }
    for side in bundle["sides"]:
        result["sides"].append(
            {
                "cylinder_face_id": side["cylinder_face_id"],
                "cylinder_surface_id": side["cylinder_surface_id"],
                "radius_mm": side["radius_mm"],
                "theta_cad_deg": math.degrees(side["theta_cad_rad"]),
                "guards": [
                    {
                        "face_id": guard["face_id"],
                        "surface_id": guard["surface_id"],
                        "shared_edge_id": guard["shared_edge_id"],
                        "interior_extent_mm": guard["interior_extent_mm"],
                    }
                    for guard in side["guards"]
                ],
            }
        )
    return result


def render_event(
    path: Path,
    case_token: str,
    points: np.ndarray,
    event: Mapping[str, Any],
    result: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> None:
    thickness = float(event["sheet_thickness_mm"])
    axis = np.asarray(event["axis"], dtype=np.float64)
    foot = np.asarray(event["sides"][0]["axis_foot_mm"], dtype=np.float64)
    first = np.asarray(event["sides"][0]["guards"][0]["radial_unit"], dtype=np.float64)
    second = np.asarray(event["sides"][0]["guards"][1]["radial_unit"], dtype=np.float64)
    direction = 1.0 if float(np.dot(axis, np.cross(first, second))) >= 0.0 else -1.0
    tangent = direction * np.cross(axis, first)
    local = points - foot
    horizontal = local @ first
    vertical = local @ tangent
    axial = (points - np.asarray(event["line_start_mm"])) @ axis
    max_radius = max(float(side["radius_mm"]) for side in event["sides"])
    local_roi = (
        (axial >= 0.0)
        & (axial <= float(event["span_mm"]))
        & (np.hypot(horizontal, vertical) <= max_radius + 4.0 * thickness)
    )
    sample_indices = np.flatnonzero(local_roi)
    if len(sample_indices) > 30000:
        sample_indices = sample_indices[:: int(math.ceil(len(sample_indices) / 30000))]

    figure, axes = plt.subplots(2, 3, figsize=(30, 24), dpi=100, facecolor="#091018")
    for axis_plot in axes.flat:
        axis_plot.set_facecolor("#091018")
        axis_plot.tick_params(colors="#d8e4ef")
        for spine in axis_plot.spines.values():
            spine.set_color("#71808f")

    cross = axes[0, 0]
    cross.scatter(horizontal[sample_indices], vertical[sample_indices], s=1.0, c="#4bd4ff", alpha=0.18, linewidths=0)
    guard_colors = ("#ffb347", "#9cff57")
    for surface_index, surface_arrays in enumerate(arrays["surfaces"]):
        for guard_index, mask in enumerate(surface_arrays["guard_masks"]):
            chosen = np.flatnonzero(mask)
            chosen = chosen[:: max(1, len(chosen) // 6000)]
            cross.scatter(
                horizontal[chosen], vertical[chosen], s=2.0,
                c=guard_colors[guard_index], alpha=0.40, linewidths=0,
            )
        chosen = np.flatnonzero(surface_arrays["owned_mask"])
        residual = surface_arrays["corrected_residuals"]
        if len(chosen) and len(residual) == len(chosen):
            order = np.arange(len(chosen))[:: max(1, len(chosen) // 8000)]
            cross.scatter(
                horizontal[chosen[order]], vertical[chosen[order]],
                c=residual[order], cmap="coolwarm", vmin=-0.5 * thickness,
                vmax=0.5 * thickness, s=4.0, alpha=0.9, linewidths=0,
            )
    theta = float(event["sides"][0]["theta_cad_rad"])
    phi = np.linspace(0.0, theta, 180)
    for side in event["sides"]:
        radius = float(side["radius_mm"])
        cross.plot(radius * np.cos(phi), radius * np.sin(phi), color="#ff2da8", linewidth=4.0)
    cross.axhline(0.0, color="#6d7885", linewidth=0.6, alpha=0.5)
    cross.axvline(0.0, color="#6d7885", linewidth=0.6, alpha=0.5)
    cross.set_aspect("equal", adjustable="box")
    cross.set_title("Fixed CAD cross-section\ncyan: local scan; magenta: native cylinders; orange/green: plane guards", color="white", fontsize=16)
    cross.set_xlabel("native radial endpoint 1 (mm)", color="white")
    cross.set_ylabel("native bend-arc direction (mm)", color="white")

    for surface_index in range(2):
        heat = axes[0, surface_index + 1]
        matrix = arrays["surfaces"][surface_index]["residual_matrix"]
        masked = np.ma.masked_invalid(matrix)
        image = heat.imshow(
            masked.T,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-0.5 * thickness,
            vmax=0.5 * thickness,
        )
        heat.set_facecolor("#303945")
        heat.set_title(
            f"{result['surfaces'][surface_index]['cylinder_face_id']} — corrected radial residual\n"
            "grey cells are unobserved; no interpolation",
            color="white", fontsize=16,
        )
        heat.set_xlabel("axial bin", color="white")
        heat.set_ylabel("native angular bin", color="white")
        colorbar = figure.colorbar(image, ax=heat, fraction=0.046, pad=0.04)
        colorbar.set_label("mm", color="white")
        colorbar.ax.tick_params(colors="white")

    profile = axes[1, 0]
    for surface_index, (surface_result, surface_arrays) in enumerate(zip(result["surfaces"], arrays["surfaces"])):
        matrix = surface_arrays["residual_matrix"]
        axial_profile = np.asarray(
            [np.nanmedian(row) if np.any(np.isfinite(row)) else np.nan for row in matrix]
        )
        profile.plot(
            np.arange(len(axial_profile)), axial_profile,
            marker="o", linewidth=2.2,
            label=f"R={surface_result['radius_mm']:.3f} mm",
        )
    profile.axhline(0.0, color="white", linewidth=1.0, alpha=0.5)
    profile.set_title("Axial median corrected cylinder residual", color="white", fontsize=16)
    profile.set_xlabel("axial bin", color="white")
    profile.set_ylabel("mm", color="white")
    profile.legend(facecolor="#111b25", labelcolor="white")

    angle_plot = axes[1, 1]
    for surface_result, surface_arrays in zip(result["surfaces"], arrays["surfaces"]):
        degrees = np.degrees(surface_arrays["angle_error_profile_rad"])
        angle_plot.plot(
            np.arange(len(degrees)), degrees, marker="o", linewidth=2.2,
            label=f"R={surface_result['radius_mm']:.3f} mm",
        )
    angle_plot.axhline(0.0, color="white", linewidth=1.0, alpha=0.5)
    angle_plot.set_title("Absolute adjacent-panel bend-angle error", color="white", fontsize=16)
    angle_plot.set_xlabel("axial bin", color="white")
    angle_plot.set_ylabel("degrees", color="white")
    angle_plot.legend(facecolor="#111b25", labelcolor="white")

    text_axis = axes[1, 2]
    text_axis.set_axis_off()
    lines = [
        f"event: {result['event_id']}",
        f"event status: {result['status']}",
        f"event anomaly score: {result['event_anomaly_score']}",
        f"thickness consistency: {result['thickness_consistency_component']}",
        "",
    ]
    for surface in result["surfaces"]:
        lines.extend(
            [
                f"{surface['cylinder_face_id']}  R={surface['radius_mm']:.3f} mm",
                f"  status: {surface['status']}",
                f"  cylinder coverage: {surface['cylinder_coverage_fraction']:.3f}",
                f"  plane coverage: {[round(v, 3) for v in surface['plane_guard_coverage_fractions']]}",
                f"  seam coverage: {[round(v, 3) for v in surface['seam_witness_coverage_fractions']]}",
                f"  bounded translation: {surface['bounded_cross_section_translation_norm_mm']}",
                f"  radius bias: {surface['corrected_radius_bias_mm']}",
                f"  form p90: {surface['corrected_axial_form_p90_mm']}",
                f"  angle p90: {surface['angle_error_p90_deg']}",
                f"  components: {surface['components']}",
                "",
            ]
        )
    text_axis.text(
        0.01, 0.99, "\n".join(lines), va="top", ha="left",
        color="#e7f0f8", fontsize=12, family="monospace",
    )
    figure.suptitle(
        f"{case_token} — {event['event_id']} — EXP27 fixed-frame differential bend conformance\n"
        "source-only; frozen pose and CAD geometry; no local search or evaluator roles",
        color="white", fontsize=22, y=0.985,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)


def load_frozen_case(
    case_token: str,
    scan_path: Path,
    exp24_root: Path,
    exp25_root: Path,
    exp25_result: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    exp24_path = exp24_root / "cases" / case_token / "semantic_result.json"
    exp25_path = exp25_root / "cases" / case_token / "semantic_result.json"
    exp24 = json.loads(exp24_path.read_text(encoding="utf-8"))
    exp25 = json.loads(exp25_path.read_text(encoding="utf-8"))
    for name, payload in (("EXP24", exp24), ("EXP25", exp25)):
        if payload.get("case_token") != case_token:
            raise ValueError(f"{name} case token mismatch")
        if payload.get("orientation_gate_pass") is not True:
            raise ValueError(f"{name} orientation gate did not pass")
        if payload.get("evaluator_data_opened") is not False:
            raise ValueError(f"{name} evaluator isolation failed")
    if exp24.get("absolute_registration_orientation_authorized") is not True:
        raise ValueError("EXP24 absolute orientation is not authorized")
    # EXP25 case payloads were frozen before their native visual review and
    # intentionally remain unauthorized.  The later immutable result is the
    # authority and binds both frozen semantic hashes; never mutate the cases.
    if exp25.get("absolute_registration_orientation_authorized") is not False:
        raise ValueError("EXP25 frozen case authorization state changed")
    expected_case_hash = exp25_result["semantic_sha256_primary_and_replay"].get(case_token)
    if expected_case_hash != EXP21.sha256_path(exp25_path):
        raise ValueError("EXP25 result does not bind the frozen case semantic")
    transform24 = np.asarray(exp24["selected_absolute_transform"], dtype=np.float64)
    transform25 = np.asarray(exp25["selected_absolute_transform"], dtype=np.float64)
    if transform24.shape != (4, 4) or not np.allclose(
        transform24, transform25, rtol=0.0, atol=TRANSFORM_EQUAL_ATOL
    ):
        raise ValueError("EXP24/EXP25 frozen transforms disagree")
    if EXP21.sha256_path(scan_path) != exp24["scan_sha256"] or exp24["scan_sha256"] != exp25["scan_sha256"]:
        raise ValueError("scan hash does not match both frozen orientation lanes")
    cloud = o3d.io.read_point_cloud(str(scan_path))
    if not cloud.has_points() or not cloud.has_normals():
        raise ValueError("EXP27 requires source scan points and normals")
    raw_points = np.asarray(cloud.points, dtype=np.float64)
    raw_normals = np.asarray(cloud.normals, dtype=np.float64)
    rotation = transform24[:3, :3]
    translation = transform24[:3, 3]
    points = raw_points @ rotation.T + translation
    normals = raw_normals @ rotation.T
    return points, normals, {
        "scan_sha256": exp24["scan_sha256"],
        "exp24_semantic_sha256": EXP21.sha256_path(exp24_path),
        "exp25_semantic_sha256": EXP21.sha256_path(exp25_path),
        "selected_candidate_id": exp24["selected_candidate_id"],
        "selected_absolute_transform": transform24.tolist(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sldprt", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--cadmpeg", required=True, type=Path)
    parser.add_argument("--exp17-summary", required=True, type=Path)
    parser.add_argument("--exp24-root", required=True, type=Path)
    parser.add_argument("--exp25-root", required=True, type=Path)
    parser.add_argument("--exp25-result", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--case-token", action="append", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    if tuple(args.case_token) != EXPECTED_CASE_TOKENS:
        raise ValueError(f"EXP27 case order must be exactly {EXPECTED_CASE_TOKENS}")
    source_hash = EXP21.sha256_path(args.source_sldprt)
    if source_hash != args.expected_source_sha256:
        raise ValueError("source SLDPRT hash mismatch")
    cadir, decoder_report = EXP25.decode_cadir(args.source_sldprt.resolve(), args.cadmpeg.resolve())
    extraction = EXP17.extract_native_roundbend_patch_axes(cadir)
    exp17_summary = json.loads(args.exp17_summary.read_text(encoding="utf-8"))
    if extraction != exp17_summary.get("extraction"):
        raise ValueError("fresh native event extraction differs from frozen EXP17")
    bundles = native_event_bundles(cadir, extraction)
    if len(bundles) != 4:
        raise ValueError("EXP27 requires exactly four source-native events")
    report_losses = decoder_report.get("losses", [])
    exp25_result = json.loads(args.exp25_result.read_text(encoding="utf-8"))
    if (
        exp25_result.get("experiment_id") != "EXP_CAD_0025"
        or exp25_result.get("post_visual_absolute_registration_orientation_authorized_for_bound_cases") is not True
        or exp25_result.get("exp24_exp25_selected_class_agreement") is not True
        or exp25_result.get("manual_native_visual_pass_count") != 2
        or set(exp25_result.get("semantic_sha256_primary_and_replay", {})) != set(EXPECTED_CASE_TOKENS)
    ):
        raise ValueError("EXP25 post-visual orientation authority is invalid")
    records = []
    for case_token in args.case_token:
        scan_path = args.scan_root / f"{case_token}.ply"
        points, normals, provenance = load_frozen_case(
            case_token, scan_path, args.exp24_root, args.exp25_root, exp25_result
        )
        capped_points, capped_normals, selected_indices = density_cap_fixed_identities(
            points, normals, float(extraction["sheet_thickness_mm"]) * VOXEL_THICKNESS_FRACTION
        )
        event_rows = []
        render_rows = []
        case_root = args.output_root / "cases" / case_token
        for bundle in bundles:
            event_result, arrays = event_conformance(capped_points, capped_normals, bundle)
            render_path = case_root / f"{bundle['event_id']}_native_conformance.png"
            render_event(render_path, case_token, capped_points, bundle, event_result, arrays)
            render_rows.append(
                {
                    "event_id": bundle["event_id"],
                    "path": str(render_path.relative_to(args.output_root)),
                    "sha256": EXP21.sha256_path(render_path),
                    "width_px": 3000,
                    "height_px": 2400,
                }
            )
            event_rows.append(event_result)
        scores = [float(row["event_anomaly_score"]) for row in event_rows if row["event_anomaly_score"] is not None]
        payload = {
            "schema": "EXP_CAD_0027_FIXED_FRAME_DIFFERENTIAL_BEND_CONFORMANCE_CASE_V1",
            "experiment_id": "EXP_CAD_0027",
            "architecture": "FIXED_FRAME_DIFFERENTIAL_BEND_CONFORMANCE_V1",
            "case_token": case_token,
            "scan_sha256": provenance["scan_sha256"],
            "source_sldprt_sha256": source_hash,
            "cadmpeg_sha256": EXP21.sha256_path(args.cadmpeg),
            "exp17_summary_sha256": EXP21.sha256_path(args.exp17_summary),
            "exp24_semantic_sha256": provenance["exp24_semantic_sha256"],
            "exp25_semantic_sha256": provenance["exp25_semantic_sha256"],
            "exp25_result_sha256": EXP21.sha256_path(args.exp25_result),
            "selected_candidate_id": provenance["selected_candidate_id"],
            "selected_absolute_transform": provenance["selected_absolute_transform"],
            "raw_scan_point_count": int(len(points)),
            "fixed_density_capped_point_count": int(len(capped_points)),
            "fixed_density_selected_index_sha256": canonical_int64_array_sha256(selected_indices),
            "sheet_thickness_mm": float(extraction["sheet_thickness_mm"]),
            "native_event_geometry": [_serializable_bundle(bundle) for bundle in bundles],
            "events": event_rows,
            "measured_event_count": int(sum(row["event_anomaly_score"] is not None for row in event_rows)),
            "part_anomaly_score": max(scores) if scores else None,
            "part_status": "MEASURED" if scores else "ABSTAIN",
            "renders": render_rows,
            "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
            "score_scope": "SOURCE_ONLY_SCREENING_INDEX_NOT_QC_TOLERANCE_NOT_ACCURACY_NOT_F1",
            "pose_refinement_performed": False,
            "local_axis_or_radius_search_performed": False,
            "unresolved_displaylist_geometry_used_for_measurement": False,
            "analytic_brep_topology_used": True,
            "evaluator_manifest_accessed": False,
            "evaluator_roles_opened": False,
            "labels_or_expected_counts_used": False,
            "generalization_evidence": False,
            "is_metric_eligible": False,
        }
        semantic_path = case_root / "semantic_result.json"
        EXP21.atomic_json(semantic_path, payload)
        records.append(
            {
                "case_token": case_token,
                "semantic_sha256": EXP21.sha256_path(semantic_path),
                "part_status": payload["part_status"],
                "measured_event_count": payload["measured_event_count"],
                "part_anomaly_score": payload["part_anomaly_score"],
                "render_sha256s": [row["sha256"] for row in render_rows],
            }
        )
    summary = {
        "schema": "EXP_CAD_0027_FIXED_FRAME_DIFFERENTIAL_BEND_CONFORMANCE_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0027",
        "architecture": "FIXED_FRAME_DIFFERENTIAL_BEND_CONFORMANCE_V1",
        "case_count": len(records),
        "event_count_per_case": len(bundles),
        "source_sldprt_sha256": source_hash,
        "cadmpeg_sha256": EXP21.sha256_path(args.cadmpeg),
        "decoder_loss_count": len(report_losses),
        "records": records,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_NATIVE_VISUAL_GATE_NOT_F1_PROGRESS",
        "evaluator_manifest_accessed": False,
        "evaluator_roles_opened": False,
        "labels_or_expected_counts_used": False,
        "generalization_evidence": False,
        "is_metric_eligible": False,
    }
    EXP21.atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
