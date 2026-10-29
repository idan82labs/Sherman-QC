#!/usr/bin/env python3
"""EXP27 V2 paired fixed-frame rotating-normal bend conformance.

This module implements the source-only measurement frozen in
``EXP_CAD_0027_PREREGISTRATION_V2.json``.  It deliberately does not consume an
evaluator manifest.  Absolute radius and plane offsets are emitted only as
diagnostics; the only directional statistic is the paired error in normal
rotation across a CAD-native bend arc.

The V1 runner remains immutable because its radius/form score has a different
scientific contract.  V2 imports only its CADIR topology and frozen-pose
loading helpers.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V1_PATH = ROOT / "scripts" / "run_exp_cad_0027_fixed_frame_differential_bend_conformance.py"
EXP21_PATH = ROOT / "scripts" / "run_exp_cad_0021_protected_registration_selector.py"
SPEC_V1 = importlib.util.spec_from_file_location("exp27_v1", V1_PATH)
V1 = importlib.util.module_from_spec(SPEC_V1)
assert SPEC_V1 and SPEC_V1.loader
SPEC_V1.loader.exec_module(V1)

EXP17 = V1.EXP17
EXP21 = V1.EXP21
EXP25 = V1.EXP25


EXPERIMENT_ID = "EXP_CAD_0027_V2"
ARCHITECTURE = "PAIRED_FIXED_FRAME_ROTATING_NORMAL_BEND_CONFORMANCE_V2"
EXPECTED_CASE_TOKENS = ("N011", "N012")
EXPECTED_PREREGISTRATION_SHA256 = (
    "78b5af922cff679f14ae69e47bb9eb3ad767df8f75236b9b77c567eddab9c997"
)

VOXEL_THICKNESS_FRACTION = 1.0 / 8.0
AXIAL_BIN_COUNT = 8
ANGULAR_BIN_COUNT = 8
ANGULAR_LOW_FRACTION = 0.08
ANGULAR_HIGH_FRACTION = 0.92
MIN_POINTS_PER_CELL = 3
MIN_OBSERVED_CELL_FRACTION = 0.75
MIN_NORMAL_SIGN_CONSISTENCY = 0.98
MAX_MEDIAN_RADIAL_NORMAL_ERROR_DEG = 20.0
MIN_MONOTONE_INCREMENT_COUNT = 6
MIN_NET_TURN_FRACTION_OF_CAD = 0.5
MIN_ELIGIBLE_AXIAL_BINS = 6
TUBE_VARIANTS_MM = (0.35, 0.40, 0.45)
CENTRAL_TUBE_MM = 0.40

SURFACE_MARGIN_FLOOR_DEG = 5.0
SURFACE_MARGIN_MAD_MULTIPLIER = 3.0 * 1.4826
MIN_SAME_SIGN_AXIAL_BINS = 6

TRANSLATION_PERTURBATION_MM = 0.25
ROTATION_PERTURBATION_DEG = 0.25
PERTURBATION_AXES = ("X", "Y", "Z")
TOPOLOGY_TOLERANCE = 1e-6
SHARED_DELTA_MAX_NORM_MM = 1.0
SHARED_DELTA_REQUIRED_RANK = 2
SHARED_DELTA_CONDITION_MAX = 10.0
EXPECTED_IMPORTED_CODE_SHA256 = {
    "EXP27_V1": "2751c57e7bcf2a42cd3d1dcccbbfdde43aa189cea5e1183c93fd1f7f7c1fc658",
    "EXP17": "be2db1631d692482a5ff87ce6e1560aa294066da64fb65afe195bd75918bd005",
    "EXP21": "9587e920cd1b3bf5a0e109b2d520e772592c10d671a368f8e5f0f517d91056e0",
    "EXP25": "09c28f30d8653b00fd77639245718e1b81243dbd4ab0bc8378a5ec0daa295cc7",
}
EXPECTED_CADMPEG_WARNING_CODES = (
    "config.incoherent-body-refs",
    "config.incomplete-snapshot",
    "config.inferred-without-native",
    "feature.native-kind-retained",
    "feature.typed-operand-incomplete",
    "tessellation.face-ownership-unresolved",
)
IMPORTED_CODE_PATHS = {
    "EXP27_V1": V1_PATH,
    "EXP17": V1.EXP17_PATH,
    "EXP21": EXP21_PATH,
    "EXP25": V1.EXP25_PATH,
}


def _finite_points_and_unit_normals(
    points: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both be finite Nx3 arrays")
    lengths = np.linalg.norm(normals, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & np.isfinite(lengths)
        & (lengths > 1e-12)
    )
    identities = np.flatnonzero(valid).astype(np.int64)
    return points[valid], normals[valid] / lengths[valid, None], identities


def canonical_int64_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def density_cap_canonical(
    points: np.ndarray,
    normals: np.ndarray,
    voxel_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one valid normal per voxel without input-order preference.

    Selection follows the frozen rule exactly: valid normals first, then the
    point closest to the voxel centre, then canonical lexicographic point and
    normal coordinates.  The original row identity is only the final tie for
    byte-identical duplicate samples.  Returned rows are in canonical voxel
    order; their ordinal positions are the identities used by even/odd
    partitions.
    """

    if not math.isfinite(voxel_mm) or voxel_mm <= 0.0:
        raise ValueError("voxel size must be finite and positive")
    valid_points, valid_normals, original_ids = _finite_points_and_unit_normals(
        points, normals
    )
    if len(valid_points) == 0:
        return valid_points, valid_normals, original_ids
    keys = np.floor(valid_points / voxel_mm).astype(np.int64)
    centres = (keys.astype(np.float64) + 0.5) * voxel_mm
    squared_distance = np.sum((valid_points - centres) ** 2, axis=1)
    # np.lexsort uses the final key as the primary key.  Voxel xyz therefore
    # appears last in reverse priority; geometric ties are resolved without
    # depending on raw acquisition order.
    order = np.lexsort(
        (
            original_ids,
            valid_normals[:, 2],
            valid_normals[:, 1],
            valid_normals[:, 0],
            valid_points[:, 2],
            valid_points[:, 1],
            valid_points[:, 0],
            squared_distance,
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
        )
    )
    ordered_keys = keys[order]
    keep = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        keep[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    chosen = order[keep]
    return valid_points[chosen], valid_normals[chosen], original_ids[chosen]


def _bin_indices(values: np.ndarray, low: float, high: float, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scaled = np.floor((values - low) / (high - low) * count).astype(np.int64)
    return np.clip(scaled, 0, count - 1)


def _radial_geometry(
    points: np.ndarray,
    side: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    axis = V1.normalized(np.asarray(side["axis"], dtype=np.float64))
    foot = np.asarray(side["axis_foot_mm"], dtype=np.float64)
    start = np.asarray(side["line_start_mm"], dtype=np.float64)
    span = float(side["span_mm"])
    relative = points - foot
    axial = (points - start) @ axis
    radial = relative - (relative @ axis)[:, None] * axis[None, :]
    radial_distance = np.linalg.norm(radial, axis=1)
    radial_unit = radial / np.maximum(radial_distance[:, None], 1e-12)
    angles, theta = V1.angle_coordinates(
        radial_unit,
        np.asarray(side["guards"][0]["radial_unit"], dtype=np.float64),
        np.asarray(side["guards"][1]["radial_unit"], dtype=np.float64),
        axis,
    )
    return {
        "axis": axis,
        "axial_mm": axial,
        "radial_distance_mm": radial_distance,
        "radial_unit": radial_unit,
        "angular_fraction": angles / theta,
        "radial_residual_mm": radial_distance - float(side["radius_mm"]),
        "theta_cad_rad": np.asarray(theta),
        "span_mm": np.asarray(span),
    }


def _signed_azimuth(
    vector: np.ndarray,
    first_radial: np.ndarray,
    second_radial: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Azimuth of an expected-outward normal in the frozen CAD arc direction."""

    axis = V1.normalized(axis)
    first = V1.normalized(first_radial)
    second = V1.normalized(second_radial)
    direction = 1.0 if float(np.dot(axis, np.cross(first, second))) >= 0.0 else -1.0
    tangent = V1.normalized(direction * np.cross(axis, first))
    projected = np.asarray(vector, dtype=np.float64)
    projected = projected - float(np.dot(projected, axis)) * axis
    projected = V1.normalized(projected)
    return float(math.atan2(float(np.dot(projected, tangent)), float(np.dot(projected, first))))


def _surface_turn_measurement(
    points: np.ndarray,
    normals: np.ndarray,
    side: Mapping[str, Any],
    *,
    owner_mask: np.ndarray,
    expected_normal_sign: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    geometry = _radial_geometry(points, side)
    axial = geometry["axial_mm"]
    angular = geometry["angular_fraction"]
    radial_unit = geometry["radial_unit"]
    span = float(geometry["span_mm"])
    selected = np.asarray(owner_mask, dtype=bool)

    axial_indices_all = _bin_indices(axial[selected], 0.0, span, AXIAL_BIN_COUNT)
    angular_indices_all = _bin_indices(
        angular[selected],
        ANGULAR_LOW_FRACTION,
        ANGULAR_HIGH_FRACTION,
        ANGULAR_BIN_COUNT,
    )
    selected_ids = np.flatnonzero(selected)
    counts = np.zeros((AXIAL_BIN_COUNT, ANGULAR_BIN_COUNT), dtype=np.int64)
    mode_azimuth = np.full((AXIAL_BIN_COUNT, ANGULAR_BIN_COUNT), np.nan)
    expected_dot = expected_normal_sign * np.sum(normals * radial_unit, axis=1)
    radial_error_deg = np.degrees(np.arccos(np.clip(expected_dot, -1.0, 1.0)))
    first = np.asarray(side["guards"][0]["radial_unit"], dtype=np.float64)
    second = np.asarray(side["guards"][1]["radial_unit"], dtype=np.float64)
    axis = np.asarray(side["axis"], dtype=np.float64)

    for axial_index in range(AXIAL_BIN_COUNT):
        for angular_index in range(ANGULAR_BIN_COUNT):
            local = (
                (axial_indices_all == axial_index)
                & (angular_indices_all == angular_index)
            )
            ids = selected_ids[local]
            counts[axial_index, angular_index] = len(ids)
            if len(ids) < MIN_POINTS_PER_CELL:
                continue
            # The surface-level and per-bin sign gates are evaluated from raw
            # oriented normals.  Multiplication by the known skin sign merely
            # maps the accepted inner-skin mode into the common outward radial
            # convention; it does not flip samples based on their observation.
            vector = np.sum(expected_normal_sign * normals[ids], axis=0)
            try:
                mode_azimuth[axial_index, angular_index] = _signed_azimuth(
                    vector, first, second, axis
                )
            except ValueError:
                continue

    observed_cell_fraction = float(np.mean(counts >= MIN_POINTS_PER_CELL))
    selected_count = int(np.count_nonzero(selected))
    surface_sign_fraction = (
        float(np.mean(expected_dot[selected] > 0.0)) if selected_count else 0.0
    )
    surface_median_error = (
        float(np.median(radial_error_deg[selected])) if selected_count else None
    )
    expected_turn = (
        (ANGULAR_BIN_COUNT - 1)
        / ANGULAR_BIN_COUNT
        * (ANGULAR_HIGH_FRACTION - ANGULAR_LOW_FRACTION)
        * float(side["theta_cad_rad"])
    )
    turn_profile = np.full(AXIAL_BIN_COUNT, np.nan)
    error_profile = np.full(AXIAL_BIN_COUNT, np.nan)
    sign_profile = np.full(AXIAL_BIN_COUNT, np.nan)
    normal_error_profile = np.full(AXIAL_BIN_COUNT, np.nan)
    monotone_increment_counts = np.zeros(AXIAL_BIN_COUNT, dtype=np.int64)
    eligible = np.zeros(AXIAL_BIN_COUNT, dtype=bool)

    surface_global_gate = (
        observed_cell_fraction >= MIN_OBSERVED_CELL_FRACTION
        and surface_sign_fraction >= MIN_NORMAL_SIGN_CONSISTENCY
        and surface_median_error is not None
        and surface_median_error <= MAX_MEDIAN_RADIAL_NORMAL_ERROR_DEG
    )
    for axial_index in range(AXIAL_BIN_COUNT):
        ids = selected_ids[axial_indices_all == axial_index]
        if len(ids):
            sign_profile[axial_index] = float(np.mean(expected_dot[ids] > 0.0))
            normal_error_profile[axial_index] = float(np.median(radial_error_deg[ids]))
        modes = mode_azimuth[axial_index]
        if (
            counts[axial_index, 0] < MIN_POINTS_PER_CELL
            or counts[axial_index, -1] < MIN_POINTS_PER_CELL
            or not np.isfinite(modes).all()
        ):
            continue
        unwrapped = np.unwrap(modes)
        increments = np.diff(unwrapped)
        monotone_count = int(np.count_nonzero(increments > 0.0))
        observed_turn = float(unwrapped[-1] - unwrapped[0])
        turn_profile[axial_index] = observed_turn
        error_profile[axial_index] = abs(observed_turn - expected_turn)
        monotone_increment_counts[axial_index] = monotone_count
        eligible[axial_index] = bool(
            surface_global_gate
            and sign_profile[axial_index] >= MIN_NORMAL_SIGN_CONSISTENCY
            and normal_error_profile[axial_index]
            <= MAX_MEDIAN_RADIAL_NORMAL_ERROR_DEG
            and monotone_count >= MIN_MONOTONE_INCREMENT_COUNT
            and observed_turn >= MIN_NET_TURN_FRACTION_OF_CAD * expected_turn
        )

    eligible_count = int(np.count_nonzero(eligible))
    reasons: list[str] = []
    if observed_cell_fraction < MIN_OBSERVED_CELL_FRACTION:
        reasons.append("OBSERVED_CELL_FRACTION_BELOW_0.75")
    if surface_sign_fraction < MIN_NORMAL_SIGN_CONSISTENCY:
        reasons.append("NORMAL_SIGN_CONSISTENCY_BELOW_0.98")
    if surface_median_error is None or surface_median_error > MAX_MEDIAN_RADIAL_NORMAL_ERROR_DEG:
        reasons.append("RADIAL_NORMAL_ERROR_ABOVE_20_DEG")
    if eligible_count < MIN_ELIGIBLE_AXIAL_BINS:
        reasons.append("FEWER_THAN_6_ELIGIBLE_AXIAL_BINS")
    status = "MEASURED_NORMAL_TURN" if not reasons else "ABSTAIN"
    radial_bias = (
        float(np.median(geometry["radial_residual_mm"][selected]))
        if selected_count
        else None
    )
    row = {
        "cylinder_face_id": str(side["cylinder_face_id"]),
        "radius_mm": float(side["radius_mm"]),
        "expected_normal_sign": int(expected_normal_sign),
        "status": status,
        "abstention_reasons": reasons,
        "owned_point_count": selected_count,
        "observed_cell_fraction": observed_cell_fraction,
        "surface_normal_sign_consistency": surface_sign_fraction,
        "surface_median_radial_normal_error_deg": surface_median_error,
        "eligible_axial_bin_count": eligible_count,
        "expected_center_to_center_turn_deg": math.degrees(expected_turn),
        "absolute_radius_bias_mm_diagnostic_only": radial_bias,
        "absolute_radius_bias_used_for_direction": False,
        "normal_turn_error_deg_by_axial_bin": [
            float(math.degrees(value)) if math.isfinite(value) else None
            for value in error_profile
        ],
        "observed_normal_turn_deg_by_axial_bin": [
            float(math.degrees(value)) if math.isfinite(value) else None
            for value in turn_profile
        ],
        "eligible_axial_bins": eligible.tolist(),
        "normal_sign_fraction_by_axial_bin": [
            float(value) if math.isfinite(value) else None for value in sign_profile
        ],
        "median_radial_normal_error_deg_by_axial_bin": [
            float(value) if math.isfinite(value) else None
            for value in normal_error_profile
        ],
        "monotone_increment_count_by_axial_bin": monotone_increment_counts.tolist(),
    }
    arrays = {
        "owner_mask": selected,
        "cell_counts": counts,
        "normal_mode_azimuth_rad": mode_azimuth,
        "turn_profile_rad": turn_profile,
        "error_profile_rad": error_profile,
        "eligible_axial_bins": eligible,
        "radial_residual_mm": geometry["radial_residual_mm"],
        "angular_fraction": geometry["angular_fraction"],
        "axial_mm": geometry["axial_mm"],
    }
    return row, arrays


def _shared_plane_delta_diagnostic(
    points: np.ndarray,
    normals: np.ndarray,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve one shared transverse delta from all four fixed plane guards.

    This result can only establish whether the fixed local frame is observable.
    It is never subtracted from cylinder points or used as direction evidence.
    """

    rows = []
    for side in event["sides"]:
        for guard in side["guards"]:
            observation = V1.plane_guard_observation(
                points,
                normals,
                guard,
                event,
                near_thickness=V1.PLANE_GUARD_NEAR_THICKNESS,
                far_thickness=V1.PLANE_GUARD_FAR_THICKNESS,
            )
            rows.append((guard, observation))
    actionable = all(
        bool(observation["actionable"])
        and observation["median_signed_offset_mm"] is not None
        for _guard, observation in rows
    )
    delta: Optional[np.ndarray] = None
    rank = 0
    condition = math.inf
    if actionable:
        matrix = np.vstack(
            [np.asarray(guard["plane_normal"], dtype=np.float64) for guard, _ in rows]
        )
        offsets = np.asarray(
            [observation["median_signed_offset_mm"] for _guard, observation in rows],
            dtype=np.float64,
        )
        delta, _residuals, rank, singular = np.linalg.lstsq(matrix, offsets, rcond=None)
        nonzero = singular[singular > np.finfo(np.float64).eps * max(matrix.shape) * singular[0]]
        condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
        actionable = bool(
            rank == SHARED_DELTA_REQUIRED_RANK
            and len(nonzero) == SHARED_DELTA_REQUIRED_RANK
            and condition <= SHARED_DELTA_CONDITION_MAX
            and np.isfinite(delta).all()
            and float(np.linalg.norm(delta)) <= SHARED_DELTA_MAX_NORM_MM
        )
    return {
        "status": "OBSERVED_DIAGNOSTIC_ONLY" if actionable else "ABSTAIN",
        "guard_count": len(rows),
        "actionable_guard_count": int(
            sum(bool(observation["actionable"]) for _guard, observation in rows)
        ),
        "rank": int(rank),
        "nonzero_singular_condition": condition if math.isfinite(condition) else None,
        "required_rank": SHARED_DELTA_REQUIRED_RANK,
        "maximum_nonzero_singular_condition": SHARED_DELTA_CONDITION_MAX,
        "maximum_shared_delta_norm_mm": SHARED_DELTA_MAX_NORM_MM,
        "shared_delta_mm_diagnostic_only": delta.tolist() if actionable and delta is not None else None,
        "shared_delta_norm_mm_diagnostic_only": (
            float(np.linalg.norm(delta)) if actionable and delta is not None else None
        ),
        "guard_offsets_mm_diagnostic_only": [
            observation["median_signed_offset_mm"] for _guard, observation in rows
        ],
        "used_for_score_correction": False,
        "used_for_direction": False,
    }


def validate_native_event_contract(event: Mapping[str, Any]) -> None:
    """Fail closed unless the two skins are the frozen concentric CAD event."""

    sides = sorted(event.get("sides", []), key=lambda row: float(row["radius_mm"]))
    if len(sides) != 2:
        raise ValueError("native event must contain exactly two cylinder skins")
    radii = [float(side["radius_mm"]) for side in sides]
    if not np.allclose(radii, [1.0, 3.0], rtol=0.0, atol=TOPOLOGY_TOLERANCE):
        raise ValueError("native event does not expose the required 1/3 mm radii")
    if not math.isclose(radii[1] - radii[0], 2.0, rel_tol=0.0, abs_tol=TOPOLOGY_TOLERANCE):
        raise ValueError("native event radius gap is not 2 mm")
    if not math.isclose(
        float(event["sheet_thickness_mm"]), 2.0, rel_tol=0.0, abs_tol=TOPOLOGY_TOLERANCE
    ):
        raise ValueError("native event sheet thickness is not 2 mm")
    first_axis = V1.normalized(np.asarray(sides[0]["axis"], dtype=np.float64))
    second_axis = V1.normalized(np.asarray(sides[1]["axis"], dtype=np.float64))
    if not (
        np.allclose(first_axis, second_axis, rtol=0.0, atol=TOPOLOGY_TOLERANCE)
        or np.allclose(first_axis, -second_axis, rtol=0.0, atol=TOPOLOGY_TOLERANCE)
    ):
        raise ValueError("native event skins do not share an axis")
    if not np.allclose(
        np.asarray(sides[0]["axis_foot_mm"], dtype=np.float64),
        np.asarray(sides[1]["axis_foot_mm"], dtype=np.float64),
        rtol=0.0,
        atol=TOPOLOGY_TOLERANCE,
    ):
        raise ValueError("native event skins do not share an axis foot")
    if not math.isclose(
        float(sides[0]["theta_cad_rad"]),
        float(sides[1]["theta_cad_rad"]),
        rel_tol=0.0,
        abs_tol=TOPOLOGY_TOLERANCE,
    ):
        raise ValueError("native event skins do not share a CAD arc angle")
    endpoint_sets = []
    for side in sides:
        guards = side.get("guards", [])
        if len(guards) != 2:
            raise ValueError("each native skin requires exactly two tangent guards")
        endpoint_sets.append(
            [V1.normalized(np.asarray(guard["radial_unit"], dtype=np.float64)) for guard in guards]
        )
    direct = all(
        np.allclose(endpoint_sets[0][index], endpoint_sets[1][index], rtol=0.0, atol=TOPOLOGY_TOLERANCE)
        for index in range(2)
    )
    swapped = all(
        np.allclose(endpoint_sets[0][index], endpoint_sets[1][1 - index], rtol=0.0, atol=TOPOLOGY_TOLERANCE)
        for index in range(2)
    )
    if not (direct or swapped):
        raise ValueError("native event skins do not share unordered arc endpoints")
    for name in ("line_start_mm", "line_end_mm"):
        if not np.allclose(
            np.asarray(sides[0][name], dtype=np.float64),
            np.asarray(sides[1][name], dtype=np.float64),
            rtol=0.0,
            atol=TOPOLOGY_TOLERANCE,
        ):
            raise ValueError(f"native event skins do not share {name}")


def event_normal_turn_measurement(
    points: np.ndarray,
    normals: np.ndarray,
    event: Mapping[str, Any],
    *,
    tube_mm: float = CENTRAL_TUBE_MM,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure both native skins of one event without fitting any geometry."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must have matching Nx3 shape")
    if tube_mm not in TUBE_VARIANTS_MM:
        raise ValueError("tube is not a frozen EXP27 V2 variant")
    validate_native_event_contract(event)
    sides = sorted(event["sides"], key=lambda row: float(row["radius_mm"]))
    radius_gap = float(sides[1]["radius_mm"]) - float(sides[0]["radius_mm"])
    thickness = float(event["sheet_thickness_mm"])
    if (
        not math.isclose(radius_gap, thickness, rel_tol=0.0, abs_tol=1e-6)
        or 2.0 * tube_mm >= radius_gap
    ):
        raise ValueError("cylinder tubes are not strictly non-overlapping")

    geometries = [_radial_geometry(points, side) for side in sides]
    masks = []
    for geometry in geometries:
        masks.append(
            (geometry["axial_mm"] >= 0.0)
            & (geometry["axial_mm"] <= float(geometry["span_mm"]))
            & (geometry["angular_fraction"] >= ANGULAR_LOW_FRACTION)
            & (geometry["angular_fraction"] <= ANGULAR_HIGH_FRACTION)
            & (np.abs(geometry["radial_residual_mm"]) <= tube_mm)
        )
    overlap = masks[0] & masks[1]
    if np.any(overlap):
        raise ValueError("a scan identity entered both frozen cylinder tubes")

    shared_delta = _shared_plane_delta_diagnostic(points, normals, event)
    surface_rows = []
    surface_arrays = []
    for side_index, (side, mask) in enumerate(zip(sides, masks)):
        expected_sign = -1.0 if side_index == 0 else 1.0
        row, arrays = _surface_turn_measurement(
            points,
            normals,
            side,
            owner_mask=mask,
            expected_normal_sign=expected_sign,
        )
        surface_rows.append(row)
        surface_arrays.append(arrays)

    reasons: list[str] = []
    if shared_delta["status"] != "OBSERVED_DIAGNOSTIC_ONLY":
        reasons.append("FOUR_SHARED_PLANE_GUARDS_NOT_ACTIONABLE")
    if any(row["status"] != "MEASURED_NORMAL_TURN" for row in surface_rows):
        reasons.append("BOTH_SKINS_NOT_MEASURED")
    status = "MEASURED_BOTH_SKINS" if not reasons else "ABSTAIN"
    result = {
        "event_id": str(event["event_id"]),
        "status": status,
        "abstention_reasons": reasons,
        "tube_mm": float(tube_mm),
        "radius_gap_mm": radius_gap,
        "strict_nonoverlap_verified": True,
        "shared_event_plane_translation": shared_delta,
        "surfaces": surface_rows,
        "absolute_radius_and_plane_offsets_are_diagnostic_only": True,
        "missing_support_used_as_defect_evidence": False,
    }
    return result, {"surfaces": surface_arrays}


def _median_absolute_deviation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def paired_event_direction(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    """Pair N011-N012 normal-turn errors for one corresponding event."""

    if first["event_id"] != second["event_id"]:
        raise ValueError("event identities do not match")
    input_event_gate = bool(
        first.get("status") == "MEASURED_BOTH_SKINS"
        and second.get("status") == "MEASURED_BOTH_SKINS"
    )
    first_surfaces = sorted(first["surfaces"], key=lambda row: float(row["radius_mm"]))
    second_surfaces = sorted(second["surfaces"], key=lambda row: float(row["radius_mm"]))
    if [row["radius_mm"] for row in first_surfaces] != [row["radius_mm"] for row in second_surfaces]:
        raise ValueError("paired event radii do not match")
    surface_rows = []
    for left, right in zip(first_surfaces, second_surfaces):
        left_values = np.asarray(
            [np.nan if value is None else float(value) for value in left["normal_turn_error_deg_by_axial_bin"]]
        )
        right_values = np.asarray(
            [np.nan if value is None else float(value) for value in right["normal_turn_error_deg_by_axial_bin"]]
        )
        left_eligible = np.asarray(left["eligible_axial_bins"], dtype=bool)
        right_eligible = np.asarray(right["eligible_axial_bins"], dtype=bool)
        paired = left_eligible & right_eligible & np.isfinite(left_values) & np.isfinite(right_values)
        deltas = left_values[paired] - right_values[paired]
        median = float(np.median(deltas)) if len(deltas) else None
        mad = _median_absolute_deviation(deltas) if len(deltas) else None
        margin = (
            max(SURFACE_MARGIN_FLOOR_DEG, SURFACE_MARGIN_MAD_MULTIPLIER * mad)
            if mad is not None
            else None
        )
        sign = 0
        same_sign_count = 0
        if median is not None and margin is not None and abs(median) >= margin:
            sign = 1 if median > 0.0 else -1
            same_sign_count = int(np.count_nonzero(sign * deltas > 0.0))
            if same_sign_count < MIN_SAME_SIGN_AXIAL_BINS:
                sign = 0
        reasons = []
        if len(deltas) < MIN_SAME_SIGN_AXIAL_BINS:
            reasons.append("FEWER_THAN_6_PAIRED_ELIGIBLE_AXIAL_BINS")
        if median is None or margin is None or abs(median) < margin:
            reasons.append("PAIRED_MEDIAN_BELOW_FROZEN_MARGIN")
        if median is not None and abs(median) >= (margin or math.inf) and same_sign_count < MIN_SAME_SIGN_AXIAL_BINS:
            reasons.append("FEWER_THAN_6_SAME_SIGN_PAIRED_DELTAS")
        surface_rows.append(
            {
                "radius_mm": float(left["radius_mm"]),
                "status": "QUALIFIED_DIRECTION" if sign else "ABSTAIN",
                "direction_sign_n011_minus_n012": sign,
                "direction": (
                    "N011_MORE_NONCONFORMING" if sign > 0 else
                    "N012_MORE_NONCONFORMING" if sign < 0 else "ABSTAIN"
                ),
                "paired_eligible_axial_bin_count": int(len(deltas)),
                "same_sign_axial_bin_count": same_sign_count,
                "paired_delta_deg_by_axial_bin": [
                    (float(left_values[index] - right_values[index]) if paired[index] else None)
                    for index in range(AXIAL_BIN_COUNT)
                ],
                "paired_delta_median_deg": median,
                "paired_delta_mad_deg": mad,
                "required_margin_deg": margin,
                "abstention_reasons": reasons,
            }
        )
    signs = [row["direction_sign_n011_minus_n012"] for row in surface_rows]
    qualified = bool(
        input_event_gate
        and len(signs) == 2
        and signs[0] != 0
        and signs[0] == signs[1]
    )
    sign = signs[0] if qualified else 0
    reasons = []
    if not input_event_gate:
        reasons.append("BOTH_CASE_EVENTS_MUST_BE_MEASURED_BOTH_SKINS")
    if not qualified:
        reasons.append("BOTH_SKINS_DO_NOT_QUALIFY_WITH_SAME_NONZERO_SIGN")
    return {
        "event_id": str(first["event_id"]),
        "status": "QUALIFIED_EVENT_DIRECTION" if qualified else "ABSTAIN",
        "direction_sign_n011_minus_n012": int(sign),
        "direction": (
            "N011_MORE_NONCONFORMING" if sign > 0 else
            "N012_MORE_NONCONFORMING" if sign < 0 else "ABSTAIN"
        ),
        "abstention_reasons": reasons,
        "surfaces": surface_rows,
    }


def pair_direction(event_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    event_rows = [paired_event_direction(first, second) for first, second in event_pairs]
    qualified = [
        int(row["direction_sign_n011_minus_n012"])
        for row in event_rows
        if row["status"] == "QUALIFIED_EVENT_DIRECTION"
    ]
    reasons = []
    if not qualified:
        reasons.append("NO_QUALIFIED_EVENT")
    if len(set(qualified)) > 1:
        reasons.append("OPPOSING_QUALIFIED_EVENTS")
    sign = qualified[0] if qualified and len(set(qualified)) == 1 else 0
    return {
        "status": "DIRECTIONAL" if sign else "ABSTAIN",
        "direction_sign_n011_minus_n012": int(sign),
        "direction": (
            "N011_MORE_NONCONFORMING" if sign > 0 else
            "N012_MORE_NONCONFORMING" if sign < 0 else "ABSTAIN"
        ),
        "qualified_event_count": len(qualified),
        "abstention_reasons": reasons,
        "events": event_rows,
    }


def _rotation_matrix(axis_index: int, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    vector = np.zeros(3, dtype=np.float64)
    vector[axis_index] = 1.0
    skew = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def individual_pose_variants() -> list[dict[str, Any]]:
    """Return the frozen 24 one-scan-at-a-time CAD-frame perturbations."""

    rows = []
    for case_index, case_token in enumerate(EXPECTED_CASE_TOKENS):
        for axis_index, axis_name in enumerate(PERTURBATION_AXES):
            for sign in (-1, 1):
                translation = np.zeros(3)
                translation[axis_index] = sign * TRANSLATION_PERTURBATION_MM
                rows.append(
                    {
                        "variant_id": f"{case_token}_T{axis_name}_{sign:+d}",
                        "case_index": case_index,
                        "rotation": np.eye(3),
                        "translation": translation,
                    }
                )
                rows.append(
                    {
                        "variant_id": f"{case_token}_R{axis_name}_{sign:+d}",
                        "case_index": case_index,
                        "rotation": _rotation_matrix(
                            axis_index, sign * ROTATION_PERTURBATION_DEG
                        ),
                        "translation": np.zeros(3),
                    }
                )
    if len(rows) != 24 or len({row["variant_id"] for row in rows}) != 24:
        raise AssertionError("EXP27 V2 requires exactly 24 individual pose variants")
    return rows


def apply_cad_frame_perturbation(
    points: np.ndarray,
    normals: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-multiply a scan-to-CAD pose, rotating about the CAD origin."""

    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("invalid perturbation")
    return points @ rotation.T + translation, normals @ rotation.T


def measure_case_events(
    points: np.ndarray,
    normals: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    *,
    tube_mm: float,
) -> list[dict[str, Any]]:
    return [
        event_normal_turn_measurement(points, normals, event, tube_mm=tube_mm)[0]
        for event in events
    ]


def measure_pair_variant(
    first_points: np.ndarray,
    first_normals: np.ndarray,
    second_points: np.ndarray,
    second_normals: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    *,
    tube_mm: float,
) -> dict[str, Any]:
    first_events = measure_case_events(first_points, first_normals, events, tube_mm=tube_mm)
    second_events = measure_case_events(second_points, second_normals, events, tube_mm=tube_mm)
    return pair_direction(list(zip(first_events, second_events)))


def robustness_matrix(
    first_points: np.ndarray,
    first_normals: np.ndarray,
    second_points: np.ndarray,
    second_normals: np.ndarray,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate 3 tubes across full/even/odd central and 24 pose variants."""

    clouds = [
        (np.asarray(first_points), np.asarray(first_normals)),
        (np.asarray(second_points), np.asarray(second_normals)),
    ]
    variants: list[dict[str, Any]] = []
    partitions = {
        "CENTRAL_FULL": (np.arange(len(clouds[0][0])), np.arange(len(clouds[1][0]))),
        "CENTRAL_EVEN": (
            np.arange(0, len(clouds[0][0]), 2), np.arange(0, len(clouds[1][0]), 2)
        ),
        "CENTRAL_ODD": (
            np.arange(1, len(clouds[0][0]), 2), np.arange(1, len(clouds[1][0]), 2)
        ),
    }
    for variant_id, indices in partitions.items():
        variants.append(
            {
                "variant_id": variant_id,
                "points_normals": [
                    (clouds[index][0][indices[index]], clouds[index][1][indices[index]])
                    for index in range(2)
                ],
            }
        )
    for pose in individual_pose_variants():
        changed = [(points.copy(), normals.copy()) for points, normals in clouds]
        index = int(pose["case_index"])
        changed[index] = apply_cad_frame_perturbation(
            changed[index][0], changed[index][1], pose["rotation"], pose["translation"]
        )
        variants.append({"variant_id": pose["variant_id"], "points_normals": changed})

    rows = []
    for variant in variants:
        for tube_mm in TUBE_VARIANTS_MM:
            (left_points, left_normals), (right_points, right_normals) = variant["points_normals"]
            result = measure_pair_variant(
                left_points,
                left_normals,
                right_points,
                right_normals,
                events,
                tube_mm=tube_mm,
            )
            rows.append(
                {
                    "variant_id": variant["variant_id"],
                    "tube_mm": tube_mm,
                    "status": result["status"],
                    "direction": result["direction"],
                    "direction_sign_n011_minus_n012": result[
                        "direction_sign_n011_minus_n012"
                    ],
                    "qualified_event_count": result["qualified_event_count"],
                    "result": result,
                }
            )
    central = next(
        row
        for row in rows
        if row["variant_id"] == "CENTRAL_FULL"
        and math.isclose(row["tube_mm"], CENTRAL_TUBE_MM)
    )
    invariant = _rows_are_direction_invariant(rows, central)
    return {
        "central": central,
        "variant_count_excluding_tube_cross_product": len(variants),
        "individual_pose_variant_count": len(individual_pose_variants()),
        "tube_cross_product_result_count": len(rows),
        "required_direction_invariance": True,
        "direction_invariant": invariant,
        "status": "PASS" if invariant else "FAIL_TERMINAL_ABSTAIN",
        "rows": rows,
    }


def _rows_are_direction_invariant(
    rows: Sequence[Mapping[str, Any]], central: Mapping[str, Any]
) -> bool:
    return bool(
        central["status"] == "DIRECTIONAL"
        and all(
            row["status"] == "DIRECTIONAL"
            and row["direction_sign_n011_minus_n012"]
            == central["direction_sign_n011_minus_n012"]
            for row in rows
        )
    )


def _json_case_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return the numeric event payload without any transient array objects."""

    # event_normal_turn_measurement already emits only JSON-safe scalars and
    # lists.  Round-tripping makes accidental ndarray leakage fail here rather
    # than after scientific execution.
    return json.loads(json.dumps(event, allow_nan=False))


def _local_cross_section(
    points: np.ndarray,
    event: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = sorted(event["sides"], key=lambda row: float(row["radius_mm"]))[0]
    axis = V1.normalized(np.asarray(side["axis"], dtype=np.float64))
    foot = np.asarray(side["axis_foot_mm"], dtype=np.float64)
    first = V1.normalized(np.asarray(side["guards"][0]["radial_unit"], dtype=np.float64))
    second = V1.normalized(np.asarray(side["guards"][1]["radial_unit"], dtype=np.float64))
    direction = 1.0 if float(np.dot(axis, np.cross(first, second))) >= 0.0 else -1.0
    tangent = V1.normalized(direction * np.cross(axis, first))
    local = np.asarray(points, dtype=np.float64) - foot
    return local @ first, local @ tangent, (points - np.asarray(event["line_start_mm"])) @ axis


def _paired_delta_profile(
    left_surface: Mapping[str, Any], right_surface: Mapping[str, Any]
) -> np.ndarray:
    left = np.asarray(
        [np.nan if value is None else float(value) for value in left_surface["normal_turn_error_deg_by_axial_bin"]]
    )
    right = np.asarray(
        [np.nan if value is None else float(value) for value in right_surface["normal_turn_error_deg_by_axial_bin"]]
    )
    eligible = (
        np.asarray(left_surface["eligible_axial_bins"], dtype=bool)
        & np.asarray(right_surface["eligible_axial_bins"], dtype=bool)
        & np.isfinite(left)
        & np.isfinite(right)
    )
    result = np.full(AXIAL_BIN_COUNT, np.nan)
    result[eligible] = left[eligible] - right[eligible]
    return result


def render_paired_event(
    path: Path,
    event: Mapping[str, Any],
    case_points: Sequence[np.ndarray],
    case_results: Sequence[Mapping[str, Any]],
    case_arrays: Sequence[Mapping[str, Any]],
    paired_result: Mapping[str, Any],
    global_radius_bias_by_case_and_surface: Sequence[Sequence[Optional[float]]],
) -> None:
    """Render every preregistered visual layer on shared paired scales."""

    if len(case_points) != 2 or len(case_results) != 2 or len(case_arrays) != 2:
        raise ValueError("paired rendering requires exactly N011 and N012")
    figure, axes = plt.subplots(3, 4, figsize=(30, 24), dpi=100, facecolor="#081018")
    for plot in axes.flat:
        plot.set_facecolor("#101a24")
        plot.tick_params(colors="#dce7ef")
        for spine in plot.spines.values():
            spine.set_color("#647587")

    # Raw local support for both scans, analytic inner/outer arcs and all four
    # fixed plane guards.  This panel is deliberately not residual-filtered.
    cross = axes[0, 0]
    colors = ("#48d7ff", "#ff9b42")
    labels = ("N011 raw local support", "N012 raw local support")
    max_radius = max(float(side["radius_mm"]) for side in event["sides"])
    thickness = float(event["sheet_thickness_mm"])
    for case_index, points in enumerate(case_points):
        horizontal, vertical, axial = _local_cross_section(points, event)
        local_mask = (
            (axial >= 0.0)
            & (axial <= float(event["span_mm"]))
            & (np.hypot(horizontal, vertical) <= max_radius + 2.0 * thickness)
        )
        chosen = np.flatnonzero(local_mask)
        chosen = chosen[:: max(1, int(math.ceil(len(chosen) / 16000)))]
        cross.scatter(
            horizontal[chosen], vertical[chosen], s=1.4, alpha=0.25,
            c=colors[case_index], linewidths=0, label=labels[case_index],
        )
    side0 = sorted(event["sides"], key=lambda row: float(row["radius_mm"]))[0]
    theta = float(side0["theta_cad_rad"])
    phi = np.linspace(0.0, theta, 240)
    for side in sorted(event["sides"], key=lambda row: float(row["radius_mm"])):
        radius = float(side["radius_mm"])
        cross.plot(
            radius * np.cos(phi), radius * np.sin(phi),
            color="#ff3aae", linewidth=3.0,
            label=f"analytic CAD arc R={radius:g} mm",
        )
        for guard in side["guards"]:
            origin = np.asarray(guard["edge_midpoint_mm"], dtype=np.float64)
            interior = np.asarray(guard["interior_direction"], dtype=np.float64)
            samples = origin + np.linspace(0.0, 6.0 * thickness, 60)[:, None] * interior
            gx, gy, _gz = _local_cross_section(samples, event)
            cross.plot(gx, gy, color="#9cff57", linewidth=1.4, alpha=0.85)
    cross.set_aspect("equal", adjustable="box")
    cross.set_title("Raw paired local support + analytic CAD arcs + plane guards", color="white", fontsize=13)
    cross.set_xlabel("CAD endpoint-1 radial coordinate (mm)", color="white")
    cross.set_ylabel("CAD arc coordinate (mm)", color="white")
    cross.legend(loc="best", fontsize=7, facecolor="#101a24", labelcolor="white")

    # Absolute radius bias remains conspicuously segregated as a diagnostic.
    # Values are the case-wide medians for each native radius over all four
    # events, not this event's local score.
    bias = axes[0, 1]
    positions = np.arange(2)
    width = 0.34
    for case_index, _result in enumerate(case_results):
        values = global_radius_bias_by_case_and_surface[case_index]
        values = [np.nan if value is None else float(value) for value in values]
        bias.bar(
            positions + (case_index - 0.5) * width,
            values,
            width,
            color=colors[case_index],
            label=EXPECTED_CASE_TOKENS[case_index],
        )
    bias.axhline(0.0, color="white", alpha=0.5, linewidth=1.0)
    bias.set_xticks(positions, ["R=1 mm skin", "R=3 mm skin"])
    bias.set_ylabel("median radial residual (mm)", color="white")
    bias.set_title(
        "CASE-WIDE RADIUS BIAS ACROSS ALL 4 EVENTS — confounder only\nnever used for direction",
        color="white", fontsize=12,
    )
    bias.legend(facecolor="#101a24", labelcolor="white")

    sorted_surfaces = [
        sorted(result["surfaces"], key=lambda row: float(row["radius_mm"]))
        for result in case_results
    ]
    for surface_index in range(2):
        profile = axes[0, surface_index + 2]
        delta = _paired_delta_profile(
            sorted_surfaces[0][surface_index], sorted_surfaces[1][surface_index]
        )
        profile.plot(np.arange(8), delta, marker="o", color="#ffe66d", linewidth=2.5)
        profile.axhline(0.0, color="white", linewidth=1.0, alpha=0.5)
        paired_surface = paired_result["surfaces"][surface_index]
        margin = paired_surface["required_margin_deg"]
        if margin is not None:
            profile.axhline(float(margin), color="#ff5a5f", linestyle="--", alpha=0.8)
            profile.axhline(-float(margin), color="#ff5a5f", linestyle="--", alpha=0.8)
        median = paired_surface["paired_delta_median_deg"]
        if median is not None:
            profile.axhline(float(median), color="#be8cff", linestyle=":", alpha=0.9)
        profile.set_title(
            f"Paired N011−N012 axial error delta — R={sorted_surfaces[0][surface_index]['radius_mm']:g} mm",
            color="white", fontsize=13,
        )
        profile.set_xlabel("axial bin", color="white")
        profile.set_ylabel("absolute normal-turn error delta (deg)", color="white")
        profile.text(
            0.02,
            0.97,
            f"{paired_surface['status']}\n"
            f"median={paired_surface['paired_delta_median_deg']}°\n"
            f"MAD={paired_surface['paired_delta_mad_deg']}°\n"
            f"required margin={paired_surface['required_margin_deg']}°\n"
            f"same-sign={paired_surface['same_sign_axial_bin_count']}/"
            f"{paired_surface['paired_eligible_axial_bin_count']}",
            transform=profile.transAxes,
            va="top",
            color="white",
            fontsize=8,
            bbox={"facecolor": "#101a24", "alpha": 0.8, "edgecolor": "#647587"},
        )

    # Four unwrapped normal-mode heatmaps with one common 0..theta scale.
    heatmaps = []
    for surface_index in range(2):
        for case_index in range(2):
            heatmaps.append(
                (
                    axes[surface_index + 1, case_index],
                    case_index,
                    surface_index,
                )
            )
    for plot, case_index, surface_index in heatmaps:
        array_surface = case_arrays[case_index]["surfaces"][surface_index]
        modes = np.degrees(np.unwrap(array_surface["normal_mode_azimuth_rad"], axis=1))
        masked = np.ma.masked_invalid(modes)
        image = plot.imshow(
            masked.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=math.degrees(theta),
        )
        plot.set_facecolor("#444b55")
        plot.set_title(
            f"{EXPECTED_CASE_TOKENS[case_index]} unwrapped normal modes — "
            f"R={sorted_surfaces[case_index][surface_index]['radius_mm']:g} mm",
            color="white", fontsize=12,
        )
        plot.set_xlabel("axial bin", color="white")
        plot.set_ylabel("angular cell", color="white")
        colorbar = figure.colorbar(image, ax=plot, fraction=0.046, pad=0.04)
        colorbar.set_label("normal azimuth (deg), shared scale", color="white")
        colorbar.ax.tick_params(colors="white")

    # Missing-cell masks encode both cases on a single categorical panel per
    # skin: 0 observed in both, 1 N011 missing, 2 N012 missing, 3 both missing.
    from matplotlib.colors import BoundaryNorm, ListedColormap

    missing_cmap = ListedColormap(["#1fba70", "#48d7ff", "#ff9b42", "#59636e"])
    missing_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], missing_cmap.N)
    for surface_index in range(2):
        plot = axes[2, surface_index + 2]
        left_counts = case_arrays[0]["surfaces"][surface_index]["cell_counts"]
        right_counts = case_arrays[1]["surfaces"][surface_index]["cell_counts"]
        code = (
            (left_counts < MIN_POINTS_PER_CELL).astype(np.int64)
            + 2 * (right_counts < MIN_POINTS_PER_CELL).astype(np.int64)
        )
        image = plot.imshow(
            code.T, origin="lower", aspect="auto", cmap=missing_cmap, norm=missing_norm
        )
        plot.set_title(
            f"Missing-cell mask — R={sorted_surfaces[0][surface_index]['radius_mm']:g} mm\n"
            "green both observed; cyan N011 missing; orange N012 missing; grey both",
            color="white", fontsize=11,
        )
        plot.set_xlabel("axial bin", color="white")
        plot.set_ylabel("angular cell", color="white")
        colorbar = figure.colorbar(image, ax=plot, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
        colorbar.ax.set_yticklabels(["both observed", "N011 missing", "N012 missing", "both missing"])
        colorbar.ax.tick_params(colors="white", labelsize=7)

    diagnostic = axes[1, 2]
    diagnostic.set_axis_off()
    diag_lines = ["SHARED PLANE DELTA — diagnostic / abstention only", ""]
    for case_index, result in enumerate(case_results):
        delta = result["shared_event_plane_translation"]
        diag_lines.extend(
            [
                EXPECTED_CASE_TOKENS[case_index],
                f"  status: {delta['status']}",
                f"  guards: {delta['actionable_guard_count']}/4",
                f"  rank / condition: {delta['rank']} / {delta['nonzero_singular_condition']}",
                f"  delta mm: {delta['shared_delta_mm_diagnostic_only']}",
                f"  norm mm: {delta['shared_delta_norm_mm_diagnostic_only']}",
                "  score correction: false",
                "",
            ]
        )
    diagnostic.text(0.01, 0.99, "\n".join(diag_lines), va="top", color="#e7f0f8", fontsize=10, family="monospace")

    decision = axes[1, 3]
    decision.set_axis_off()
    decision_lines = [
        "PAIRED EVENT DECISION",
        f"status: {paired_result['status']}",
        f"direction: {paired_result['direction']}",
        f"reasons: {paired_result['abstention_reasons']}",
        "",
    ]
    for case_index, result in enumerate(case_results):
        decision_lines.append(f"{EXPECTED_CASE_TOKENS[case_index]}: {result['status']}")
        for surface in sorted_surfaces[case_index]:
            decision_lines.append(
                f"  R={surface['radius_mm']:g}: {surface['status']}; "
                f"cells={surface['observed_cell_fraction']:.3f}; "
                f"eligible={surface['eligible_axial_bin_count']}/8"
            )
    decision.text(0.01, 0.99, "\n".join(decision_lines), va="top", color="#e7f0f8", fontsize=10, family="monospace")

    figure.suptitle(
        f"EXP27 V2 paired fixed-frame rotating-normal audit — {event['event_id']}\n"
        "source-only, frozen CAD pose; shared scales; grey/missing evidence never implies defect",
        color="white", fontsize=20, y=0.99,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)


def _validate_preregistration(path: Path) -> dict[str, Any]:
    if EXP21.sha256_path(path) != EXPECTED_PREREGISTRATION_SHA256:
        raise ValueError("EXP27 V2 preregistration hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("EXP27 V2 preregistration identity mismatch")
    return payload


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = EXP21.sha256_path(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def validate_imported_code_bindings() -> dict[str, str]:
    actual = {name: EXP21.sha256_path(path) for name, path in IMPORTED_CODE_PATHS.items()}
    if actual != EXPECTED_IMPORTED_CODE_SHA256:
        raise ValueError(
            f"imported helper code binding mismatch: {actual} != {EXPECTED_IMPORTED_CODE_SHA256}"
        )
    return actual


def validate_proper_rigid_transform(transform: np.ndarray) -> dict[str, Any]:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("selected transform is not a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-10):
        raise ValueError("selected transform has invalid homogeneous row")
    rotation = transform[:3, :3]
    singular = np.linalg.svd(rotation, compute_uv=False)
    determinant = float(np.linalg.det(rotation))
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if (
        determinant <= 0.0
        or not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-8)
        or not np.allclose(singular, np.ones(3), rtol=0.0, atol=1e-8)
        or orthogonality_error > 1e-8
    ):
        raise ValueError("selected transform is not proper rigid with unit scale")
    return {
        "determinant_recomputed": determinant,
        "singular_values_recomputed": singular.tolist(),
        "orthogonality_max_abs_error_recomputed": orthogonality_error,
        "proper_rigid_unit_scale_recomputed": True,
    }


def validate_decoder_provenance(report: Mapping[str, Any]) -> dict[str, Any]:
    decoded = report.get("decode_report")
    if not isinstance(decoded, Mapping):
        raise ValueError("cadmpeg decoder provenance is missing")
    losses = decoded.get("losses")
    if not isinstance(losses, list):
        raise ValueError("cadmpeg decoder losses are missing")
    codes = tuple(
        sorted(
            str(row.get("code", {}).get("code"))
            for row in losses
            if isinstance(row, Mapping)
        )
    )
    severities = [str(row.get("severity")) for row in losses if isinstance(row, Mapping)]
    if (
        decoded.get("geometry_transferred") is not True
        or decoded.get("container_only") is not False
        or len(losses) != 6
        or codes != EXPECTED_CADMPEG_WARNING_CODES
        or any(value == "error" for value in severities)
        or any(value != "warning" for value in severities)
    ):
        raise ValueError("cadmpeg decoder provenance differs from the frozen warning inventory")
    return {
        "geometry_transferred": True,
        "container_only": False,
        "warning_count": len(losses),
        "warning_codes": list(codes),
        "error_severity_count": 0,
        "unresolved_displaylist_count_from_bound_warning": 24,
    }


def execute_source_only(args: argparse.Namespace, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    imported_code_sha256 = validate_imported_code_bindings()
    frozen = preregistration["frozen_inputs"]
    _require_hash(args.source_sldprt, frozen["source_sldprt_sha256"], "source SLDPRT")
    _require_hash(args.cadmpeg, frozen["cadmpeg_sha256"], "cadmpeg")
    _require_hash(args.exp17_summary, frozen["exp17_summary_sha256"], "EXP17 summary")
    _require_hash(args.exp24_result, frozen["exp24_result_sha256"], "EXP24 result")
    _require_hash(args.exp25_result, frozen["exp25_result_sha256"], "EXP25 result")
    output_root = args.output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root must not contain prior execution artifacts")
    output_root.mkdir(parents=True, exist_ok=True)

    cadir, decoder_report = EXP25.decode_cadir(args.source_sldprt.resolve(), args.cadmpeg.resolve())
    decoder_provenance = validate_decoder_provenance(decoder_report)
    extraction = EXP17.extract_native_roundbend_patch_axes(cadir)
    frozen_exp17 = json.loads(args.exp17_summary.read_text(encoding="utf-8"))
    if extraction != frozen_exp17.get("extraction"):
        raise ValueError("fresh native event extraction differs from frozen EXP17")
    bundles = V1.native_event_bundles(cadir, extraction)
    if len(bundles) != 4:
        raise ValueError("EXP27 V2 requires exactly four native events")
    for bundle in bundles:
        validate_native_event_contract(bundle)

    exp25_result = json.loads(args.exp25_result.read_text(encoding="utf-8"))
    case_points: list[np.ndarray] = []
    case_normals: list[np.ndarray] = []
    case_provenance: list[dict[str, Any]] = []
    case_central_results: list[list[dict[str, Any]]] = []
    case_central_arrays: list[list[dict[str, Any]]] = []
    voxel_mm = float(extraction["sheet_thickness_mm"]) * VOXEL_THICKNESS_FRACTION
    for case_token in EXPECTED_CASE_TOKENS:
        scan_path = args.scan_root / f"{case_token}.ply"
        points, normals, provenance = V1.load_frozen_case(
            case_token,
            scan_path,
            args.exp24_root,
            args.exp25_root,
            exp25_result,
        )
        if provenance["scan_sha256"] != frozen["scan_sha256"][case_token]:
            raise ValueError(f"{case_token} scan hash differs from preregistration")
        if provenance["exp24_semantic_sha256"] != frozen["exp24_case_semantic_sha256"][case_token]:
            raise ValueError(f"{case_token} EXP24 semantic differs from preregistration")
        if provenance["exp25_semantic_sha256"] != frozen["exp25_case_semantic_sha256"][case_token]:
            raise ValueError(f"{case_token} EXP25 semantic differs from preregistration")
        rigid_audit = validate_proper_rigid_transform(
            np.asarray(provenance["selected_absolute_transform"], dtype=np.float64)
        )
        capped_points, capped_normals, selected_raw_ids = density_cap_canonical(
            points, normals, voxel_mm
        )
        central_results = []
        central_arrays = []
        for bundle in bundles:
            result, arrays = event_normal_turn_measurement(
                capped_points, capped_normals, bundle, tube_mm=CENTRAL_TUBE_MM
            )
            central_results.append(result)
            central_arrays.append(arrays)
        case_points.append(capped_points)
        case_normals.append(capped_normals)
        case_provenance.append(
            {
                **provenance,
                "raw_point_count": int(len(points)),
                "valid_canonical_density_point_count": int(len(capped_points)),
                "selected_raw_identity_sha256_audit_only": canonical_int64_array_sha256(
                    selected_raw_ids
                ),
                "canonical_partition_identity_definition": "ORDINAL_IN_CANONICAL_VOXEL_ORDER",
                "selected_transform_rigid_audit": rigid_audit,
            }
        )
        case_central_results.append(central_results)
        case_central_arrays.append(central_arrays)

    central_event_pairs = list(zip(case_central_results[0], case_central_results[1]))
    central_pair = pair_direction(central_event_pairs)
    robustness = robustness_matrix(
        case_points[0], case_normals[0], case_points[1], case_normals[1], bundles
    )
    # The separately materialized central result must equal the corresponding
    # robustness entry; otherwise visual and robustness evidence diverged.
    if central_pair != robustness["central"]["result"]:
        raise AssertionError("central result differs between render and robustness lanes")

    paired_event_results = [
        paired_event_direction(left, right) for left, right in central_event_pairs
    ]
    render_rows = []
    for event_index, bundle in enumerate(bundles):
        render_path = output_root / "paired_native_visuals" / f"{bundle['event_id']}_paired_normal_turn.png"
        render_paired_event(
            render_path,
            bundle,
            case_points,
            [case_central_results[0][event_index], case_central_results[1][event_index]],
            [case_central_arrays[0][event_index], case_central_arrays[1][event_index]],
            paired_event_results[event_index],
        )
        render_rows.append(
            {
                "event_id": bundle["event_id"],
                "path": str(render_path.relative_to(output_root)),
                "sha256": EXP21.sha256_path(render_path),
                "width_px": 3000,
                "height_px": 2400,
                "required_layer_count": 7,
                "paired_shared_scales": True,
                "manual_original_resolution_review": "PENDING",
            }
        )

    for case_index, case_token in enumerate(EXPECTED_CASE_TOKENS):
        case_payload = {
            "schema": "EXP_CAD_0027_V2_PAIRED_ROTATING_NORMAL_CASE_V1",
            "experiment_id": EXPERIMENT_ID,
            "architecture": ARCHITECTURE,
            "case_token": case_token,
            "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
            "source_sldprt_sha256": frozen["source_sldprt_sha256"],
            "cadmpeg_sha256": frozen["cadmpeg_sha256"],
            "exp17_summary_sha256": frozen["exp17_summary_sha256"],
            "exp24_result_sha256": frozen["exp24_result_sha256"],
            "exp25_result_sha256": frozen["exp25_result_sha256"],
            "provenance": case_provenance[case_index],
            "voxel_size_mm": voxel_mm,
            "central_tube_mm": CENTRAL_TUBE_MM,
            "events": [_json_case_event(row) for row in case_central_results[case_index]],
            "measured_both_skin_event_count": int(
                sum(row["status"] == "MEASURED_BOTH_SKINS" for row in case_central_results[case_index])
            ),
            "absolute_radius_and_plane_offsets_used_for_direction": False,
            "missing_support_used_as_defect_evidence": False,
            "pose_refinement_performed": False,
            "local_axis_radius_angle_or_pose_fit_performed": False,
            "evaluator_manifest_accessed": False,
            "evaluator_roles_opened": False,
            "labels_or_expected_counts_used": False,
            "generalization_evidence": False,
            "production_promotion_authorized": False,
        }
        case_path = output_root / "cases" / case_token / "semantic_result.json"
        EXP21.atomic_json(case_path, case_payload)

    success_numeric = bool(
        central_pair["status"] == "DIRECTIONAL"
        and robustness["direction_invariant"] is True
    )
    summary = {
        "schema": "EXP_CAD_0027_V2_PAIRED_ROTATING_NORMAL_SOURCE_ONLY_SUMMARY_V1",
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "implementation_sha256": EXP21.sha256_path(Path(__file__)),
        "imported_code_sha256": imported_code_sha256,
        "test_sha256": EXP21.sha256_path(
            ROOT / "tests" / "test_exp_cad_0027_v2_paired_rotating_normal_conformance.py"
        ),
        "case_order": list(EXPECTED_CASE_TOKENS),
        "case_semantic_sha256": {
            case_token: EXP21.sha256_path(output_root / "cases" / case_token / "semantic_result.json")
            for case_token in EXPECTED_CASE_TOKENS
        },
        "event_count": len(bundles),
        "decoder_provenance": decoder_provenance,
        "central_pair_result": central_pair,
        "robustness": robustness,
        "numeric_success_criterion_pass": success_numeric,
        "renders": render_rows,
        "visual_status": "AWAITING_MANUAL_ORIGINAL_RESOLUTION_REVIEW",
        "final_source_only_status": (
            "NUMERIC_PASS_PENDING_VISUAL_GATE" if success_numeric
            else "TERMINAL_ABSTAIN_NUMERIC_FAILURE_NO_RESCUE"
        ),
        "absolute_radius_and_plane_offsets_used_for_direction": False,
        "missing_support_used_as_defect_evidence": False,
        "evaluator_manifest_accessed": False,
        "evaluator_roles_opened": False,
        "labels_or_expected_counts_used": False,
        "generalization_evidence": False,
        "maximum_gpi": "GPI-C",
        "generalization_progress_indicator": (
            "GPI-C_PENDING_SOURCE_ONLY_NATIVE_VISUAL_GATE_NOT_F1_PROGRESS"
            if success_numeric
            else "GPI-C_NEGATIVE_SOURCE_ONLY_MECHANISM_ABSTENTION_NOT_F1_PROGRESS"
        ),
        "production_promotion_authorized": False,
    }
    summary_path = output_root / "SOURCE_ONLY_SUMMARY.json"
    EXP21.atomic_json(summary_path, summary)
    return {
        "summary_path": str(summary_path),
        "summary_sha256": EXP21.sha256_path(summary_path),
        "central_status": central_pair["status"],
        "central_direction": central_pair["direction"],
        "direction_invariant": robustness["direction_invariant"],
        "render_count": len(render_rows),
        "evaluator_roles_opened": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--execute-source-only", action="store_true")
    parser.add_argument("--source-sldprt", type=Path)
    parser.add_argument("--cadmpeg", type=Path)
    parser.add_argument("--exp17-summary", type=Path)
    parser.add_argument("--exp24-root", type=Path)
    parser.add_argument("--exp24-result", type=Path)
    parser.add_argument("--exp25-root", type=Path)
    parser.add_argument("--exp25-result", type=Path)
    parser.add_argument("--scan-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    preregistration = _validate_preregistration(args.preregistration)
    imported_code_sha256 = validate_imported_code_bindings()
    if args.execute_source_only:
        required = (
            "source_sldprt", "cadmpeg", "exp17_summary", "exp24_root",
            "exp24_result", "exp25_root", "exp25_result", "scan_root", "output_root",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"--execute-source-only requires: {', '.join(missing)}")
        result = execute_source_only(args, preregistration)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "architecture": ARCHITECTURE,
                "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
                "imported_code_sha256": imported_code_sha256,
                "numeric_engine_implemented": True,
                "individual_pose_variant_count": len(individual_pose_variants()),
                "directional_source_execution_authorized": True,
                "implemented_native_visual_layers": list(
                    preregistration["visual_gate"]["required_layers"]
                ),
                "blocked_pending_after_source_execution": [
                    "MANUAL_ORIGINAL_RESOLUTION_REVIEW_BINDING"
                ],
                "evaluator_roles_opened": False,
                "production_promotion_authorized": bool(
                    preregistration["production_promotion_authorized"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
