"""V63 deletion-only localized turn-corridor ownership certificate.

This module implements the single prospective intervention authorized after
the consumed V5R2 diagnosis.  It audits only final V52-origin recoveries and
never creates, moves, reorders, or re-NMSes a line.  Source points, source
indices, routed profiles, and profile normals are transient; the public audit
contains bounded scalar records only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from backend.f1_mesh_section_turn_locus_reselection_v55 import (
    MeshSectionIndex,
    _StationUnusable,
    _section_profile,
)


PROFILE = "V63_LOCALIZED_TURN_CORRIDOR_OWNERSHIP_V1"
SHORT_NAME = "LTCO_V1"
V52_ROUTE = "cylindrical_consensus_recovery_v52"
PASS_TOKEN = "V63_LTCO_RETAIN_RESEARCH_ONLY"
TERMINAL_RETIRE_TOKEN = "V63_LTCO_TERMINAL_RETIRE"
UNDERACTIVATED_RETIRE_TOKEN = "V63_LTCO_UNDERACTIVATED_RETIRE"

MAXIMUM_AUDITED_CANDIDATES = 32
MAXIMUM_RAW_SLICE_POINTS = 4096
MAXIMUM_PROFILE_POINTS = 65
MAXIMUM_PUBLIC_AUDIT_BYTES_PER_CANDIDATE = 2048
STATION_FRACTIONS = tuple(0.10 + 0.10 * index for index in range(9))

AXIAL_SLAB_HALF_WIDTH_OVER_D = 3.0
PROFILE_SPACING_OVER_D = 2.0
MINIMUM_PROFILE_HALF_EXTENT_OVER_D = 12.0
MAXIMUM_PROFILE_HALF_EXTENT_OVER_D = 64.0
MAXIMUM_PROFILE_HALF_EXTENT_OVER_B = 0.10
WIDE_RADIUS_MULTIPLIER = 2.0
MINIMUM_PROFILE_SAMPLE_COUNT = 13
MINIMUM_TRANSVERSE_NORMAL_LENGTH = 0.866
MAXIMUM_INVALID_TRANSVERSE_NORMAL_FRACTION = 0.10
NORMAL_LOOKUP_RADIUS_OVER_D = 2.0

SMOOTHING_KERNEL = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)
PEAK_SEARCH_RADIUS_EDGES = 3
SHOULDER_RUN_LENGTH = 3
SHOULDER_LEVEL_FRACTION = 0.25
MINIMUM_MONOTONICITY = 0.85
MINIMUM_SCALE_TURN_OVER_CANDIDATE = 0.50
MAXIMUM_STATION_WIDTH_FRACTION = 0.30
MAXIMUM_CENTER_OVER_WIDTH = 0.20
MINIMUM_FINE_COARSE_IOU = 0.60
MINIMUM_WIDTH_SCALE_RATIO = 2.0 / 3.0
MAXIMUM_WIDTH_SCALE_RATIO = 3.0 / 2.0
MINIMUM_PASSING_STATIONS = 7
MINIMUM_MEDIAN_TURN_OVER_CANDIDATE = 0.70
MAXIMUM_MEDIAN_WIDTH_FRACTION = 0.25
MAXIMUM_WIDTH_ROBUST_CV = 0.35
ROBUST_SIGMA_FACTOR = 1.4826


class V63CapacityFail(RuntimeError):
    """The prospective run is invalid because the frozen candidate cap fired."""


class V63InputContractError(RuntimeError):
    """The frozen source-only/provenance contract is inconsistent."""


@dataclass(frozen=True)
class Lobe:
    start_edge: int
    end_edge: int
    start_u: float
    end_u: float
    q_sum: float
    signed_turn: float
    turn: float
    monotonicity: float
    center: float
    width: float
    passed: bool


@dataclass(frozen=True)
class StationArtifact:
    candidate_id: str
    station_index: int
    valid: bool
    passed: bool
    reason: str
    u: np.ndarray | None
    theta: np.ndarray | None
    theta_fine: np.ndarray | None
    theta_coarse: np.ndarray | None
    q_fine: np.ndarray | None
    q_coarse: np.ndarray | None
    fine_interval: tuple[float, float] | None
    coarse_interval: tuple[float, float] | None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def frozen_configuration() -> dict[str, Any]:
    return {
        "profile_id": PROFILE,
        "scope": {
            "deletion_only": True,
            "origin": V52_ROUTE,
            "v16_passthrough": True,
            "second_nms": False,
        },
        "caps": {
            "audited_candidates_per_scan": MAXIMUM_AUDITED_CANDIDATES,
            "raw_slice_points": MAXIMUM_RAW_SLICE_POINTS,
            "profile_points": MAXIMUM_PROFILE_POINTS,
            "public_audit_bytes_per_candidate": MAXIMUM_PUBLIC_AUDIT_BYTES_PER_CANDIDATE,
        },
        "profile_parameters": {
            "station_fractions": list(STATION_FRACTIONS),
            "axial_slab_half_width_over_d": AXIAL_SLAB_HALF_WIDTH_OVER_D,
            "spacing_over_d": PROFILE_SPACING_OVER_D,
            "minimum_half_extent_over_d": MINIMUM_PROFILE_HALF_EXTENT_OVER_D,
            "maximum_half_extent_over_d": MAXIMUM_PROFILE_HALF_EXTENT_OVER_D,
            "maximum_half_extent_over_bbox_span": MAXIMUM_PROFILE_HALF_EXTENT_OVER_B,
            "wide_radius_multiplier": WIDE_RADIUS_MULTIPLIER,
            "minimum_samples": MINIMUM_PROFILE_SAMPLE_COUNT,
            "minimum_transverse_normal_length": MINIMUM_TRANSVERSE_NORMAL_LENGTH,
            "maximum_invalid_transverse_normal_fraction": MAXIMUM_INVALID_TRANSVERSE_NORMAL_FRACTION,
            "normal_lookup_radius_over_d": NORMAL_LOOKUP_RADIUS_OVER_D,
        },
        "certificate": {
            "smoothing_kernel": list(SMOOTHING_KERNEL),
            "peak_search_radius_edges": PEAK_SEARCH_RADIUS_EDGES,
            "shoulder_run_length": SHOULDER_RUN_LENGTH,
            "shoulder_level_fraction": SHOULDER_LEVEL_FRACTION,
            "minimum_monotonicity": MINIMUM_MONOTONICITY,
            "minimum_scale_turn_over_candidate": MINIMUM_SCALE_TURN_OVER_CANDIDATE,
            "maximum_station_width_fraction": MAXIMUM_STATION_WIDTH_FRACTION,
            "maximum_center_over_width": MAXIMUM_CENTER_OVER_WIDTH,
            "minimum_fine_coarse_iou": MINIMUM_FINE_COARSE_IOU,
            "minimum_width_scale_ratio": MINIMUM_WIDTH_SCALE_RATIO,
            "maximum_width_scale_ratio": MAXIMUM_WIDTH_SCALE_RATIO,
            "minimum_passing_stations": MINIMUM_PASSING_STATIONS,
            "minimum_median_turn_over_candidate": MINIMUM_MEDIAN_TURN_OVER_CANDIDATE,
            "maximum_median_width_fraction": MAXIMUM_MEDIAN_WIDTH_FRACTION,
            "maximum_width_robust_cv": MAXIMUM_WIDTH_ROBUST_CV,
            "robust_sigma_factor": ROBUST_SIGMA_FACTOR,
        },
    }


CONFIGURATION_SHA256 = hashlib.sha256(_canonical_json(frozen_configuration())).hexdigest()


def estimate_scan_spacing(points: np.ndarray) -> float:
    """Frozen median sixth-other-neighbor spacing with deterministic ties."""

    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 7:
        raise V63InputContractError("INVALID_SOURCE_POINTS")
    if not np.isfinite(source).all():
        raise V63InputContractError("NONFINITE_SOURCE_POINTS")
    count = len(source)
    sample_count = min(count, 4096)
    sample_indices = np.floor(
        (np.arange(sample_count, dtype=np.float64) + 0.5) * count / sample_count
    ).astype(np.int64)
    tree = cKDTree(source)
    distances, _ = tree.query(source[sample_indices], k=7, workers=1)
    values = []
    for source_index, row_distances in zip(sample_indices, distances):
        # ``cKDTree.query`` does not promise source-index ordering for points
        # tied at the kth distance.  Expand the boundary ball, then apply the
        # frozen (distance, source_index) ordering ourselves.
        boundary = float(np.max(np.atleast_1d(row_distances)))
        if not math.isfinite(boundary):
            raise V63InputContractError("INSUFFICIENT_SPACING_NEIGHBORS")
        neighbor_indices = tree.query_ball_point(
            source[int(source_index)],
            r=float(np.nextafter(boundary, math.inf)),
            workers=1,
        )
        ordered = sorted(
            (
                (
                    float(np.linalg.norm(source[int(index)] - source[int(source_index)])),
                    int(index),
                )
                for index in neighbor_indices
                if int(index) != int(source_index)
            ),
            key=lambda item: (item[0], item[1]),
        )
        if len(ordered) < 6:
            raise V63InputContractError("INSUFFICIENT_SPACING_NEIGHBORS")
        values.append(ordered[5][0])
    spacing = float(np.median(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise V63InputContractError("INVALID_SCAN_SPACING")
    return spacing


def _canonical_line(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = np.asarray(row.get("line_start"), dtype=np.float64)
    end = np.asarray(row.get("line_end"), dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite((start, end)).all():
        raise V63InputContractError("INVALID_V52_LINE")
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 0.0:
        raise V63InputContractError("INVALID_V52_LINE")
    direction = delta / length
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        start, end = end, start
        direction = -direction
    return start, end, direction, length


def _smooth(values: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(values, dtype=np.float64), (2, 2), mode="reflect")
    return np.convolve(padded, np.asarray(SMOOTHING_KERNEL), mode="valid")


def _extract_lobe(
    u: np.ndarray,
    theta: np.ndarray,
    *,
    candidate_turn: float,
    profile_half_extent: float,
    spacing: float,
) -> Lobe | None:
    delta = np.diff(theta)
    q = np.abs(delta)
    edge_u = 0.5 * (u[:-1] + u[1:])
    if not len(q) or not np.isfinite(q).all():
        return None
    k0 = min(range(len(q)), key=lambda index: (abs(float(edge_u[index])), index))
    lower = max(0, k0 - PEAK_SEARCH_RADIUS_EDGES)
    upper = min(len(q), k0 + PEAK_SEARCH_RADIUS_EDGES + 1)
    peak = min(
        range(lower, upper),
        key=lambda index: (-float(q[index]), abs(float(edge_u[index])), index),
    )
    height = float(q[peak])
    if not math.isfinite(height) or height <= 0.0:
        return None
    threshold = SHOULDER_LEVEL_FRACTION * height
    left_run = None
    for end in range(peak - 1, SHOULDER_RUN_LENGTH - 2, -1):
        start = end - SHOULDER_RUN_LENGTH + 1
        if bool(np.all(q[start : end + 1] <= threshold)):
            left_run = (start, end)
            break
    right_run = None
    for start in range(peak + 1, len(q) - SHOULDER_RUN_LENGTH + 1):
        end = start + SHOULDER_RUN_LENGTH - 1
        if bool(np.all(q[start : end + 1] <= threshold)):
            right_run = (start, end)
            break
    if left_run is None or right_run is None:
        return None
    start_edge = left_run[1] + 1
    end_edge = right_run[0] - 1
    if start_edge > end_edge:
        return None
    selected = slice(start_edge, end_edge + 1)
    q_sum = float(np.sum(q[selected]))
    signed_turn = float(np.sum(delta[selected]))
    turn = abs(signed_turn)
    if not math.isfinite(q_sum) or q_sum <= 0.0:
        return None
    center = float(np.sum(edge_u[selected] * q[selected]) / q_sum)
    start_u = float(u[start_edge])
    end_u = float(u[end_edge + 1])
    width = end_u - start_u
    monotonicity = turn / q_sum
    passed = bool(
        monotonicity >= MINIMUM_MONOTONICITY
        and turn >= MINIMUM_SCALE_TURN_OVER_CANDIDATE * candidate_turn
        and width / max(2.0 * profile_half_extent, 1e-12) <= MAXIMUM_STATION_WIDTH_FRACTION
        and abs(center) <= max(2.0 * spacing, MAXIMUM_CENTER_OVER_WIDTH * width)
    )
    return Lobe(
        start_edge=start_edge,
        end_edge=end_edge,
        start_u=start_u,
        end_u=end_u,
        q_sum=q_sum,
        signed_turn=signed_turn,
        turn=turn,
        monotonicity=monotonicity,
        center=center,
        width=width,
        passed=passed,
    )


def _interval_iou(left: Lobe, right: Lobe) -> float:
    overlap = max(0.0, min(left.end_u, right.end_u) - max(left.start_u, right.start_u))
    union = max(left.end_u, right.end_u) - min(left.start_u, right.start_u)
    return overlap / max(union, 1e-12)


def evaluate_station_profile(
    u: np.ndarray,
    normals: np.ndarray,
    direction: np.ndarray,
    *,
    candidate_turn: float,
    profile_half_extent: float,
    spacing: float,
) -> tuple[dict[str, Any], StationArtifact]:
    profile_u = np.asarray(u, dtype=np.float64)
    source_normals = np.asarray(normals, dtype=np.float64)
    axis = np.asarray(direction, dtype=np.float64)
    if (
        profile_u.ndim != 1
        or source_normals.shape != (len(profile_u), 3)
        or axis.shape != (3,)
        or len(profile_u) < MINIMUM_PROFILE_SAMPLE_COUNT
        or not np.isfinite(profile_u).all()
        or not np.isfinite(source_normals).all()
        or not np.isfinite(axis).all()
        or not math.isfinite(float(candidate_turn))
        or not math.isfinite(float(profile_half_extent))
        or not math.isfinite(float(spacing))
    ):
        raise _StationUnusable("NONFINITE_OR_INVALID_PROFILE")
    if not bool(np.all(np.diff(profile_u) > 0.0)):
        raise _StationUnusable("NONMONOTONE_PROFILE_ARCLENGTH")
    transverse = source_normals - (source_normals @ axis)[:, None] * axis
    lengths = np.linalg.norm(transverse, axis=1)
    invalid_fraction = float(np.mean(lengths < MINIMUM_TRANSVERSE_NORMAL_LENGTH))
    if invalid_fraction > MAXIMUM_INVALID_TRANSVERSE_NORMAL_FRACTION:
        raise _StationUnusable("TRANSVERSE_NORMAL_DEGENERACY")
    if np.any(lengths <= 0.0):
        raise _StationUnusable("TRANSVERSE_NORMAL_DEGENERACY")
    projected = transverse / lengths[:, None]
    for index in range(1, len(projected)):
        if float(np.dot(projected[index - 1], projected[index])) < 0.0:
            projected[index] = -projected[index]
    theta = np.zeros(len(projected), dtype=np.float64)
    for index in range(len(projected) - 1):
        delta = math.atan2(
            float(np.dot(axis, np.cross(projected[index], projected[index + 1]))),
            float(np.dot(projected[index], projected[index + 1])),
        )
        theta[index + 1] = theta[index] + delta
    fine = _smooth(theta)
    coarse = _smooth(fine)
    fine_lobe = _extract_lobe(
        profile_u,
        fine,
        candidate_turn=candidate_turn,
        profile_half_extent=profile_half_extent,
        spacing=spacing,
    )
    coarse_lobe = _extract_lobe(
        profile_u,
        coarse,
        candidate_turn=candidate_turn,
        profile_half_extent=profile_half_extent,
        spacing=spacing,
    )
    persistence = bool(
        fine_lobe is not None
        and coarse_lobe is not None
        and fine_lobe.passed
        and coarse_lobe.passed
        and _interval_iou(fine_lobe, coarse_lobe) >= MINIMUM_FINE_COARSE_IOU
        and abs(fine_lobe.center - coarse_lobe.center) <= 2.0 * spacing
        and MINIMUM_WIDTH_SCALE_RATIO
        <= coarse_lobe.width / max(fine_lobe.width, 1e-12)
        <= MAXIMUM_WIDTH_SCALE_RATIO
        and fine_lobe.signed_turn * coarse_lobe.signed_turn > 0.0
    )
    record = {
        "R_j": float(profile_half_extent),
        "w_f": None if fine_lobe is None else float(fine_lobe.width),
        "w_c": None if coarse_lobe is None else float(coarse_lobe.width),
        "mu_f": None if fine_lobe is None else float(fine_lobe.center),
        "mu_c": None if coarse_lobe is None else float(coarse_lobe.center),
        "T_f": None if fine_lobe is None else float(fine_lobe.turn),
        "T_c": None if coarse_lobe is None else float(coarse_lobe.turn),
        "M_f": None if fine_lobe is None else float(fine_lobe.monotonicity),
        "M_c": None if coarse_lobe is None else float(coarse_lobe.monotonicity),
        "passed": persistence,
        "reason": "PASS" if persistence else "FINE_COARSE_CERTIFICATE_FAIL",
    }
    artifact = StationArtifact(
        candidate_id="",
        station_index=-1,
        valid=True,
        passed=persistence,
        reason=record["reason"],
        u=profile_u,
        theta=theta,
        theta_fine=fine,
        theta_coarse=coarse,
        q_fine=np.abs(np.diff(fine)),
        q_coarse=np.abs(np.diff(coarse)),
        fine_interval=None if fine_lobe is None else (fine_lobe.start_u, fine_lobe.end_u),
        coarse_interval=None if coarse_lobe is None else (coarse_lobe.start_u, coarse_lobe.end_u),
    )
    return record, artifact


def _nearest_normals(
    profile: np.ndarray,
    slice_points: np.ndarray,
    slice_normals: np.ndarray,
    slice_indices: np.ndarray,
    *,
    maximum_distance: float,
) -> np.ndarray:
    tree = cKDTree(slice_points)
    neighbor_count = min(8, len(slice_points))
    distances, local_indices = tree.query(profile, k=neighbor_count, workers=1)
    if neighbor_count == 1:
        distances = np.asarray(distances)[:, None]
        local_indices = np.asarray(local_indices)[:, None]
    selected = []
    for row_distances, row_local_indices in zip(distances, local_indices):
        ordered = sorted(
            (
                (float(distance), int(slice_indices[int(local_index)]), int(local_index))
                for distance, local_index in zip(row_distances, row_local_indices)
                if math.isfinite(float(distance))
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not ordered or ordered[0][0] > maximum_distance:
            raise _StationUnusable("PROFILE_NORMAL_SOURCE_GAP")
        selected.append(slice_normals[ordered[0][2]])
    return np.asarray(selected, dtype=np.float64)


def _station_profile(
    mesh: MeshSectionIndex,
    points: np.ndarray,
    normals: np.ndarray,
    support_indices: frozenset[int],
    station: np.ndarray,
    direction: np.ndarray,
    *,
    spacing: float,
    maximum_half_extent: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    delta = points - station
    axial = np.abs(delta @ direction)
    distance = np.linalg.norm(delta, axis=1)
    mask = (axial <= AXIAL_SLAB_HALF_WIDTH_OVER_D * spacing) & (
        distance <= maximum_half_extent + AXIAL_SLAB_HALF_WIDTH_OVER_D * spacing
    )
    raw_indices = np.flatnonzero(mask)
    if len(raw_indices) == 0:
        raise _StationUnusable("EMPTY_AXIAL_SLICE")
    ordered = sorted(
        (int(index) for index in raw_indices),
        key=lambda index: (float(distance[index]), index),
    )
    ordered = ordered[:MAXIMUM_RAW_SLICE_POINTS]
    slice_indices = np.asarray(ordered, dtype=np.int64)
    anchor_options = [index for index in ordered if index in support_indices]
    if not anchor_options:
        raise _StationUnusable("MISSING_STATION_SOURCE_SUPPORT")
    anchor_index = min(anchor_options, key=lambda index: (float(distance[index]), index))
    parameters = {
        "radius": maximum_half_extent + AXIAL_SLAB_HALF_WIDTH_OVER_D * spacing,
        "profile_half_span": maximum_half_extent,
        "section_epsilon": max(mesh.scale.bbox_diagonal, 1.0) * 2.0**-40,
        "attachment_tolerance": maximum_half_extent + AXIAL_SLAB_HALF_WIDTH_OVER_D * spacing,
    }
    u, profile = _section_profile(
        mesh,
        station,
        direction,
        parameters,
        seed_point=points[anchor_index],
        output_spacing=PROFILE_SPACING_OVER_D * spacing,
        maximum_output_points=MAXIMUM_PROFILE_POINTS,
        allow_partial_symmetric_extent=True,
        minimum_section_segments_per_side=1,
    )
    if len(u) < MINIMUM_PROFILE_SAMPLE_COUNT:
        raise _StationUnusable("INSUFFICIENT_PROFILE_SAMPLES")
    half_extent = float(max(abs(float(u[0])), abs(float(u[-1]))))
    if half_extent < MINIMUM_PROFILE_HALF_EXTENT_OVER_D * spacing:
        raise _StationUnusable("INSUFFICIENT_PROFILE_EXTENT")
    profile_normals = _nearest_normals(
        profile,
        points[slice_indices],
        normals[slice_indices],
        slice_indices,
        maximum_distance=NORMAL_LOOKUP_RADIUS_OVER_D * spacing,
    )
    return u, profile_normals, half_extent


def _empty_station(reason: str) -> dict[str, Any]:
    return {
        "R_j": None,
        "w_f": None,
        "w_c": None,
        "mu_f": None,
        "mu_c": None,
        "T_f": None,
        "T_c": None,
        "M_f": None,
        "M_c": None,
        "passed": False,
        "reason": reason,
    }


def _candidate_audit(
    prediction: Mapping[str, Any],
    line_object: Mapping[str, Any],
    mesh: MeshSectionIndex,
    points: np.ndarray,
    normals: np.ndarray,
    support_indices: frozenset[int],
    *,
    spacing: float,
    bbox_span: float,
    wide_radius: float,
) -> tuple[dict[str, Any], list[StationArtifact]]:
    start, end, direction, length = _canonical_line(prediction)
    candidate_id = str(line_object.get("candidate_id") or prediction.get("source_hypothesis_id") or "")
    try:
        candidate_turn = math.radians(float(line_object["two_sided_normal_angle_deg"]))
    except (KeyError, TypeError, ValueError):
        candidate_turn = math.nan
    maximum_half_extent = min(
        WIDE_RADIUS_MULTIPLIER * float(wide_radius),
        MAXIMUM_PROFILE_HALF_EXTENT_OVER_D * spacing,
        MAXIMUM_PROFILE_HALF_EXTENT_OVER_B * bbox_span,
    )
    missing = None
    if not math.isfinite(candidate_turn) or candidate_turn <= 0.0:
        missing = "MISSING_TWO_SIDED_ANGLE"
    elif not math.isfinite(wide_radius) or wide_radius <= 0.0:
        missing = "MISSING_WIDE_SUPPORT_RADIUS"
    elif not support_indices:
        missing = "MISSING_SOURCE_SUPPORT"
    elif maximum_half_extent < MINIMUM_PROFILE_HALF_EXTENT_OVER_D * spacing:
        missing = "INSUFFICIENT_REQUESTED_PROFILE_EXTENT"

    station_records: list[dict[str, Any]] = []
    artifacts: list[StationArtifact] = []
    valid_mask = 0
    pass_mask = 0
    for station_index, fraction in enumerate(STATION_FRACTIONS):
        if missing is not None:
            record = _empty_station(missing)
            artifact = StationArtifact(
                candidate_id, station_index, False, False, missing,
                None, None, None, None, None, None, None, None,
            )
        else:
            station = start + fraction * (end - start)
            try:
                u, station_normals, half_extent = _station_profile(
                    mesh,
                    points,
                    normals,
                    support_indices,
                    station,
                    direction,
                    spacing=spacing,
                    maximum_half_extent=maximum_half_extent,
                )
                record, raw_artifact = evaluate_station_profile(
                    u,
                    station_normals,
                    direction,
                    candidate_turn=candidate_turn,
                    profile_half_extent=half_extent,
                    spacing=spacing,
                )
                artifact = StationArtifact(
                    candidate_id=candidate_id,
                    station_index=station_index,
                    valid=True,
                    passed=bool(record["passed"]),
                    reason=str(record["reason"]),
                    u=raw_artifact.u,
                    theta=raw_artifact.theta,
                    theta_fine=raw_artifact.theta_fine,
                    theta_coarse=raw_artifact.theta_coarse,
                    q_fine=raw_artifact.q_fine,
                    q_coarse=raw_artifact.q_coarse,
                    fine_interval=raw_artifact.fine_interval,
                    coarse_interval=raw_artifact.coarse_interval,
                )
                valid_mask |= 1 << station_index
                if bool(record["passed"]):
                    pass_mask |= 1 << station_index
            except _StationUnusable as error:
                record = _empty_station(error.reason)
                artifact = StationArtifact(
                    candidate_id, station_index, False, False, error.reason,
                    None, None, None, None, None, None, None, None,
                )
        station_records.append(record)
        artifacts.append(artifact)

    passing = [record for record in station_records if bool(record["passed"])]
    passed = False
    decision_reason = "FEWER_THAN_SEVEN_PASSING_STATIONS"
    if len(passing) >= MINIMUM_PASSING_STATIONS:
        turns = np.asarray([float(record["T_c"]) for record in passing])
        centers = np.asarray([abs(float(record["mu_c"])) for record in passing])
        widths = np.asarray([float(record["w_c"]) for record in passing])
        extents = np.asarray([float(record["R_j"]) for record in passing])
        median_width = float(np.median(widths))
        robust_cv = (
            ROBUST_SIGMA_FACTOR * float(np.median(np.abs(widths - median_width)))
            / max(median_width, 1e-12)
        )
        checks = (
            float(np.median(turns)) >= MINIMUM_MEDIAN_TURN_OVER_CANDIDATE * candidate_turn,
            float(np.median(centers)) <= spacing,
            float(np.median(widths / (2.0 * extents))) <= MAXIMUM_MEDIAN_WIDTH_FRACTION,
            robust_cv <= MAXIMUM_WIDTH_ROBUST_CV,
        )
        passed = all(checks)
        decision_reason = "PASS" if passed else "CANDIDATE_AGGREGATE_CERTIFICATE_FAIL"
    scalar = {
        "candidate_id": candidate_id,
        "bend_id": str(prediction.get("bend_id") or ""),
        "decision": "PASS" if passed else "REJECT",
        "reason": decision_reason,
        "valid_mask": valid_mask,
        "pass_mask": pass_mask,
        "passing_station_count": len(passing),
        "R_j": [record["R_j"] for record in station_records],
        "w_f": [record["w_f"] for record in station_records],
        "w_c": [record["w_c"] for record in station_records],
        "mu_f": [record["mu_f"] for record in station_records],
        "mu_c": [record["mu_c"] for record in station_records],
        "T_f": [record["T_f"] for record in station_records],
        "T_c": [record["T_c"] for record in station_records],
        "M_f": [record["M_f"] for record in station_records],
        "M_c": [record["M_c"] for record in station_records],
        "reasons": [record["reason"] for record in station_records],
    }
    # Include the byte-count field itself in the bounded public payload.
    scalar["serialized_bytes"] = 0
    while True:
        scalar_bytes = len(_canonical_json(scalar))
        if scalar["serialized_bytes"] == scalar_bytes:
            break
        scalar["serialized_bytes"] = scalar_bytes
    if scalar_bytes > MAXIMUM_PUBLIC_AUDIT_BYTES_PER_CANDIDATE:
        raise V63InputContractError("CANDIDATE_AUDIT_OVERFLOW")
    return scalar, artifacts


def apply_v63(
    v62_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    mesh: MeshSectionIndex,
    source_points: np.ndarray,
    source_normals: np.ndarray,
    transient_wide_support: Mapping[str, Sequence[int]],
) -> tuple[dict[str, Any], list[StationArtifact]]:
    """Apply LTCO to final V52-origin lines and return transient plot data."""

    if v62_result.get("source_only") is not True or v62_result.get("research_only") is not True:
        raise V63InputContractError("V63_REQUIRES_SOURCE_ONLY_RESEARCH_V62")
    points = np.asarray(source_points, dtype=np.float64)
    normals = np.asarray(source_normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise V63InputContractError("INVALID_BOUNDED_SOURCE_GEOMETRY")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise V63InputContractError("NONFINITE_BOUNDED_SOURCE_GEOMETRY")
    if len(points) > 200_000:
        raise V63InputContractError("SOURCE_POINT_BUDGET_EXCEEDED")
    spacing = estimate_scan_spacing(points)
    bbox_span = float(np.max(np.ptp(points, axis=0)))
    if not math.isfinite(bbox_span) or bbox_span <= 0.0:
        raise V63InputContractError("INVALID_BBOX_SPAN")
    predictions = [deepcopy(row) for row in v62_result.get("predictions") or ()]
    audited = [row for row in predictions if str(row.get("selection_route") or "") == V52_ROUTE]
    if len(audited) > MAXIMUM_AUDITED_CANDIDATES:
        raise V63CapacityFail("CAPACITY_FAIL")
    line_objects = {
        str(row.get("candidate_id") or ""): row for row in pool.get("line_objects") or ()
    }
    wide_candidates = {
        str(row.get("candidate_id") or ""): row
        for row in pool.get("candidates") or ()
        if str(row.get("candidate_source") or "") == "cylindrical_wide"
    }
    try:
        wide_radius = float(
            pool["generator_diagnostics"]["cylindrical_wide"]["parameters"]["corridor_radius_mm"]
        )
    except (KeyError, TypeError, ValueError):
        wide_radius = math.nan

    candidate_audits = []
    all_artifacts: list[StationArtifact] = []
    kept_v52_bend_ids: set[str] = set()
    for prediction in audited:
        hypothesis_id = str(prediction.get("source_hypothesis_id") or "")
        line_object = line_objects.get(hypothesis_id)
        if line_object is None:
            line_object = {"candidate_id": hypothesis_id}
        member_ids = sorted(
            str(value)
            for value in line_object.get("cluster_member_ids") or ()
            if str(value).startswith("cylindrical_wide::")
        )
        support: set[int] = set()
        for member_id in member_ids:
            if member_id not in wide_candidates:
                continue
            raw_id = member_id.split("::", 1)[1]
            support.update(int(value) for value in transient_wide_support.get(raw_id, ()))
        scalar, artifacts = _candidate_audit(
            prediction,
            line_object,
            mesh,
            points,
            normals,
            frozenset(support),
            spacing=spacing,
            bbox_span=bbox_span,
            wide_radius=wide_radius,
        )
        candidate_audits.append(scalar)
        all_artifacts.extend(artifacts)
        if scalar["decision"] == "PASS":
            kept_v52_bend_ids.add(str(prediction.get("bend_id") or ""))

    output_predictions = [
        row
        for row in predictions
        if str(row.get("selection_route") or "") != V52_ROUTE
        or str(row.get("bend_id") or "") in kept_v52_bend_ids
    ]
    non_v52_before = [row for row in predictions if str(row.get("selection_route") or "") != V52_ROUTE]
    non_v52_after = [row for row in output_predictions if str(row.get("selection_route") or "") != V52_ROUTE]
    if _canonical_json(non_v52_before) != _canonical_json(non_v52_after):
        raise V63InputContractError("NON_V52_BYTE_PRESERVATION_FAILED")
    audit = {
        "configuration_sha256": CONFIGURATION_SHA256,
        "scan_spacing_mm": spacing,
        "scan_bbox_span_mm": bbox_span,
        "input_prediction_count": len(predictions),
        "v52_origin_count": len(audited),
        "v52_retained_count": len(kept_v52_bend_ids),
        "v52_deleted_count": len(audited) - len(kept_v52_bend_ids),
        "non_v52_retained_count": len(non_v52_after),
        "non_v52_byte_preservation": True,
        "second_nms_used": False,
        "candidate_order_preserved": True,
        "candidate_audits": candidate_audits,
    }
    result = deepcopy(dict(v62_result))
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v63_localized_turn_corridor_ownership_result",
            "profile": PROFILE,
            "v62_retained": True,
            "predictions": output_predictions,
            "candidate_count": len(output_predictions),
            "audit": audit,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
            "source_point_serialization": False,
        }
    )
    result["v63_result_sha256"] = hashlib.sha256(
        _canonical_json(
            {
                "profile": PROFILE,
                "configuration_sha256": CONFIGURATION_SHA256,
                "predictions": output_predictions,
                "audit": audit,
            }
        )
    ).hexdigest()
    return result, all_artifacts
