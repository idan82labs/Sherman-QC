"""Finite *sample-supported* cylindrical instances for the opt-in offline path.

This consumes a fresh scan fit and its private support rows, not a serialized
measurement assertion. Sampling gates are explicit engineering policy, not a
scanner calibration or proof of continuous material / manufacturing identity.
No CAD feature count, nominal radius, or nominal span enters this adapter.
"""

from dataclasses import dataclass
import math

import numpy as np


MAX_POINTS = 200_000
MAX_SLABS = 2048
LINE_CONVENTION = "CYLINDER_CENTER_AXIS_SUPPORT"


@dataclass(frozen=True)
class SupportPolicy:
    max_axial_gap_mm: float
    max_angular_gap_deg: float
    min_arc_sweep_deg: float
    min_local_points: int

    def __post_init__(self):
        for name, lower, upper in (
            ("max_axial_gap_mm", 1e-8, 1e6),
            ("max_angular_gap_deg", 1e-8, 90.),
            ("min_arc_sweep_deg", 1e-8, 160.),
        ):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not lower <= value <= upper):
                raise ValueError(f"invalid {name}")
        if type(self.min_local_points) is not int or not 3 <= self.min_local_points <= 512:
            raise ValueError("min_local_points must be an integer in 3..512")


class _Unavailable(ValueError):
    pass


def _rows(indices, count):
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise _Unavailable("INVALID_SUPPORT_ROWS")
    raw = np.asarray(indices)
    if (raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0
            or raw.size > MAX_POINTS or int(raw.min()) < 0 or int(raw.max()) >= count):
        raise _Unavailable("INVALID_SUPPORT_ROWS")
    result = raw.astype(np.int64, copy=False)
    if len(np.unique(result)) != len(result):
        raise _Unavailable("REPEATED_SUPPORT_ROW_ID")
    return result


def _vector(value):
    raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in "fiu":
        raise _Unavailable("INVALID_MEASURED_AXIS")
    vector = raw.astype(np.float64)
    if not np.isfinite(vector).all() or float(np.max(np.abs(vector))) > 1e6:
        raise _Unavailable("INVALID_MEASURED_AXIS")
    return vector


def _build(points, measurement, arc_row_indices, flange_row_indices, feature_id, policy):
    if not isinstance(feature_id, str) or not feature_id.strip() or len(feature_id) > 512:
        raise _Unavailable("INVALID_FEATURE_ID")
    if not isinstance(measurement, dict):
        raise _Unavailable("INVALID_SCAN_MEASUREMENT")
    if (not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3
            or points.dtype.kind not in "fiu" or not 1 <= len(points) <= MAX_POINTS):
        raise _Unavailable("INVALID_POINT_ARRAY_OR_BUDGET")
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all() or float(np.max(np.abs(points))) > 1e6:
        raise _Unavailable("INVALID_POINT_NUMERICS")
    if not isinstance(flange_row_indices, (list, tuple)) or len(flange_row_indices) != 2:
        raise _Unavailable("REQUIRE_TWO_FLANGE_SUPPORTS")
    ids = [_rows(arc_row_indices, len(points)), *[_rows(rows, len(points))
                                               for rows in flange_row_indices]]
    # A row cannot simultaneously be curved support and either planar support;
    # the two flange roles must also be distinct for this narrow single bend.
    if len(np.unique(np.concatenate(ids))) != sum(map(len, ids)):
        raise _Unavailable("OVERLAPPING_SUPPORT_ROLES")
    subsets = [np.unique(points[rows], axis=0) for rows in ids]
    if any(len(rows) < policy.min_local_points for rows in subsets):
        raise _Unavailable("INSUFFICIENT_UNIQUE_SUPPORT")
    # Copies of the same position with different raw IDs do not supply two roles.
    if len(np.unique(np.vstack(subsets), axis=0)) != sum(map(len, subsets)):
        raise _Unavailable("OVERLAPPING_SUPPORT_POSITIONS")
    center = _vector(measurement.get("measured_axis_point"))
    axis = _vector(measurement.get("measured_axis_direction"))
    if not math.isclose(float(np.linalg.norm(axis)), 1., rel_tol=0., abs_tol=1e-10):
        raise _Unavailable("NON_UNIT_MEASURED_AXIS")
    radius = measurement.get("measured_radius_mm")
    residual = measurement.get("circle_max_residual_mm")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in (radius, residual)):
        raise _Unavailable("INVALID_MEASURED_RADIUS_OR_RESIDUAL")
    if not 1e-8 <= radius <= 1e6 or not 0 <= residual < radius:
        raise _Unavailable("INVALID_MEASURED_RADIUS_OR_RESIDUAL")
    guard = 128 * np.finfo(float).eps * max(1., float(np.max(np.abs(points))),
                                          float(np.max(np.abs(center))), radius)
    if guard >= 1e-4 * min(radius, policy.max_axial_gap_mm):
        raise _Unavailable("NUMERIC_RESOLUTION_TOO_COARSE")
    local = [rows - center for rows in subsets]
    axial = [rows @ axis for rows in local]
    radial = local[0] - axial[0][:, None] * axis
    if float(np.max(np.abs(np.linalg.norm(radial, axis=1) - radius))) > residual + guard:
        raise _Unavailable("ARC_ROWS_INCONSISTENT_WITH_SCAN_FIT")
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    wrapped = np.mod(np.arctan2(radial @ v, radial @ u), 2 * math.pi)
    unique = np.unique(wrapped)
    if len(unique) < policy.min_local_points:
        raise _Unavailable("INSUFFICIENT_DISTINCT_ANGULAR_SUPPORT")
    circular_gaps = np.diff(np.r_[unique, unique[0] + 2 * math.pi])
    start = unique[(int(np.argmax(circular_gaps)) + 1) % len(unique)]
    angles = np.rad2deg(np.mod(wrapped - start, 2 * math.pi))
    # Roundoff around the unwrap origin must not create a fictitious 360 degree
    # sweep. This is a numerical guard, never a sampling tolerance adjustment.
    angular_guard = math.degrees(guard / max(radius, 1e-8))
    angles[angles > 360. - angular_guard] = 0.
    if float(np.ptp(angles)) > 180. + angular_guard:
        raise _Unavailable("UNSUPPORTED_ARC_WINDING")
    stations = np.unique(axial[0])
    if len(stations) < 2 or float(np.ptp(stations)) <= guard:
        raise _Unavailable("NO_FINITE_AXIAL_SUPPORT")
    max_gap = float(np.diff(stations).max())
    if max_gap > policy.max_axial_gap_mm + guard:
        raise _Unavailable("DISCONNECTED_AXIAL_SUPPORT")
    lower, upper = float(stations[0]), float(stations[-1])
    slabs = max(1, math.ceil((upper - lower) / policy.max_axial_gap_mm))
    if slabs > MAX_SLABS:
        raise _Unavailable("AXIAL_SLAB_BUDGET")
    width = (upper - lower) / slabs
    ordering = [np.argsort(coordinates) for coordinates in axial]
    ordered = [coordinates[order] for coordinates, order in zip(axial, ordering)]
    coverage = []
    for slab in range(slabs):
        left, right = lower + slab * width, lower + (slab + 1) * width
        # Closed slab boundaries share a boundary station. This is intentional:
        # sampling evidence supports both adjacent neighborhoods, and a tiny
        # rotation roundoff must not create an empty artificial half-open bin.
        members = [order[np.searchsorted(values, left - guard, side="left"):
                         np.searchsorted(values, right + guard, side="right")]
                   for values, order in zip(ordered, ordering)]
        counts = [len(rows) for rows in members]
        if min(counts) < policy.min_local_points:
            raise _Unavailable("INSUFFICIENT_LOCAL_ARC_OR_FLANGE_SUPPORT")
        angular = np.unique(angles[members[0]])
        if len(angular) < policy.min_local_points:
            raise _Unavailable("INSUFFICIENT_LOCAL_ANGULAR_STATIONS")
        sweep = float(np.ptp(angular))
        gap = float(np.diff(angular).max())
        if sweep + angular_guard < policy.min_arc_sweep_deg:
            raise _Unavailable("INSUFFICIENT_LOCAL_ANGULAR_COVERAGE")
        if gap > policy.max_angular_gap_deg + angular_guard:
            raise _Unavailable("DISCONNECTED_LOCAL_ANGULAR_SUPPORT")
        coverage.append({"slab_index": slab, "arc_points": counts[0],
                         "flange_points": counts[1:], "arc_sweep_deg": sweep,
                         "max_angular_gap_deg": gap})
    endpoints = center + np.array([[lower], [upper]]) * axis
    return {"bend_id": feature_id, "feature_id": feature_id,
            "line_start": endpoints[0].tolist(), "line_end": endpoints[1].tolist(),
            "line_convention": LINE_CONVENTION, "unit": "mm",
            "endpoint_semantics": "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
            "observation_status": "SAMPLE_SUPPORTED_CONDITIONAL",
            "continuous_surface_coverage": "NOT_ESTABLISHED",
            "physical_trim_endpoints": None, "manufacturing_event_identity": "UNKNOWN",
            "axial_support_length_mm": upper - lower,
            "unique_arc_support_count": len(subsets[0]),
            "unique_flange_support_counts": [len(rows) for rows in subsets[1:]],
            "max_axial_sample_gap_mm": max_gap, "local_coverage": coverage}


def build_finite_support_instance(points, measurement, arc_row_indices,
                                  flange_row_indices, *, feature_id, policy):
    """Return one conditional sampled-support segment, or explicit unavailable.

    This does not split disconnected intervals into multiple bends or expand
    endpoints to CAD trims. The caller must reject cross-feature curved-support
    reuse before publishing instances. Shared *planar context* alone is not a
    manufacturing identity or a reason to merge two separate curved roots.
    """
    if not isinstance(policy, SupportPolicy):
        raise ValueError("policy must be SupportPolicy")
    try:
        instance = _build(points, measurement, arc_row_indices, flange_row_indices,
                          feature_id, policy)
        return {"status": "SUPPORTED", "instance": instance, "reasons": []}
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        return {"status": "UNAVAILABLE", "instance": None,
                "reasons": [str(error) or type(error).__name__]}
