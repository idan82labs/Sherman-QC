"""All-row radius constraints for one STIPULATED axis-line family, research only.

This is not dense support discovery, an optimizer, a fitted-axis certificate or
a replacement for the 256-raw-point local engine. Every supplied row belongs to
the caller's single-cylinder support hypothesis; none is sampled or trimmed.
"""
from __future__ import annotations

import math

import numpy as np


MAX_TOTAL_ROWS = 2_000_000
MAX_CHUNK_ROWS = 65_536
MAX_CHUNKS = 4096
_EPS = float(np.finfo(float).eps)
_MIN_NORMAL = float(np.finfo(float).tiny)


def _real(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating)))


def _number(value, name):
    if not _real(value):
        raise ValueError(f"{name} must be an explicit finite nonnegative number")
    try:
        converted = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable") from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return converted


def _integer(value, name, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an explicit integer")
    value = int(value)
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside the declared supported budget")
    return value


def _vector(value, name, length):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.ndim != 1 or len(value) != length):
        raise ValueError(f"{name} must contain exactly {length} coordinates")
    if any(not _real(x) for x in value):
        raise ValueError(f"{name} must contain real coordinates")
    if any(isinstance(x, (int, np.integer)) and abs(int(x)) > 2**53 for x in value):
        raise ValueError(f"{name} integer precision is unsupported")
    try:
        result = np.array(value, dtype=float, copy=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _fail(result, reason):
    result.update(status="ABSTAINED", radius_interval=None, constraint_extrema=None,
                  search_complete_within_declared_policy=False, reasons=[reason])
    return result


def analyze_dense_axis_tube_radius(
    chunks, *, unit, axis, axis_point, axis_angle_bound, axis_point_error_bound,
    position_error_bound, radius_bounds, max_total_rows, max_chunk_rows, max_chunks,
):
    """Intersect every row's necessary constraints under ONE shared-line family.

    Hypothetical true line: direction within alpha of nominal a (unoriented),
    and containing a point within b of nominal o. Measurement errors <= epsilon.
    For v=p-o and d=||v cross a||, every true radius obeys
    |R-d| <= epsilon + b + sin(alpha)*||v||, before numerical padding.

    The per-row outer constraints relax shared parameters; a nonempty intersection
    is NOT a common-line witness. Empty excludes only the declared family/support
    and radius prior. EOF means complete supplied stream, not physical coverage.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be m or mm")
    a = _vector(axis, "axis", 3)
    norm = math.hypot(*a)
    if not math.isclose(norm, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("axis must already be a unit vector")
    a /= norm  # Numerical cleanup only; not axis estimation.
    o = _vector(axis_point, "axis_point", 3)
    prior = _vector(radius_bounds, "radius_bounds", 2)
    if not 0 < prior[0] <= prior[1]:
        raise ValueError("radius_bounds must satisfy 0 < lower <= upper")
    alpha = _number(axis_angle_bound, "axis_angle_bound")
    if alpha >= math.pi/2:
        raise ValueError("axis_angle_bound must be less than pi/2 radians")
    b = _number(axis_point_error_bound, "axis_point_error_bound")
    epsilon = _number(position_error_bound, "position_error_bound")
    row_cap = _integer(max_total_rows, "max_total_rows", MAX_TOTAL_ROWS)
    chunk_cap = _integer(max_chunk_rows, "max_chunk_rows", MAX_CHUNK_ROWS)
    count_cap = _integer(max_chunks, "max_chunks", MAX_CHUNKS)
    if not math.isfinite(epsilon+b):
        raise ValueError("combined position and axis-point bounds must be representable")
    sine = math.sin(alpha)
    origin_magnitude = float(np.max(np.abs(o)))
    result = {
        "schema_version": 1, "kind": "DENSE_AXIS_TUBE_RADIUS_CONSTRAINTS_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": unit, "radius_interval": None, "constraint_extrema": None,
        "radius_interval_scope": "NECESSARY_OUTER_CONSTRAINT_FOR_DECLARED_AXIS_FAMILY_SUPPORT_AND_PRIOR",
        "input_row_count": 0, "stream_fully_consumed": False,
        "search_complete_within_declared_policy": False,
        "physical_geometry_is_verified": False, "physical_bend_count": None,
        "shared_model_feasibility_is_established": False,
        "position_error_bound": epsilon,
        "hypothesis": {
            "axis": a.tolist(), "axis_point": o.tolist(),
            "axis_angle_bound_radians": alpha, "axis_point_error_bound": b,
            "radius_prior": prior.tolist(),
            "model": "ONE_INFINITE_CIRCULAR_CYLINDER_WITHIN_DECLARED_AXIS_POINT_BALL_AND_DIRECTION_CONE",
            "axis_and_point_are_verified": False, "support_is_verified": False,
            "axis_point_bound_is_inferred_from_measurement_epsilon": False,
        },
        "policy": {
            "max_total_rows": row_cap, "max_chunk_rows": chunk_cap, "max_chunks": count_cap,
            "all_supplied_rows_are_in_the_single_support_hypothesis": True,
            "raw_duplicates_consume_budget": True, "sampling_or_residual_trimming": False,
            "retained_raw_rows_scope": "NO_RAW_ARCHIVE_BEYOND_TRANSIENT_CHUNK_WORKSPACE",
            "input_row_count_scope": "FULLY_VALIDATED_AND_CONSTRAINED_CHUNKS_EXCLUDES_REJECTED_CHUNK",
            "extremum_tie_rule": "EARLIEST_STREAM_ROW_WITH_EXACT_EQUAL_VALUE",
            "producer_allocation_or_blocking_time_is_bounded": False,
            "eof_required_for_any_geometric_conclusion": True,
        },
        "diagnostics": {
            "received_chunks": 0, "processed_chunks": 0, "peak_chunk_rows": 0,
            "constrained_raw_rows": 0, "retained_raw_rows": 0,
            "max_numerical_allowance": 0., "max_propagated_error_bound": 0.,
            "max_nominal_axis_distance": 0., "max_origin_distance": 0.,
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION", "DECLARED_SUPPORT_AND_AXIS_FAMILY_NOT_DISCOVERED",
            "AXIS_POINT_BALL_REQUIRES_SEPARATE_JUSTIFICATION_NOT_FIT_RESIDUAL_OR_EPSILON",
            "NONEMPTY_OUTER_INTERSECTION_NOT_SHARED_MODEL_FEASIBILITY",
            "EMPTY_INTERSECTION_EXCLUDES_ONLY_DECLARED_FAMILY_SUPPORT_AND_RADIUS_PRIOR",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
            "NO_TANGENCY_SURFACE_IDENTITY_FINITE_COVERAGE_OR_FORMING_STATE_CLAIM",
            "NO_TOPOLOGY_GRAPH_OR_PHYSICAL_EVENT_COUNT", "NO_COMPRESSION_OR_PROPOSAL_DISCOVERY",
            "STREAM_ROWS_AUDITED_NOT_DURABLY_RETAINED_MEASUREMENT_ARCHIVE",
            "EOF_NOT_PHYSICAL_CAPTURE_COVERAGE",
        ],
        "reasons": [],
    }
    try:
        iterator = iter(chunks)
    except TypeError:
        raise ValueError("chunks must be an explicit cooperative iterable of NumPy arrays") from None
    except Exception:
        return _fail(result, "INPUT_STREAM_FAILED")
    diagnostics = result["diagnostics"]
    max_lower, min_upper, lower_row, upper_row = -math.inf, math.inf, None, None
    while True:
        try:
            raw = next(iterator)
        except StopIteration:
            result["stream_fully_consumed"] = True
            break
        except Exception:
            return _fail(result, "INPUT_STREAM_FAILED")
        diagnostics["received_chunks"] += 1
        if diagnostics["received_chunks"] > count_cap:
            return _fail(result, "CHUNK_COUNT_BUDGET_EXCEEDED")
        if (not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.shape[1] != 3
                or raw.dtype.kind not in "iuf" or raw.dtype.itemsize > 8):
            return _fail(result, "INVALID_CHUNK_ARRAY")
        n = len(raw)
        if n > chunk_cap:
            return _fail(result, "CHUNK_ROW_BUDGET_EXCEEDED")
        if result["input_row_count"]+n > row_cap:
            return _fail(result, "TOTAL_ROW_BUDGET_EXCEEDED")
        if raw.dtype.kind in "iu" and (np.any(raw > 2**53) or np.any(raw < -2**53)):
            return _fail(result, "INTEGER_COORDINATE_PRECISION_UNSUPPORTED")
        p = np.asarray(raw, dtype=float)
        if not np.isfinite(p).all():
            return _fail(result, "NONFINITE_CHUNK_COORDINATE")
        if n:
            # Row-wise arithmetic makes the constraint independent of chunk
            # partitioning; no dense matrix, covariance sum or fit is needed.
            with np.errstate(over="ignore", invalid="ignore"):
                v = p-o
                footprint = np.hypot(np.hypot(v[:, 0], v[:, 1]), v[:, 2])
                cross = np.cross(v, a)
                d = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])
                propagated = epsilon+b+sine*footprint
                arithmetic_scale = np.max(np.abs(p), axis=1)+origin_magnitude+footprint+d+propagated
                guard = 512*_EPS*arithmetic_scale
                effective = propagated+guard
                low, high = np.maximum(0., d-effective), d+effective
            if any(not np.isfinite(x).all() for x in (v, footprint, d, propagated, guard, low, high)):
                return _fail(result, "NUMERICAL_RANGE_UNSUPPORTED")
            # A relative guard can vanish or enter subnormal arithmetic while
            # distances still round by absolute subnormal units. Do not claim an
            # outer interval in that unsupported range; exact all-zero scale is
            # harmless and does not need an arithmetic allowance.
            if np.any((arithmetic_scale > 0) & (guard < _MIN_NORMAL)):
                return _fail(result, "NUMERICAL_RANGE_UNSUPPORTED")
            offset = result["input_row_count"]
            li, ui = int(np.argmax(low)), int(np.argmin(high))
            if float(low[li]) > max_lower:
                max_lower, lower_row = float(low[li]), offset+li
            if float(high[ui]) < min_upper:
                min_upper, upper_row = float(high[ui]), offset+ui
            diagnostics["max_numerical_allowance"] = max(diagnostics["max_numerical_allowance"], float(np.max(guard)))
            diagnostics["max_propagated_error_bound"] = max(diagnostics["max_propagated_error_bound"], float(np.max(propagated)))
            diagnostics["max_nominal_axis_distance"] = max(diagnostics["max_nominal_axis_distance"], float(np.max(d)))
            diagnostics["max_origin_distance"] = max(diagnostics["max_origin_distance"], float(np.max(footprint)))
        result["input_row_count"] += n
        diagnostics["constrained_raw_rows"] += n
        diagnostics["processed_chunks"] += 1
        diagnostics["peak_chunk_rows"] = max(diagnostics["peak_chunk_rows"], n)
    result["search_complete_within_declared_policy"] = True
    if not result["input_row_count"]:
        result["reasons"] = ["NO_SUPPLIED_ROWS"]
        return result
    lower, upper = max(float(prior[0]), max_lower), min(float(prior[1]), min_upper)
    result["constraint_extrema"] = {
        "max_sample_lower": max_lower, "min_sample_upper": min_upper,
        "max_sample_lower_row": lower_row, "min_sample_upper_row": upper_row,
        "prior_intersection_lower": lower, "prior_intersection_upper": upper,
        "scope": "ALL_ROW_NECESSARY_CONSTRAINTS_NOT_FEASIBLE_SET_OR_COMPLETE_PROVENANCE_ARCHIVE",
    }
    if lower > upper:
        result["status"] = "DECLARED_AXIS_TUBE_MODEL_EXCLUDED"
        result["reasons"] = ["EMPTY_NECESSARY_RADIUS_INTERSECTION"]
    else:
        result["status"] = "NECESSARY_RADIUS_CONSTRAINTS"
        result["radius_interval"] = [lower, upper]
    return result
