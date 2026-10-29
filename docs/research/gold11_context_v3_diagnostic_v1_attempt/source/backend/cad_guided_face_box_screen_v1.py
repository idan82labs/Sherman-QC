"""Rejection-only represented-point distance bound for an enclosing face box.

If the caller binds every triangle of a finite face to B=[lower,upper], then
L(p)=max(0,max_j(lower_j-p_j,p_j-upper_j)) <= distance(p,face).
Outward-down subtraction preserves this lower bound under the same conditional
IEEE binary64 model as distance V3. L>gate proves a row outside; the complement
is only POSSIBLE support. No box row is admitted as measured/finite support.

This helper has no native geometry, file access, pose estimation or Gold. The
caller must preserve byte/row/face binding and charge each screen row before use.
The helper cannot authenticate that its supplied box encloses the intended face.
"""

import numpy as np

from .cad_guided_triangle_distance_bounds_v3 import (
    ARITHMETIC_MODEL, MAX_COORDINATE, _arithmetic_sentinels,
)


MAX_ROWS = 65_536
MAX_SUPPORT_ROWS = 1_000_000


class _Refusal(ValueError):
    pass


def _scalar(value, low, high, name):
    if type(value) not in (float, int) or type(value) is int and abs(value) > 2**53:
        raise _Refusal("INVALID_"+name)
    represented = float(value)
    if not np.isfinite(represented) or not low <= represented <= high:
        raise _Refusal("INVALID_"+name)
    return represented


def _owned(value, shape, name, *, reject_subnormals):
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64) or value.shape != shape:
        raise _Refusal("INVALID_"+name)
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    absolute = np.abs(result)
    if (not np.isfinite(result).all() or np.any(absolute > MAX_COORDINATE)
            or reject_subnormals and np.any((absolute > 0) & (absolute < np.finfo(float).tiny))):
        raise _Refusal("UNSUPPORTED_NUMERIC_"+name)
    return result


def screen_face_box_rows(points, box_lower, box_upper, *, distance_gate_mm, max_rows=MAX_ROWS):
    """Screen one bounded chunk, returning no finite-surface query results.

Ordinary float64 points must already be the same represented transformed rows
used by the downstream query. Nonzero subnormal original points refuse, matching
V3 input admission; gradual-underflow interval intermediates remain supported.
Box endpoints may be subnormal to accommodate outward-expanded zero boundaries.
Chunk counts must be aggregated over ALL support rows before deciding exclusion.
Unavailable clears every output; neither NaN nor malformed rows count outside.
"""
    result = dict(status="UNAVAILABLE", reason=None, input_rows=None,
        attempted_rows=0, completed_rows=0, definitely_outside_count=None,
        distance_lower_mm=None, definitely_outside_mask=None,
        arithmetic_model=ARITHMETIC_MODEL,
        geometry_scope="CONDITIONAL_CALLER_BOUND_ENCLOSING_BOX_REJECTION_ONLY",
        finite_surface_support_admitted=False)
    try:
        if type(max_rows) is not int or not 0 <= max_rows <= MAX_ROWS:
            raise _Refusal("INVALID_ROW_BUDGET")
        if type(points) is not np.ndarray or points.ndim != 2 or points.shape[1:] != (3,) or len(points) > max_rows:
            raise _Refusal("INVALID_OR_OVER_CAP_POINTS")
        p = _owned(points, (len(points), 3), "POINTS", reject_subnormals=True)
        lower = _owned(box_lower, (3,), "BOX_LOWER", reject_subnormals=False)
        upper = _owned(box_upper, (3,), "BOX_UPPER", reject_subnormals=False)
        if np.any(lower > upper):
            raise _Refusal("INVERTED_BOX")
        gate = _scalar(distance_gate_mm, 0., MAX_COORDINATE, "DISTANCE_GATE")
        result["input_rows"] = len(p)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if not _arithmetic_sentinels():
                raise _Refusal("UNSUPPORTED_ARITHMETIC_MODEL_SENTINEL")
            result["attempted_rows"] = len(p)
            offset = np.maximum(np.nextafter(lower-p, -np.inf),
                                np.nextafter(p-upper, -np.inf))
            bound = np.maximum(0., np.max(offset, axis=1))
            if not np.isfinite(bound).all():
                raise _Refusal("UNSUPPORTED_BOX_BOUND_ARITHMETIC")
            outside = bound > gate
        result.update(status="COMPLETE", reason="CONDITIONAL_BOX_DISTANCE_LOWER_BOUND",
            completed_rows=len(p), definitely_outside_count=int(np.count_nonzero(outside)),
            distance_lower_mm=bound, definitely_outside_mask=outside)
    except (_Refusal, ValueError, TypeError, OverflowError, FloatingPointError) as error:
        result["reason"] = str(error) if isinstance(error, _Refusal) else "UNSUPPORTED_BOX_INPUT_OR_ARITHMETIC"
    return result


def decide_face_box_exclusion(*, support_count, definitely_outside_count, minimum_coverage):
    """Use complete support counts only; a chunk is not a whole-support proof.

The caller authenticates/aggregates counts. Integers <=1e6 are exactly represented.
Upward division bounds the possible inlier fraction; equality never excludes.
No nonexcluded result admits a point, a pose, or a finite face correspondence.
"""
    if (type(support_count) is not int or not 0 < support_count <= MAX_SUPPORT_ROWS
            or type(definitely_outside_count) is not int
            or not 0 <= definitely_outside_count <= support_count):
        raise ValueError("INVALID_COMPLETE_SUPPORT_COUNTS")
    minimum = _scalar(minimum_coverage, 0., 1., "MINIMUM_COVERAGE")
    if not _arithmetic_sentinels():
        raise ValueError("UNSUPPORTED_ARITHMETIC_MODEL_SENTINEL")
    possible = support_count-definitely_outside_count
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        upper = min(1., float(np.nextafter(np.float64(possible)/np.float64(support_count), np.inf)))
    return dict(decision="BOUND_EXCLUDED" if upper < minimum else "NEEDS_SURFACE_QUERY",
        support_count=support_count, definitely_outside_count=definitely_outside_count,
        possible_inlier_count_upper=possible, possible_inlier_fraction_upper=upper,
        minimum_coverage=minimum, finite_surface_support_admitted=False,
        count_scope="CALLER_COMPLETE_SUPPORT_ROW_DENOMINATOR_REQUIRED")
