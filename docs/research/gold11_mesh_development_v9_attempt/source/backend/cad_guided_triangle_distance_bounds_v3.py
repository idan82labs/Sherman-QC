"""Local feasible-point bounds on distance to represented triangle convex hulls.

This additive NumPy-only helper does not find the closest point. A caller supplies
one exactly feasible binary64 barycentric pair per point/triangle pair. An outward
primal/dual check encloses the true finite-hull distance even for a poor candidate.
For r=p-q, h=max_i r.(v_i-q), convex support and Cauchy give
max(0, (||r||^2-h)/||r||) <= distance(p, hull(vertices)) <= ||r||.

The arithmetic statement is conditional on IEEE-754 binary64 round-to-nearest,
gradual underflow and correctly rounded basic operations/sqrt. Each arithmetic
operation in the enclosure is rounded outward with nextafter; no reassociation
or fast-math is permitted. Small runtime sentinels are not binary attestation.
This is NOT scanner/CAD uncertainty, a native-surface certificate, normal/region
qualification, pose validation, or production authority. Frozen V2 is unchanged.
"""

import numpy as np


MAX_PAIRS = 65_536
MAX_COORDINATE = 1e100
ARITHMETIC_MODEL = "CONDITIONAL_BINARY64_RN_GRADUAL_UNDERFLOW_CORRECT_SQRT"
RESULT_ARRAY_KEYS = (
    "distance_lower", "distance_upper", "closest_points", "closest_point_error",
    "candidate_distance", "candidate_distance_lower", "candidate_distance_upper",
)


class _Refusal(ValueError):
    pass


def _down(value):
    return np.nextafter(value, -np.inf)


def _up(value):
    return np.nextafter(value, np.inf)


def _point(value):
    return value, value


def _add(a, b):
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _sub(a, b):
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _mul(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return _down(np.minimum.reduce(products)), _up(np.maximum.reduce(products))


def _divide_positive(a, denominator):
    # Denominator is an exact positive represented number, NOT an interval.
    return _down(a[0] / denominator), _up(a[1] / denominator)


def _square(a):
    lo_square, hi_square = a[0] * a[0], a[1] * a[1]
    lower = np.where((a[0] <= 0) & (a[1] >= 0), 0.,
                     np.maximum(0., _down(np.minimum(lo_square, hi_square))))
    return lower, _up(np.maximum(lo_square, hi_square))


def _sum3(a):
    return _add(_add((a[0][..., 0], a[1][..., 0]),
                     (a[0][..., 1], a[1][..., 1])),
                (a[0][..., 2], a[1][..., 2]))


def _dot(a, b):
    return _sum3(_mul(a, b))


def _norm_squared(a):
    lo, hi = _sum3(_square(a))
    return np.maximum(0., lo), np.maximum(0., hi)


def _sqrt_nonnegative(a):
    return np.maximum(0., _down(np.sqrt(np.maximum(0., a[0])))), _up(np.sqrt(a[1]))


def _midpoint(a):
    # Used for a displayed approximation only; no certificate depends on it.
    return a[0] / 2. + a[1] / 2.


def _array(value, shape, cap, name):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(np.float64)
            or value.ndim != len(shape) + 1 or value.shape[1:] != shape
            or len(value) > cap):
        raise _Refusal("INVALID_OR_OVER_CAP_" + name)
    owned = np.array(value, dtype=np.float64, copy=True, order="C")
    absolute = np.abs(owned)
    if (not np.isfinite(owned).all() or np.any(absolute > MAX_COORDINATE)
            or np.any((absolute > 0) & (absolute < np.finfo(np.float64).tiny))):
        raise _Refusal("UNSUPPORTED_NUMERIC_" + name)
    return owned


def _feasible(coefficients):
    if np.any(coefficients < 0) or np.any(coefficients > 1):
        return False
    a, b = coefficients[:, 0], coefficients[:, 1]
    # Knuth TwoSum: represented s + tail equals the exact a+b under the model.
    # In particular (1 + tiny_positive) rounded to 1 is NOT feasible.
    s = a + b
    bv = s - a
    tail = (a - (s - bv)) + (b - bv)
    return not np.any((s > 1) | ((s == 1) & (tail > 0)))


def _arithmetic_sentinels():
    tiny = np.float64(np.finfo(np.float64).tiny)
    half = np.multiply(tiny, np.float64(.5))
    smallest = np.nextafter(np.float64(0), np.float64(1))
    return bool(half > 0 and half < tiny and smallest > 0
                and np.multiply(half, np.float64(2)) == tiny
                and np.sqrt(np.float64(4)) == np.float64(2)
                and np.add(np.float64(1), np.float64(2**-53)) == np.float64(1))


def enclose_triangle_distances(points, triangles, barycentric, *, max_pairs=MAX_PAIRS):
    """Bound N independent finite-hull distances from N feasible candidates.

    Inputs are ordinary owned-on-entry native float64 ndarrays: (N,3), (N,3,3),
    (N,2). Original nonzero subnormal values, nonfinite/out-of-range coordinates,
    infeasible coefficients or a bad/overflowed budget refuse the WHOLE result.
    Degenerate hulls are mathematically allowed HERE; a surface caller retains
    its own edge/shape admission. No primitive is dropped or silently repaired.

    `closest_points` names the materialized supplied feasible candidate, NOT a
    proved closest point. `closest_point_error` bounds its Euclidean restoration
    error relative to the exact barycentric candidate. Candidate-distance bounds
    refer to that feasible point; distance_lower/upper bound the hull optimum.
    The lower bound can be loose (even zero) for a poor candidate. A downstream
    caller must enforce its own interval-width and threshold-ambiguity handling.

    Local scale is a power of two chosen separately for each point/triangle pair.
    Subtraction, division by scale, certificate and world restoration all retain
    outward intervals. Underflow is enclosed under the declared gradual-underflow
    model; unsupported overflow/nonfinite arithmetic clears all output rows.
    """
    diagnostics = dict(arithmetic_model=ARITHMETIC_MODEL, input_pairs=None,
        attempted_pairs=0, completed_pairs=0, maximum_distance_width=None,
        maximum_restoration_error=None, full_result_cleared=False,
        meaning="FINITE_REPRESENTED_CONVEX_HULL_DISTANCE_ONLY_NOT_CLOSEST_POINT_PROOF")
    cleared = {key: None for key in RESULT_ARRAY_KEYS}
    try:
        if type(max_pairs) is not int or not 0 <= max_pairs <= MAX_PAIRS:
            raise _Refusal("INVALID_PAIR_BUDGET")
        p = _array(points, (3,), max_pairs, "POINTS")
        vertices = _array(triangles, (3, 3), max_pairs, "TRIANGLES")
        coefficients = _array(barycentric, (2,), max_pairs, "BARYCENTRIC")
        if len(p) != len(vertices) or len(p) != len(coefficients):
            raise _Refusal("PAIR_COUNT_MISMATCH")
        diagnostics["input_pairs"] = len(p)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if not _arithmetic_sentinels():
                raise _Refusal("UNSUPPORTED_ARITHMETIC_MODEL_SENTINEL")
            if not _feasible(coefficients):
                raise _Refusal("INFEASIBLE_EXACT_BARYCENTRIC_CANDIDATE")
            diagnostics["attempted_pairs"] = len(p)
            anchor = vertices[:, 0]
            raw_edges = _sub(_point(vertices), _point(anchor[:, None, :]))
            raw_point = _sub(_point(p), _point(anchor))
            # Nominal differences choose a chart only; enclosures never discard
            # subtraction error, even if cancellation produces a subnormal edge.
            edge_nominal = vertices - anchor[:, None, :]
            point_nominal = p - anchor
            magnitude = np.maximum(np.max(np.abs(edge_nominal), axis=(1, 2)),
                                   np.max(np.abs(point_nominal), axis=1))
            _, exponent = np.frexp(magnitude)
            scale = np.ldexp(np.ones(len(p)), np.where(magnitude == 0, 0, exponent))
            if np.any(scale <= 0) or not np.isfinite(scale).all():
                raise _Refusal("UNSUPPORTED_LOCAL_SCALE")
            local_vertices = _divide_positive(raw_edges, scale[:, None, None])
            local_p = _divide_positive(raw_point, scale[:, None])
            first = (local_vertices[0][:, 1], local_vertices[1][:, 1])
            second = (local_vertices[0][:, 2], local_vertices[1][:, 2])
            q = _add(_mul(_point(coefficients[:, 0, None]), first),
                     _mul(_point(coefficients[:, 1, None]), second))
            residual = _sub(local_p, q)
            squared = _norm_squared(residual)
            norm = _sqrt_nonnegative(squared)
            vertex_offsets = _sub(local_vertices, (q[0][:, None, :], q[1][:, None, :]))
            support_dot = _dot((residual[0][:, None, :], residual[1][:, None, :]), vertex_offsets)
            support_upper = np.maximum(0., np.max(support_dot[1], axis=1))
            numerator_lower = _down(squared[0] - support_upper)
            # A positive numerator already proves a nonzero residual; when it
            # is unresolved/nonpositive, zero is the sound lower bound.
            lower = np.zeros(len(p))
            resolved = (numerator_lower > 0) & (norm[1] > 0)
            lower[resolved] = np.maximum(0., _down(numerator_lower[resolved] / norm[1][resolved]))
            lower_world = np.maximum(0., _down(lower * scale))
            upper_world = _up(norm[1] * scale)
            candidate_lower = np.maximum(0., _down(norm[0] * scale))
            candidate_upper = upper_world.copy()
            candidate_distance = _midpoint(norm) * scale
            world_q = _add(_point(anchor), _mul(q, _point(scale[:, None])))
            materialized = _midpoint(world_q)
            error_vector = _sub(_point(materialized), world_q)
            restoration_error = _sqrt_nonnegative(_norm_squared(error_vector))[1]
            arrays = dict(distance_lower=lower_world, distance_upper=upper_world,
                closest_points=materialized, closest_point_error=restoration_error,
                candidate_distance=candidate_distance, candidate_distance_lower=candidate_lower,
                candidate_distance_upper=candidate_upper)
            if (any(not np.isfinite(value).all() for value in arrays.values())
                    or np.any(lower_world > upper_world)
                    or np.any(candidate_lower > candidate_upper)):
                raise _Refusal("UNSUPPORTED_ENCLOSURE_ARITHMETIC")
            diagnostics.update(completed_pairs=len(p),
                maximum_distance_width=float(np.max(upper_world-lower_world)) if len(p) else 0.,
                maximum_restoration_error=float(np.max(restoration_error)) if len(p) else 0.)
        return dict(status="COMPLETE", reason="CONDITIONAL_LOCAL_DISTANCE_ENCLOSURE", **arrays,
                    diagnostics=diagnostics)
    except (_Refusal, ValueError, TypeError, OverflowError, FloatingPointError) as error:
        diagnostics["full_result_cleared"] = True
        return dict(status="UNAVAILABLE", reason=str(error) if isinstance(error, _Refusal)
                    else "UNSUPPORTED_ENCLOSURE_INPUT_OR_ARITHMETIC", **cleared, diagnostics=diagnostics)
