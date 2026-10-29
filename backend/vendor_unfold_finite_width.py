"""Pure finite-width side certificate for a *tessellated* flange chart.

This module does not import a CAD kernel, read files, flatten geometry, or alter
meshes.  The caller supplies native tessellation and an already signed tangent
(towards the developed bend).  Every triangle is clipped to the expanded finite
axial slab; an affine side function attains its extrema at the clipped polygon
vertices.  Fractions make that clipping exact for the supplied binary floats,
including triangles with no original vertex inside the slab.

This is not a continuous-CAD or manufacturing certificate. Native face/edge
ownership, complete skin coverage, exact seam incidence/closure, proper chart
transforms, and final overlap checking of the ENTIRE unchanged developed mesh
remain mandatory external gates. In particular, material outside this slab is
not removed and may still overlap another developed face.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Real


GEOM_TOL = 1e-5  # Same side tolerance as the frozen visual-skin kernel.
MAX_VERTICES = 600_000
MAX_TRIANGLES = 200_000
MAX_TOTAL_TRIANGLES = 1_000_000
MAX_TOTAL_PROJECTED_VERTICES = 3_000_000
MAX_TOTAL_CLIP_EDGE_TESTS = 12_000_000
MAX_POLYGON_VERTICES = 6
MAX_FRACTION_BITS = 4096
_FRAME_EPS = 64 * math.ulp(1.0)
_TOL = Fraction.from_float(GEOM_TOL)
_FRAME_BOUND = Fraction.from_float(_FRAME_EPS)


class FiniteWidthUnavailable(ValueError):
    """Fail-closed malformed, unsupported, over-budget, or wrong-side chart."""


def _require(condition, reason):
    if not condition:
        raise FiniteWidthUnavailable(reason)


class FiniteWidthBudget:
    """Fixed-cap counters shared across all certificates in one development.

    Callers cannot select larger caps. Rejected work remains charged. This is a
    deterministic operation cap, not a substitute for the enclosing worker's
    wall-clock/process-tree memory limits.
    """

    __slots__ = ("_triangles", "_vertices", "_edges")

    def __init__(self):
        self._triangles = self._vertices = self._edges = 0

    def _charge(self, kind, amount=1):
        attr, limit = {
            "triangles": ("_triangles", MAX_TOTAL_TRIANGLES),
            "projected_vertices": ("_vertices", MAX_TOTAL_PROJECTED_VERTICES),
            "clip_edge_tests": ("_edges", MAX_TOTAL_CLIP_EDGE_TESTS),
        }[kind]
        current = getattr(self, attr)
        _require(current + amount <= limit, "FINITE_WIDTH_WORK_BUDGET_EXCEEDED")
        setattr(self, attr, current + amount)

    def snapshot(self):
        return {"triangles": self._triangles, "projected_vertices": self._vertices,
                "clip_edge_tests": self._edges}


def _finite(value):
    _require(isinstance(value, Real) and not isinstance(value, bool),
             "INVALID_FINITE_WIDTH_NUMBER")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise FiniteWidthUnavailable("INVALID_FINITE_WIDTH_NUMBER") from exc
    _require(math.isfinite(number), "NONFINITE_FINITE_WIDTH_NUMBER")
    _require(math.ulp(number) <= GEOM_TOL / 64, "ILL_CONDITIONED_COORDINATE_RESOLUTION")
    return number


def _sequence(value, length=None):
    _require(not isinstance(value, (str, bytes, dict)), "INVALID_FINITE_WIDTH_TABLE")
    try:
        count = len(value)
    except (TypeError, OverflowError) as exc:
        raise FiniteWidthUnavailable("INVALID_FINITE_WIDTH_TABLE") from exc
    _require(length is None or count == length, "INVALID_FINITE_WIDTH_TABLE")
    return count


def _bounded(value):
    _require(value.numerator.bit_length() <= MAX_FRACTION_BITS and
             value.denominator.bit_length() <= MAX_FRACTION_BITS,
             "ILL_CONDITIONED_RATIONAL_CLIP")
    return value


def _vector(value):
    _sequence(value, 3)
    return tuple(Fraction.from_float(_finite(x)) for x in value)


def _sub(a, b):
    return tuple(_bounded(x-y) for x, y in zip(a, b))


def _dot(a, b):
    return _bounded(sum((_bounded(x*y) for x, y in zip(a, b)), Fraction(0)))


def _cross(a, b):
    return tuple(_bounded(a[i]*b[j]-a[j]*b[i]) for i, j in ((1, 2), (2, 0), (0, 1)))


def _upper_float(value):
    result = float(value)
    _require(math.isfinite(result), "ILL_CONDITIONED_CERTIFICATE_OUTPUT")
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _clip(polygon, bound, keep_above, budget):
    """Sutherland-Hodgman on exact (axial, signed-tangent) coordinates."""
    if not polygon:
        return []
    output = []
    previous = polygon[-1]
    previous_inside = previous[0] >= bound if keep_above else previous[0] <= bound
    for current in polygon:
        budget._charge("clip_edge_tests")
        current_inside = current[0] >= bound if keep_above else current[0] <= bound
        if current_inside != previous_inside:
            denominator = _bounded(current[0] - previous[0])
            _require(denominator != 0, "ILL_CONDITIONED_RATIONAL_CLIP")
            alpha = _bounded((bound-previous[0]) / denominator)
            _require(0 <= alpha <= 1, "ILL_CONDITIONED_RATIONAL_CLIP")
            side = _bounded(previous[1] + alpha*(current[1]-previous[1]))
            output.append((bound, side))
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    # Equal boundary points are not additional polygon vertices.
    compact = []
    for point in output:
        if not compact or point != compact[-1]:
            compact.append(point)
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    _require(len(compact) <= MAX_POLYGON_VERTICES, "FINITE_WIDTH_POLYGON_BUDGET_EXCEEDED")
    return compact


def certify_finite_width_flange(vertices, triangles, *, cylinder_origin,
                                cylinder_axis, tangent, seam_endpoints,
                                axial_min, axial_max, side="parent", budget=None):
    """Certify only the finite axial slab's tessellated tangent halfspace.

    ``vertices`` and ``triangles`` are Nx3/Mx3 tables, not flattened arrays.
    ``axial_min/max`` are cylinder-axis coordinates relative to the origin.
    ``tangent`` includes the bend's signed sweep direction. Parent material
    must be on its nonpositive side; child material on its nonnegative side.
    Seam endpoints may be reversed but must cover the declared axial span.
    Numeric checks do not establish native topology identity or seam ownership.

    Refusal raises FiniteWidthUnavailable. Success returns diagnostics only;
    no output vertices/triangles, AVAILABLE claim, or manufacturing approval.
    """
    _require(isinstance(side, str) and side in ("parent", "child"), "INVALID_FLANGE_SIDE")
    _require(budget is None or type(budget) is FiniteWidthBudget, "INVALID_FINITE_WIDTH_BUDGET")
    budget = budget if budget is not None else FiniteWidthBudget()
    before = budget.snapshot()
    count = _sequence(vertices)
    triangle_count = _sequence(triangles)
    _require(0 < count <= MAX_VERTICES and 0 < triangle_count <= MAX_TRIANGLES,
             "FINITE_WIDTH_INPUT_BUDGET_EXCEEDED")
    origin, axis, direction = map(_vector, (cylinder_origin, cylinder_axis, tangent))
    axis2, tangent2 = _dot(axis, axis), _dot(direction, direction)
    _require(abs(axis2-1) <= _FRAME_BOUND and abs(tangent2-1) <= _FRAME_BOUND and
             abs(_dot(axis, direction)) <= _FRAME_BOUND, "INVALID_FINITE_WIDTH_FRAME")
    axis_norm_lower = 1-abs(axis2-1)
    tangent_norm_lower = 1-abs(tangent2-1)
    tangent_norm_upper = 1+abs(tangent2-1)
    low = Fraction.from_float(_finite(axial_min))
    high = Fraction.from_float(_finite(axial_max))
    _require(high-low > 4*_TOL, "INVALID_OR_ILL_CONDITIONED_AXIAL_SPAN")
    _sequence(seam_endpoints, 2)
    seam = tuple(_vector(point) for point in seam_endpoints)
    axial = sorted(_dot(_sub(point, origin), axis) for point in seam)
    _require(abs(axial[0]-low) <= _TOL*axis_norm_lower and
             abs(axial[1]-high) <= _TOL*axis_norm_lower, "SEAM_DOES_NOT_COVER_AXIAL_SPAN")
    seam_delta = _sub(seam[1], seam[0])
    _require(_dot(_cross(seam_delta, axis), _cross(seam_delta, axis)) <=
             (_TOL/8)**2*axis2, "NONAXIAL_FINITE_WIDTH_SEAM")
    anchor = tuple(_bounded((a+b)/2) for a, b in zip(*seam))
    anchor_error = max(abs(_dot(_sub(point, anchor), direction)) for point in seam)
    _require(anchor_error <= _TOL/8, "UNCERTAIN_FINITE_WIDTH_TANGENT_BOUNDARY")
    points = []
    chart = []
    scale = max(Fraction(1), abs(low), abs(high))
    polarity = 1 if side == "parent" else -1
    for row in vertices:
        point = _vector(row)
        budget._charge("projected_vertices")
        points.append(point)
        relative_origin, relative_anchor = _sub(point, origin), _sub(point, anchor)
        scale = max(scale, sum(map(abs, relative_origin)), sum(map(abs, relative_anchor)))
        chart.append((_dot(relative_origin, axis), polarity*_dot(relative_anchor, direction)))
    # Frame residual times chart scale must not exhaust even a small portion of
    # the tolerance; exact clipping cannot cure an unresolved native frame.
    frame_error = _bounded(8*scale*(abs(axis2-1)+abs(tangent2-1)+abs(_dot(axis, direction))))
    _require(frame_error <= _TOL/64, "ILL_CONDITIONED_FINITE_WIDTH_FRAME")
    remaining = _bounded(_TOL*tangent_norm_lower-anchor_error-frame_error)
    slab_low, slab_high = _bounded(low-_TOL), _bounded(high+_TOL)
    intersecting = 0
    maximum = None
    for row in triangles:
        _sequence(row, 3)
        _require(all(isinstance(index, Integral) and not isinstance(index, bool) for index in row),
                 "INVALID_FINITE_WIDTH_TRIANGLE_INDEX")
        indices = tuple(int(index) for index in row)
        _require(len(set(indices)) == 3 and all(0 <= index < count for index in indices),
                 "INVALID_FINITE_WIDTH_TRIANGLE_INDEX")
        budget._charge("triangles")
        a, b, c = (points[index] for index in indices)
        _require(any(_cross(_sub(b, a), _sub(c, a))), "DEGENERATE_FINITE_WIDTH_TRIANGLE")
        polygon = _clip([chart[index] for index in indices], slab_low, True, budget)
        polygon = _clip(polygon, slab_high, False, budget)
        if polygon:
            intersecting += 1
            local = max(point[1] for point in polygon)
            maximum = local if maximum is None else max(maximum, local)
            _require(local <= remaining, "FLANGE_EXTENDS_INTO_FINITE_DEVELOPED_BEND")
    _require(intersecting > 0, "NO_TRIANGLE_IN_BEND_AXIAL_SLAB")
    total = budget.snapshot()
    return {"status": "CERTIFIED", "model": "NATIVE_TESSELLATION_FINITE_AXIAL_SLAB",
            "side": side, "geometry_mutated": False, "manufacturing_ready": False,
            "exact_binary_clipping": True, "tolerance_mm": GEOM_TOL,
            "axial_bounds": [float(low), float(high)],
            "expanded_axial_bounds": [float(slab_low), float(slab_high)],
            "intersecting_triangles": intersecting,
            "max_oriented_side_distance_mm": _upper_float(
                maximum/(tangent_norm_lower if maximum >= 0 else tangent_norm_upper)),
            "charged_uncertainty_mm": _upper_float((anchor_error+frame_error)/tangent_norm_lower),
            "work": {key: total[key]-before[key] for key in total},
            "shared_work": total,
            "required_external_gates": ["native_ownership_and_complete_skin_coverage",
                                        "native_seam_identity_and_chart_closure",
                                        "whole_unchanged_developed_mesh_overlap"]}
