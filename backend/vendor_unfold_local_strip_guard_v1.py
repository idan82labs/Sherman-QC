"""Pure finite bend-strip exclusion with an independently witnessed seam side.

This separately versioned guard generalizes the global halfspace condition.
It proves a supplied native tessellated flange does not occupy the finite
developed cylinder strip, and that its full native seam is a mesh boundary with
local material on the correct side. Distant flange material may return beyond
the strip. Every source triangle is retained and clipped exactly for the test.

The certificate alone never admits a flat preview: native seam identity, both
complete skin graphs, source material coverage, proper cylinder chart closure,
and final overlap of ALL unchanged developed faces remain external obligations.
No files, native CAD, fitted geometry or manufacturing authority are accessed.
"""
from __future__ import annotations

from fractions import Fraction
import math
from numbers import Integral

from backend import vendor_unfold_finite_width as exact


GEOM_TOL = exact.GEOM_TOL
MODEL = "NATIVE_TESSELLATED_FLANGE_FINITE_DEVELOPED_STRIP_WITH_LOCAL_SEAM_SIDE"
MAX_LOCAL_VERTICES = exact.MAX_VERTICES
MAX_LOCAL_TRIANGLES = exact.MAX_TRIANGLES
CAPS = {"triangles": 1_000_000, "projected_vertices": 3_000_000,
        "clip_edge_tests": 20_000_000, "seam_edge_tests": 3_000_000}
_TOL = Fraction.from_float(GEOM_TOL)


class LocalStripUnavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise LocalStripUnavailable(reason)


class LocalStripBudget:
    """One fixed-cap counter across every parent/child/bend in a development."""
    def __init__(self):
        self._work = {key: 0 for key in CAPS}

    def charge(self, key, amount=1):
        _require(self._work[key]+amount <= CAPS[key], "LOCAL_STRIP_WORK_BUDGET_EXCEEDED")
        self._work[key] += amount

    def snapshot(self):
        return self._work.copy()


def _clip(polygon, bound, keep_above, budget):
    if not polygon:
        return []
    previous = polygon[-1]
    previous_inside = previous[0] >= bound if keep_above else previous[0] <= bound
    output = []
    for current in polygon:
        budget.charge("clip_edge_tests")
        current_inside = current[0] >= bound if keep_above else current[0] <= bound
        if current_inside != previous_inside:
            denominator = exact._bounded(current[0]-previous[0])
            _require(denominator != 0, "ILL_CONDITIONED_LOCAL_STRIP_CLIP")
            alpha = exact._bounded((bound-previous[0])/denominator)
            _require(0 <= alpha <= 1, "ILL_CONDITIONED_LOCAL_STRIP_CLIP")
            second = exact._bounded(previous[1]+alpha*(current[1]-previous[1]))
            output.append((bound, second))
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    compact = []
    for point in output:
        if not compact or point != compact[-1]:
            compact.append(point)
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    # Clipping a triangle by the four sides of a rectangle gives <=7 vertices.
    _require(len(compact) <= 7, "LOCAL_STRIP_POLYGON_BUDGET_EXCEEDED")
    return compact


def _positive_area(polygon):
    if len(polygon) < 3:
        return False
    twice = Fraction(0)
    for i, point in enumerate(polygon):
        next_point = polygon[(i+1) % len(polygon)]
        twice = exact._bounded(twice+exact._bounded(
            point[0]*next_point[1]-next_point[0]*point[1]))
    return twice != 0


def _lower_float(value):
    number = float(value)
    _require(math.isfinite(number), "ILL_CONDITIONED_LOCAL_STRIP_OUTPUT")
    if Fraction.from_float(number) > value:
        number = math.nextafter(number, -math.inf)
    return number


def certify_local_finite_strip(vertices, triangles, *, cylinder_origin,
                               cylinder_axis, signed_tangent, seam_endpoints,
                               axial_min, axial_max, developed_strip_width,
                               side="parent", budget=None):
    """Certify local seam side and absence of material in a finite bend strip.

    ``signed_tangent`` points along the developed bend from its parent seam.
    For a child use the transported signed tangent in the child native frame;
    child polarity is inverted internally. ``developed_strip_width`` must be
    the same selected native cylinder radius times absolute native sweep used
    by the eventual development (verified by caller, never an adjustable gap).

    Exclusion covers axial [min-tol,max+tol], signed intrusion coordinate
    [tol-uncertainty,width+tol+uncertainty]. The remaining seam tolerance is the
    incumbent numerical geometry allowance. Local boundary triangles must
    contain genuine material at least tol+uncertainty away on the correct side.
    """
    try:
        _require(side in ("parent", "child"), "INVALID_LOCAL_STRIP_SIDE")
        _require(budget is None or type(budget) is LocalStripBudget,
                 "INVALID_LOCAL_STRIP_BUDGET")
        budget = budget if budget is not None else LocalStripBudget()
        before = budget.snapshot()
        count, triangle_count = exact._sequence(vertices), exact._sequence(triangles)
        _require(0 < count <= MAX_LOCAL_VERTICES and 0 < triangle_count <= MAX_LOCAL_TRIANGLES,
                 "LOCAL_STRIP_INPUT_BUDGET_EXCEEDED")
        origin, axis, tangent = map(exact._vector, (cylinder_origin, cylinder_axis, signed_tangent))
        axis2, tangent2 = exact._dot(axis, axis), exact._dot(tangent, tangent)
        orthogonality = exact._dot(axis, tangent)
        _require(abs(axis2-1) <= exact._FRAME_BOUND and abs(tangent2-1) <= exact._FRAME_BOUND
                 and abs(orthogonality) <= exact._FRAME_BOUND, "INVALID_LOCAL_STRIP_FRAME")
        low, high, strip_width = (Fraction.from_float(exact._finite(value))
                                  for value in (axial_min, axial_max, developed_strip_width))
        _require(high-low > 4*_TOL and strip_width > 4*_TOL,
                 "INVALID_LOCAL_STRIP_INTERVAL")
        exact._sequence(seam_endpoints, 2)
        seam = tuple(exact._vector(point) for point in seam_endpoints)
        seam_axial = sorted(exact._dot(exact._sub(point, origin), axis) for point in seam)
        _require(abs(seam_axial[0]-low) <= _TOL*(1-abs(axis2-1))
                 and abs(seam_axial[1]-high) <= _TOL*(1-abs(axis2-1)),
                 "LOCAL_SEAM_DOES_NOT_COVER_AXIAL_SPAN")
        delta = exact._sub(seam[1], seam[0])
        perpendicular = exact._cross(delta, axis)
        _require(exact._dot(perpendicular, perpendicular) <= (_TOL/8)**2*axis2,
                 "NONAXIAL_LOCAL_STRIP_SEAM")
        anchor = tuple(exact._bounded((a+b)/2) for a, b in zip(*seam))
        anchor_error = max(abs(exact._dot(exact._sub(point, anchor), tangent)) for point in seam)
        _require(anchor_error <= _TOL/8, "UNCERTAIN_LOCAL_STRIP_SEAM")
        points, chart = [], []
        scale, polarity = max(Fraction(1), abs(low), abs(high), strip_width), 1 if side == "parent" else -1
        for row in vertices:
            point = exact._vector(row)
            budget.charge("projected_vertices")
            points.append(point)
            relative_origin, relative_anchor = exact._sub(point, origin), exact._sub(point, anchor)
            scale = max(scale, sum(map(abs, relative_origin)), sum(map(abs, relative_anchor)))
            chart.append((exact._dot(relative_origin, axis),
                          polarity*exact._dot(relative_anchor, tangent)))
        frame_error = exact._bounded(8*scale*(abs(axis2-1)+abs(tangent2-1)+abs(orthogonality)))
        _require(frame_error <= _TOL/64, "ILL_CONDITIONED_LOCAL_STRIP_FRAME")
        uncertainty = exact._bounded(anchor_error+frame_error)
        clearance = exact._bounded(_TOL*(1-abs(tangent2-1))-uncertainty)
        strip_high = exact._bounded(strip_width+_TOL+uncertainty)
        edge_owners, seam_candidates, intersections, remote_triangles = {}, [], 0, 0
        for row in triangles:
            exact._sequence(row, 3)
            _require(all(isinstance(i, Integral) and not isinstance(i, bool) and 0 <= i < count for i in row)
                     and len(set(map(int, row))) == 3, "INVALID_LOCAL_STRIP_TRIANGLE")
            ids = list(map(int, row))
            budget.charge("triangles")
            polygon = [chart[i] for i in ids]
            polygon = _clip(polygon, exact._bounded(low-_TOL), True, budget)
            polygon = _clip(polygon, exact._bounded(high+_TOL), False, budget)
            if polygon:
                intersections += 1
                # Swap axial/intrusion coordinates for the two strip sides.
                strip = [(s, a) for a, s in polygon]
                strip = _clip(strip, clearance, True, budget)
                strip = _clip(strip, strip_high, False, budget)
                _require(not _positive_area(strip), "FLANGE_INTERSECTS_FINITE_DEVELOPED_STRIP")
                if max(s for _, s in polygon) > strip_high:
                    remote_triangles += 1
            for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                budget.charge("seam_edge_tests")
                key = tuple(sorted((ids[i], ids[j])))
                edge_owners[key] = edge_owners.get(key, 0)+1
                _require(edge_owners[key] <= 2, "NON_MANIFOLD_LOCAL_MESH_EDGE")
                a, b = chart[ids[i]], chart[ids[j]]
                if (abs(a[1])+uncertainty <= _TOL/4 and abs(b[1])+uncertainty <= _TOL/4
                        and low-_TOL <= a[0] <= high+_TOL
                        and low-_TOL <= b[0] <= high+_TOL):
                    left, right = max(low, min(a[0], b[0])), min(high, max(a[0], b[0]))
                    if right-left > _TOL:
                        third_side = chart[ids[k]][1]
                        _require(third_side <= -_TOL-uncertainty,
                                 "LOCAL_SEAM_MATERIAL_SIDE_NOT_PROVEN")
                        seam_candidates.append((key, left, right))
        _require(intersections > 0, "NO_FLANGE_MATERIAL_IN_LOCAL_AXIAL_SPAN")
        intervals = sorted((left, right) for key, left, right in seam_candidates if edge_owners[key] == 1)
        _require(intervals, "LOCAL_NATIVE_SEAM_NOT_MESH_BOUNDARY")
        covered = low
        for left, right in intervals:
            _require(left <= covered+_TOL, "LOCAL_NATIVE_SEAM_MESH_COVERAGE_GAP")
            covered = max(covered, right)
        _require(covered >= high-_TOL, "LOCAL_NATIVE_SEAM_MESH_COVERAGE_GAP")
        work = budget.snapshot()
        return {"status": "CERTIFIED", "mechanism_version": 1, "model": MODEL,
            "side": side, "geometry_mutated": False, "manufacturing_ready": False,
            "exact_binary_clipping": True, "tolerance_mm": GEOM_TOL,
            "axial_bounds": [float(low), float(high)],
            "developed_strip_width_mm": float(strip_width),
            # Diagnostic floats describe a SUBSET of the exact checked strip:
            # round the lower side up and the upper side down, never expanding
            # an exclusion claim into an unchecked one-ULP sliver.
            "certified_strip_intrusion_bounds": [exact._upper_float(clearance), _lower_float(strip_high)],
            "charged_uncertainty_mm": exact._upper_float(uncertainty),
            "intersecting_axial_triangles": intersections,
            "retained_remote_side_triangles": remote_triangles,
            "native_seam_boundary_interval_count": len(intervals),
            "native_seam_local_material_side_checked": True,
            "whole_supplied_mesh_finite_strip_checked": True,
            "work": {key: work[key]-before[key] for key in CAPS}, "shared_work": work,
            "required_external_gates": ["native_seam_identity_and_same_radius_sweep_strip_width",
                "native_ownership_and_complete_skin_material_coverage",
                "proper_chart_transform_and_native_seam_closure",
                "whole_unchanged_developed_mesh_overlap"]}
    except exact.FiniteWidthUnavailable as exc:
        raise LocalStripUnavailable(str(exc)) from exc
