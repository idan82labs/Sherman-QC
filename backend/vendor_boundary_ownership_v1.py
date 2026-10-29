"""Pure coefficient/incidence proof for polygonal constant-thickness cut walls.

No native imports, file access or runtime integration. A native adapter must
retain exact, byte-bound face/edge identities, oriented uses and full curves.
This sufficient proof accepts tensor-parameterized normal walls even when
generator speed varies with the transverse parameter. It deliberately does
not accept curved outlines, partial trims, steps, shoulders or forming skins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class BoundaryOwnershipUnavailable(ValueError):
    """A typed refusal; never permission to omit a face."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise BoundaryOwnershipUnavailable(reason)


@dataclass(frozen=True)
class Budget:
    max_faces: int = 512
    max_edges: int = 2048
    max_poles: int = 8192
    max_degree: int = 12
    max_pairwise_checks: int = 32768
    tolerance: float = 1e-7


@dataclass(frozen=True)
class Curve:
    degree: int
    knots: tuple[float, ...]  # expanded clamped knot vector
    poles: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class EdgeUse:
    edge_id: int
    direction: int  # direction relative to native edge parameterization


@dataclass(frozen=True)
class Face:
    face_id: int
    uses: tuple[EdgeUse, ...]
    # Ordered native wires, outer wire first. Each tuple preserves edge use.
    wires: tuple[tuple[EdgeUse, ...], ...] = ()


@dataclass(frozen=True)
class SkinPair:
    pair_id: int
    lower_face: int
    upper_face: int
    origin: np.ndarray
    normal: np.ndarray
    thickness: float


@dataclass(frozen=True)
class WallPatch:
    face_id: int
    pair_id: int
    u_degree: int
    v_degree: int
    u_knots: tuple[float, ...]
    v_knots: tuple[float, ...]
    poles: np.ndarray
    weights: np.ndarray
    # v=min, u=max, v=max, u=min; native edge identity + increasing u/v.
    rails: tuple[EdgeUse, EdgeUse, EdgeUse, EdgeUse]


def _basis(degree, knots, count, budget):
    _require(type(degree) is int and 1 <= degree <= budget.max_degree,
             "UNSUPPORTED_SPLINE_DEGREE")
    values = np.asarray(knots, dtype=float)
    _require(values.ndim == 1 and len(values) == count + degree + 1
             and np.isfinite(values).all() and np.all(np.diff(values) >= 0)
             and values[0] < values[-1], "INVALID_SPLINE_BASIS")
    _require(np.all(values[:degree + 1] == values[0])
             and np.all(values[-degree - 1:] == values[-1])
             and np.all(values[degree + 1:-degree - 1] > values[0])
             and np.all(values[degree + 1:-degree - 1] < values[-1]),
             "FULL_CLAMPED_CHART_REQUIRED")
    _require(all(np.count_nonzero(values == x) <= degree
                 for x in np.unique(values[degree + 1:-degree - 1])),
             "DISCONTINUOUS_SPLINE_BASIS")
    return values


def _curve(curve, budget):
    p = np.asarray(curve.poles, dtype=float)
    w = np.asarray(curve.weights, dtype=float)
    _require(p.ndim < 1 or len(p) <= budget.max_poles, "BOUNDARY_POLE_BUDGET_EXCEEDED")
    _require(p.ndim == 2 and p.shape[1:] == (3,) and len(p) >= 2
             and w.shape == (len(p),) and np.isfinite(p).all()
             and np.isfinite(w).all() and np.all(w > 0)
             and float(w.max()) / float(w.min()) <= 1e12,
             "INVALID_NATIVE_CURVE_COEFFICIENTS")
    _basis(curve.degree, curve.knots, len(p), budget)
    return p, w / w[0]


def _same_curve(native, generated, direction, budget):
    p, w = _curve(native, budget)
    q, z = _curve(generated, budget)
    _require(type(direction) is int and direction in (-1, 1), "INVALID_EDGE_DIRECTION")
    knots = np.asarray(native.knots)
    if direction == -1:
        p, w = p[::-1], w[::-1]
        w = w / w[0]
        knots = knots[0] + knots[-1] - knots[::-1]
    _require(native.degree == generated.degree and p.shape == q.shape
             and np.array_equal(knots, np.asarray(generated.knots))
             and np.array_equal(w, z), "NATIVE_RAIL_BASIS_MISMATCH")
    error = float(np.linalg.norm(p - q, axis=1).max())
    _require(error <= budget.tolerance, "NATIVE_RAIL_GEOMETRY_MISMATCH")
    return error


def _line_support(p, normal, budget):
    """Positive rational basis + monotone collinear poles prove a full line."""
    projected = p - np.outer(p @ normal, normal)
    a, b = projected[0], projected[-1]
    axis = b - a
    length = float(np.linalg.norm(axis))
    _require(length > 8 * budget.tolerance, "DEGENERATE_BOUNDARY_SUPPORT")
    axis /= length
    scalar = (projected - a) @ axis
    perpendicular = projected - a - np.outer(scalar, axis)
    _require(np.linalg.norm(perpendicular, axis=1).max() <= budget.tolerance
             and np.all(np.diff(scalar) > budget.tolerance),
             "NONPOLYGONAL_OR_FOLDING_BOUNDARY_SUPPORT")
    return a, b, float(np.linalg.norm(perpendicular, axis=1).max())


def _patch_proof(patch, pair, curves, budget):
    p, w = np.asarray(patch.poles, dtype=float), np.asarray(patch.weights, dtype=float)
    _require(p.ndim != 3 or p.shape[0] * p.shape[1] <= budget.max_poles,
             "BOUNDARY_POLE_BUDGET_EXCEEDED")
    _require(p.ndim == 3 and p.shape[2] == 3 and min(p.shape[:2]) >= 2
             and w.shape == p.shape[:2] and np.isfinite(p).all()
             and np.isfinite(w).all() and np.all(w > 0)
             and float(w.max()) / float(w.min()) <= 1e12,
             "INVALID_WALL_COEFFICIENTS")
    _basis(patch.u_degree, patch.u_knots, len(p), budget)
    _basis(patch.v_degree, patch.v_knots, p.shape[1], budget)
    _require(np.array_equal(w, np.repeat(w[:, :1], w.shape[1], axis=1)),
             "COUPLED_RATIONAL_WALL_WEIGHTS")
    n = pair.normal
    height = (p - pair.origin) @ n
    transverse = p - np.expand_dims(height, 2) * n
    error = float(np.linalg.norm(transverse - transverse[:, :1], axis=2).max())
    _require(error <= budget.tolerance, "UNPROVEN_NORMAL_FIBERS")
    low, high = float(height[0, 0]), float(height[0, -1])
    _require(np.max(np.abs(height[:, 0] - low)) <= budget.tolerance
             and np.max(np.abs(height[:, -1] - high)) <= budget.tolerance,
             "VARYING_PHYSICAL_GENERATOR_ENDPOINTS")
    _require(low >= -budget.tolerance and high <= pair.thickness + budget.tolerance
             and high - low > 8 * budget.tolerance,
             "WALL_OUTSIDE_THICKNESS_SLAB")
    _require(np.all(np.diff(height, axis=1) > budget.tolerance),
             "NONMONOTONE_OR_SINGULAR_NORMAL_FIBERS")
    a, b, support_error = _line_support(p[:, 0], n, budget)
    a = a + float(pair.origin @ n) * n
    b = b + float(pair.origin @ n) * n
    generated = (
        Curve(patch.u_degree, patch.u_knots, p[:, 0], w[:, 0]),
        Curve(patch.v_degree, patch.v_knots, p[-1], w[-1]),
        Curve(patch.u_degree, patch.u_knots, p[:, -1], w[:, -1]),
        Curve(patch.v_degree, patch.v_knots, p[0], w[0]),
    )
    rail_errors = []
    for use, curve in zip(patch.rails, generated):
        _require(use.edge_id in curves, "NATIVE_RAIL_MISSING")
        rail_errors.append(_same_curve(curves[use.edge_id], curve, use.direction, budget))
    return {"face_id": patch.face_id, "pair_id": pair.pair_id,
            "support_start": a, "support_end": b,
            "axial_min": low, "axial_max": high,
            "polygonal_support_error_bound": support_error,
            "normal_fiber_error_bound": error,
            "native_rail_error_bound": max(rail_errors)}


def _near(a, b, tolerance):
    return np.linalg.norm(a - b) <= tolerance


def _skin_loops(face, pair, height, curves, budget):
    _require(bool(face.wires), "NATIVE_SKIN_WIRES_REQUIRED")
    flat = [use for wire in face.wires for use in wire]
    _require(sorted((x.edge_id, x.direction) for x in flat)
             == sorted((x.edge_id, x.direction) for x in face.uses),
             "SKIN_WIRE_COVERAGE_MISMATCH")
    loops = []
    for wire in face.wires:
        _require(len(wire) >= 3, "NONPOLYGONAL_SKIN_WIRE")
        points = []
        previous = None
        for use in wire:
            p, _ = _curve(curves[use.edge_id], budget)
            _require(np.max(np.abs((p - pair.origin) @ pair.normal - height))
                     <= budget.tolerance, "SKIN_RAIL_NOT_ON_PAIRED_PLANE")
            _line_support(p, pair.normal, budget)
            start, end = (p[0], p[-1]) if use.direction == 1 else (p[-1], p[0])
            if previous is not None:
                _require(_near(previous, start, budget.tolerance), "OPEN_NATIVE_SKIN_WIRE")
            points.append(start - height * pair.normal)
            previous = end
        _require(_near(previous, points[0] + height * pair.normal, budget.tolerance),
                 "OPEN_NATIVE_SKIN_WIRE")
        loops.append(np.asarray(points))
    return loops


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _polygon_checks(loops, normal, budget):
    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    x = np.cross(normal, helper); x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    origin = loops[0][0]
    projected = [np.stack([(loop - origin) @ x, (loop - origin) @ y], axis=1)
                 for loop in loops]
    edges = [(li, i, loop[i], loop[(i + 1) % len(loop)])
             for li, loop in enumerate(projected) for i in range(len(loop))]
    checks = 0
    for k, (li, i, a, b) in enumerate(edges):
        for lj, j, c, d in edges[k + 1:]:
            checks += 1
            _require(checks <= budget.max_pairwise_checks, "BOUNDARY_INTERSECTION_BUDGET_EXCEEDED")
            if li == lj and ((i - j) % len(projected[li]) in (1, len(projected[li]) - 1)):
                first, second = b - a, d - c
                angular = abs(_cross(first, second)) / (np.linalg.norm(first) * np.linalg.norm(second))
                _require(angular > 1e-12 or first @ second >= 0,
                         "BACKTRACKING_NATIVE_BOUNDARY")
                continue
            # A strict separating axis is a conservative disjointness proof.
            axes = [b - a, d - c]
            separated = False
            for axis in axes:
                axis = np.array([-axis[1], axis[0]])
                axis /= np.linalg.norm(axis)
                ab, cd = np.array([a @ axis, b @ axis]), np.array([c @ axis, d @ axis])
                if ab.max() + 4 * budget.tolerance < cd.min() or cd.max() + 4 * budget.tolerance < ab.min():
                    separated = True
            # Collinear disjoint segments need the tangent separating axis.
            for axis in axes:
                axis = axis / np.linalg.norm(axis)
                ab, cd = np.array([a @ axis, b @ axis]), np.array([c @ axis, d @ axis])
                if ab.max() + 4 * budget.tolerance < cd.min() or cd.max() + 4 * budget.tolerance < ab.min():
                    separated = True
            _require(separated, "SELF_INTERSECTING_OR_UNSEPARATED_BOUNDARY")
    areas = [sum(_cross(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))) / 2
             for loop in projected]
    _require(abs(areas[0]) > budget.tolerance ** 2
             and all(a * areas[0] < 0 for a in areas[1:]), "INVALID_NATIVE_WIRE_ORIENTATION")

    def inside(point, loop):
        hits = 0
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            if (a[1] > point[1]) != (b[1] > point[1]):
                intercept = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                hits += intercept > point[0]
        return bool(hits % 2)

    for i, hole in enumerate(projected[1:], 1):
        _require(inside(hole[0], projected[0]), "HOLE_OUTSIDE_OUTER_WIRE")
        _require(all(not inside(hole[0], other) and not inside(other[0], hole)
                     for j, other in enumerate(projected[1:], 1) if i != j),
                 "NESTED_OR_OVERLAPPING_HOLES_UNSUPPORTED")
    return checks, areas[0]


def certify_boundary_ownership(
    faces: Sequence[Face], skin_pairs: Sequence[SkinPair],
    wall_patches: Sequence[WallPatch], edge_curves: Mapping[int, Curve],
    *, budget: Budget = Budget(),
) -> dict:
    """Certify a complete polygonal slab inventory, or raise a typed refusal.

    Inputs are internal native-adapter evidence, not client receipt fields.
    Every source face must be one analytic plane skin or one full rectangular
    wall chart. Exact native trims/locations/solid validity must be established
    by the adapter before using this certificate in any runtime. This module
    does not establish that upstream extraction was honest or byte-bound.
    """
    _require(np.isfinite(budget.tolerance) and budget.tolerance > 0
             and all(type(v) is int and v > 0 for v in
                     (budget.max_faces, budget.max_edges, budget.max_poles,
                      budget.max_degree, budget.max_pairwise_checks)), "INVALID_PROOF_BUDGET")
    _require(0 < len(faces) <= budget.max_faces and 0 < len(edge_curves) <= budget.max_edges,
             "BOUNDARY_INVENTORY_BUDGET_EXCEEDED")
    by_face = {f.face_id: f for f in faces}
    _require(len(by_face) == len(faces) and all(type(i) is int and i >= 0 for i in by_face),
             "DUPLICATE_OR_INVALID_NATIVE_FACE_ID")
    pairs = {p.pair_id: p for p in skin_pairs}
    _require(len(pairs) == len(skin_pairs) and bool(pairs)
             and all(type(i) is int and i >= 0 for i in pairs), "INVALID_SKIN_PAIR_IDENTITIES")
    roles = {}
    normalized_pairs = {}
    for pair in skin_pairs:
        origin, normal = np.asarray(pair.origin, dtype=float), np.asarray(pair.normal, dtype=float)
        _require(origin.shape == normal.shape == (3,) and np.isfinite(origin).all()
                 and np.isfinite(normal).all() and abs(np.linalg.norm(normal) - 1) <= 1e-12
                 and np.isfinite(pair.thickness) and pair.thickness > 8 * budget.tolerance,
                 "INVALID_SKIN_PAIR_GEOMETRY")
        for face_id, role in ((pair.lower_face, "LOWER_PLANE_SKIN"),
                              (pair.upper_face, "UPPER_PLANE_SKIN")):
            _require(face_id in by_face and face_id not in roles, "DUPLICATE_OR_MISSING_SKIN_OWNERSHIP")
            roles[face_id] = role
        normalized_pairs[pair.pair_id] = SkinPair(pair.pair_id, pair.lower_face,
                                                 pair.upper_face, origin, normal, pair.thickness)
    owners = {}
    for face in faces:
        _require(len(face.uses) >= 3 and len({u.edge_id for u in face.uses}) == len(face.uses),
                 "REPEATED_OR_INCOMPLETE_NATIVE_EDGE_USES")
        for use in face.uses:
            _require(type(use.edge_id) is int and use.edge_id in edge_curves
                     and type(use.direction) is int and use.direction in (-1, 1),
                     "INVALID_NATIVE_EDGE_USE")
            owners.setdefault(use.edge_id, []).append((face.face_id, use.direction))
    _require(set(owners) == set(edge_curves), "UNUSED_NATIVE_EDGE_GEOMETRY")
    _require(all(len(uses) == 2 and uses[0][0] != uses[1][0]
                 and uses[0][1] == -uses[1][1] for uses in owners.values()),
             "OPEN_NONMANIFOLD_OR_INCOHERENT_EDGE_INCIDENCE")
    pole_count = sum(len(_curve(c, budget)[0]) for c in edge_curves.values())
    pole_count += sum(np.asarray(p.poles).shape[0] * np.asarray(p.poles).shape[1]
                      for p in wall_patches if np.asarray(p.poles).ndim == 3)
    _require(pole_count <= budget.max_poles, "BOUNDARY_POLE_BUDGET_EXCEEDED")
    proofs = []
    for patch in wall_patches:
        _require(patch.face_id in by_face and patch.face_id not in roles
                 and patch.pair_id in normalized_pairs, "DUPLICATE_OR_MISSING_WALL_OWNERSHIP")
        face = by_face[patch.face_id]
        _require(len(patch.rails) == 4 and len(face.uses) == 4
                 and {u.edge_id for u in face.uses} == {u.edge_id for u in patch.rails},
                 "PARTIAL_OR_NONRECTANGULAR_WALL_TRIM")
        expected = [p.direction * sign for p, sign in zip(patch.rails, (1, 1, -1, -1))]
        actual = {u.edge_id: u.direction for u in face.uses}
        signs = [actual[u.edge_id] * value for u, value in zip(patch.rails, expected)]
        _require(len(set(signs)) == 1, "INCOHERENT_WALL_CHART_ORIENTATION")
        proof = _patch_proof(patch, normalized_pairs[patch.pair_id], edge_curves, budget)
        roles[patch.face_id] = "CERTIFIED_NORMAL_CUT_WALL"
        proofs.append(proof)
    _require(set(roles) == set(by_face), "UNEXPLAINED_NATIVE_MATERIAL_SURFACE")
    intersections = 0
    for pair in normalized_pairs.values():
        lower = _skin_loops(by_face[pair.lower_face], pair, 0, edge_curves, budget)
        upper = _skin_loops(by_face[pair.upper_face], pair, pair.thickness, edge_curves, budget)
        lower_checks, lower_area = _polygon_checks(lower, pair.normal, budget)
        upper_checks, upper_area = _polygon_checks(upper, pair.normal, budget)
        intersections += lower_checks + upper_checks
        _require(lower_area < 0 < upper_area, "INVERTED_PAIRED_SKIN_ORIENTATION")
        _require(intersections <= budget.max_pairwise_checks, "BOUNDARY_INTERSECTION_BUDGET_EXCEEDED")
        # Match geometric line supports across both skin loops, independent of winding.
        lower_segments = [(loop[i], loop[(i + 1) % len(loop)]) for loop in lower for i in range(len(loop))]
        upper_segments = [(loop[i], loop[(i + 1) % len(loop)]) for loop in upper for i in range(len(loop))]
        _require(len(lower_segments) == len(upper_segments), "UNEQUAL_PAIRED_SKIN_FOOTPRINTS")
        def same(a, b, c, d):
            return (_near(a, c, budget.tolerance) and _near(b, d, budget.tolerance)) or (
                _near(a, d, budget.tolerance) and _near(b, c, budget.tolerance))
        _require(all(sum(same(a, b, c, d) for c, d in upper_segments) == 1
                     for a, b in lower_segments), "UNEQUAL_PAIRED_SKIN_FOOTPRINTS")
        members = [p for p in proofs if p["pair_id"] == pair.pair_id]
        assignments = {i: [] for i in range(len(lower_segments))}
        for p in members:
            matches = [i for i, (a, b) in enumerate(lower_segments)
                       if same(a, b, p["support_start"], p["support_end"])]
            _require(len(matches) == 1, "WALL_SUPPORT_NOT_ON_SKIN_BOUNDARY")
            assignments[matches[0]].append(p)
        for group in assignments.values():
            intervals = sorted((p["axial_min"], p["axial_max"]) for p in group)
            _require(bool(intervals) and abs(intervals[0][0]) <= budget.tolerance
                     and abs(intervals[-1][1] - pair.thickness) <= budget.tolerance,
                     "INCOMPLETE_NORMAL_FIBER_COVERAGE")
            _require(all(abs(a[1] - b[0]) <= budget.tolerance for a, b in zip(intervals, intervals[1:])),
                     "OVERLAPPING_OR_GAPPED_NORMAL_FIBER_COVERAGE")
        # Each skin edge must directly own a certified wall, never another skin.
        for skin in (pair.lower_face, pair.upper_face):
            for use in by_face[skin].uses:
                other = next(f for f, _ in owners[use.edge_id] if f != skin)
                _require(roles[other] == "CERTIFIED_NORMAL_CUT_WALL"
                         and next(p.pair_id for p in wall_patches if p.face_id == other) == pair.pair_id,
                         "SKIN_BOUNDARY_OWNERSHIP_MISMATCH")
    # Disconnected slab pairs may collide; this first version certifies one pair only.
    _require(len(normalized_pairs) == 1, "MULTIPAIR_GLOBAL_OWNERSHIP_UNSUPPORTED")
    return {"schema": "NORMAL_CUT_WALL_OWNERSHIP_V1",
            "scope": "COMPLETE_POLYGONAL_SINGLE_SLAB",
            "source_face_roles": {str(k): roles[k] for k in sorted(roles)},
            "native_face_count": len(faces), "native_edge_count": len(owners),
            "coefficient_pole_count": pole_count, "pairwise_checks": intersections,
            "wall_certificates": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in p.items()} for p in proofs],
            "geometry_tolerance": budget.tolerance,
            "precision_scope": "LOCAL_COEFFICIENT_BOUNDS_NOT_PHYSICAL_OR_CONTINUOUS_TRIM_ACCURACY",
            "manufacturing_blank_established": False,
            "native_adapter_required": ["BYTE_BOUND_SOURCE", "EXACT_FULL_CHART_TRIMS",
                                        "ANALYTIC_SKIN_PLANES", "LOCATED_ORIENTED_BREP",
                                        "VALID_SINGLE_SOLID"]}
