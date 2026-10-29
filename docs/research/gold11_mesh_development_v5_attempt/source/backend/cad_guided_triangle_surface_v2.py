"""Bounded closest-point queries on finite oriented tessellation triangles.

This additive V2 primitive does not import CAD kernels or application services.
Triangles are an approximate input surface, NOT analytic CAD, calibrated geometry,
native outward normals, registration or observed-state evidence. No V1/R0 path is
changed. The interior-or-boundary minimization follows the formulation in Eberly,
DistancePoint3Triangle3 (Geometric Tools, 2020); all three finite segment candidates
are evaluated explicitly instead of copying its rounded quadratic pseudocode.
https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

Arithmetic restrictions and guards below define an engineering support regime;
they are not interval-arithmetic certificates or calibrated error bounds.
"""

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np


MAX_TRIANGLES = 500_000
MAX_QUERY_ROWS = 200_000
MAX_COORDINATE = 1e100
MIN_EDGE_LENGTH = 1e-100
MIN_SHAPE_QUALITY = 1e-8
ARITHMETIC_FACTOR = 4096.0
MAX_GUARD_TO_MIN_EDGE = 1e-5
RESULT_ARRAY_KEYS = (
    "closest_points", "distances", "triangle_indices", "source_face_ids", "oriented_normals",
    "normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous", "near_tie_counts",
    "numerical_distance_guard", "closest_regions",
)


@dataclass(frozen=True)
class TriangleSurfaceBudgets:
    max_triangles: int = MAX_TRIANGLES
    max_query_rows: int = MAX_QUERY_ROWS
    leaf_size: int = 16
    max_build_nodes: int = 131_071
    max_build_triangle_visits: int = 10_000_000
    max_node_visits: int = 32_000_000
    max_triangle_pairs: int = 32_000_000
    max_normal_pairs: int = 32_000_000
    max_work_units: int = 512_000_000
    max_pair_batch: int = 65_536
    max_ties_per_query: int = 32


@dataclass(frozen=True)
class _TriangleSurfaceIndex:
    triangles: np.ndarray
    source_face_ids: tuple
    normals: np.ndarray
    edge_vectors: np.ndarray
    edge_units: np.ndarray
    edge_lengths: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    c_u: np.ndarray
    c_v: np.ndarray
    permutation: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    left: np.ndarray
    right: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    minimum_edge: float
    minimum_quality: float
    maximum_edge: float
    coordinate_magnitude: float


class _Refusal(Exception):
    pass


def _norm(values):
    return np.hypot(np.hypot(values[..., 0], values[..., 1]), values[..., 2])


def _budgets(value):
    if value is None:
        return TriangleSurfaceBudgets()
    if type(value) is not TriangleSurfaceBudgets:
        raise _Refusal("INVALID_BUDGETS")
    maxima = TriangleSurfaceBudgets()
    for item in fields(value):
        number = getattr(value, item.name)
        maximum = getattr(maxima, item.name)
        if type(number) is not int or number < 0 or number > maximum:
            raise _Refusal("INVALID_BUDGETS")
    if value.leaf_size < 8 or value.max_pair_batch < 1 or value.max_ties_per_query < 1:
        raise _Refusal("INVALID_BUDGETS")
    return value


def _positions(value, trailing_shape, cap, name):
    # Inspect before conversion so booleans, complex values and large integer
    # coordinates cannot silently become different binary64 geometry.
    if not isinstance(value, np.ndarray):
        original = np.asarray(value, dtype=object)
        if (original.ndim != len(trailing_shape) + 1 or original.shape[1:] != trailing_shape
                or len(original) > cap):
            raise _Refusal(f"INVALID_OR_OVER_CAP_{name}")
        for scalar in original.flat:
            if (isinstance(scalar, (bool, np.bool_))
                    or not isinstance(scalar, (float, int, np.floating, np.integer))):
                raise _Refusal(f"INVALID_SCALAR_{name}")
            if isinstance(scalar, (int, np.integer)) and abs(int(scalar)) > 2**53:
                raise _Refusal(f"INEXACT_INTEGER_{name}")
        array = np.asarray(value)
    else:
        array = value
    if (array.ndim != len(trailing_shape) + 1 or array.shape[1:] != trailing_shape
            or len(array) > cap or array.dtype.kind not in "fiu" or array.dtype.itemsize > 8):
        raise _Refusal(f"INVALID_OR_OVER_CAP_{name}")
    if array.dtype.kind in "iu" and array.size and (
            np.any(array > 2**53) or (array.dtype.kind == "i" and np.any(array < -(2**53)))):
        raise _Refusal(f"INEXACT_INTEGER_{name}")
    array = np.array(array, dtype=np.float64, copy=True, order="C")
    absolute = np.abs(array)
    if (not np.isfinite(array).all() or np.any(absolute > MAX_COORDINATE)
            or np.any((absolute > 0) & (absolute < np.finfo(float).tiny))):
        raise _Refusal(f"UNSUPPORTED_NUMERIC_{name}")
    return array


def _identities(values, count):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != count:
        raise _Refusal("INVALID_SOURCE_FACE_IDENTITIES")
    result = []
    for value in values:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            if not 0 <= int(value) <= 2**53:
                raise _Refusal("INVALID_SOURCE_FACE_IDENTITIES")
            result.append(int(value))
        elif isinstance(value, str) and 0 < len(value) <= 128 and all(32 <= ord(char) <= 126 for char in value):
            result.append(value)
        else:
            raise _Refusal("INVALID_SOURCE_FACE_IDENTITIES")
    return tuple(result)


def _read_only(array):
    # A write=False owning ndarray can be re-enabled. Immutable bytes also bind
    # every base/view so accidental mutation cannot detach geometry from its BVH.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def build_triangle_surface(triangles, source_face_ids, *, budgets: Optional[TriangleSurfaceBudgets] = None):
    """Snapshot a finite triangle soup and build its deterministic median BVH.

    Source face IDs are caller-supplied attribution labels only. A single invalid
    or ill-conditioned triangle rejects the entire soup; no silent hole creation.
    Build work counts triangle rows examined at tree nodes, not CPU instructions.
    """
    diagnostics = {"build_nodes": 0, "build_triangle_visits": 0}
    try:
        budget = _budgets(budgets)
        vertices = _positions(triangles, (3, 3), budget.max_triangles, "TRIANGLES")
        if not len(vertices):
            raise _Refusal("EMPTY_TRIANGLE_SURFACE")
        identities = _identities(source_face_ids, len(vertices))
        edges = np.stack((vertices[:, 1] - vertices[:, 0],
                          vertices[:, 2] - vertices[:, 1],
                          vertices[:, 0] - vertices[:, 2]), axis=1)
        lengths = _norm(edges)
        maximum = np.max(lengths, axis=1)
        if np.any(lengths < MIN_EDGE_LENGTH) or not np.isfinite(lengths).all():
            raise _Refusal("DEGENERATE_OR_UNSUPPORTED_TRIANGLE_EDGE")
        cross = np.cross(edges[:, 0] / maximum[:, None], -edges[:, 2] / maximum[:, None])
        quality = _norm(cross)
        if np.any(quality < MIN_SHAPE_QUALITY) or not np.isfinite(quality).all():
            raise _Refusal("ILL_CONDITIONED_TRIANGLE")
        normals = cross / quality[:, None]
        units = edges / lengths[..., None]
        basis_u = units[:, 0]
        basis_v = np.cross(normals, basis_u)
        c = -edges[:, 2]
        c_u = np.sum(c * basis_u, axis=1)
        c_v = np.sum(c * basis_v, axis=1)
        if np.any(c_v <= 0) or not np.isfinite(c_v).all():
            raise _Refusal("UNSUPPORTED_TRIANGLE_BASIS")
        minimum_edge = float(np.min(lengths))
        minimum_quality = float(np.min(quality))
        coordinate_magnitude = float(np.max(np.abs(vertices)))
        maximum_edge = float(np.max(lengths))
        guard = ARITHMETIC_FACTOR * np.finfo(float).eps * (coordinate_magnitude + maximum_edge) / minimum_quality
        if (not np.isfinite(guard) or guard < np.finfo(float).tiny
                or guard > MAX_GUARD_TO_MIN_EDGE * minimum_edge):
            raise _Refusal("UNSUPPORTED_TRIANGLE_ARITHMETIC_CONDITIONING")
        triangle_lower = vertices.min(axis=1)
        triangle_upper = vertices.max(axis=1)
        centers = vertices[:, 0] / 3. + vertices[:, 1] / 3. + vertices[:, 2] / 3.
        order = np.arange(len(vertices), dtype=np.int64)
        node_lower, node_upper, left, right, starts, stops = [], [], [], [], [], []

        def build(start, stop):
            count = stop - start
            if (diagnostics["build_nodes"] + 1 > budget.max_build_nodes
                    or diagnostics["build_triangle_visits"] + count > budget.max_build_triangle_visits):
                raise _Refusal("BUILD_BUDGET_EXCEEDED")
            diagnostics["build_nodes"] += 1
            diagnostics["build_triangle_visits"] += count
            ids = order[start:stop]
            node = len(left)
            node_lower.append(np.min(triangle_lower[ids], axis=0))
            node_upper.append(np.max(triangle_upper[ids], axis=0))
            left.append(-1)
            right.append(-1)
            starts.append(start)
            stops.append(stop)
            if count > budget.leaf_size:
                extent = np.ptp(centers[ids], axis=0)
                axis = int(np.argmax(extent))
                permutation = np.argsort(centers[ids, axis], kind="stable")
                order[start:stop] = ids[permutation]
                middle = start + count // 2
                left[node] = build(start, middle)
                right[node] = build(middle, stop)
            return node

        build(0, len(vertices))
        index = _TriangleSurfaceIndex(
            *[_read_only(array) for array in (vertices,)], identities,
            *[_read_only(array) for array in (normals, edges, units, lengths, basis_u, basis_v, c_u, c_v, order,
                                             np.asarray(node_lower), np.asarray(node_upper),
                                             np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64),
                                             np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64))],
            minimum_edge, minimum_quality, maximum_edge, coordinate_magnitude)
        return {"status": "COMPLETE", "reason": "FINITE_TRIANGLE_INDEX_BUILT", "index": index,
                "diagnostics": diagnostics}
    except (_Refusal, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc) if isinstance(exc, _Refusal) else "INVALID_SURFACE_INPUT",
                "index": None, "diagnostics": diagnostics}


def _closest_pairs(index, points, ids, guards):
    """Cartesian batch; return finite closest points and direct hypot distances."""
    triangles = index.triangles[ids]
    delta = points[:, None, :] - triangles[None, :, 0, :]
    beta = np.sum(delta * index.basis_v[ids][None, :, :], axis=2) / index.c_v[ids][None, :]
    alpha = ((np.sum(delta * index.basis_u[ids][None, :, :], axis=2)
              - beta * index.c_u[ids][None, :]) / index.edge_lengths[ids, 0][None, :])
    inside = (alpha >= 0) & (beta >= 0) & (alpha + beta <= 1)
    a = np.where(inside, alpha, 0.)
    b = np.where(inside, beta, 0.)
    interior = (triangles[None, :, 0, :]
                + a[..., None] * index.edge_vectors[ids, 0][None, :, :]
                - b[..., None] * index.edge_vectors[ids, 2][None, :, :])
    best_points = interior
    best_distances = np.where(inside, _norm(points[:, None, :] - interior), np.inf)
    edge_candidates = []
    for edge in range(3):
        start = triangles[:, edge]
        end = triangles[:, (edge + 1) % 3]
        offset = points[:, None, :] - start[None, :, :]
        along = np.sum(offset * index.edge_units[ids, edge][None, :, :], axis=2)
        parameter = np.clip(along / index.edge_lengths[ids, edge][None, :], 0., 1.)
        candidate = start[None, :, :] + parameter[..., None] * index.edge_vectors[ids, edge][None, :, :]
        candidate = np.where((parameter == 1)[..., None], end[None, :, :], candidate)
        edge_candidates.append(candidate)
        distance = _norm(points[:, None, :] - candidate)
        use = distance < best_distances
        best_distances = np.where(use, distance, best_distances)
        best_points = np.where(use[..., None], candidate, best_points)
    if not np.isfinite(best_distances).all() or not np.isfinite(best_points).all():
        raise _Refusal("UNSUPPORTED_PAIR_ARITHMETIC")
    # A facet tangent normal is not the finite-distance gradient on its boundary.
    # These are triangulation regions, not certified smooth/native surface roles.
    edge_gap = np.minimum.reduce([_norm(best_points - edge_point) for edge_point in edge_candidates])
    vertex_gap = np.minimum.reduce([_norm(best_points - triangles[None, :, vertex, :]) for vertex in range(3)])
    regions = np.full(best_distances.shape, "FACE", dtype="U6")
    regions[edge_gap <= 2. * guards[:, None]] = "EDGE"
    regions[vertex_gap <= 2. * guards[:, None]] = "VERTEX"
    return best_points, best_distances, regions


def query_closest_surface(index, points, *, near_tie_distance=0.0,
                          normal_similarity_cos=0.9961946980917455,
                          budgets: Optional[TriangleSurfaceBudgets] = None):
    """Closest finite surface points with global-nearest, all-pair tie diagnostics.

    Two complete BVH passes are required: nearest first, then a fixed nearest+
    tie-distance sweep. Opposite winding is an oriented-normal ambiguity; parallel
    equidistant sheets are a closest-point ambiguity even with identical normals.
    Any malformed input, unsupported arithmetic or exhausted budget clears every
    result row. A COMPLETE result does not establish CAD/calibration accuracy.
    The normal flag means possibly incompatible under the numerical dot guard,
    not a certified angular violation. FACE/EDGE/VERTEX describe tessellation
    regions; a facet normal must not replace the finite-distance boundary gradient.
    Work units are box pairs + 8*triangle pairs + 3*near-tie normal/point pairs.
    """
    statistics = {"node_visits": 0, "triangle_pairs": 0, "normal_pairs": 0, "work_units": 0,
                  "passes_completed": 0, "maximum_pair_batch": 0,
                  "tie_batches_completed": 0, "maximum_tie_storage_rows": 0}
    unavailable = {key: None for key in RESULT_ARRAY_KEYS}
    try:
        budget = _budgets(budgets)
        if not isinstance(index, _TriangleSurfaceIndex):
            raise _Refusal("INVALID_TRIANGLE_INDEX")
        if len(index.triangles) > budget.max_triangles:
            raise _Refusal("INDEX_EXCEEDS_QUERY_TRIANGLE_BUDGET")
        query = _positions(points, (3,), budget.max_query_rows, "QUERY_ROWS")
        for value in (near_tie_distance, normal_similarity_cos):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)):
                raise _Refusal("INVALID_TIE_POLICY")
        if (not np.isfinite(near_tie_distance) or not 0 <= near_tie_distance <= MAX_COORDINATE
                or not np.isfinite(normal_similarity_cos) or not -1 <= normal_similarity_cos <= 1):
            raise _Refusal("INVALID_TIE_POLICY")
        magnitude = np.max(np.abs(query), axis=1) if len(query) else np.empty(0)
        guard = ARITHMETIC_FACTOR * np.finfo(float).eps * (
            magnitude + index.coordinate_magnitude + index.maximum_edge) / index.minimum_quality
        if (not np.isfinite(guard).all() or np.any(guard < np.finfo(float).tiny)
                or np.any(guard > MAX_GUARD_TO_MIN_EDGE * index.minimum_edge)):
            raise _Refusal("UNSUPPORTED_QUERY_ARITHMETIC_CONDITIONING")
        best = np.full(len(query), np.inf)
        best_ids = np.full(len(query), -1, dtype=np.int64)
        best_points = np.zeros((len(query), 3))
        best_regions = np.full(len(query), "FACE", dtype="U6")
        near_tie_counts = np.zeros(len(query), dtype=np.int64)

        def charge(name, count, weight):
            if (statistics[name] + count > getattr(budget, "max_" + name)
                    or statistics["work_units"] + weight * count > budget.max_work_units):
                raise _Refusal("QUERY_BUDGET_EXCEEDED_" + name.upper())
            statistics[name] += count
            statistics["work_units"] += weight * count

        def lower_bound(node, rows):
            charge("node_visits", len(rows), 1)
            offset = np.maximum(np.maximum(index.lower[node] - query[rows],
                                           query[rows] - index.upper[node]), 0.)
            return np.maximum(0., _norm(offset) - guard[rows])

        def walk(tie_pass, seed_rows):
            stack = [(0, seed_rows)]
            while stack:
                node, rows = stack.pop()
                if not len(rows):
                    continue
                lower = lower_bound(node, rows)
                extra = float(near_tie_distance) if tie_pass else 0.
                rows = rows[lower <= best[rows] + extra + 2. * guard[rows]]
                if not len(rows):
                    continue
                left, right = int(index.left[node]), int(index.right[node])
                if left >= 0:
                    lower_left, lower_right = lower_bound(left, rows), lower_bound(right, rows)
                    left_first = lower_left <= lower_right
                    # Every query's near subtree executes before its far subtree.
                    stack.append((right, rows[left_first]))
                    stack.append((left, rows[~left_first]))
                    stack.append((right, rows[~left_first]))
                    stack.append((left, rows[left_first]))
                    continue
                leaf_ids = index.permutation[index.starts[node]:index.stops[node]]
                for tri_start in range(0, len(leaf_ids), budget.max_pair_batch):
                    ids = leaf_ids[tri_start:tri_start + budget.max_pair_batch]
                    row_batch = max(1, budget.max_pair_batch // len(ids))
                    for row_start in range(0, len(rows), row_batch):
                        batch = rows[row_start:row_start + row_batch]
                        pairs = len(batch) * len(ids)
                        charge("triangle_pairs", pairs, 8)
                        statistics["maximum_pair_batch"] = max(statistics["maximum_pair_batch"], pairs)
                        closest, distances, regions = _closest_pairs(index, query[batch], ids, guard[batch])
                        for column, triangle_id in enumerate(ids):
                            distance = distances[:, column]
                            if tie_pass:
                                chosen = distance <= best[batch] + float(near_tie_distance) + 2. * guard[batch]
                                selected = batch[chosen]
                                local_rows = selected - tie_start
                                slots = tie_counts[local_rows]
                                if np.any(slots >= budget.max_ties_per_query):
                                    raise _Refusal("NEAR_TIE_BUDGET_EXCEEDED")
                                tie_ids[local_rows, slots] = triangle_id
                                tie_points[local_rows, slots] = closest[chosen, column]
                                tie_counts[local_rows] += 1
                            else:
                                use = ((distance < best[batch]) |
                                       ((distance == best[batch]) & (triangle_id < best_ids[batch])))
                                selected = batch[use]
                                best[selected] = distance[use]
                                best_ids[selected] = triangle_id
                                best_points[selected] = closest[use, column]
                                best_regions[selected] = regions[use, column]
        walk(False, np.arange(len(query), dtype=np.int64))
        statistics["passes_completed"] = 1
        if np.any(best_ids < 0) or not np.isfinite(best).all():
            raise _Refusal("CLOSEST_SEARCH_INCOMPLETE")
        normal_ambiguous = np.zeros(len(query), dtype=bool)
        point_ambiguous = np.zeros(len(query), dtype=bool)
        face_ambiguous = np.zeros(len(query), dtype=bool)
        normal_guard = ARITHMETIC_FACTOR * np.finfo(float).eps / index.minimum_quality
        if not np.isfinite(normal_guard) or normal_guard < np.finfo(float).tiny:
            raise _Refusal("UNSUPPORTED_NORMAL_ARITHMETIC")
        statistics["normal_dot_guard"] = normal_guard
        tie_batch_size = max(1, budget.max_pair_batch // budget.max_ties_per_query)
        for tie_start in range(0, len(query), tie_batch_size):
            tie_stop = min(len(query), tie_start + tie_batch_size)
            count = tie_stop - tie_start
            tie_ids = np.full((count, budget.max_ties_per_query), -1, dtype=np.int64)
            tie_points = np.zeros((count, budget.max_ties_per_query, 3))
            tie_counts = np.zeros(count, dtype=np.int64)
            statistics["maximum_tie_storage_rows"] = max(statistics["maximum_tie_storage_rows"], count)
            walk(True, np.arange(tie_start, tie_stop, dtype=np.int64))
            if np.any(tie_counts < 1):
                raise _Refusal("NEAR_TIE_SEARCH_INCOMPLETE")
            near_tie_counts[tie_start:tie_stop] = tie_counts
            maximum_ties = int(np.max(tie_counts))
            for second in range(1, maximum_ties):
                local_rows = np.flatnonzero(tie_counts > second)
                rows = local_rows + tie_start
                second_ids = tie_ids[local_rows, second]
                for first in range(second):
                    charge("normal_pairs", len(rows), 3)
                    first_ids = tie_ids[local_rows, first]
                    dot = np.sum(index.normals[first_ids] * index.normals[second_ids], axis=1)
                    normal_ambiguous[rows] |= dot < float(normal_similarity_cos) + normal_guard
                    separation = _norm(tie_points[local_rows, first] - tie_points[local_rows, second])
                    point_ambiguous[rows] |= separation > 2. * guard[rows]
                    face_ambiguous[rows] |= np.fromiter(
                        (index.source_face_ids[int(a)] != index.source_face_ids[int(b)]
                         for a, b in zip(first_ids, second_ids)), dtype=bool, count=len(rows))
            statistics["tie_batches_completed"] += 1
        statistics["passes_completed"] = 2
        return {"status": "COMPLETE", "reason": "FINITE_TRIANGLE_CLOSEST_QUERY_COMPLETE",
                "closest_points": best_points, "distances": best, "triangle_indices": best_ids,
                "source_face_ids": tuple(index.source_face_ids[int(value)] for value in best_ids),
                "oriented_normals": index.normals[best_ids].copy(), "normal_ambiguous": normal_ambiguous,
                "closest_point_ambiguous": point_ambiguous, "face_identity_ambiguous": face_ambiguous,
                "near_tie_counts": near_tie_counts, "numerical_distance_guard": guard,
                "closest_regions": best_regions,
                "diagnostics": statistics}
    except (_Refusal, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc) if isinstance(exc, _Refusal) else "INVALID_QUERY_INPUT",
                **unavailable, "diagnostics": statistics}
