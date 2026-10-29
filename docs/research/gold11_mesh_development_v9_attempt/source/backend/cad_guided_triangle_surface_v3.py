"""V3 finite-triangle BVH with conditional local distance enclosures.

Additive, NumPy-only research primitive; V2 and all workflow routes remain frozen.
The supplied tessellation is represented geometry, NOT analytic/native CAD or
calibrated geometry. Distance statements inherit the binary64 arithmetic model
of cad_guided_triangle_distance_bounds_v3. Normals and region labels remain
conditioned engineering estimates, explicitly NOT interval certificates.

A first pass minimizes feasible-candidate upper bounds. A second pass fixes that
upper bound and checks every possibly competitive triangle using its own lower
bound and its own edge-relative interval-width requirement. A wide but sound
interval on a provably irrelevant primitive is not an error or a deleted face.
The minimizer proposal chart uses each triangle's rounded maximum edge length;
it does not claim exact scaling. The independent certificate helper chooses its
own per-pair power-of-two chart and encloses the original represented geometry.
"""

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from .cad_guided_triangle_distance_bounds_v3 import enclose_triangle_distances


MAX_TRIANGLES = 500_000
MAX_QUERY_ROWS = 200_000
MAX_COORDINATE = 1e100
MIN_EDGE_LENGTH = 1e-100
MIN_SHAPE_QUALITY = 1e-8
ARITHMETIC_FACTOR = 4096.0
MAX_INTERVAL_TO_MIN_EDGE = 1e-5
RESULT_ARRAY_KEYS = (
    "closest_points", "distances", "triangle_indices", "source_face_ids", "oriented_normals",
    "normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous", "near_tie_counts",
    "numerical_distance_guard", "closest_regions", "distance_lower", "distance_upper",
    "triangle_distance_lower", "triangle_distance_upper", "closest_point_error",
    "closest_point_restoration_error", "normal_ambiguity_uncertain",
    "closest_point_ambiguity_uncertain", "region_uncertain",
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
    max_certificate_pairs: int = 128_000_000
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
    edge_lengths: np.ndarray
    edge_units: np.ndarray
    local_edges: np.ndarray
    scales: np.ndarray
    minimum_edges: np.ndarray
    qualities: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    c_u: np.ndarray
    c_v: np.ndarray
    normal_guards: np.ndarray
    geometric_guards: np.ndarray
    altitudes: np.ndarray
    triangle_lower: np.ndarray
    triangle_upper: np.ndarray
    permutation: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    left: np.ndarray
    right: np.ndarray
    starts: np.ndarray
    stops: np.ndarray


class _Refusal(Exception):
    pass


def _down(value):
    return np.nextafter(value, -np.inf)


def _up(value):
    return np.nextafter(value, np.inf)


def _norm(values):
    return np.hypot(np.hypot(values[..., 0], values[..., 1]), values[..., 2])


def _budgets(value):
    if value is None:
        return TriangleSurfaceBudgets()
    if type(value) is not TriangleSurfaceBudgets:
        raise _Refusal("INVALID_BUDGETS")
    maxima, copied = TriangleSurfaceBudgets(), {}
    for item in fields(value):
        number = getattr(value, item.name)
        if type(number) is not int or not 0 <= number <= getattr(maxima, item.name):
            raise _Refusal("INVALID_BUDGETS")
        copied[item.name] = number
    if copied["leaf_size"] < 8 or copied["max_pair_batch"] < 1 or copied["max_ties_per_query"] < 1:
        raise _Refusal("INVALID_BUDGETS")
    return TriangleSurfaceBudgets(**copied)


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
    """Snapshot all primitives; preserve original edge/shape gates without a soup-wide guard."""
    diagnostics = dict(build_nodes=0, build_triangle_visits=0)
    try:
        budget = _budgets(budgets)
        vertices = _positions(triangles, (3, 3), budget.max_triangles, "TRIANGLES")
        if not len(vertices):
            raise _Refusal("EMPTY_TRIANGLE_SURFACE")
        identities = _identities(source_face_ids, len(vertices))
        edges = np.stack((vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 1],
                          vertices[:, 0]-vertices[:, 2]), axis=1)
        lengths = _norm(edges)
        if np.any(lengths < MIN_EDGE_LENGTH) or not np.isfinite(lengths).all():
            raise _Refusal("DEGENERATE_OR_UNSUPPORTED_TRIANGLE_EDGE")
        scales = np.max(lengths, axis=1)
        local = edges/scales[:, None, None]
        cross = np.cross(local[:, 0], -local[:, 2])
        quality = _norm(cross)
        if np.any(quality < MIN_SHAPE_QUALITY) or not np.isfinite(quality).all():
            raise _Refusal("ILL_CONDITIONED_TRIANGLE")
        normals = cross/quality[:, None]
        units = edges/lengths[..., None]
        basis_u = units[:, 0]
        basis_v = np.cross(normals, basis_u)
        c_u = np.sum(-local[:, 2]*basis_u, axis=1)
        c_v = np.sum(-local[:, 2]*basis_v, axis=1)
        normal_guards = _up(ARITHMETIC_FACTOR*np.finfo(float).eps/quality)
        geometric_guards = _up(normal_guards*scales)
        # Opposite-edge altitudes, used only for engineering region diagnostics.
        altitudes = quality[:, None]*scales[:, None]*(scales[:, None]/lengths[:, [1, 2, 0]])
        if (np.any(c_v <= 0) or any(not np.isfinite(a).all() for a in
                (normals, units, basis_v, c_u, c_v, normal_guards, geometric_guards, altitudes))):
            raise _Refusal("UNSUPPORTED_TRIANGLE_BASIS")
        tri_lower, tri_upper = vertices.min(axis=1), vertices.max(axis=1)
        centers = tri_lower+.5*(tri_upper-tri_lower)  # Ordering only, never a distance bound.
        order = np.arange(len(vertices), dtype=np.int64)
        node_lower, node_upper, left, right, starts, stops = [], [], [], [], [], []
        def build(start, stop):
            count = stop-start
            if (diagnostics["build_nodes"]+1 > budget.max_build_nodes
                    or diagnostics["build_triangle_visits"]+count > budget.max_build_triangle_visits):
                raise _Refusal("BUILD_BUDGET_EXCEEDED")
            diagnostics["build_nodes"] += 1
            diagnostics["build_triangle_visits"] += count
            ids = order[start:stop]
            node = len(left)
            node_lower.append(np.min(tri_lower[ids], axis=0))
            node_upper.append(np.max(tri_upper[ids], axis=0))
            left.append(-1)
            right.append(-1)
            starts.append(start)
            stops.append(stop)
            if count > budget.leaf_size:
                axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
                order[start:stop] = ids[np.argsort(centers[ids, axis], kind="stable")]
                middle = start+count//2
                left[node], right[node] = build(start, middle), build(middle, stop)
            return node
        build(0, len(vertices))
        arrays = (vertices, normals, edges, lengths, units, local, scales, np.min(lengths, axis=1),
            quality, basis_u, basis_v, c_u, c_v, normal_guards, geometric_guards, altitudes,
            tri_lower, tri_upper, order, np.asarray(node_lower), np.asarray(node_upper),
            np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64),
            np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64))
        frozen = [_read_only(array) for array in arrays]
        index = _TriangleSurfaceIndex(frozen[0], identities, *frozen[1:])
        return dict(status="COMPLETE", reason="LOCAL_TRIANGLE_INDEX_BUILT", index=index,
                    diagnostics=diagnostics)
    except (_Refusal, TypeError, ValueError, OverflowError, FloatingPointError) as error:
        return dict(status="UNAVAILABLE", reason=str(error) if isinstance(error, _Refusal)
                    else "INVALID_SURFACE_INPUT", index=None, diagnostics=diagnostics)


def _feasible_mask(coefficients):
    a, b = coefficients[:, 0], coefficients[:, 1]
    s = a+b
    bv = s-a
    tail = (a-(s-bv))+(b-bv)
    ordinary = np.isfinite(coefficients).all(axis=1)
    ordinary &= ~np.any((np.abs(coefficients) > 0) &
                       (np.abs(coefficients) < np.finfo(float).tiny), axis=1)
    return ordinary & (a >= 0) & (b >= 0) & (a <= 1) & (b <= 1) & ((s < 1) | ((s == 1) & (tail <= 0)))


def _edge_coefficients(parameter, edge):
    # Edge1's rounded complementary pair can sum to MORE than one exactly.
    # Shrinking one coordinate yields an explicit feasible nearby candidate;
    # the distance certificate, not a claim of exact edge projection, governs it.
    t = np.clip(parameter, 0., 1.)
    t = np.where((t > 0) & (t < np.finfo(float).tiny), 0., t)
    coefficients = np.zeros((len(t), 2))
    if edge == 0:
        coefficients[:, 0] = t
    elif edge == 1:
        coefficients[:, 0], coefficients[:, 1] = 1.-t, t
        bad = ~_feasible_mask(coefficients)
        coefficients[bad, 0] = np.nextafter(coefficients[bad, 0], 0.)
    else:
        coefficients[:, 1] = 1.-t
    # A subnormal complement is conservatively replaced by the corresponding
    # endpoint coefficient; no primitive or source identity is removed.
    coefficients[(coefficients > 0) & (coefficients < np.finfo(float).tiny)] = 0.
    if not _feasible_mask(coefficients).all():
        raise _Refusal("UNSUPPORTED_FEASIBLE_EDGE_CANDIDATE")
    return coefficients


def _optimizer_error(upper, lower, restoration):
    # Strong convexity of squared distance on a convex set:
    # ||q-proj_T(p)||^2 <= ||p-q||^2 - dist(p,T)^2.
    upper_square = _up(upper*upper)
    lower_square = np.maximum(0., _down(lower*lower))
    gap = np.maximum(0., _up(upper_square-lower_square))
    return _up(_up(np.sqrt(gap))+restoration)


def _paired_candidates(index, points, ids, charge, pair_cap):
    """One row per point/triangle pair; four feasible primal/dual witnesses."""
    count = len(points)
    scale = index.scales[ids]
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        local_p = (points-index.triangles[ids, 0])/scale[:, None]
        beta = np.sum(local_p*index.basis_v[ids], axis=1)/index.c_v[ids]
        alpha = (np.sum(local_p*index.basis_u[ids], axis=1)-beta*index.c_u[ids]) / (
            index.edge_lengths[ids, 0]/scale)
        interior = np.column_stack((alpha, beta))
        interior[~_feasible_mask(interior)] = 0.
        choices = [interior]
        for edge in range(3):
            offset = (points-index.triangles[ids, edge])/scale[:, None]
            along = np.sum(offset*index.edge_units[ids, edge], axis=1)
            parameter = along/(index.edge_lengths[ids, edge]/scale)
            choices.append(_edge_coefficients(parameter, edge))
        lower = np.zeros(count)
        upper = np.full(count, np.inf)
        closest = np.zeros((count, 3))
        nominal = np.zeros(count)
        restoration = np.zeros(count)
        coefficients = np.zeros((count, 2))
        for candidate in choices:
            charge("certificate_pairs", count, 1)
            certificate = enclose_triangle_distances(points, index.triangles[ids], candidate, max_pairs=pair_cap)
            if certificate["status"] != "COMPLETE":
                raise _Refusal("UNAVAILABLE_PAIR_DISTANCE_CERTIFICATE")
            lower = np.maximum(lower, certificate["distance_lower"])
            use = certificate["distance_upper"] < upper
            upper[use] = certificate["distance_upper"][use]
            closest[use] = certificate["closest_points"][use]
            nominal[use] = certificate["candidate_distance"][use]
            restoration[use] = certificate["closest_point_error"][use]
            coefficients[use] = candidate[use]
        if np.any(lower > upper):
            raise _Refusal("INCONSISTENT_PAIR_DISTANCE_ENCLOSURE")
        optimizer_error = _optimizer_error(upper, lower, restoration)
        weights = np.column_stack((np.maximum(0., 1.-coefficients[:, 0]-coefficients[:, 1]), coefficients))
        edge_gap = np.min(weights*index.altitudes[ids], axis=1)
        local_q = (coefficients[:, :1]*index.local_edges[ids, 0]
                   -coefficients[:, 1:]*index.local_edges[ids, 2])
        vertex_gap = np.minimum.reduce((_norm(local_q),
            _norm(local_q-index.local_edges[ids, 0]), _norm(local_q+index.local_edges[ids, 2])))*scale
        geometric_guard = index.geometric_guards[ids]
        regions = np.full(count, "FACE", dtype="U6")
        regions[edge_gap <= 2.*geometric_guard] = "EDGE"
        regions[vertex_gap <= 2.*geometric_guard] = "VERTEX"
        uncertain = edge_gap <= _up(optimizer_error+2.*geometric_guard)
        if any(not np.isfinite(a).all() for a in (lower, upper, closest, nominal, restoration, optimizer_error)):
            raise _Refusal("UNSUPPORTED_PAIR_OUTPUT_ARITHMETIC")
        return dict(lower=lower, upper=upper, closest=closest, nominal=nominal,
            restoration=restoration, optimizer_error=optimizer_error, regions=regions,
            region_uncertain=uncertain)


def _box_lower(lower, upper, points):
    """Outward L-infinity AABB lower bound, also a Euclidean lower bound."""
    offset = np.maximum(_down(lower-points), _down(points-upper))
    return np.maximum(0., np.max(offset, axis=-1))


def query_closest_surface(index, points, *, near_tie_distance=0.0,
                          normal_similarity_cos=0.9961946980917455,
                          budgets: Optional[TriangleSurfaceBudgets] = None):
    """Two-pass finite search with conditional per-pair distance certificates.

    Pass1 minimizes feasible upper bounds U*. Pass2 freezes U*: every primitive
    with Ltri<=up(U*+tau) is a possible competitor and must pass its OWN width
    gate. Wide valid intervals may be excluded only by a sound lower bound.
    An unavailable unexcluded certificate or exhausted budget clears all rows.

    Work units are BVH box rows + triangle-box rows +8*evaluated triangle pairs
    +certificate pairs +3*near-tie normal/point pairs. Each triangle evaluation
    has four supplied feasible candidates. This is bounded abstract work, not a
    CPU-instruction count. The V3 certificate budget is additive and versioned.

    Distances are nominal distances to the selected feasible candidate, with
    global hull-distance bounds beside them. Closest-point error bounds distance
    to the SELECTED TRIANGLE'S exact optimizer, not a globally unique point.
    Normal/region/point-comparison flags remain engineering diagnostics; their
    uncertainty flags are not permission to interpret them as certified roles.
    No workflow consumes this V3 primitive by default.
    """
    statistics = dict(node_visits=0, triangle_box_pairs=0, triangle_pairs=0,
        certificate_pairs=0, normal_pairs=0, work_units=0, passes_completed=0,
        maximum_pair_batch=0, maximum_certificate_batch=0, tie_batches_completed=0,
        maximum_tie_storage_rows=0, interval_width_rejections=0,
        certificate_failures=0, proof_pruned_triangle_pairs=0)
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
        tau = float(near_tie_distance)
        count = len(query)
        best_upper = np.full(count, np.inf)
        global_lower = np.full(count, np.inf)
        best_ids = np.full(count, -1, dtype=np.int64)
        best_points = np.zeros((count, 3))
        best_nominal = np.zeros(count)
        best_triangle_lower = np.zeros(count)
        best_restoration = np.zeros(count)
        best_error = np.zeros(count)
        best_regions = np.full(count, "FACE", dtype="U6")
        region_uncertain = np.zeros(count, dtype=bool)
        normal_ambiguous = np.zeros(count, dtype=bool)
        point_ambiguous = np.zeros(count, dtype=bool)
        face_ambiguous = np.zeros(count, dtype=bool)
        normal_uncertain = np.zeros(count, dtype=bool)
        point_uncertain = np.zeros(count, dtype=bool)
        near_tie_counts = np.zeros(count, dtype=np.int64)

        def charge(name, number, weight):
            if (statistics[name]+number > getattr(budget, "max_"+name)
                    or statistics["work_units"]+weight*number > budget.max_work_units):
                raise _Refusal("QUERY_BUDGET_EXCEEDED_"+name.upper())
            statistics[name] += number
            statistics["work_units"] += weight*number

        def charge_triangle_boxes(number):
            if statistics["work_units"]+number > budget.max_work_units:
                raise _Refusal("QUERY_BUDGET_EXCEEDED_TRIANGLE_BOX_WORK")
            statistics["triangle_box_pairs"] += number
            statistics["work_units"] += number

        def node_lower(node, rows):
            charge("node_visits", len(rows), 1)
            return _box_lower(index.lower[node], index.upper[node], query[rows])

        def threshold(rows, tie_pass):
            return _up(fixed_upper[rows]+tau) if tie_pass else best_upper[rows]

        def leaf(ids, rows, tie_pass):
            # Bound both sorting and all temporary pair arrays by max_pair_batch.
            # Per-query near-box ordering also avoids evaluating a safely
            # excludable extreme-scale distractor before a local candidate.
            for tri_start in range(0, len(ids), budget.max_pair_batch):
                tri_ids = ids[tri_start:tri_start+budget.max_pair_batch]
                row_batch = max(1, budget.max_pair_batch//len(tri_ids))
                for row_start in range(0, len(rows), row_batch):
                    batch = rows[row_start:row_start+row_batch]
                    box_pairs = len(batch)*len(tri_ids)
                    charge_triangle_boxes(box_pairs)
                    lower = _box_lower(index.triangle_lower[tri_ids][None, :, :],
                        index.triangle_upper[tri_ids][None, :, :], query[batch, None, :])
                    ordering = np.argsort(lower, axis=1, kind="stable")
                    for rank in range(len(tri_ids)):
                        columns = ordering[:, rank]
                        candidate_ids = tri_ids[columns]
                        candidate_lower = lower[np.arange(len(batch)), columns]
                        chosen = candidate_lower <= threshold(batch, tie_pass)
                        statistics["proof_pruned_triangle_pairs"] += int(np.count_nonzero(~chosen))
                        selected, selected_ids = batch[chosen], candidate_ids[chosen]
                        if not len(selected):
                            continue
                        charge("triangle_pairs", len(selected), 8)
                        statistics["maximum_pair_batch"] = max(statistics["maximum_pair_batch"], len(selected))
                        statistics["maximum_certificate_batch"] = max(statistics["maximum_certificate_batch"], len(selected))
                        try:
                            pairs = _paired_candidates(index, query[selected], selected_ids, charge, budget.max_pair_batch)
                        except _Refusal as error:
                            if str(error) == "UNAVAILABLE_PAIR_DISTANCE_CERTIFICATE":
                                statistics["certificate_failures"] += 1
                            raise
                        # min of valid lower bounds over all visited primitives,
                        # not merely the selected min-upper primitive. A truly
                        # nearer primitive cannot have been pruned by a bound>U*.
                        global_lower[selected] = np.minimum(global_lower[selected], pairs["lower"])
                        if not tie_pass:
                            use = ((pairs["upper"] < best_upper[selected]) |
                                ((pairs["upper"] == best_upper[selected]) & (selected_ids < best_ids[selected])))
                            target = selected[use]
                            best_upper[target] = pairs["upper"][use]
                            best_ids[target] = selected_ids[use]
                            best_points[target] = pairs["closest"][use]
                            best_nominal[target] = pairs["nominal"][use]
                            best_triangle_lower[target] = pairs["lower"][use]
                            best_restoration[target] = pairs["restoration"][use]
                            best_error[target] = pairs["optimizer_error"][use]
                            best_regions[target] = pairs["regions"][use]
                            region_uncertain[target] = pairs["region_uncertain"][use]
                            continue
                        competitive = pairs["lower"] <= threshold(selected, True)
                        statistics["proof_pruned_triangle_pairs"] += int(np.count_nonzero(~competitive))
                        target = selected[competitive]
                        relevant_ids = selected_ids[competitive]
                        if not len(target):
                            continue
                        # Deferred admission: a wide interval is fatal only when
                        # it cannot soundly exclude the triangle after U* froze.
                        width = _up(pairs["upper"][competitive]-pairs["lower"][competitive])
                        allowed = _down(MAX_INTERVAL_TO_MIN_EDGE*index.minimum_edges[relevant_ids])
                        if np.any(width > allowed):
                            statistics["interval_width_rejections"] += int(np.count_nonzero(width > allowed))
                            raise _Refusal("UNSUPPORTED_COMPETITIVE_TRIANGLE_INTERVAL_WIDTH")
                        local_rows = target-tie_start
                        slots = tie_counts[local_rows]
                        if np.any(slots >= budget.max_ties_per_query):
                            raise _Refusal("NEAR_TIE_BUDGET_EXCEEDED")
                        tie_ids[local_rows, slots] = relevant_ids
                        tie_points[local_rows, slots] = pairs["closest"][competitive]
                        tie_errors[local_rows, slots] = pairs["optimizer_error"][competitive]
                        tie_counts[local_rows] += 1

        def walk(tie_pass, seed_rows):
            stack = [(0, seed_rows)]
            while stack:
                node, rows = stack.pop()
                if not len(rows):
                    continue
                lower = node_lower(node, rows)
                rows = rows[lower <= threshold(rows, tie_pass)]
                if not len(rows):
                    continue
                left, right = int(index.left[node]), int(index.right[node])
                if left >= 0:
                    near_left = node_lower(left, rows) <= node_lower(right, rows)
                    stack.append((right, rows[near_left]))
                    stack.append((left, rows[~near_left]))
                    stack.append((right, rows[~near_left]))
                    stack.append((left, rows[near_left]))
                else:
                    leaf(index.permutation[index.starts[node]:index.stops[node]], rows, tie_pass)

        walk(False, np.arange(count, dtype=np.int64))
        statistics["passes_completed"] = 1
        if np.any(best_ids < 0) or not np.isfinite(best_upper).all():
            raise _Refusal("CLOSEST_SEARCH_INCOMPLETE")
        fixed_upper = best_upper.copy()
        # First-pass certificates may be deliberately weak for primitives that
        # a later, smaller U* excludes. Rebuild the global lower envelope from
        # the complete fixed-U* pass: every true minimizing triangle survives
        # its sound AABB bounds, so its L <= d* remains in that envelope.
        global_lower.fill(np.inf)
        tie_batch_size = max(1, budget.max_pair_batch//budget.max_ties_per_query)
        for tie_start in range(0, count, tie_batch_size):
            tie_stop = min(count, tie_start+tie_batch_size)
            row_count = tie_stop-tie_start
            tie_ids = np.full((row_count, budget.max_ties_per_query), -1, dtype=np.int64)
            tie_points = np.zeros((row_count, budget.max_ties_per_query, 3))
            tie_errors = np.zeros((row_count, budget.max_ties_per_query))
            tie_counts = np.zeros(row_count, dtype=np.int64)
            statistics["maximum_tie_storage_rows"] = max(statistics["maximum_tie_storage_rows"], row_count)
            walk(True, np.arange(tie_start, tie_stop, dtype=np.int64))
            if np.any(tie_counts < 1):
                raise _Refusal("NEAR_TIE_SEARCH_INCOMPLETE")
            near_tie_counts[tie_start:tie_stop] = tie_counts
            for second in range(1, int(np.max(tie_counts))):
                local_rows = np.flatnonzero(tie_counts > second)
                rows = local_rows+tie_start
                second_ids = tie_ids[local_rows, second]
                for first in range(second):
                    charge("normal_pairs", len(rows), 3)
                    first_ids = tie_ids[local_rows, first]
                    dot = np.sum(index.normals[first_ids]*index.normals[second_ids], axis=1)
                    dot_guard = _up(index.normal_guards[first_ids]+index.normal_guards[second_ids])
                    normal_ambiguous[rows] |= dot < float(normal_similarity_cos)+dot_guard
                    normal_uncertain[rows] |= np.abs(dot-float(normal_similarity_cos)) <= dot_guard
                    separation = _norm(tie_points[local_rows, first]-tie_points[local_rows, second])
                    error_sum = _up(tie_errors[local_rows, first]+tie_errors[local_rows, second])
                    point_threshold = 2.*np.maximum(index.geometric_guards[first_ids], index.geometric_guards[second_ids])
                    # Engineering separation test plus explicit unresolved band;
                    # a False flag is NOT a certificate of coincident optimizers.
                    distinct = separation > _up(error_sum+point_threshold)
                    point_ambiguous[rows] |= distinct
                    point_uncertain[rows] |= ~distinct & (_up(separation+error_sum) > point_threshold)
                    face_ambiguous[rows] |= np.fromiter((index.source_face_ids[int(a)] != index.source_face_ids[int(b)]
                        for a, b in zip(first_ids, second_ids)), dtype=bool, count=len(rows))
            statistics["tie_batches_completed"] += 1
        statistics["passes_completed"] = 2
        if not np.isfinite(global_lower).all() or np.any(global_lower > fixed_upper):
            raise _Refusal("GLOBAL_DISTANCE_ENCLOSURE_INCOMPLETE")
        numerical_guard = _up(np.maximum(np.abs(best_nominal-global_lower), np.abs(fixed_upper-best_nominal)))
        return dict(status="COMPLETE", reason="CONDITIONAL_LOCAL_FINITE_TRIANGLE_QUERY_COMPLETE",
            closest_points=best_points, distances=best_nominal, triangle_indices=best_ids,
            source_face_ids=tuple(index.source_face_ids[int(value)] for value in best_ids),
            oriented_normals=index.normals[best_ids].copy(), normal_ambiguous=normal_ambiguous,
            closest_point_ambiguous=point_ambiguous, face_identity_ambiguous=face_ambiguous,
            near_tie_counts=near_tie_counts, numerical_distance_guard=numerical_guard,
            closest_regions=best_regions, distance_lower=global_lower, distance_upper=fixed_upper,
            triangle_distance_lower=best_triangle_lower, triangle_distance_upper=fixed_upper.copy(),
            closest_point_error=best_error, closest_point_restoration_error=best_restoration,
            normal_ambiguity_uncertain=normal_uncertain, closest_point_ambiguity_uncertain=point_uncertain,
            region_uncertain=region_uncertain, diagnostics=statistics,
            distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
            closest_point_error_scope="SELECTED_TRIANGLE_OPTIMIZER_NOT_GLOBAL_UNIQUE_WITNESS",
            normal_region_status="ENGINEERING_NOT_INTERVAL_CERTIFIED",
            tie_semantics="POSSIBLE_COMPETITORS_AT_FIXED_GLOBAL_UPPER_PLUS_TAU")
    except (_Refusal, TypeError, ValueError, OverflowError, FloatingPointError) as error:
        return dict(status="UNAVAILABLE", reason=str(error) if isinstance(error, _Refusal)
            else "INVALID_QUERY_INPUT_OR_UNSUPPORTED_ARITHMETIC", **unavailable, diagnostics=statistics)
