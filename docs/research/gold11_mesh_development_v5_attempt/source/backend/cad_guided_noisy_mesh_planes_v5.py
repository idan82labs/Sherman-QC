"""Bounded raw-mesh CORE + fixed-cone CLOSURE proposals, research only.

Position-error balls are caller stipulations, not scanner calibration. A TLS
orientation with no all-row offset witness is WITNESS_NOT_FOUND, not proof of
infeasibility. Numerical enclosures are disclosed engineering estimates, not
directed-rounding certificates. No registration, CAD answer, IO or app imports.
"""

from collections import deque
from dataclasses import dataclass, fields
import hashlib
import json
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class NoisyMeshPlanePolicyV5:
    tile_triangles: tuple = (128, 1024)
    halo_rings: int = 1
    min_support_rows: int = 3
    max_normal_angle_radians: float = math.pi / 18.
    min_inplane_covariance_ratio: float = 1e-8


@dataclass(frozen=True)
class NoisyMeshPlaneBudgetsV5:
    max_raw_rows: int = 1_000_000
    max_triangles: int = 2_000_000
    max_edge_records: int = 6_000_000
    max_core_records: int = 65_536
    max_core_triangle_rows: int = 4096
    max_triangle_memberships: int = 20_000_000
    max_raw_memberships: int = 32_000_000
    max_full_support_row_tests: int = 64_000_000
    max_cone_row_tests: int = 16_000_000
    max_closure_triangle_tests: int = 32_000_000
    max_closure_edge_tests: int = 96_000_000
    max_fit_evaluations: int = 65_552
    max_closure_schedules: int = 16
    max_retained_array_bytes: int = 536_870_912
    max_metadata_bytes: int = 67_108_864
    chunk_rows: int = 65_536


class _Unavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unavailable(reason)


def _configuration(policy, budgets):
    _require(type(policy) is NoisyMeshPlanePolicyV5, "INVALID_NOISY_PLANE_POLICY_TYPE")
    _require(type(budgets) is NoisyMeshPlaneBudgetsV5, "INVALID_NOISY_PLANE_BUDGET_TYPE")
    gate = {f.name: getattr(policy, f.name) for f in fields(policy)}
    _require(type(gate["tile_triangles"]) is tuple
             and all(type(x) is int for x in gate["tile_triangles"])
             and gate["tile_triangles"] == (128, 1024)
             and type(gate["halo_rings"]) is int and gate["halo_rings"] == 1,
             "UNSUPPORTED_PROSPECTIVE_TILING_POLICY")
    _require(type(gate["min_support_rows"]) is int
             and 3 <= gate["min_support_rows"] <= 1_000_000, "INVALID_SUPPORT_POLICY")
    for name, high in (("max_normal_angle_radians", math.pi / 4.),
                       ("min_inplane_covariance_ratio", 1.)):
        x = gate[name]
        _require(type(x) in (int, float) and math.isfinite(x) and 0 < x <= high,
                 "INVALID_NOISY_PLANE_SCALAR_POLICY")
        gate[name] = float(x)
    maximum = NoisyMeshPlaneBudgetsV5()
    cap = {f.name: getattr(budgets, f.name) for f in fields(budgets)}
    _require(all(type(x) is int and 0 <= x <= getattr(maximum, name)
                 for name, x in cap.items()) and cap["chunk_rows"] > 0,
             "INVALID_NOISY_PLANE_BUDGET")
    return gate, cap


def _owned(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _digest(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _charge(stats, cap, field, count, maximum):
    _require(type(count) is int and count >= 0, "INVALID_INTERNAL_WORK_CHARGE")
    _require(stats[field] + count <= cap[maximum], maximum.upper() + "_EXCEEDED")
    stats[field] += count


def _retained(array, stats, cap):
    _charge(stats, cap, "retained_array_bytes", int(array.size)*8, "max_retained_array_bytes")
    return _owned(array, "<i8")


def _metadata(value):
    if isinstance(value, np.ndarray):
        return dict(shape=list(value.shape), dtype=value.dtype.str, sha256=_digest(value))
    if isinstance(value, dict):
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    return value


def _append(target, record, stats, cap):
    size = len(json.dumps(_metadata(record), allow_nan=False, separators=(",", ":")).encode())
    _charge(stats, cap, "metadata_bytes", size, "max_metadata_bytes")
    target.append(record)


def _base():
    return dict(schema_version="noisy_mesh_planes_v5", status="UNAVAILABLE", reasons=[],
        result_representation="ARRAY_NATIVE_IMMUTABLE_LINEAGE_NOT_JSON_REPORT",
        uncertainty_status="STIPULATED_UNCALIBRATED", numerical_status="ENGINEERING_NOT_CERTIFIED",
        search_exhaustive=False, physical_plane_count=None, observed_bend_count=None,
        production_use_allowed=False, accepted_registration=False,
        finite_support_representation="ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON",
        raw_coordinate_welding=False, residual_trimming=False, physical_flatness_verified=False,
        cores=[], closures=[], proposals=[], partitions=[], input_digests={},
        statistics=dict(input_raw_rows=0, input_triangles=0, edge_records=0,
            shared_edge_pairs=0, core_count=0, cores_evaluated=0, qualified_core_count=0,
            closures_scheduled=0, closures_evaluated=0, unique_closure_count=0,
            qualified_closure_count=0, unscheduled_qualified_core_count=0,
            triangle_memberships=0, raw_memberships=0, full_support_row_tests=0,
            cone_row_tests=0, closure_triangle_tests=0, closure_edge_tests=0,
            fit_evaluations=0, retained_array_bytes=0, metadata_bytes=0,
            core_construction_complete=False, core_evaluation_complete=False,
            scheduled_closure_evaluation_complete=False, all_qualified_core_closures_evaluated=False))


def _failure(result, error):
    stats = result["statistics"]
    stats["discarded_core_records"] = len(result["cores"])
    stats["discarded_closure_records"] = len(result["closures"])
    stats["discarded_proposals"] = len(result["proposals"])
    stats.update(core_construction_complete=False, core_evaluation_complete=False,
                 scheduled_closure_evaluation_complete=False,
                 all_qualified_core_closures_evaluated=False,
                 evidence_scope="ATTEMPTED_WORK_ONLY_ALL_PROPOSALS_INVALIDATED")
    result.update(status="UNAVAILABLE", reasons=[str(error) or type(error).__name__],
                  cores=[], closures=[], proposals=[], partitions=[])
    return result


def _inputs(points, triangle_raw_rows, error, cap):
    _require(type(points) is np.ndarray and points.ndim == 2 and points.shape[1:] == (3,)
             and points.dtype.kind in "fiu", "INVALID_RAW_POINTS")
    _require(type(triangle_raw_rows) is np.ndarray and triangle_raw_rows.ndim == 2
             and triangle_raw_rows.shape[1:] == (3,) and triangle_raw_rows.dtype.kind in "iu",
             "INVALID_RAW_TRIANGLES")
    n, m = len(points), len(triangle_raw_rows)
    _require(3 <= n <= cap["max_raw_rows"] and 1 <= m <= cap["max_triangles"], "RAW_INPUT_CAP_EXCEEDED")
    _require(3*m <= cap["max_edge_records"], "RAW_EDGE_CAP_EXCEEDED")
    _require(np.all(triangle_raw_rows >= 0) and np.all(triangle_raw_rows < n), "RAW_INDEX_OUT_OF_RANGE")
    if type(error) in (int, float):
        _require(math.isfinite(error) and 0 <= error <= 1000., "INVALID_POSITION_ERROR_BOUND")
        epsilon = np.full(n, float(error), dtype="<f8")
    else:
        _require(type(error) is np.ndarray and error.ndim == 1 and len(error) == n
                 and error.dtype.kind in "fiu", "INVALID_POSITION_ERROR_ARRAY")
        epsilon = error
    p, t, epsilon = _owned(points, "<f8"), _owned(triangle_raw_rows, "<i8"), _owned(epsilon, "<f8")
    # Early caller-array checks only bound preparation. Revalidate the owned
    # snapshots before any indexing: mutation during an earlier snapshot must
    # not introduce negative NumPy aliases or change the admitted dimensions.
    _require(p.shape == (n, 3) and t.shape == (m, 3) and epsilon.shape == (n,),
             "OWNED_INPUT_SHAPE_CHANGED")
    _require(np.all(t >= 0) and np.all(t < len(p)), "OWNED_RAW_INDEX_OUT_OF_RANGE")
    _require(np.isfinite(p).all() and np.max(np.abs(p)) <= 1e6, "UNSUPPORTED_POSITION_RANGE")
    _require(np.isfinite(epsilon).all() and np.all(epsilon >= 0.) and np.all(epsilon <= 1000.),
             "INVALID_POSITION_ERROR_BOUND")
    _require(np.all(t[:, 0] != t[:, 1]) and np.all(t[:, 0] != t[:, 2])
             and np.all(t[:, 1] != t[:, 2]), "REPEATED_RAW_TRIANGLE_VERTEX")
    for start in range(0, m, cap["chunk_rows"]):
        xyz = p[t[start:start+cap["chunk_rows"]]]
        a, b = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
        area2 = np.linalg.norm(np.cross(a, b), axis=1)
        edge2 = np.maximum.reduce((np.sum(a*a, axis=1), np.sum(b*b, axis=1), np.sum((a-b)**2, axis=1)))
        _require(np.isfinite(area2).all() and np.all(area2 > 1e-12*edge2) and np.all(area2 > 0.),
                 "DEGENERATE_OR_UNSTABLE_RAW_TRIANGLE")
    return p, t, epsilon


def _adjacency(triangles, vertex_count, stats):
    m = len(triangles)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    keys = edges[:, 0]*vertex_count+edges[:, 1]
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1])+1]
    counts = np.diff(np.r_[starts, len(order)])
    _require(np.all(counts <= 2), "NONMANIFOLD_SHARED_RAW_EDGE_FAN")
    pairs = starts[counts == 2]
    left, right = order[pairs] % m, order[pairs+1] % m
    graph = coo_matrix((np.ones(len(left), dtype=np.uint8), (left, right)), shape=(m, m))
    graph = (graph+graph.T).tocsr()
    graph.sort_indices()
    stats.update(edge_records=3*m, shared_edge_pairs=len(left))
    return graph, left, right


def _tile_partition(graph, size, maximum_blocks):
    """Prospective BFS blocks; the quota partitions topology, never residuals.

    Lowest original triangle row breaks seed/frontier ties. All remaining rows
    form later blocks; no support disappears when a block reaches its size.
    """
    labels = np.full(graph.shape[0], -1, dtype=np.int64)
    blocks = []
    for seed in range(len(labels)):
        if labels[seed] >= 0:
            continue
        _require(len(blocks) < maximum_blocks, "CORE_RECORD_BUDGET_EXCEEDED")
        queue, rows = deque([seed]), []
        while queue and len(rows) < size:
            row = queue.popleft()
            if labels[row] >= 0:
                continue
            labels[row] = len(blocks)
            rows.append(row)
            for neighbor in graph.indices[graph.indptr[row]:graph.indptr[row+1]]:
                if labels[neighbor] < 0:
                    queue.append(int(neighbor))
        blocks.append(np.asarray(sorted(rows), dtype=np.int64))
    return labels, blocks


def _analyze_support(points, epsilon, gate):
    """Private exact-source math seam; validated finite owned arrays in mm.

    Covariance uses each original raw identity once, including equal positions
    with different identities. Numerical allowances NEVER modify epsilon.
    """
    result = dict(status="REJECTED", reasons=[], raw_support_count=len(points),
        witness_status="NOT_EVALUATED", cone_status="UNRESOLVED", normal=None,
        observed_centroid_mm=None, witness_anchor_mm=None, witness_offset_from_anchor_mm=None,
        normal_angle_bound_radians=None, same_physical_plane_verified=False,
        uncertainty_status="STIPULATED_UNCALIBRATED",
        fit_measure="UNIFORM_ORIGINAL_RAW_IDENTITIES_NOT_XYZ_DEDUPLICATION",
        numerical_bound_status="ENGINEERING_ENCLOSURE_NOT_DIRECTED_ROUNDING_CERTIFICATE")
    if len(points) < gate["min_support_rows"]:
        result["reasons"] = ["INSUFFICIENT_RAW_SUPPORT"]
        return result
    anchor = points[0]
    delta = points-anchor
    scale = float(np.max(np.abs(delta)))
    if not math.isfinite(scale) or scale <= 1e-12:
        result["reasons"] = ["UNRESOLVED_SUPPORT_EXTENT"]
        return result
    local = delta/scale
    mean = np.mean(local, axis=0)
    centered = local-mean
    covariance = centered.T@centered/len(points)
    covariance = .5*(covariance+covariance.T)
    eigenvalues, axes = np.linalg.eigh(covariance)
    _require(np.isfinite(eigenvalues).all() and np.isfinite(axes).all(), "NONFINITE_EIGENSYSTEM")
    normal = axes[:, 0]/np.linalg.norm(axes[:, 0])
    if normal[np.argmax(np.abs(normal))] < 0.:
        normal = -normal
    u = np.finfo(float).eps
    gamma = ((len(points)+8)*u)/(1.-(len(points)+8)*u)
    coordinate_guard = 128*u*(float(np.max(np.abs(points)))+scale)
    centroid_guard = coordinate_guard+(gamma+8*u)*scale*float(np.max(np.linalg.norm(local, axis=1)))
    local_guard = coordinate_guard/scale
    residual = float(np.linalg.norm(covariance@axes-axes*eigenvalues, ord=2))
    orthogonality = float(np.linalg.norm(axes.T@axes-np.eye(3), ord=2))
    covariance_arithmetic_guard = (
        64*local_guard*(float(np.max(np.linalg.norm(centered, axis=1)))+local_guard)
        + (8*gamma+512*u)*max(1., float(np.linalg.norm(covariance, ord=2))))
    covariance_guard = (covariance_arithmetic_guard + residual
                       + orthogonality*max(1., float(np.max(np.abs(eigenvalues)))))
    _require(math.isfinite(covariance_guard) and covariance_guard < 1e-4,
             "NUMERICAL_COVARIANCE_RESOLUTION_UNSUPPORTED")
    centroid = anchor+scale*mean
    projected = delta@normal
    low = float(np.max(projected-epsilon))
    high = float(np.min(projected+epsilon))
    projection_guard = 8*coordinate_guard+128*u*(float(np.max(np.abs(projected)))+float(np.max(epsilon)))
    offset = low+.5*(high-low)
    distances = np.abs(projected-offset)
    margin = float(np.min(epsilon-distances))
    # An exact coordinate plane is an algebraic witness over the represented
    # doubles; do not invent epsilon to make other touching intervals pass.
    axis = int(np.argmax(np.abs(normal)))
    exact_coordinate_plane = bool(np.count_nonzero(normal) == 1 and abs(normal[axis]) == 1.
                                  and np.all(points[:, axis] == points[0, axis]))
    if low > high+2*projection_guard:
        witness = "WITNESS_NOT_FOUND"
    elif margin > projection_guard or (exact_coordinate_plane and low <= high and margin >= 0.):
        witness = "FOUND"
    else:
        witness = "NUMERICALLY_UNRESOLVED_OFFSET_INTERVAL"
    ebar = float(np.mean(epsilon))
    b_squared = float(np.mean(((epsilon+ebar)/scale)**2))
    # Positive-sum rounding allowance is separate from the stipulated errors.
    # A length-dependent gamma also covers a non-pairwise reduction; no 1/sqrt(N)
    # statistical improvement is assumed. Enclosures remain engineering bounds.
    ebar_guard = (gamma+8*u)*float(np.max(epsilon))
    b_upper_terms = ((epsilon+ebar+ebar_guard)/scale)**2
    b_upper_mean = float(np.mean(b_upper_terms))
    b_guard = (8*gamma+128*u)*max(b_upper_mean, np.finfo(float).tiny)
    b_upper = b_upper_mean+b_guard
    gap_lower = float(eigenvalues[1]-eigenvalues[0]-2*covariance_guard)
    numerator_upper = b_upper-float(eigenvalues[0])+covariance_guard
    # The covariance-to-cone formula concerns the true minimum eigendirection.
    # Account separately for the COMPUTED normal: projecting (C-lambda0 I)n
    # into the other eigenspaces bounds its sine error by residual/separation.
    computed_normal_residual = float(np.linalg.norm(covariance@normal-eigenvalues[0]*normal))
    normal_residual_upper = (computed_normal_residual+covariance_arithmetic_guard
                            + 256*u*max(1., float(np.linalg.norm(covariance, ord=2))))
    normal_separation_lower = float(eigenvalues[1]-covariance_guard-eigenvalues[0])
    normal_numeric_angle = None
    if normal_separation_lower > 0. and normal_residual_upper < normal_separation_lower:
        normal_numeric_angle = math.asin(normal_residual_upper/normal_separation_lower)
    cone_reasons = []
    alpha = None
    if gap_lower <= 0. or numerator_upper < 0. or normal_numeric_angle is None:
        cone_reasons.append("CONDITIONAL_CONE_NUMERICALLY_OR_GEOMETRICALLY_UNRESOLVED")
    else:
        ratio = numerator_upper/gap_lower
        if ratio >= 1.:
            cone_reasons.append("CONDITIONAL_NORMAL_CONE_UNRESOLVED")
        else:
            alpha = math.asin(math.sqrt(ratio))+normal_numeric_angle
            if alpha > gate["max_normal_angle_radians"]:
                cone_reasons.append("CONDITIONAL_NORMAL_CONE_EXCEEDS_POLICY")
    ratio_lower = max(0., float(eigenvalues[1])-covariance_guard)/max(
        np.finfo(float).tiny, float(eigenvalues[2])+covariance_guard)
    if ratio_lower < gate["min_inplane_covariance_ratio"]:
        cone_reasons.append("INPLANE_SUPPORT_UNRESOLVED")
    result.update(witness_status=witness, cone_status="AVAILABLE" if not cone_reasons else "UNRESOLVED",
        normal=normal.tolist(), observed_centroid_mm=centroid.tolist(), witness_anchor_mm=anchor.tolist(),
        witness_offset_from_anchor_mm=offset, offset_interval_mm=[low, high],
        full_support_max_abs_residual_mm=float(np.max(distances)),
        raw_support_rms_mm=float(np.sqrt(np.mean(distances*distances))),
        minimum_error_ball_margin_mm=margin, mean_position_error_bound_mm=ebar,
        normal_angle_bound_radians=alpha, covariance_mm2=(covariance*scale**2).tolist(),
        covariance_eigenvalues_mm2=(eigenvalues*scale**2).tolist(),
        cone_B_squared_mm2=b_squared*scale**2, cone_gap_lower_mm2=gap_lower*scale**2,
        cone_B_squared_upper_mm2=b_upper*scale**2,
        cone_numerator_upper_mm2=numerator_upper*scale**2,
        inplane_covariance_ratio_lower=ratio_lower,
        numerical=dict(coordinate_guard_mm=coordinate_guard, projection_guard_mm=projection_guard,
            centroid_guard_mm=centroid_guard, positive_sum_gamma=gamma,
            covariance_guard_mm2=covariance_guard*scale**2,
            covariance_arithmetic_guard_mm2=covariance_arithmetic_guard*scale**2,
            eigensystem_residual_scaled=residual, orthogonality_residual=orthogonality,
            mean_error_bound_guard_mm=ebar_guard, B_squared_rounding_guard_mm2=b_guard*scale**2,
            computed_normal_residual_scaled=computed_normal_residual,
            computed_normal_residual_upper_scaled=normal_residual_upper,
            computed_normal_separation_lower_scaled=normal_separation_lower,
            computed_normal_angle_guard_radians=normal_numeric_angle,
            position_error_bound_was_increased=False, exact_coordinate_plane_witness=exact_coordinate_plane))
    result["reasons"] = ([witness] if witness != "FOUND" else [])+cone_reasons
    if not result["reasons"]:
        result["status"] = "PROPOSED"
    _require(all(math.isfinite(x) for x in (low, high, offset, margin, b_squared, ebar, b_upper, ebar_guard))
             and np.isfinite(centroid).all(), "NONFINITE_SUPPORT_ANALYSIS")
    return result


def _fit(points, epsilon, rows, gate, cap, stats):
    _charge(stats, cap, "fit_evaluations", 1, "max_fit_evaluations")
    _charge(stats, cap, "full_support_row_tests", len(rows), "max_full_support_row_tests")
    return _analyze_support(points[rows], epsilon[rows], gate)


def _support_rows(triangles, rows, stats, cap):
    raw = np.unique(triangles[rows].reshape(-1))  # IDs only; never XYZ.
    _charge(stats, cap, "raw_memberships", len(raw), "max_raw_memberships")
    return _retained(raw, stats, cap)


def _schedule(cores, maximum):
    """Declared incomplete schedule, independent of closure/registration results.

    Alternate scales. Within a scale prefer unoriented normal novelty, then
    observed-centroid spread, then smaller cone and original core ID. Centroids
    rank exploration in ONE frame; they are never matched across frames.
    """
    eligible = [row for row in cores if row["analysis"]["status"] == "PROPOSED"]
    if not eligible or maximum == 0:
        return []
    normals = np.asarray([row["analysis"]["normal"] for row in eligible])
    centers = np.asarray([row["analysis"]["observed_centroid_mm"] for row in eligible])
    alphas = np.asarray([row["analysis"]["normal_angle_bound_radians"] for row in eligible])
    ids = np.asarray([row["core_id"] for row in eligible])
    scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), np.finfo(float).tiny)
    available, chosen = np.ones(len(eligible), dtype=bool), []
    for slot in range(min(maximum, len(eligible))):
        family = (128, 1024)[slot % 2]
        pool = np.flatnonzero(available & np.asarray([row["tile_triangles"] == family for row in eligible]))
        if not len(pool):
            pool = np.flatnonzero(available)
        if chosen:
            novelty = 1.-np.max(np.abs(normals[pool]@normals[chosen].T), axis=1)
            spread = np.min(np.linalg.norm(centers[pool, None]-centers[None, chosen], axis=2), axis=1)/scale
        else:
            novelty, spread = np.zeros(len(pool)), np.zeros(len(pool))
        order = np.lexsort((ids[pool], alphas[pool], -spread, -novelty))
        chosen.append(int(pool[order[0]]))
        available[chosen[-1]] = False
    return [int(ids[index]) for index in chosen]


def _cone_mask(points, epsilon, analysis, cap, stats):
    """Necessary per-row possibility, NOT a jointly feasible plane witness."""
    normal = np.asarray(analysis["normal"])
    center = np.asarray(analysis["observed_centroid_mm"])
    alpha, ebar = analysis["normal_angle_bound_radians"], analysis["mean_position_error_bound_mm"]
    allowed = np.empty(len(points), dtype=bool)
    boundary, largest_guard = 0, 0.
    for start in range(0, len(points), cap["chunk_rows"]):
        stop = min(len(points), start+cap["chunk_rows"])
        _charge(stats, cap, "cone_row_tests", stop-start, "max_cone_row_tests")
        v = points[start:stop]-center
        signed = v@normal
        tangent = np.linalg.norm(v-signed[:, None]*normal, axis=1)
        lower = np.maximum(0., np.abs(signed)*math.cos(alpha)-tangent*math.sin(alpha))
        guard = (8*analysis["numerical"]["centroid_guard_mm"]
                 + analysis["numerical"]["mean_error_bound_guard_mm"]
                 + 512*np.finfo(float).eps*(float(np.max(np.abs(points[start:stop])))
                                          + float(np.max(np.linalg.norm(v, axis=1)))))
        threshold = epsilon[start:stop]+ebar
        _require(np.isfinite(lower).all() and math.isfinite(guard), "NONFINITE_CONE_PROPOSAL")
        # Keep arithmetically uncertain POSSIBLE rows. Final fixed-normal witness
        # has a separate fail-closed interval test and does not enlarge epsilon.
        allowed[start:stop] = lower-guard <= threshold
        boundary += int(np.count_nonzero(np.abs(lower-threshold) <= guard))
        largest_guard = max(largest_guard, guard)
    return allowed, dict(possible_raw_rows=int(np.count_nonzero(allowed)),
        numerical_boundary_sensitive_rows=boundary, maximum_arithmetic_guard_mm=largest_guard,
        pointwise_possibility_only=True, joint_compatibility_verified=False,
        position_error_bound_was_increased=False)


def extract_noisy_scan_plane_proposals_v5(points, triangle_raw_rows, *, position_error_bounds_mm,
        policy=NoisyMeshPlanePolicyV5(), budgets=NoisyMeshPlaneBudgetsV5()):
    """Construct every declared core; retain scheduled closures as alternatives.

    Arrays and all policies are revalidated/snapshotted. Equal XYZ rows remain
    distinct. The output is an internal array payload with no IO/native access.
    """
    result = _base()
    stats = result["statistics"]
    try:
        gate, cap = _configuration(policy, budgets)
        result.update(policy=gate, budgets=cap, unit="mm", coordinate_frame="RAW_SCAN_UNREGISTERED",
            unit_assumption="PROVISIONAL_ARRAY_MM_UNVERIFIED",
            closure_schedule_policy="ALTERNATING_TILE_SCALE_NORMAL_NOVELTY_THEN_CENTROID_SPREAD",
            tiling_tie_policy="ORIGINAL_TRIANGLE_ROW_ORDER_NOT_ROW_PERMUTATION_INVARIANT")
        p, t, epsilon = _inputs(points, triangle_raw_rows, position_error_bounds_mm, cap)
        stats.update(input_raw_rows=len(p), input_triangles=len(t))
        result["input_digests"] = dict(points_sha256=_digest(p), triangle_raw_rows_sha256=_digest(t),
                                        position_error_bounds_mm_sha256=_digest(epsilon))
        graph, left, right = _adjacency(t, len(p), stats)
        referenced = np.zeros(len(p), dtype=bool)
        referenced[t.reshape(-1)] = True
        unreferenced = _retained(np.flatnonzero(~referenced), stats, cap)
        result["raw_input_accounting"] = dict(referenced_raw_row_count=int(np.count_nonzero(referenced)),
            unreferenced_raw_vertex_rows=unreferenced,
            unreferenced_raw_vertex_rows_sha256=_digest(unreferenced),
            unreferenced_disposition="INPUT_RETAINED_NOT_TRIANGLE_SUPPORT")
        for size in gate["tile_triangles"]:
            labels, blocks = _tile_partition(graph, size, cap["max_core_records"]-len(result["cores"]))
            _require(len(result["cores"])+len(blocks) <= cap["max_core_records"], "CORE_RECORD_BUDGET_EXCEEDED")
            first_core_id = len(result["cores"])
            label_rows = _retained(labels+first_core_id, stats, cap)
            _append(result["partitions"], dict(tile_triangles=size, triangle_core_labels=label_rows,
                triangle_core_labels_sha256=_digest(label_rows)), stats, cap)
            for base in blocks:
                neighbors = [graph.indices[graph.indptr[row]:graph.indptr[row+1]] for row in base]
                rows = np.unique(np.concatenate([base, *neighbors]))
                _require(len(rows) <= cap["max_core_triangle_rows"], "CORE_TRIANGLE_ROW_BUDGET_EXCEEDED")
                _charge(stats, cap, "triangle_memberships", len(base)+len(rows), "max_triangle_memberships")
                base_rows, rows = _retained(base, stats, cap), _retained(rows, stats, cap)
                raw = _support_rows(t, rows, stats, cap)
                analysis = _fit(p, epsilon, raw, gate, cap, stats)
                core_id = len(result["cores"])
                proposed = analysis["status"] == "PROPOSED"
                record = dict(core_id=core_id, tile_triangles=size, base_triangle_rows=base_rows,
                    triangle_rows=rows, raw_vertex_rows=raw, triangle_rows_sha256=_digest(rows),
                    raw_vertex_rows_sha256=_digest(raw), analysis=analysis,
                    closure_disposition="NOT_SCHEDULED_BY_POLICY" if proposed else "CORE_NOT_QUALIFIED",
                    closure_id=None)
                _append(result["cores"], record, stats, cap)
                stats["cores_evaluated"] += 1
                stats["qualified_core_count"] += int(proposed)
                if proposed:
                    _append(result["proposals"], dict(kind="CORE", record_id=core_id), stats, cap)
        stats.update(core_count=len(result["cores"]), core_construction_complete=True, core_evaluation_complete=True)
        schedule = _schedule(result["cores"], cap["max_closure_schedules"])
        result["scheduled_core_ids"] = schedule
        stats["closures_scheduled"] = len(schedule)
        stats["unscheduled_qualified_core_count"] = stats["qualified_core_count"]-len(schedule)
        seen = {}
        for core_id in schedule:
            core = result["cores"][core_id]
            allowed, cone = _cone_mask(p, epsilon, core["analysis"], cap, stats)
            _charge(stats, cap, "closure_triangle_tests", len(t), "max_closure_triangle_tests")
            eligible = np.all(allowed[t], axis=1)
            _require(np.all(eligible[core["triangle_rows"]]), "NUMERICAL_SEED_CONE_SELF_EXCLUSION")
            _charge(stats, cap, "closure_edge_tests", len(left), "max_closure_edge_tests")
            edge = eligible[left] & eligible[right]
            closure_graph = coo_matrix((np.ones(int(np.count_nonzero(edge)), dtype=np.uint8),
                (left[edge], right[edge])), shape=(len(t), len(t))).tocsr()
            _, labels = connected_components(closure_graph, directed=False, return_labels=True)
            group = labels[int(core["triangle_rows"][0])]
            _require(np.all(labels[core["triangle_rows"]] == group), "CORE_TOPOLOGY_INCONSISTENT")
            rows = np.flatnonzero((labels == group) & eligible)
            key = _digest(np.asarray(rows, dtype="<i8"))
            # Hash narrows lookup; exact original triangle-ID equality decides.
            duplicate = next((index for index in seen.get(key, [])
                              if np.array_equal(result["closures"][index]["triangle_rows"], rows)), None)
            if duplicate is not None:
                closure = result["closures"][duplicate]
                closure["source_core_ids"].append(core_id)
                core.update(closure_disposition="EXACT_SUPPORT_DUPLICATE", closure_id=duplicate)
            else:
                _charge(stats, cap, "triangle_memberships", len(rows), "max_triangle_memberships")
                rows = _retained(rows, stats, cap)
                raw = _support_rows(t, rows, stats, cap)
                analysis = _fit(p, epsilon, raw, gate, cap, stats)
                closure_id = len(result["closures"])
                record = dict(closure_id=closure_id, source_core_ids=[core_id], triangle_rows=rows,
                    raw_vertex_rows=raw, triangle_rows_sha256=_digest(rows), raw_vertex_rows_sha256=_digest(raw),
                    analysis=analysis)
                _append(result["closures"], record, stats, cap)
                seen.setdefault(key, []).append(closure_id)
                proposed = analysis["status"] == "PROPOSED"
                core.update(closure_disposition="PROPOSED" if proposed else "WHOLE_CLOSURE_REJECTED_CORE_RETAINED",
                            closure_id=closure_id)
                stats["qualified_closure_count"] += int(proposed)
                if proposed:
                    _append(result["proposals"], dict(kind="CLOSURE", record_id=closure_id), stats, cap)
            core["cone_proposal"] = cone
            stats["closures_evaluated"] += 1
        stats.update(unique_closure_count=len(result["closures"]), scheduled_closure_evaluation_complete=True,
                     all_qualified_core_closures_evaluated=stats["unscheduled_qualified_core_count"] == 0)
        result.update(status="COMPLETE", reasons=[] if result["proposals"] else ["NO_PROPOSAL_WITHIN_DECLARED_POLICY"])
        # Include fields added after initial record admission. Arrays remain
        # arrays; only bounded metadata is serialized for this final check.
        for _ in range(3):
            metadata_bytes = len(json.dumps(_metadata(result), allow_nan=False, separators=(",", ":")).encode())
            _require(metadata_bytes <= cap["max_metadata_bytes"], "FINAL_METADATA_BUDGET_EXCEEDED")
            if stats["metadata_bytes"] == metadata_bytes:
                break
            stats["metadata_bytes"] = metadata_bytes
        _require(stats["metadata_bytes"] == len(json.dumps(_metadata(result), allow_nan=False,
                 separators=(",", ":")).encode()), "METADATA_SIZE_ACCOUNTING_UNRESOLVED")
        return result
    except Exception as error:
        return _failure(result, error)
