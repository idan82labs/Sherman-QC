"""Near-roundoff coplanar mesh proposals, not noisy-plane qualification.

Scan adjacency uses shared ORIGINAL vertex-row edges only. Pairwise compatible
triangles are compressed once; each whole component is then area-TLS fitted and
checked against ALL its original support vertices. A failed curved chain is
not trimmed, subdivided or salvaged. Other independently valid components stay.
CAD uses every triangle of each import-local face, not analytic native planes.

Finite support is the retained triangle union, including holes/disconnections;
no convex hull, coordinate welding, repair, material-side or flange label exists.
"""

from dataclasses import dataclass, fields
import hashlib
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class CoplanarPatchPolicyV4:
    max_pair_normal_sine: float = 1e-8
    max_pair_anchored_distance_mm: float = 1e-8
    max_full_support_distance_mm: float = 1e-8
    min_support_vertices: int = 12
    min_second_extent_mm: float = .25
    min_inplane_covariance_ratio: float = 1e-8


@dataclass(frozen=True)
class CoplanarPatchBudgetsV4:
    max_vertices: int = 1_000_000
    max_triangles: int = 2_000_000
    max_edge_records: int = 6_000_000
    max_components: int = 65536
    max_output_patches: int = 4096
    max_support_memberships: int = 6_000_000
    max_full_vertex_tests: int = 12_000_000
    chunk_triangles: int = 65536


class _Unavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unavailable(reason)


def _configuration(policy, budgets):
    _require(type(budgets) is CoplanarPatchBudgetsV4, "INVALID_PATCH_BUDGET_TYPE")
    maxima = CoplanarPatchBudgetsV4()
    cap = {field.name: getattr(budgets, field.name) for field in fields(maxima)}
    _require(all(type(value) is int and 0 <= value <= getattr(maxima, name)
                 for name, value in cap.items()) and cap["chunk_triangles"] > 0,
             "INVALID_PATCH_BUDGET")
    _require(type(policy) is CoplanarPatchPolicyV4, "INVALID_PATCH_POLICY_TYPE")
    gate = {field.name: getattr(policy, field.name) for field in fields(policy)}
    _require(type(gate["min_support_vertices"]) is int
             and 3 <= gate["min_support_vertices"] <= 1_000_000, "INVALID_PATCH_SUPPORT_POLICY")
    for name, high in (("max_pair_normal_sine", 1e-4),
                       ("max_pair_anchored_distance_mm", 1e-3),
                       ("max_full_support_distance_mm", 1e-3),
                       ("min_second_extent_mm", 1e6),
                       ("min_inplane_covariance_ratio", 1.)):
        value = gate[name]
        _require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= high,
                 "INVALID_NEAR_ROUNDOFF_PATCH_POLICY")
        gate[name] = float(value)
    return gate, cap


def _immutable(value, dtype):
    owned = np.asarray(value, dtype=dtype)
    return np.frombuffer(owned.tobytes(order="C"), dtype=dtype).reshape(owned.shape)


def _digest(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _array_shape(value, tail, kinds, reason):
    _require(type(value) is np.ndarray and value.ndim == len(tail)+1
             and value.shape[1:] == tail and value.dtype.kind in kinds, reason)


def _input_caps(count, cap, *, scan):
    _require(1 <= count <= cap["max_triangles"], "PATCH_TRIANGLE_BUDGET_EXCEEDED")
    _require(3*count <= cap["max_support_memberships"], "PATCH_MEMBERSHIP_BUDGET_EXCEEDED")
    _require(3*count <= cap["max_full_vertex_tests"], "PATCH_VERTEX_TEST_BUDGET_EXCEEDED")
    if scan:
        _require(3*count <= cap["max_edge_records"], "PATCH_EDGE_BUDGET_EXCEEDED")


def _base(mode):
    return dict(schema_version="mesh_plane_proposals_v4", status="UNAVAILABLE", reasons=[],
                search_scope="NEAR_ROUNDOFF_MESH_COPLANAR_PROPOSALS_ONLY", mode=mode,
                completeness_for_noisy_planes=False, accepted_registration=False,
                production_use_allowed=False, material_topology_verified=False,
                finite_support_representation="ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON",
                plane_classification="TRIANGULATED_SUPPORT_FIT_NOT_ANALYTIC_OR_FLANGE_CLASS",
                patches=[], components=[], triangle_component_labels=None, digests={},
                statistics=dict(input_triangles=0, edge_records=0, shared_edge_pairs=0,
                    compatible_edge_pairs=0, component_count=0, components_evaluated=0,
                    qualified_patch_count=0, rejected_component_count=0,
                    support_memberships=0, full_vertex_tests=0, component_search_complete=False))


def _fail(result, error):
    statistics = result["statistics"]
    statistics["qualified_patches_before_fatal_clear"] = statistics["qualified_patch_count"]
    statistics["qualified_patch_count"] = 0
    statistics["component_search_complete"] = False
    statistics["evidence_scope"] = "ATTEMPTED_WORK_ONLY_NOT_AVAILABLE_COMPONENTS"
    result.update(status="UNAVAILABLE", reasons=[str(error)], patches=[], components=[],
                  triangle_component_labels=None)
    result["digests"].pop("triangle_component_labels_sha256", None)
    return result


def _geometry(get_triangles, count, chunk):
    normals, areas = np.empty((count, 3)), np.empty(count)
    for start in range(0, count, chunk):
        stop = min(count, start+chunk)
        triangles = get_triangles(slice(start, stop))
        _require(np.isfinite(triangles).all() and np.max(np.abs(triangles)) <= 1e6,
                 "INVALID_PATCH_POSITION")
        a, b = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        cross = np.cross(a, b)
        length = np.linalg.norm(cross, axis=1)
        edge_squared = np.maximum.reduce((np.sum(a*a, axis=1), np.sum(b*b, axis=1),
                                          np.sum((a-b)**2, axis=1)))
        _require(np.isfinite(length).all() and np.all(length > 1e-12*edge_squared)
                 and np.all(length > 0.), "DEGENERATE_OR_UNSTABLE_PATCH_TRIANGLE")
        normals[start:stop], areas[start:stop] = cross/length[:, None], .5*length
    return normals, areas


def _scan_components(points, triangles, normals, gate, cap, statistics):
    count, vertices = len(triangles), len(points)
    # int64 is sufficient under the exact vertex/input caps, before allocation.
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    keys = edges[:, 0]*vertices+edges[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])+1]
    sizes = np.diff(np.r_[starts, len(order)])
    statistics["edge_records"] = 3*count
    _require(np.all(sizes <= 2), "NONMANIFOLD_SHARED_RAW_EDGE_FAN")
    pair_starts = starts[sizes == 2]
    left, right = order[pair_starts] % count, order[pair_starts+1] % count
    shared = edges[order[pair_starts]]
    statistics.update(edge_records=3*count, shared_edge_pairs=len(left))
    compatible = np.zeros(len(left), dtype=bool)
    chunk = cap["chunk_triangles"]
    for start in range(0, len(left), chunk):
        sl = slice(start, min(len(left), start+chunk))
        i, j, edge = left[sl], right[sl], shared[sl]
        anchor = points[edge[:, 0]]
        opposite_i = np.sum(triangles[i], axis=1)-np.sum(edge, axis=1)
        opposite_j = np.sum(triangles[j], axis=1)-np.sum(edge, axis=1)
        sine = np.linalg.norm(np.cross(normals[i], normals[j]), axis=1)
        distance_i = np.abs(np.sum(normals[i]*(points[opposite_j]-anchor), axis=1))
        distance_j = np.abs(np.sum(normals[j]*(points[opposite_i]-anchor), axis=1))
        compatible[sl] = ((sine <= gate["max_pair_normal_sine"])
                          & (distance_i <= gate["max_pair_anchored_distance_mm"])
                          & (distance_j <= gate["max_pair_anchored_distance_mm"]))
    statistics["compatible_edge_pairs"] = int(np.count_nonzero(compatible))
    graph = coo_matrix((np.ones(statistics["compatible_edge_pairs"], dtype=np.uint8),
                        (left[compatible], right[compatible])), shape=(count, count)).tocsr()
    group_count, labels = connected_components(graph, directed=False, return_labels=True)
    statistics["component_count"] = int(group_count)
    _require(group_count <= cap["max_components"], "PATCH_COMPONENT_BUDGET_EXCEEDED")
    return labels.astype(np.int64, copy=False), int(group_count)


def _area_fits(get_triangles, areas, labels, group_count, chunk):
    """Exact triangle-area moments, evaluated about local anchors in two passes.

    Uniform triangle barycentrics have E[lambda_i^2]=1/6 and
    E[lambda_i lambda_j]=1/12. Thus for q_i about a FIXED anchor and s=sum q_i:
      E[q]=s/3; E[qq.T]=(sum q_i q_i.T+s s.T)/12.
    Area-weight these moments. The second pass recenters near the area centroid
    before covariance subtraction; world-origin second moments are never used.
    """
    first = np.full(group_count, len(labels), dtype=np.int64)
    np.minimum.at(first, labels, np.arange(len(labels)))
    anchors = get_triangles(first)[:, 0].copy()
    total_area, first_moment = np.zeros(group_count), np.zeros((group_count, 3))
    for start in range(0, len(labels), chunk):
        sl = slice(start, min(len(labels), start+chunk))
        group, weight = labels[sl], areas[sl]
        q = get_triangles(sl)-anchors[group, None, :]
        np.add.at(total_area, group, weight)
        np.add.at(first_moment, group, weight[:, None]*np.sum(q, axis=1)/3.)
    _require(np.isfinite(total_area).all() and np.all(total_area > 0.), "INVALID_PATCH_AREA")
    centers = anchors+first_moment/total_area[:, None]
    centered_first, second_moment = np.zeros_like(first_moment), np.zeros((group_count, 3, 3))
    for start in range(0, len(labels), chunk):
        sl = slice(start, min(len(labels), start+chunk))
        group, weight = labels[sl], areas[sl]
        q = get_triangles(sl)-centers[group, None, :]
        summed = np.sum(q, axis=1)
        moment = (np.einsum("mki,mkj->mij", q, q)
                  + summed[:, :, None]*summed[:, None, :])/12.
        np.add.at(centered_first, group, weight[:, None]*summed/3.)
        np.add.at(second_moment, group, weight[:, None, None]*moment)
    mean = centered_first/total_area[:, None]
    covariance = second_moment/total_area[:, None, None]-mean[:, :, None]*mean[:, None, :]
    covariance = .5*(covariance+np.swapaxes(covariance, 1, 2))
    centers += mean
    _require(np.isfinite(covariance).all() and np.isfinite(centers).all(), "NONFINITE_PATCH_MOMENTS")
    eigenvalues, axes = np.linalg.eigh(covariance)
    _require(np.isfinite(eigenvalues).all() and np.isfinite(axes).all()
             and np.all(eigenvalues[:, 2] > 0.)
             and np.all(eigenvalues[:, 0] >= -256*np.finfo(float).eps*eigenvalues[:, 2]),
             "NUMERICALLY_INVALID_PATCH_COVARIANCE")
    normals = axes[:, :, 0].copy()
    sign = np.where(normals[np.arange(group_count), np.argmax(np.abs(normals), axis=1)] < 0., -1., 1.)
    normals *= sign[:, None]
    return dict(area=total_area, center=centers, covariance=covariance,
                eigenvalues=eigenvalues, normals=normals, extent_axes=axes[:, :, [2, 1]])


def _finish(result, get_triangles, areas, labels, group_count, gate, cap, *,
            points=None, triangles=None, face_ids=None, local_triangle_ids=None):
    scan = points is not None
    statistics = result["statistics"]
    statistics["component_count"] = group_count
    labels = _immutable(labels, "<i8")
    result["triangle_component_labels"] = labels
    result["digests"]["triangle_component_labels_sha256"] = _digest(labels)
    fits = _area_fits(get_triangles, areas, labels, group_count, cap["chunk_triangles"])
    if scan:
        # One membership per (component, ORIGINAL raw vertex row), not XYZ weld.
        membership = np.unique((labels[:, None]*len(points)+triangles).reshape(-1))
        member_groups, member_rows = membership//len(points), membership % len(points)
    else:
        # CAD triangles have no shared native vertex-row identity in this API.
        member_groups = np.repeat(labels, 3)
        member_rows = np.arange(3*len(labels), dtype=np.int64)
    statistics["support_memberships"] = len(member_rows)
    _require(len(member_rows) <= cap["max_support_memberships"], "PATCH_MEMBERSHIP_BUDGET_EXCEEDED")
    _require(len(member_rows) <= cap["max_full_vertex_tests"], "PATCH_VERTEX_TEST_BUDGET_EXCEEDED")
    counts = np.bincount(member_groups, minlength=group_count)
    maximum, squared = np.zeros(group_count), np.zeros(group_count)
    low, high = np.full((group_count, 2), np.inf), np.full((group_count, 2), -np.inf)
    chunk = 3*cap["chunk_triangles"]
    for start in range(0, len(member_rows), chunk):
        sl = slice(start, min(len(member_rows), start+chunk))
        group, rows = member_groups[sl], member_rows[sl]
        vertices = points[rows] if scan else get_triangles(rows//3)[np.arange(len(rows)), rows % 3]
        delta = vertices-fits["center"][group]
        residual = np.sum(delta*fits["normals"][group], axis=1)
        projected = np.einsum("mi,mij->mj", delta, fits["extent_axes"][group])
        _require(np.isfinite(residual).all() and np.isfinite(projected).all(), "NONFINITE_PATCH_SUPPORT_RESIDUAL")
        np.maximum.at(maximum, group, np.abs(residual))
        np.add.at(squared, group, residual*residual)
        np.minimum.at(low, group, projected)
        np.maximum.at(high, group, projected)
        statistics["full_vertex_tests"] += len(rows)
    ranges = high-low
    extents = np.sort(ranges, axis=1)[:, ::-1]
    ratios = np.maximum(fits["eigenvalues"][:, 1], 0.)/fits["eigenvalues"][:, 2]
    repeated = (fits["eigenvalues"][:, 2]-fits["eigenvalues"][:, 1]
                <= 1e-10*fits["eigenvalues"][:, 2])
    # For every unit in-plane direction, variance >= lambda_min_inplane.
    # Popoviciu's variance <= range^2/4 therefore gives this invariant
    # sufficient lower bound on EVERY directional support width. It is not a
    # numerical certificate, nor a tight estimate of the minimum caliper width.
    width_lower = 2.*np.sqrt(np.maximum(fits["eigenvalues"][:, 1], 0.))
    rms = np.sqrt(squared/counts)
    triangle_order = np.argsort(labels, kind="stable")
    triangle_counts = np.bincount(labels, minlength=group_count)
    triangle_starts = np.r_[0, np.cumsum(triangle_counts)]
    member_starts = np.r_[0, np.cumsum(counts)] if scan else None
    qualified = []
    for group in range(group_count):
        reasons = []
        if counts[group] < (gate["min_support_vertices"] if scan else 3):
            reasons.append("INSUFFICIENT_SUPPORT_VERTICES")
        if ratios[group] < gate["min_inplane_covariance_ratio"]:
            reasons.append("DEGENERATE_INPLANE_SUPPORT")
        if maximum[group] > gate["max_full_support_distance_mm"]:
            reasons.append("FULL_COMPONENT_MAX_RESIDUAL_EXCEEDS_GATE")
        if extents[group, 1] < gate["min_second_extent_mm"]:
            reasons.append("INSUFFICIENT_SECOND_EXTENT")
        if repeated[group] and width_lower[group] < gate["min_second_extent_mm"]:
            reasons.append("REPEATED_EIGENSPACE_EXTENT_UNRESOLVED")
        disposition = dict(component_id=group, disposition="REJECTED" if reasons else "PROPOSED",
            reasons=reasons, triangle_count=int(triangle_counts[group]), support_count=int(counts[group]),
            support_rms_mm=float(rms[group]), full_support_max_abs_residual_mm=float(maximum[group]),
            support_extent_mm=extents[group].tolist(), inplane_covariance_ratio=float(ratios[group]),
            inplane_min_width_lower_bound_mm=float(width_lower[group]),
            extent_admission_basis="INVARIANT_WIDTH_SUFFICIENT_BOUND_FOR_REPEATED_EIGENSPACE"
                if repeated[group] else "DISTINCT_AREA_COVARIANCE_AXES")
        if not scan:
            disposition["source_face_id"] = int(face_ids[group])
        result["components"].append(disposition)
        if not reasons:
            qualified.append(group)
    statistics.update(components_evaluated=group_count, qualified_patch_count=len(qualified),
                      rejected_component_count=group_count-len(qualified), component_search_complete=True)
    _require(len(qualified) <= cap["max_output_patches"], "PATCH_OUTPUT_BUDGET_EXCEEDED")
    for group in qualified:
        rows = triangle_order[triangle_starts[group]:triangle_starts[group+1]]
        triangle_rows = _immutable(rows, "<i8")
        patch = dict(result["components"][group], plane_id=group if scan else int(face_ids[group]),
            normal=fits["normals"][group].tolist(),
            offset_mm=float(np.dot(fits["normals"][group], fits["center"][group])),
            area_centroid_mm=fits["center"][group].tolist(),
            surface_covariance_mm2=fits["covariance"][group].tolist(),
            surface_area_mm2=float(fits["area"][group]),
            area_measure="TESSELLATION_AREA_WITH_MULTIPLICITY_NOT_MATERIAL_AREA",
            support_extent_axes=fits["extent_axes"][group][:, np.argsort(-ranges[group], kind="stable")].T.tolist(),
            extent_axis_status="ARBITRARY_WITHIN_REPEATED_EIGENSPACE" if repeated[group]
                               else "AREA_COVARIANCE_INPLANE_EIGENVECTORS",
            plane_fit_measure="UNIFORM_TRIANGLE_AREA_TLS",
            rms_measure="UNIQUE_RAW_VERTICES" if scan else "TRIANGLE_CORNERS_WITH_MULTIPLICITY",
            triangle_rows=triangle_rows, triangle_rows_sha256=_digest(triangle_rows))
        if scan:
            raw = _immutable(member_rows[member_starts[group]:member_starts[group+1]], "<i8")
            patch.update(raw_vertex_rows=raw, raw_vertex_rows_sha256=_digest(raw))
        else:
            raw = _immutable(local_triangle_ids[rows], "<i8")
            patch.update(source_triangle_ids=raw, source_triangle_ids_sha256=_digest(raw))
        result["patches"].append(patch)
    result.update(status="AVAILABLE", reasons=[])
    return result


def extract_coplanar_scan_patches_v4(points, triangle_raw_rows, *,
        policy=CoplanarPatchPolicyV4(), budgets=CoplanarPatchBudgetsV4()):
    """Pure array scan seam; immutable support lineage, no source-file access."""
    result = _base("SCAN_SHARED_RAW_EDGE_COMPONENTS")
    try:
        gate, cap = _configuration(policy, budgets)
        result.update(policy=gate, budgets=cap, coordinate_frame="SCAN_INPUT_UNREGISTERED")
        _array_shape(points, (3,), "fiu", "INVALID_PATCH_POINTS")
        _array_shape(triangle_raw_rows, (3,), "iu", "INVALID_PATCH_RAW_TRIANGLES")
        result["statistics"]["input_triangles"] = len(triangle_raw_rows)
        _require(3 <= len(points) <= cap["max_vertices"], "PATCH_VERTEX_BUDGET_EXCEEDED")
        _input_caps(len(triangle_raw_rows), cap, scan=True)
        _require(np.all(triangle_raw_rows >= 0) and np.all(triangle_raw_rows < len(points)), "PATCH_RAW_INDEX_OUT_OF_RANGE")
        vertices = _immutable(points, "<f8")
        triangles = _immutable(triangle_raw_rows, "<i8")
        _require(np.all(triangles >= 0) and np.all(triangles < len(vertices)), "PATCH_RAW_INDEX_OUT_OF_RANGE")
        _require(np.isfinite(vertices).all() and np.max(np.abs(vertices)) <= 1e6, "INVALID_PATCH_POSITION")
        _require(np.all(triangles[:, 0] != triangles[:, 1]) and np.all(triangles[:, 0] != triangles[:, 2])
                 and np.all(triangles[:, 1] != triangles[:, 2]), "DUPLICATE_RAW_TRIANGLE_VERTEX")
        result["statistics"]["input_triangles"] = len(triangles)
        result["digests"].update(points_sha256=_digest(vertices), triangle_raw_rows_sha256=_digest(triangles))
        get = lambda rows: vertices[triangles[rows]]
        normals, areas = _geometry(get, len(triangles), cap["chunk_triangles"])
        labels, count = _scan_components(vertices, triangles, normals, gate, cap, result["statistics"])
        return _finish(result, get, areas, labels, count, gate, cap, points=vertices, triangles=triangles)
    except (_Unavailable, ValueError, TypeError, AttributeError, OverflowError, FloatingPointError,
            MemoryError, np.linalg.LinAlgError) as error:
        return _fail(result, error)


def fit_cad_face_planes_v4(triangles, *, source_face_indices, source_triangle_indices,
        policy=CoplanarPatchPolicyV4(), budgets=CoplanarPatchBudgetsV4()):
    """Fit each COMPLETE imported face triangle union; not an analytic plane."""
    result = _base("CAD_COMPLETE_IMPORT_LOCAL_FACE_UNIONS")
    try:
        gate, cap = _configuration(policy, budgets)
        result.update(policy=gate, budgets=cap, coordinate_frame="CAD_INPUT_UNREGISTERED")
        _array_shape(triangles, (3, 3), "fiu", "INVALID_CAD_PATCH_TRIANGLES")
        result["statistics"]["input_triangles"] = len(triangles)
        _input_caps(len(triangles), cap, scan=False)
        for value in (source_face_indices, source_triangle_indices):
            _array_shape(value, (), "iu", "INVALID_CAD_PATCH_SOURCE_IDS")
            _require(len(value) == len(triangles) and np.all(value >= 0)
                     and (value.dtype.kind == "i" or np.all(value <= np.uint64(2**63-1))),
                     "INVALID_CAD_PATCH_SOURCE_IDS")
        values = _immutable(triangles, "<f8")
        faces, local = _immutable(source_face_indices, "<i8"), _immutable(source_triangle_indices, "<i8")
        _require(np.all(faces >= 0) and np.all(local >= 0), "INVALID_CAD_PATCH_SOURCE_IDS")
        order = np.lexsort((local, faces))
        _require(not np.any((faces[order[1:]] == faces[order[:-1]]) & (local[order[1:]] == local[order[:-1]])),
                 "DUPLICATE_CAD_FACE_TRIANGLE_ID")
        face_ids, labels = np.unique(faces, return_inverse=True)
        result["statistics"]["component_count"] = len(face_ids)
        _require(len(face_ids) <= cap["max_components"], "PATCH_COMPONENT_BUDGET_EXCEEDED")
        result["statistics"]["input_triangles"] = len(values)
        result["digests"].update(triangles_sha256=_digest(values), source_face_indices_sha256=_digest(faces),
                                  source_triangle_indices_sha256=_digest(local))
        get = lambda rows: values[rows]
        _, areas = _geometry(get, len(values), cap["chunk_triangles"])
        return _finish(result, get, areas, labels, len(face_ids), gate, cap,
                       face_ids=face_ids, local_triangle_ids=local)
    except (_Unavailable, ValueError, TypeError, AttributeError, OverflowError, FloatingPointError,
            MemoryError, np.linalg.LinAlgError) as error:
        return _fail(result, error)
