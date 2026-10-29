"""Bounded approximate-CAD sampling for provisional whole-part registration.

No file, application, model, scan, label or reference access. OCP is lazy and
requires explicit opt-in. The native shape remains the geometry source; points
are interior samples of its TRIANGULATION, not certified native-surface points.
Meshing can reuse or mutate caches on the caller-owned shape. External process
wall/CPU/RSS supervision is mandatory: post-mesh count caps cannot bound the
mesher's internal allocations before it returns.

The area-preserving square-to-triangle sqrt warp is described in PBRT 3e:
https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
OCCT constructor meshing and triangulation/location semantics:
https://occt3d.com/dev/doc/refman/html/class_b_rep_mesh___incremental_mesh.html
https://occt3d.com/dev/doc/refman/html/class_b_rep___tool.html
These describe API/math, not validation of the installed OCP binary.
"""

import hashlib
import json
import math

import numpy as np


MAX_FACES = 1_024
MAX_TOPOLOGY_NODES = 32_768
MAX_TOPOLOGY_DEPTH = 32
MAX_TRIANGLES = 500_000
MAX_MESH_NODES = 1_000_000
MAX_ABS_POSITION = 1e6
MIN_SAMPLES = 36
MAX_SAMPLES = 100_000
LOCATION_TOLERANCE = 1e-9
MIN_LINEAR_DEFLECTION_MM = 1e-4
MAX_LINEAR_DEFLECTION_MM = 10.0
MIN_ANGULAR_DEFLECTION_RADIANS = 1e-4
MAX_ANGULAR_DEFLECTION_RADIANS = 1.0
METHOD = "AREA_CDF_MIDPOINT_HALTON23_POSITIVE_BARYCENTRIC_V1"
# OCCT7.9 IMeshData_Status.hxx:128 is an informational cache-reuse bit.
# All other nonzero bits remain outside this conservative sampler contract.
NATIVE_MESH_REUSED_FLAG = 0x80


class CADSamplingUnavailable(ValueError):
    """Bounded approximate-CAD preparation refused; never a part defect."""


def _require(condition, code):
    if not condition:
        raise CADSamplingUnavailable(code)


def _immutable(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _digest(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _sample_count(value):
    _require(type(value) is int and MIN_SAMPLES <= value <= MAX_SAMPLES, "CAD_SAMPLE_COUNT_BUDGET")
    return value


def _scalar(value, minimum, maximum, code):
    _require(type(value) in (int, float), code)
    try:
        result = float(value)
    except OverflowError as exc:
        raise CADSamplingUnavailable(code) from exc
    _require(math.isfinite(result) and minimum <= result <= maximum, code)
    return result


def _triangle_array(value):
    _require(type(value) is np.ndarray and value.ndim == 3 and value.shape[1:] == (3, 3)
             and 1 <= len(value) <= MAX_TRIANGLES and value.dtype.kind in "fiu",
             "INVALID_TRIANGLE_ARRAY")
    result = np.array(value, dtype=np.float64, copy=True)
    _require(np.isfinite(result).all() and np.max(np.abs(result)) <= MAX_ABS_POSITION,
             "INVALID_TRIANGLE_NUMERICS")
    return result


def _id_array(value, length, minimum, maximum, code):
    _require(type(value) is np.ndarray and value.shape == (length,) and value.dtype.kind in "iu", code)
    # Check the original unsigned/signed values BEFORE converting to int64.
    _require(np.all(value >= minimum) and np.all(value <= maximum), code)
    return np.array(value, dtype=np.int64, copy=True)


def _triangle_geometry(triangles):
    """Area and winding normal using scale-safe cross-product normalization.

    Common per-triangle scaling prevents squaring microscopic edge lengths or
    overflowing large cross products. A nonrepresentable positive area declines
    the entire input; no degenerate/numerically lost triangle is discarded.
    """
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    scale = np.maximum(np.max(np.abs(a), axis=1), np.max(np.abs(b), axis=1))
    _require(np.isfinite(scale).all() and np.all(scale > 0), "DEGENERATE_TRIANGLE")
    cross = np.cross(a / scale[:, None], b / scale[:, None])
    cross_length = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])
    _require(np.isfinite(cross_length).all() and np.all(cross_length > 0), "DEGENERATE_TRIANGLE")
    areas = (.5 * cross_length * scale) * scale
    _require(np.isfinite(areas).all() and np.all(areas > 0), "TRIANGLE_AREA_NOT_REPRESENTABLE")
    normals = cross / cross_length[:, None]
    _require(np.isfinite(normals).all(), "INVALID_TRIANGLE_NORMAL")
    return areas, normals


def _radical_inverse_positive(indices, base):
    """Exact bounded digit reversal, followed by one float64 division.

    A common denominator also pads short digit strings with trailing zeros in
    the reversed numerator. At indices <=100000 the largest denominator is
    177147 (base 3), so every integer intermediate is exactly representable in
    both int64 and float64. Repeated rounded reciprocals are not accumulated.
    """
    _require(type(base) is int and base in (2, 3), "INVALID_RADICAL_INVERSE_BASE")
    _require(type(indices) is np.ndarray and indices.ndim == 1 and indices.dtype.kind in "iu"
             and 1 <= len(indices) <= MAX_SAMPLES and np.all(indices >= 1)
             and np.all(indices <= MAX_SAMPLES), "INVALID_RADICAL_INVERSE_INDICES")
    remaining = np.array(indices, dtype=np.int64, copy=True)
    numerator = np.zeros(len(indices), dtype=np.int64)
    denominator = 1
    while np.any(remaining):
        numerator = numerator * base + remaining % base
        remaining //= base
        denominator *= base
    result = numerator.astype(np.float64) / denominator
    _require(np.all(result > 0) and np.all(result < 1), "NONINTERIOR_SEQUENCE_VALUE")
    return result


def _affine_interior_points(triangles, weights):
    """Add the anchor last; a constant coordinate is preserved exactly.

    For represented vertices/weights, the reference is the real affine formula
    p0 + w1*(p1-p0) + w2*(p2-p0). The six explicit float64 operations have a
    coordinatewise forward-error bound u*abs(p0) + gamma4*(abs(w1)*abs(p1-p0)
    + abs(w2)*abs(p2-p0)) + 8*eta, where u=2**-53, gamma4=4*u/(1-4*u), and
    eta=2**-1075, including gradual underflow. Each edge term has four rounding
    factors and the anchor has one. Existing coordinate/weight limits exclude
    overflow. This arithmetic bound is NOT a native-surface/metrology bound.
    """
    anchor = triangles[:, 0]
    first_edge = triangles[:, 1] - anchor
    second_edge = triangles[:, 2] - anchor
    first_term = weights[:, 1, None] * first_edge
    second_term = weights[:, 2, None] * second_edge
    return anchor + (first_term + second_term)


def sample_triangle_interiors_v1(triangles, *, source_face_indices, source_triangle_indices,
                                 sample_count=32_768):
    """Deterministically sample approximate oriented triangles, not vertices.

    Every triangle is checked before area allocation. Global area-CDF midpoint
    strata choose source triangles; a fixed Halton(2,3) sequence with indices
    1..N supplies positive sqrt-warp barycentric weights. Allocation is area-
    distributed, not guaranteed to include every triangle or every source face.
    IDs are zero-based face occurrences and one-based local triangle indices.

    The caller supplies already oriented triangles in one coordinate frame.
    Normals use the same triangle winding as sampled positions. No analytic
    surface normals or mathematically certified point/surface error are claimed.
    """
    base = {"schema": "CAD_TRIANGLE_INTERIOR_SAMPLING_V1", "status": "UNAVAILABLE",
            "points": None, "normals": None, "sample_triangle_rows": None,
            "sample_source_face_indices": None, "sample_source_triangle_indices": None,
            "sample_barycentric_weights": None, "reasons": [], "production_use_allowed": False,
            "evidence_class": "APPROXIMATE_TRIANGULATED_CAD_REGISTRATION_ONLY",
            "native_surface_point_status": "NOT_CERTIFIED",
            "normal_status": "ORIENTED_TRIANGLE_NORMAL_NOT_ANALYTIC_SURFACE_NORMAL",
            "calibrated_error_status": "NOT_ESTABLISHED",
            "sample_identity_status": "INPUT_TRIANGLE_ORDER_LOCAL_NOT_UNIVERSAL"}
    try:
        count = _sample_count(sample_count)
        triangle_values = _triangle_array(triangles)
        faces = _id_array(source_face_indices, len(triangle_values), 0, MAX_FACES - 1,
                          "INVALID_SOURCE_FACE_IDS")
        triangle_ids = _id_array(source_triangle_indices, len(triangle_values), 1, MAX_TRIANGLES,
                                 "INVALID_SOURCE_TRIANGLE_IDS")
        pairs = faces * (MAX_TRIANGLES + 1) + triangle_ids
        _require(len(np.unique(pairs)) == len(pairs), "DUPLICATE_SOURCE_TRIANGLE_ID")
        areas, triangle_normals = _triangle_geometry(triangle_values)
        total_area = math.fsum(areas.tolist())
        _require(math.isfinite(total_area) and total_area > 0, "INVALID_TOTAL_TRIANGLE_AREA")
        cumulative = np.cumsum(areas, dtype=np.float64)
        _require(np.isfinite(cumulative).all() and cumulative[-1] > 0
                 and np.all(np.diff(cumulative) >= 0), "INVALID_AREA_CDF")
        # Midpoints avoid exactly zero/total. Tiny positive triangles may have
        # no float64 CDF increment; disclose this rather than silently claiming
        # a sample/coverage guarantee for them.
        targets = (np.arange(count, dtype=np.float64) + .5) / count * cumulative[-1]
        selected = np.searchsorted(cumulative, targets, side="right")
        _require(np.all(selected < len(triangle_values)), "AREA_CDF_SELECTION_OUT_OF_RANGE")
        indices = np.arange(1, count + 1, dtype=np.int64)
        u = _radical_inverse_positive(indices, 2)
        v = _radical_inverse_positive(indices, 3)
        root = np.sqrt(u)
        first_weight, second_weight = root * (1.0 - v), root * v
        weights = np.column_stack((1.0 - (first_weight + second_weight), first_weight, second_weight))
        _require(np.all(weights > 0) and np.all(weights < 1)
                 and np.allclose(np.sum(weights, axis=1), 1., rtol=0, atol=4e-16),
                 "INVALID_BARYCENTRIC_WEIGHTS")
        chosen = triangle_values[selected]
        points = _affine_interior_points(chosen, weights)
        _require(np.isfinite(points).all(), "INVALID_SAMPLED_POINT")
        _require(not np.any(np.all(points[:, None, :] == chosen, axis=2)),
                 "BARYCENTRIC_POINT_COLLAPSED_TO_VERTEX")
        normals = triangle_normals[selected]
        all_faces, area_per_face = np.unique(faces), np.bincount(faces, weights=areas)
        sampled_faces, counts_per_face = np.unique(faces[selected], return_counts=True)
        sample_per_triangle = np.bincount(selected, minlength=len(triangle_values))
        points, normals = _immutable(points, "<f8"), _immutable(normals, "<f8")
        selected = _immutable(selected, "<i8")
        weights = _immutable(weights, "<f8")
        sample_faces = _immutable(faces[selected], "<i8")
        sample_triangles = _immutable(triangle_ids[selected], "<i8")
        triangle_values = _immutable(triangle_values, "<f8")
        faces, triangle_ids = _immutable(faces, "<i8"), _immutable(triangle_ids, "<i8")
        policy = {"method": METHOD, "sample_count": count, "sequence_start_index": 1,
                  "radical_inverse_bases": [2, 3], "allocation": "GLOBAL_AREA_CDF_MIDPOINT_STRATA",
                  "radical_inverse_evaluation": "EXACT_INT64_DIGIT_REVERSAL_SINGLE_FLOAT64_DIVISION",
                  "area_cdf_precision": "FLOAT64_WITH_LOST_INCREMENT_DISCLOSURE",
                  "warp": "SQRT_U_POSITIVE_BARYCENTRIC",
                  "first_barycentric_weight": "ONE_MINUS_FLOAT64_SUM_OF_LAST_TWO",
                  "position_evaluation": "ANCHOR_PLUS_SUM_OF_WEIGHTED_EDGE_DIFFERENCES",
                  "vertex_resampling": False}
        policy_bytes = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return {**base, "status": "AVAILABLE", "points": points, "normals": normals,
                "sample_triangle_rows": selected, "sample_source_face_indices": sample_faces,
                "sample_source_triangle_indices": sample_triangles, "sample_barycentric_weights": weights,
                "sampling_policy": policy,
                "coverage": {"sample_count": count, "triangle_count": len(triangle_values),
                    "face_count": len(all_faces), "total_triangulated_area_coordinate_units_squared": total_area,
                    "sampled_triangle_count": int(np.count_nonzero(sample_per_triangle)),
                    "unsampled_triangle_count": int(np.count_nonzero(sample_per_triangle == 0)),
                    "unsampled_source_face_indices": np.setdiff1d(all_faces, sampled_faces).tolist(),
                    "sample_count_by_source_face": {int(f): int(n) for f, n in zip(sampled_faces, counts_per_face)},
                    "area_by_source_face": {int(f): float(area_per_face[f]) for f in all_faces},
                    "cdf_lost_increment_triangle_count": int(np.count_nonzero(np.diff(cumulative) == 0)),
                    "cdf_sum_relative_difference_from_fsum": abs(float(cumulative[-1]) - total_area) / total_area,
                    "every_face_sampled": len(sampled_faces) == len(all_faces),
                    "continuous_surface_coverage": "NOT_ESTABLISHED",
                    "interior_status": "STRICT_POSITIVE_WEIGHTS_FLOAT64_POSITION_NOT_INTERVAL_CERTIFIED"},
                "digests": {"oriented_triangle_xyz_sha256": _digest(triangle_values),
                    "source_face_ids_sha256": _digest(faces), "source_triangle_ids_sha256": _digest(triangle_ids),
                    "sample_points_sha256": _digest(points), "sample_normals_sha256": _digest(normals),
                    "sample_triangle_rows_sha256": _digest(selected),
                    "sampling_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
                    "array_encoding": "C_CONTIGUOUS_LITTLE_ENDIAN_FLOAT64_OR_INT64",
                    "scope": "INPUT_TRIANGLE_AND_SAMPLE_BYTES_NOT_NATIVE_SOURCE_AUTHORIZATION"}}
    except CADSamplingUnavailable as exc:
        return {**base, "reasons": [str(exc)]}


def _proper_location_matrix(value):
    _require(type(value) is np.ndarray and value.shape == (3, 4) and value.dtype.kind in "fiu",
             "INVALID_NATIVE_LOCATION")
    matrix = np.asarray(value, dtype=np.float64)
    _require(np.isfinite(matrix).all() and np.max(np.abs(matrix[:, 3])) <= MAX_ABS_POSITION,
             "INVALID_NATIVE_LOCATION")
    rotation = matrix[:, :3]
    determinant = float(np.linalg.det(rotation))
    _require(math.isfinite(determinant) and determinant > 0, "REFLECTED_OR_SINGULAR_NATIVE_LOCATION")
    _require(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=LOCATION_TOLERANCE)
             and abs(determinant - 1.) <= LOCATION_TOLERANCE,
             "NONRIGID_NATIVE_LOCATION")
    return matrix


def _oriented_transformed_triangles(nodes, local_indices, location_matrix, *, reversed_face):
    """Pure location/winding seam: indices are one-based and location applied once."""
    _require(type(nodes) is np.ndarray and nodes.ndim == 2 and nodes.shape[1] == 3
             and 3 <= len(nodes) <= MAX_MESH_NODES and nodes.dtype.kind in "fiu", "INVALID_MESH_NODES")
    _require(type(local_indices) is np.ndarray and local_indices.ndim == 2 and local_indices.shape[1] == 3
             and 1 <= len(local_indices) <= MAX_TRIANGLES and local_indices.dtype.kind in "iu",
             "INVALID_NATIVE_TRIANGLE_INDICES")
    _require(np.all(local_indices >= 1) and np.all(local_indices <= len(nodes)), "INVALID_NATIVE_TRIANGLE_INDICES")
    _require(np.all(local_indices[:, 0] != local_indices[:, 1])
             and np.all(local_indices[:, 0] != local_indices[:, 2])
             and np.all(local_indices[:, 1] != local_indices[:, 2]), "DEGENERATE_NATIVE_TRIANGLE_INDICES")
    _require(type(reversed_face) is bool, "INVALID_FACE_ORIENTATION")
    matrix = _proper_location_matrix(location_matrix)
    values = np.asarray(nodes, dtype=np.float64)
    _require(np.isfinite(values).all() and np.max(np.abs(values)) <= MAX_ABS_POSITION, "INVALID_MESH_NODE_NUMERICS")
    transformed = values @ matrix[:, :3].T + matrix[:, 3]
    _require(np.isfinite(transformed).all() and np.max(np.abs(transformed)) <= MAX_ABS_POSITION,
             "INVALID_TRANSFORMED_MESH_NODES")
    indices = np.array(local_indices, dtype=np.int64, copy=True) - 1
    if reversed_face:
        indices[:, [1, 2]] = indices[:, [2, 1]]
    return transformed[indices]


def _native_triangles(shape, linear_deflection, angular_deflection):
    """Only native seam. Called after policy/opt-in checks, under an external cap."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Iterator

    _require(isinstance(shape, TopoDS_Shape) and not shape.IsNull(), "NATIVE_SHAPE_REQUIRED")

    def matrix_of(location):
        transform = location.Transformation()
        matrix = np.array([[transform.Value(row, column) for column in range(1, 5)]
                           for row in range(1, 4)], dtype=np.float64)
        return _proper_location_matrix(matrix)

    faces, stack, topology_nodes = [], [(shape, 0)], 0
    while stack:
        current, depth = stack.pop()
        topology_nodes += 1
        _require(topology_nodes <= MAX_TOPOLOGY_NODES and depth <= MAX_TOPOLOGY_DEPTH, "NATIVE_TOPOLOGY_BUDGET")
        matrix_of(current.Location())
        if current.ShapeType() == TopAbs_FACE:
            _require(len(faces) < MAX_FACES, "NATIVE_FACE_BUDGET")
            face = TopoDS.Face_s(current)
            _require(face.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED), "UNSUPPORTED_FACE_ORIENTATION")
            _require(not any(face.IsSame(other) for other in faces), "REPEATED_NATIVE_FACE_OCCURRENCE")
            faces.append(TopoDS.Face_s(face.Oriented(face.Orientation())))
        children, iterator = [], TopoDS_Iterator(current, True, True)
        while iterator.More():
            _require(topology_nodes + len(stack) + len(children) < MAX_TOPOLOGY_NODES, "NATIVE_TOPOLOGY_BUDGET")
            child = iterator.Value()
            children.append((child.Oriented(child.Orientation()), depth + 1))
            iterator.Next()
        stack.extend(reversed(children))
    _require(bool(faces), "NO_NATIVE_FACES")
    preexisting = 0
    for face in faces:
        location = TopLoc_Location()
        if BRep_Tool.Triangulation_s(face, location) is not None:
            preexisting += 1
    # The constructor calls Perform; do not execute a second implicit retry.
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, False)
    _require(mesher.IsDone() is True, "NATIVE_MESHING_NOT_DONE")
    status_flags = mesher.GetStatusFlags()
    if (type(status_flags) is not int or status_flags < 0
            or status_flags & ~NATIVE_MESH_REUSED_FLAG):
        error = CADSamplingUnavailable("NATIVE_MESHING_STATUS_FLAGS")
        error.native_mesher_status_flags = status_flags if type(status_flags) is int else None
        raise error
    blocks, face_ids, triangle_ids, mesh_nodes, triangle_count, face_records = [], [], [], 0, 0, []
    for face_index, face in enumerate(faces):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        _require(triangulation is not None, "NATIVE_FACE_TRIANGULATION_MISSING")
        node_count, count = triangulation.NbNodes(), triangulation.NbTriangles()
        _require(type(node_count) is int and type(count) is int and node_count >= 3 and count >= 1,
                 "INVALID_NATIVE_MESH_COUNTS")
        mesh_nodes += node_count
        triangle_count += count
        _require(mesh_nodes <= MAX_MESH_NODES and triangle_count <= MAX_TRIANGLES, "NATIVE_MESH_CARDINALITY_BUDGET")
        matrix = matrix_of(location)
        nodes = np.empty((node_count, 3), dtype=np.float64)
        for index in range(1, node_count + 1):
            point = triangulation.Node(index)
            nodes[index - 1] = (point.X(), point.Y(), point.Z())
        local_indices = np.empty((count, 3), dtype=np.int64)
        for index in range(1, count + 1):
            values = triangulation.Triangle(index).Get()
            _require(len(values) == 3 and all(type(value) is int and 1 <= value <= node_count for value in values),
                     "INVALID_NATIVE_TRIANGLE_INDICES")
            local_indices[index - 1] = values
        reversed_face = face.Orientation() == TopAbs_REVERSED
        blocks.append(_oriented_transformed_triangles(nodes, local_indices, matrix, reversed_face=reversed_face))
        face_ids.append(np.full(count, face_index, dtype=np.int64))
        triangle_ids.append(np.arange(1, count + 1, dtype=np.int64))
        face_records.append({"source_face_index": face_index, "node_count": node_count, "triangle_count": count,
                             "orientation": "REVERSED" if reversed_face else "FORWARD"})
    return np.concatenate(blocks), np.concatenate(face_ids), np.concatenate(triangle_ids), {
        "face_count": len(faces), "topology_node_occurrences": topology_nodes,
        "mesh_node_occurrences": mesh_nodes, "triangle_count": triangle_count,
        "preexisting_triangulated_faces": preexisting, "mesher_status_flags": status_flags,
        "mesher_cache_reused": bool(status_flags & NATIVE_MESH_REUSED_FLAG),
        "face_records": face_records,
    }


def sample_native_cad_surface_v1(shape, *, allow_ocp=False, native_binding="ocp", sample_count=32_768,
                                 linear_deflection_mm=.10, angular_deflection_radians=.20):
    """Opt-in approximate native CAD samples and normals for one registration cloud.

    Deflection parameter names require the caller's mm-scale CAD import contract;
    they do not verify source units. Positive rigid locations are supported and
    face winding is honored; reflected/nonrigid locations fail closed. A mirror
    baked into native geometry is judged by its resulting face orientations,
    not inferred from a filename or handedness label.
    """
    metadata = {
        "schema": "CAD_GUIDED_NATIVE_SURFACE_SAMPLING_V1", "status": "UNAVAILABLE", "reasons": [],
        "points": None, "normals": None, "production_use_allowed": False,
        "evidence_class": "APPROXIMATE_TRIANGULATED_CAD_REGISTRATION_ONLY",
        "native_surface_point_status": "NOT_CERTIFIED", "source_unit_status": "CALLER_IMPORT_CONTRACT_NOT_VERIFIED_HERE",
        "frame_status": "NATIVE_MODEL_COORDINATES_NOT_SCAN_REGISTRATION",
        "native_validity_status": "NOT_ANALYTICALLY_OR_TOPOLOGICALLY_QUALIFIED",
        "source_identity_status": "CALLER_OWNED_NATIVE_HANDLE_NOT_BYTE_BOUND_HERE",
        "cache_effect": "MESHER_MAY_REUSE_OR_MUTATE_CALLER_NATIVE_TRIANGULATION_CACHE",
        "native_mesher_resource_enforcement": "EXTERNAL_SUPERVISOR_REQUIRED_BEFORE_POST_MESH_COUNT_CHECKS",
        "replay_scope": "SAME_NATIVE_TOPOLOGY_ENUMERATION_CACHE_AND_KERNEL_NOT_UNIVERSAL_EXPORT_IDENTITY",
        "calibrated_error_status": "NOT_ESTABLISHED",
    }
    if type(allow_ocp) is not bool or not allow_ocp or type(native_binding) is not str or native_binding != "ocp":
        return {**metadata, "reasons": ["EXPLICIT_OCP_BINDING_REQUIRED"]}
    try:
        _sample_count(sample_count)
        linear = _scalar(linear_deflection_mm, MIN_LINEAR_DEFLECTION_MM, MAX_LINEAR_DEFLECTION_MM,
                         "LINEAR_DEFLECTION_POLICY")
        angular = _scalar(angular_deflection_radians, MIN_ANGULAR_DEFLECTION_RADIANS, MAX_ANGULAR_DEFLECTION_RADIANS,
                          "ANGULAR_DEFLECTION_POLICY")
        triangles, faces, triangle_ids, native_facts = _native_triangles(shape, linear, angular)
        sampled = sample_triangle_interiors_v1(triangles, source_face_indices=faces,
                                              source_triangle_indices=triangle_ids, sample_count=sample_count)
        _require(sampled["status"] == "AVAILABLE", sampled["reasons"][0] if sampled["reasons"] else "CAD_SAMPLING_FAILED")
        mesh_policy = {"method": "OCP_BRepMesh_IncrementalMesh_CONSTRUCTOR_ONCE", "linear_deflection_mm": linear,
            "angular_deflection_radians": angular, "relative_deflection": False, "parallel": False,
            "max_faces": MAX_FACES, "max_topology_nodes": MAX_TOPOLOGY_NODES,
            "max_topology_depth": MAX_TOPOLOGY_DEPTH, "max_mesh_nodes": MAX_MESH_NODES,
            "max_triangles": MAX_TRIANGLES, "location_tolerance": LOCATION_TOLERANCE,
            "reflected_locations": "REFUSE", "nonrigid_locations": "REFUSE",
            "face_order": "DEPTH_FIRST_NATIVE_CHILD_ENUMERATION", "reversed_face_winding": "SWAP_INDICES_2_3",
            "native_shape_cache_cleanup": False, "deflection_error_is_certified": False,
            "accepted_status_mask": NATIVE_MESH_REUSED_FLAG,
            "accepted_status_meaning": "NO_ERROR_OR_INFORMATIONAL_REUSED_ONLY"}
        policy_bytes = json.dumps(mesh_policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return {**sampled, **metadata, "status": "AVAILABLE", "points": sampled["points"], "normals": sampled["normals"],
                "sample_triangle_rows": sampled["sample_triangle_rows"],
                "sample_source_face_indices": sampled["sample_source_face_indices"],
                "sample_source_triangle_indices": sampled["sample_source_triangle_indices"],
                "sample_barycentric_weights": sampled["sample_barycentric_weights"],
                "sampling_policy": sampled["sampling_policy"], "coverage": sampled["coverage"],
                "native_facts": native_facts, "mesher_policy": mesh_policy,
                "digests": {**sampled["digests"], "mesher_policy_sha256": hashlib.sha256(policy_bytes).hexdigest()}}
    except CADSamplingUnavailable as exc:
        return {**metadata, "reasons": [str(exc)],
                "native_mesher_status_flags": getattr(exc, "native_mesher_status_flags", None)}
    except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
        return {**metadata, "reasons": ["NATIVE_CAD_SAMPLING_UNAVAILABLE"]}
