"""V6 metadata successor on the unchanged frozen V5 capacity construction.

No physical input, 108-case study, native geometry or generated mesh files.
The two small prerequisites run normally; the one full-size case is opt-in.
"""
import hashlib
import json
import math
import os
import resource
import sys
import time

import numpy as np
import pytest

from backend.cad_guided_noisy_mesh_planes_v7 import extract_noisy_scan_plane_proposals_v7


CAPACITY_FLAG = "SHERMAN_RUN_NOISY_MESH_CAPACITY_V6"
CAPACITY_DIMENSIONS = (500, 250, 175)
CAPACITY_COUNTS = (900_000, 1_800_000)
REVIEWED_COHORT_MAXIMUM = (863_115, 1_701_535)  # Released counts, not file access.
POSITION_ERROR_MM = .05
NOISE_DENOMINATOR = 8192
CHUNK_ROWS = 65536
RESOURCE_LIMITS = dict(wall_seconds=120, worker_cpu_seconds=90,
                       sampled_process_group_rss_bytes=4*1024**3, output_bytes=1024**2)


def _boundary(low, high):
    forward = np.arange(low, high, dtype=np.int64)
    backward = np.arange(high, low, -1, dtype=np.int64)
    width = high-low
    return np.vstack((np.column_stack((forward, np.full(width, low))),
                      np.column_stack((np.full(width, high), forward)),
                      np.column_stack((backward, np.full(width, high))),
                      np.column_stack((np.full(width, low), backward))))


def _construct(n, h, k):
    """Shared-index boundary of an annular prism, not surface/row padding.

    Adapted as a NEW fixture from the reviewed V4 construction, without importing
    or editing its test/module. V=2(n+h)(n-h+2k), F=2V, and all rows are used.
    Face labels are evaluator truth only; they never enter the extractor.
    """
    assert all(type(value) is int for value in (n, h, k))
    assert 2 <= h < n <= 500 and 1 <= k <= 175 and (n-h) % 2 == 0
    margin, high = (n-h)//2, (n+h)//2
    vertex_count = 2*(n+h)*(n-h+2*k)
    triangle_count = 2*vertex_count
    axis = np.arange(n+1, dtype=np.int64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    keep = ~((x > margin) & (x < high) & (y > margin) & (y < high))
    xy = np.column_stack((x[keep], y[keep]))
    face_vertices = (n+1)**2-(h-1)**2
    assert len(xy) == face_vertices
    bottom_ids = np.full((n+1, n+1), -1, dtype=np.int64)
    bottom_ids[keep] = np.arange(face_vertices, dtype=np.int64)
    truth = np.empty((vertex_count, 3), dtype=np.float64)
    truth[:face_vertices, :2], truth[:face_vertices, 2] = xy, 0.
    truth[face_vertices:2*face_vertices, :2] = xy
    truth[face_vertices:2*face_vertices, 2] = k
    outer, inner = _boundary(0, n), _boundary(margin, high)
    boundary = np.vstack((outer, inner))
    perimeter = 4*(n+h)
    assert len(boundary) == perimeter
    side_ids = np.empty((k+1, perimeter), dtype=np.int64)
    side_ids[0] = bottom_ids[boundary[:, 1], boundary[:, 0]]
    side_ids[-1] = side_ids[0]+face_vertices
    for layer in range(1, k):
        start = 2*face_vertices+(layer-1)*perimeter
        side_ids[layer] = np.arange(start, start+perimeter, dtype=np.int64)
        truth[start:start+perimeter, :2] = boundary
        truth[start:start+perimeter, 2] = layer
    assert 2*face_vertices+(k-1)*perimeter == vertex_count
    cells = ~((x[:-1, :-1] >= margin) & (x[:-1, :-1] < high)
              & (y[:-1, :-1] >= margin) & (y[:-1, :-1] < high))
    a, b = bottom_ids[:-1, :-1][cells], bottom_ids[:-1, 1:][cells]
    c, d = bottom_ids[1:, 1:][cells], bottom_ids[1:, :-1][cells]
    horizontal = 2*(n*n-h*h)
    assert all(np.all(values >= 0) for values in (a, b, c, d))
    triangles = np.empty((triangle_count, 3), dtype=np.int64)
    labels = np.empty(triangle_count, dtype=np.int8)
    triangles[:horizontal:2] = np.column_stack((a, c, b))
    triangles[1:horizontal:2] = np.column_stack((a, d, c))
    triangles[horizontal:2*horizontal:2] = np.column_stack((a, b, c))+face_vertices
    triangles[horizontal+1:2*horizontal:2] = np.column_stack((a, c, d))+face_vertices
    labels[:horizontal], labels[horizontal:2*horizontal] = 0, 1
    next_boundary = np.r_[np.arange(1, len(outer)), 0,
                          np.arange(len(outer)+1, perimeter), len(outer)]
    low_a, low_b = side_ids[:-1].reshape(-1), side_ids[:-1, next_boundary].reshape(-1)
    high_a, high_b = side_ids[1:].reshape(-1), side_ids[1:, next_boundary].reshape(-1)
    wall_start = 2*horizontal
    triangles[wall_start::2] = np.column_stack((low_a, low_b, high_b))
    triangles[wall_start+1::2] = np.column_stack((low_a, high_b, high_a))
    walls = triangles[wall_start:].reshape(k, perimeter, 2, 3)
    inner_walls = walls[:, len(outer):]
    inner_walls[..., [1, 2]] = inner_walls[..., [2, 1]]
    wall_labels = np.r_[np.repeat(np.arange(2, 6), n), np.repeat(np.arange(6, 10), h)]
    labels[wall_start:].reshape(k, perimeter, 2)[:] = wall_labels[None, :, None]
    assert wall_start+2*k*perimeter == triangle_count
    return truth, triangles, labels


def _observe(truth):
    """Binary-exact, nonzero raw-ID perturbations; no face label or fitted value."""
    observed = truth.copy()
    for start in range(0, len(truth), CHUNK_ROWS):
        stop = min(len(truth), start+CHUNK_ROWS)
        rows = np.arange(start, stop, dtype=np.int64)
        for axis, (multiplier, shift, modulus) in enumerate(((5, 1, 32), (7, 3, 34), (11, 5, 36))):
            numerator = 2*((multiplier*rows+shift) % modulus)-(modulus-1)
            observed[start:stop, axis] += numerator/NOISE_DENOMINATOR
    return observed


def _faces(n, h, k):
    margin, high = (n-h)//2, (n+h)//2
    horizontal = 2*(n*n-h*h)
    return [dict(face_id=0, axis=2, offset=0, triangles=horizontal),
            dict(face_id=1, axis=2, offset=k, triangles=horizontal),
            *[dict(face_id=index+2, axis=axis, offset=offset, triangles=2*width*k)
              for index, (axis, offset, width) in enumerate((
                  (1, 0, n), (0, n, n), (1, n, n), (0, 0, n),
                  (1, margin, h), (0, high, h), (1, high, h), (0, margin, h)))]]


def _digest(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _immutable_sorted_rows(array, count):
    assert array.dtype == np.dtype("<i8") and array.ndim == 1
    assert not array.flags.writeable
    base = array
    while isinstance(base, np.ndarray):
        base = base.base
    assert isinstance(base, bytes)
    if len(array):
        assert 0 <= array[0] <= array[-1] < count
        assert np.all(array[1:] > array[:-1])


def _witness_and_normal(analysis, raw, observed, true_axis=None):
    assert analysis["status"] == "PROPOSED" and analysis["witness_status"] == "FOUND"
    assert analysis["cone_status"] == "AVAILABLE"
    assert analysis["raw_support_count"] == len(raw)
    normal = np.asarray(analysis["normal"])
    anchor = np.asarray(analysis["witness_anchor_mm"])
    offset = analysis["witness_offset_from_anchor_mm"]
    assert np.isfinite(normal).all() and abs(np.linalg.norm(normal)-1.) < 1e-12
    maximum = 0.
    for start in range(0, len(raw), CHUNK_ROWS):
        values = observed[raw[start:start+CHUNK_ROWS]]
        residual = np.abs((values-anchor) @ normal-offset)
        assert np.all(residual <= POSITION_ERROR_MM)  # No added numerical/error slack.
        maximum = max(maximum, float(np.max(residual)))
    alpha = analysis["normal_angle_bound_radians"]
    assert 0 <= alpha <= math.pi/18
    assert analysis["numerical"]["position_error_bound_was_increased"] is False
    assert analysis["same_physical_plane_verified"] is False
    angle = None
    if true_axis is not None:
        truth_normal = np.eye(3)[true_axis]
        angle = math.atan2(float(np.linalg.norm(np.cross(normal, truth_normal))), abs(float(normal[true_axis])))
        assert angle <= alpha
    return maximum, angle


def _verify(truth, observed, triangles, labels, result, dimensions):
    n, h, k = dimensions
    vertices, count = len(truth), len(triangles)
    assert result["status"] == "COMPLETE", result["reasons"]
    assert result["physical_plane_count"] is None and result["observed_bend_count"] is None
    assert result["accepted_registration"] is False and result["production_use_allowed"] is False
    assert result["search_exhaustive"] is False and result["physical_flatness_verified"] is False
    assert result["raw_coordinate_welding"] is False and result["residual_trimming"] is False
    assert result["unit_assumption"] == "PROVISIONAL_ARRAY_MM_UNVERIFIED"
    assert result["uncertainty_status"] == "STIPULATED_UNCALIBRATED"
    assert result["finite_support_representation"] == "ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON"
    assert result["policy"]["tile_triangles"] == (128, 1024)
    assert result["policy"]["max_normal_angle_radians"] == math.pi/18
    assert result["input_digests"]["points_sha256"] == _digest(observed)
    assert result["input_digests"]["triangle_raw_rows_sha256"] == _digest(triangles)
    expected_faces = _faces(n, h, k)
    assert np.array_equal(np.bincount(labels, minlength=10), [face["triangles"] for face in expected_faces])
    referenced = np.zeros(vertices, dtype=bool)
    maximum_displacement = 0.
    normal_noise_maxima = np.zeros(10)
    for start in range(0, vertices, CHUNK_ROWS):
        delta = observed[start:start+CHUNK_ROWS]-truth[start:start+CHUNK_ROWS]
        assert np.all(delta != 0.) and np.all(np.abs(delta) < .005)
        norms = np.linalg.norm(delta, axis=1)
        assert np.all(norms < POSITION_ERROR_MM)
        maximum_displacement = max(maximum_displacement, float(np.max(norms)))
    for start in range(0, count, CHUNK_ROWS):
        ids = triangles[start:start+CHUNK_ROWS]
        referenced[ids.reshape(-1)] = True
        for face in expected_faces:
            chosen = ids[labels[start:start+CHUNK_ROWS] == face["face_id"]]
            if not len(chosen):
                continue
            assert np.all(truth[chosen, face["axis"]] == face["offset"])
            normal_noise_maxima[face["face_id"]] = max(normal_noise_maxima[face["face_id"]],
                float(np.max(np.abs(observed[chosen, face["axis"]]-face["offset"]))))
    assert referenced.all() and np.all(normal_noise_maxima > .003)
    accounting = result["raw_input_accounting"]
    assert accounting["referenced_raw_row_count"] == vertices
    assert len(accounting["unreferenced_raw_vertex_rows"]) == 0
    _immutable_sorted_rows(accounting["unreferenced_raw_vertex_rows"], vertices)
    assert accounting["unreferenced_raw_vertex_rows_sha256"] == _digest(accounting["unreferenced_raw_vertex_rows"])
    assert accounting["unreferenced_disposition"] == "INPUT_RETAINED_NOT_TRIANGLE_SUPPORT"
    assert len(result["partitions"]) == 2
    assert [row["tile_triangles"] for row in result["partitions"]] == [128, 1024]
    retained_bytes = 0
    for partition in result["partitions"]:
        mapping = partition["triangle_core_labels"]
        assert mapping.shape == (count,) and not mapping.flags.writeable
        owner = mapping
        while isinstance(owner, np.ndarray):
            owner = owner.base
        assert isinstance(owner, bytes)
        assert _digest(mapping) == partition["triangle_core_labels_sha256"]
        retained_bytes += mapping.nbytes
        coverage = np.zeros(count, dtype=np.uint32)
        for core in result["cores"]:
            if core["tile_triangles"] != partition["tile_triangles"]:
                continue
            base = core["base_triangle_rows"]
            _immutable_sorted_rows(base, count)
            assert len(base) <= partition["tile_triangles"]
            assert np.all(mapping[base] == core["core_id"])
            coverage[base] += 1
        assert np.all(coverage == 1)
    face_core_counts = np.zeros(10, dtype=np.int64)
    representatives = [None]*10
    geometry_counts = dict(pure_face_qualified_cores=0, pure_face_rejected_cores=0,
                          mixed_face_qualified_cores=0, mixed_face_rejected_cores=0,
                          qualified_closures=0, rejected_closures=0)
    rejection_reasons = {}
    maximum_witness_residual, maximum_true_normal_error = 0., 0.
    proposal_ids = {(row["kind"], row["record_id"]) for row in result["proposals"]}
    assert len(proposal_ids) == len(result["proposals"])
    expected_proposal_ids = set()
    for kind, records in (("CORE", result["cores"]), ("CLOSURE", result["closures"])):
        for index, record in enumerate(records):
            assert record["core_id" if kind == "CORE" else "closure_id"] == index
            rows, raw = record["triangle_rows"], record["raw_vertex_rows"]
            _immutable_sorted_rows(rows, count)
            _immutable_sorted_rows(raw, vertices)
            assert len(rows) and len(raw)
            assert _digest(rows) == record["triangle_rows_sha256"]
            assert _digest(raw) == record["raw_vertex_rows_sha256"]
            retained_bytes += rows.nbytes+raw.nbytes
            if kind == "CORE":
                retained_bytes += record["base_triangle_rows"].nbytes
                assert np.all(np.isin(record["base_triangle_rows"], rows))
                assert np.array_equal(raw, np.unique(triangles[rows].reshape(-1)))
            else:
                marked = np.zeros(vertices, dtype=bool)
                for start in range(0, len(rows), CHUNK_ROWS):
                    marked[triangles[rows[start:start+CHUNK_ROWS]].reshape(-1)] = True
                assert np.array_equal(raw, np.flatnonzero(marked))
                assert record["source_core_ids"]
                for source in record["source_core_ids"]:
                    assert result["cores"][source]["closure_id"] == index
            faces = np.unique(labels[rows])
            pure = len(faces) == 1
            analysis = record["analysis"]
            qualified = analysis["status"] == "PROPOSED"
            assert ((kind, index) in proposal_ids) == qualified
            if qualified:
                expected_proposal_ids.add((kind, index))
            if kind == "CORE":
                key = ("pure_face_" if pure else "mixed_face_")+("qualified_cores" if qualified else "rejected_cores")
            else:
                key = "qualified_closures" if qualified else "rejected_closures"
            geometry_counts[key] += 1
            if not qualified:
                for reason in analysis["reasons"]:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0)+1
                continue
            axis = expected_faces[int(faces[0])]["axis"] if pure else None
            residual, angle = _witness_and_normal(analysis, raw, observed, axis)
            maximum_witness_residual = max(maximum_witness_residual, residual)
            if angle is not None:
                maximum_true_normal_error = max(maximum_true_normal_error, angle)
            if kind == "CORE" and pure:
                face_id = int(faces[0])
                face_core_counts[face_id] += 1
                if len(raw) >= 50 and representatives[face_id] is None:
                    representatives[face_id] = dict(face_id=face_id, core_id=index,
                        triangle_rows=len(rows), raw_rows=len(raw),
                        true_normal_error_radians=angle,
                        conditional_cone_radians=analysis["normal_angle_bound_radians"],
                        whole_support_max_residual_mm=residual)
    assert proposal_ids == expected_proposal_ids
    assert all(row is not None for row in representatives), "Missing required >=50-row pure-face CORE representative"
    stats = result["statistics"]
    assert result["schema_version"] == "noisy_mesh_planes_v7"
    assert result["metadata_encoding"]["changed_encoded_representation"] is True
    assert 0 < stats["metadata_expanded_bytes"] <= 268_435_456
    assert 0 < stats["metadata_node_visits"] <= 64_000_000
    assert 0 < stats["metadata_array_bytes_hashed"] <= 4*1024**3
    assert stats["input_raw_rows"] == vertices and stats["input_triangles"] == count
    assert stats["edge_records"] == 3*count and stats["shared_edge_pairs"] == 3*count//2
    assert stats["core_count"] == stats["cores_evaluated"] == len(result["cores"])
    assert stats["qualified_core_count"] == geometry_counts["pure_face_qualified_cores"]+geometry_counts["mixed_face_qualified_cores"]
    assert stats["unique_closure_count"] == len(result["closures"])
    assert stats["qualified_closure_count"] == geometry_counts["qualified_closures"]
    assert stats["closures_scheduled"] == stats["closures_evaluated"] == len(result["scheduled_core_ids"]) <= 16
    assert len(set(result["scheduled_core_ids"])) == len(result["scheduled_core_ids"])
    assert all(("CORE", core_id) in proposal_ids for core_id in result["scheduled_core_ids"])
    assert stats["unscheduled_qualified_core_count"] == stats["qualified_core_count"]-stats["closures_scheduled"]
    assert stats["core_construction_complete"] and stats["core_evaluation_complete"]
    assert stats["scheduled_closure_evaluation_complete"]
    assert stats["all_qualified_core_closures_evaluated"] == (stats["unscheduled_qualified_core_count"] == 0)
    assert stats["retained_array_bytes"] == retained_bytes
    for field, limit in (("triangle_memberships", "max_triangle_memberships"), ("raw_memberships", "max_raw_memberships"),
        ("full_support_row_tests", "max_full_support_row_tests"), ("cone_row_tests", "max_cone_row_tests"),
        ("closure_triangle_tests", "max_closure_triangle_tests"), ("closure_edge_tests", "max_closure_edge_tests"),
        ("fit_evaluations", "max_fit_evaluations"), ("retained_array_bytes", "max_retained_array_bytes"),
        ("metadata_bytes", "max_metadata_bytes")):
        assert 0 <= stats[field] <= result["budgets"][limit]
    return dict(all_raw_rows_accounted=True, all_triangle_rows_partitioned_at_both_scales=True,
        all_retained_lineage_checked=True, all_qualified_support_witnesses_checked=True,
        all_qualified_pure_face_normal_cones_checked=True,
        face_qualified_core_counts=face_core_counts.tolist(), representatives=representatives,
        geometry_dispositions=geometry_counts, rejected_record_reason_counts=rejection_reasons,
        maximum_displacement_mm=maximum_displacement, normal_noise_maxima_mm=normal_noise_maxima.tolist(),
        maximum_qualified_witness_residual_mm=maximum_witness_residual,
        maximum_qualified_pure_face_normal_error_radians=maximum_true_normal_error,
        statistics=stats)


def test_small_constructor_has_shared_closed_oriented_edges_and_genuine_noise():
    truth, triangles, labels = _construct(6, 2, 3)
    observed = _observe(truth)
    assert truth.shape == (160, 3) and triangles.shape == (320, 3)
    assert np.array_equal(np.unique(triangles), np.arange(160))
    assert len(np.unique(truth, axis=0)) == 160
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    winding = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    assert np.all(counts == 2) and np.all(np.bincount(inverse, weights=winding) == 0)
    assert len(truth)-len(unique)+len(triangles) == 0
    neighbors = [set() for _ in truth]
    for left, right in unique:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    visited, frontier = {0}, [0]
    while frontier:
        for neighbor in neighbors[frontier.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    assert len(visited) == len(truth)
    delta = observed-truth
    assert np.all(delta != 0.) and np.all(np.abs(delta) < .005)
    integer_noise = delta*NOISE_DENOMINATOR
    assert np.all(integer_noise == np.rint(integer_noise))
    assert np.all(np.sum(integer_noise*integer_noise, axis=1) <= 3275)
    assert np.all(np.linalg.norm(delta, axis=1) < POSITION_ERROR_MM)
    assert set(labels.tolist()) == set(range(10))


def test_small_actual_noisy_prism_oracle_includes_all_ten_required_representatives():
    dimensions = (64, 24, 32)
    truth, triangles, labels = _construct(*dimensions)
    assert (len(truth), len(triangles)) == (18_304, 36_608)
    observed = _observe(truth)
    result = extract_noisy_scan_plane_proposals_v7(observed, triangles, position_error_bounds_mm=POSITION_ERROR_MM)
    _verify(truth, observed, triangles, labels, result, dimensions)


@pytest.mark.skipif(os.environ.get(CAPACITY_FLAG) != "1",
                    reason="separate reviewed 120 s / 90 worker CPU / 4 GiB capacity gate required")
def test_fixed_900000_row_1800000_triangle_noisy_capacity_once():
    started = time.perf_counter()
    truth, triangles, labels = _construct(*CAPACITY_DIMENSIONS)
    observed = _observe(truth)
    assert (len(truth), len(triangles)) == CAPACITY_COUNTS
    assert all(actual >= reviewed for actual, reviewed in zip(CAPACITY_COUNTS, REVIEWED_COHORT_MAXIMUM))
    extraction_started = time.perf_counter()
    result = extract_noisy_scan_plane_proposals_v7(observed, triangles, position_error_bounds_mm=POSITION_ERROR_MM)
    extraction_seconds = time.perf_counter()-extraction_started
    receipt = dict(schema="NOISY_MESH_CAPACITY_V6_INVENTED_ANNULAR_PRISM",
        dimensions_cells=list(CAPACITY_DIMENSIONS), raw_rows=len(truth), triangle_rows=len(triangles),
        expected_unique_shared_raw_edges=2_700_000, expected_extractor_edge_records=5_400_000,
        position_error_bound_mm=POSITION_ERROR_MM, noise_numerator_maxima=[31, 33, 35],
        noise_denominator=NOISE_DENOMINATOR, extraction_status=result["status"],
        extraction_reasons=result["reasons"], extraction_seconds=extraction_seconds,
        qualification_scope="ONE_STIPULATED_NOISY_FULL_SIZE_CONSTRUCTION_NOT_108_STUDY_OR_PHYSICAL_QUALIFICATION",
        physical_plane_count=None, observed_bend_count=None, production_use_allowed=False,
        limits=RESOURCE_LIMITS)
    try:
        receipt.update(_verify(truth, observed, triangles, labels, result, CAPACITY_DIMENSIONS), oracle_status="PASSED")
    except Exception as error:
        receipt.update(oracle_status="FAILED", oracle_error_type=type(error).__name__,
                       oracle_error=str(error)[:256], statistics=result["statistics"])
        encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
        assert len(encoded.encode()) < 32768
        print("NOISY_MESH_CAPACITY_V6_RECEIPT="+encoded, flush=True)
        raise
    receipt["construction_extraction_verification_seconds"] = time.perf_counter()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt["worker_lifetime_peak_rss_bytes_after_verification"] = int(peak if sys.platform == "darwin" else peak*1024)
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) < 32768
    print("NOISY_MESH_CAPACITY_V6_RECEIPT="+encoded, flush=True)
