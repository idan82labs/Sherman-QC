"""Prospective exact-count capacity construction; no native or physical inputs.

The opt-in case is a genuine closed rectangular annular prism, not a periodic
wrap, duplicated face padding, point thinning or an invented degenerate mesh.
Root owns source review, source binding and the one serial supervised execution.
Ordinary controls stay small; the 800000/1600000 case requires a distinct flag.
"""

import json
import os
import resource
import sys
import time

import numpy as np
import pytest

from backend.cad_guided_mesh_plane_proposals_v4 import extract_coplanar_scan_patches_v4


CAPACITY_DIMENSIONS = (500, 300, 150)
CAPACITY_COUNTS = (800_000, 1_600_000)
RESOURCE_LIMITS = dict(wall_seconds=120, cpu_seconds=90,
                       process_tree_rss_bytes=4*1024**3, output_bytes=1024**2)
CAPACITY_FLAG = "SHERMAN_RUN_MESH_PLANE_CAPACITY_V4"


def _boundary_square(low, high):
    """CCW integer boundary, each corner once and no implicit XYZ welding."""
    forward = np.arange(low, high, dtype=np.int64)
    backward = np.arange(high, low, -1, dtype=np.int64)
    width = high-low
    return np.vstack((np.column_stack((forward, np.full(width, low))),
                      np.column_stack((np.full(width, high), forward)),
                      np.column_stack((backward, np.full(width, high))),
                      np.column_stack((np.full(width, low), backward))))


def _annular_prism(outer_cells, hole_cells, height_cells):
    """Construct a closed 10-face frame using shared raw boundary row IDs.

    For outer n, hole h and height k, V=2(n+h)(n-h+2k), F=2V.
    The hole is centered, with positive integer wall thickness. Top/bottom
    triangulations and all eight wall grids share the SAME boundary vertex IDs.
    """
    n, h, k = outer_cells, hole_cells, height_cells
    assert all(type(value) is int for value in (n, h, k))
    assert 2 <= h < n <= 500 and 1 <= k <= 150 and (n-h) % 2 == 0
    margin = (n-h)//2
    high = margin+h
    vertex_count = 2*(n+h)*(n-h+2*k)
    triangle_count = 2*vertex_count

    axis = np.arange(n+1, dtype=np.int64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    keep = ~((x > margin) & (x < high) & (y > margin) & (y < high))
    xy = np.column_stack((x[keep], y[keep]))
    face_vertices = len(xy)
    assert face_vertices == (n+1)**2-(h-1)**2
    bottom_ids = np.full((n+1, n+1), -1, dtype=np.int64)
    bottom_ids[keep] = np.arange(face_vertices, dtype=np.int64)

    points = np.empty((vertex_count, 3), dtype=np.float64)
    points[:face_vertices, :2], points[:face_vertices, 2] = xy, 0.
    points[face_vertices:2*face_vertices, :2] = xy
    points[face_vertices:2*face_vertices, 2] = k

    outer = _boundary_square(0, n)
    inner = _boundary_square(margin, high)
    boundary = np.vstack((outer, inner))
    perimeter = len(boundary)
    assert perimeter == 4*(n+h)
    side_ids = np.empty((k+1, perimeter), dtype=np.int64)
    side_ids[0] = bottom_ids[boundary[:, 1], boundary[:, 0]]
    side_ids[-1] = side_ids[0]+face_vertices
    for layer in range(1, k):
        start = 2*face_vertices+(layer-1)*perimeter
        rows = np.arange(start, start+perimeter, dtype=np.int64)
        side_ids[layer] = rows
        points[start:start+perimeter, :2] = boundary
        points[start:start+perimeter, 2] = layer
    assert 2*face_vertices+(k-1)*perimeter == vertex_count

    cells = ~((x[:-1, :-1] >= margin) & (x[:-1, :-1] < high)
              & (y[:-1, :-1] >= margin) & (y[:-1, :-1] < high))
    a = bottom_ids[:-1, :-1][cells]
    b = bottom_ids[:-1, 1:][cells]
    c = bottom_ids[1:, 1:][cells]
    d = bottom_ids[1:, :-1][cells]
    horizontal_cells = len(a)
    assert horizontal_cells == n*n-h*h
    assert min(int(values.min()) for values in (a, b, c, d)) >= 0
    horizontal_triangles = 2*horizontal_cells
    triangles = np.empty((triangle_count, 3), dtype=np.int64)
    # Bottom is outward -z, top outward +z.
    triangles[:horizontal_triangles:2] = np.column_stack((a, c, b))
    triangles[1:horizontal_triangles:2] = np.column_stack((a, d, c))
    start = horizontal_triangles
    triangles[start:2*start:2] = np.column_stack((a, b, c))+face_vertices
    triangles[start+1:2*start:2] = np.column_stack((a, c, d))+face_vertices

    # Wrap separately around the outer and inner boundary loops. This is the
    # ordinary closure of physical wall perimeters, NOT a planar periodic seam.
    outer_next = np.r_[np.arange(1, len(outer)), 0]
    inner_next = np.r_[np.arange(len(outer)+1, perimeter), len(outer)]
    next_boundary = np.r_[outer_next, inner_next]
    low_a, low_b = side_ids[:-1].reshape(-1), side_ids[:-1, next_boundary].reshape(-1)
    high_a, high_b = side_ids[1:].reshape(-1), side_ids[1:, next_boundary].reshape(-1)
    wall_start = 2*horizontal_triangles
    triangles[wall_start::2] = np.column_stack((low_a, low_b, high_b))
    triangles[wall_start+1::2] = np.column_stack((low_a, high_b, high_a))
    walls = triangles[wall_start:].reshape(k, perimeter, 2, 3)
    # A hole's outward surface points into the hole, reversing its CCW walls.
    inner_walls = walls[:, len(outer):]
    inner_walls[..., [1, 2]] = inner_walls[..., [2, 1]]
    assert wall_start+2*k*perimeter == triangle_count
    return points, triangles


def _expected_faces(n, h, k):
    margin = (n-h)//2
    face_vertices = (n+1)**2-(h-1)**2
    records = []
    for height in (0, k):
        records.append(dict(axis=2, offset=height, other_axis=None,
            other_bounds=None, area=n*n-h*h, triangles=2*(n*n-h*h), support=face_vertices,
            center=[n/2., n/2., float(height)],
            covariance=np.diag([(n*n+h*h)/12., (n*n+h*h)/12., 0.])))
    for width, bounds, planes in ((n, (0, n), (0, n)),
                                  (h, (margin, margin+h), (margin, margin+h))):
        for axis in (0, 1):
            other = 1-axis
            for offset in planes:
                center = np.array([n/2., n/2., k/2.])
                center[axis] = offset
                covariance = np.zeros((3, 3))
                covariance[other, other], covariance[2, 2] = width*width/12., k*k/12.
                records.append(dict(axis=axis, offset=offset, other_axis=other,
                    other_bounds=bounds, area=width*k, triangles=2*width*k,
                    support=(width+1)*(k+1), center=center, covariance=covariance))
    assert len(records) == 10
    return records


def _assert_lineage_and_geometry(points, triangles, result, n, h, k):
    """Score after extraction; no expected face records enter the extractor."""
    assert result["status"] == "AVAILABLE", result["reasons"]
    assert result["reasons"] == [] and len(result["patches"]) == 10
    assert result["completeness_for_noisy_planes"] is False
    assert result["material_topology_verified"] is False
    assert result["accepted_registration"] is False and result["production_use_allowed"] is False
    expected = _expected_faces(n, h, k)
    expected_by_key = {(row["axis"], row["offset"]): row for row in expected}
    seen = set()
    triangle_coverage = np.zeros(len(triangles), dtype=np.uint8)
    raw_coverage = np.zeros(len(points), dtype=np.uint8)
    compact = []
    labels = result["triangle_component_labels"]
    assert labels.shape == (len(triangles),) and not labels.flags.writeable
    for patch in result["patches"]:
        normal = np.asarray(patch["normal"])
        axis = int(np.argmax(np.abs(normal)))
        signed_axis = np.zeros(3)
        signed_axis[axis] = 1.
        np.testing.assert_allclose(normal, signed_axis, rtol=0., atol=1e-12)
        offset = int(round(patch["offset_mm"]))
        key = (axis, offset)
        assert key in expected_by_key and key not in seen
        seen.add(key)
        truth = expected_by_key[key]
        assert patch["offset_mm"] == pytest.approx(offset, abs=1e-10)
        assert patch["triangle_count"] == truth["triangles"]
        assert patch["support_count"] == truth["support"]
        assert patch["surface_area_mm2"] == float(truth["area"])
        assert patch["area_measure"] == "TESSELLATION_AREA_WITH_MULTIPLICITY_NOT_MATERIAL_AREA"
        np.testing.assert_allclose(patch["area_centroid_mm"], truth["center"], rtol=0., atol=5e-8)
        # A conservative source-frozen floating summation comparison, not an
        # error-bar or a change to the extractor's 1e-8 mm maximum gate.
        np.testing.assert_allclose(patch["surface_covariance_mm2"], truth["covariance"], rtol=0., atol=2e-4)
        assert patch["full_support_max_abs_residual_mm"] <= 1e-8
        raw = patch["raw_vertex_rows"]
        rows = patch["triangle_rows"]
        assert raw.dtype == rows.dtype == np.dtype("<i8")
        assert not raw.flags.writeable and not rows.flags.writeable
        assert len(raw) == truth["support"] and len(rows) == truth["triangles"]
        assert np.all(raw[1:] > raw[:-1]) and np.all(rows[1:] > rows[:-1])
        assert raw[0] >= 0 and raw[-1] < len(points)
        assert rows[0] >= 0 and rows[-1] < len(triangles)
        assert np.all(labels[rows] == patch["component_id"])

        mask = points[:, axis] == offset
        if truth["other_axis"] is not None:
            other = truth["other_axis"]
            lower, upper = truth["other_bounds"]
            mask &= (points[:, other] >= lower) & (points[:, other] <= upper)
        assert np.array_equal(raw, np.flatnonzero(mask))
        raw_coverage[raw] += 1
        triangle_coverage[rows] += 1
        # Every original triangle is checked, in bounded chunks; no sampled
        # subset or successful-patch-only denominator replaces full support.
        for start in range(0, len(rows), 65536):
            coordinates = points[triangles[rows[start:start+65536]]]
            assert np.all(coordinates[:, :, axis] == offset)
            if truth["other_axis"] is not None:
                values = coordinates[:, :, truth["other_axis"]]
                assert np.all(values >= lower) and np.all(values <= upper)
        compact.append(dict(axis=axis, offset_mm=offset, triangle_count=len(rows),
                            support_count=len(raw), area_mm2=patch["surface_area_mm2"],
                            maximum_residual_mm=patch["full_support_max_abs_residual_mm"]))

    assert seen == set(expected_by_key)
    assert np.all(triangle_coverage == 1)
    crease_edges = 8*(n+h+k)
    coverage_histogram = {multiplicity: int(np.count_nonzero(raw_coverage == multiplicity))
                          for multiplicity in (0, 1, 2, 3)}
    assert coverage_histogram == {0: 0, 1: len(points)-crease_edges+8,
                                  2: crease_edges-24, 3: 16}
    memberships = len(points)+crease_edges+8
    statistics = result["statistics"]
    required = dict(input_triangles=len(triangles), edge_records=3*len(triangles),
        shared_edge_pairs=3*len(triangles)//2,
        compatible_edge_pairs=3*len(triangles)//2-crease_edges,
        component_count=10, components_evaluated=10, qualified_patch_count=10,
        rejected_component_count=0, support_memberships=memberships,
        full_vertex_tests=memberships, component_search_complete=True)
    for name, value in required.items():
        assert statistics[name] == value, name
    assert len(result["components"]) == 10
    assert all(row["disposition"] == "PROPOSED" and row["reasons"] == []
               for row in result["components"])
    return dict(geometry=sorted(compact, key=lambda row: (row["axis"], row["offset_mm"])),
                statistics=required, raw_coverage_histogram=coverage_histogram,
                triangle_coverage_histogram={1: len(triangles)},
                all_raw_rows_accounted=True, all_triangle_rows_accounted=True)


def test_small_constructor_is_closed_oriented_manifold_without_padding_or_unused_rows():
    points, triangles = _annular_prism(6, 2, 3)
    assert points.shape == (160, 3) and triangles.shape == (320, 3)
    assert np.array_equal(np.unique(triangles), np.arange(len(points)))
    assert len(np.unique(points, axis=0)) == len(points)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    orientations = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    undirected, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                          return_inverse=True, return_counts=True)
    assert np.all(counts == 2)
    assert np.all(np.bincount(inverse, weights=orientations) == 0)
    assert len(points)-len(undirected)+len(triangles) == 0
    coordinates = points[triangles]
    crosses = np.cross(coordinates[:, 1]-coordinates[:, 0], coordinates[:, 2]-coordinates[:, 0])
    assert np.all(np.linalg.norm(crosses, axis=1) == 1.)


def test_small_actual_extractor_checks_every_face_and_raw_row_against_analytic_oracle():
    n, h, k = 8, 4, 2
    points, triangles = _annular_prism(n, h, k)
    assert points.shape == (192, 3) and triangles.shape == (384, 3)
    result = extract_coplanar_scan_patches_v4(points, triangles)
    _assert_lineage_and_geometry(points, triangles, result, n, h, k)


@pytest.mark.skipif(os.environ.get(CAPACITY_FLAG) != "1",
                    reason="separate 120 s / 90 s CPU / 4 GiB capacity authorization required")
def test_exact_800000_vertex_1600000_triangle_capacity_receipt():
    started = time.perf_counter()
    n, h, k = CAPACITY_DIMENSIONS
    points, triangles = _annular_prism(n, h, k)
    assert (len(points), len(triangles)) == CAPACITY_COUNTS
    extraction_started = time.perf_counter()
    result = extract_coplanar_scan_patches_v4(points, triangles)
    extraction_seconds = time.perf_counter()-extraction_started
    receipt = _assert_lineage_and_geometry(points, triangles, result, n, h, k)
    elapsed = time.perf_counter()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak*1024)
    receipt.update(schema="MESH_PLANE_CAPACITY_V4_INVENTED_ANNULAR_PRISM",
        raw_vertices=len(points), triangles=len(triangles), dimensions_cells=[n, h, k],
        construction_and_extraction_and_verification_seconds=elapsed,
        extraction_seconds=extraction_seconds, worker_lifetime_peak_rss_bytes=peak_bytes,
        measurement_scope="WORKER_LIFETIME_RSS_AFTER_ALL_LINEAGE_CHECKS_PARENT_ENFORCES_FULL_PROCESS_GROUP",
        qualification_scope="ONE_EXACT_AXIS_ALIGNED_MANIFOLD_CONSTRUCTION_NOT_NOISY_OR_WORST_CASE",
        limits=RESOURCE_LIMITS)
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) < 16384
    print("MESH_PLANE_CAPACITY_V4_RECEIPT="+encoded)
