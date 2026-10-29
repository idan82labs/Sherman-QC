"""ONE prospective dense-support context capacity outcome; source-only authoring.

Exactly900000 unique invented XYZ rows on two500x900 integer planar grids,
1794404 scan triangles, two450000-row supports and1800000 indexed memberships.
No input files, physical inference, native CAD, plane extraction, refinement,
measurement fit or scoring. Plane declarations are explicit fixture premises.
This complements, never reruns/replaces, the qualified mostly-coincident900k
fixture. Root must freeze94 total integration outcomes and the exact independent
PRE/archive before one45CPU/55work/60wall/1GiB/1MiB source gate. These values are
prospective caps, not authorization or predicted runtime/RSS.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ROWS = 900_000
SUPPORT_ROWS = 450_000
TRIANGLES_PER_SUPPORT = 897_202
MEMBERSHIPS = 1_800_000
SLOTS = ("WHOLE_MIN", "WHOLE_MAX", "FACE0_MIN", "FACE0_MAX", "FACE1_MIN", "FACE1_MAX",
    "WHOLE_CONTEXT_MIDPOINT", "ASSIGNED_CONTEXT_INTERSECTION_MIDPOINT")


@pytest.fixture(scope="module")
def context_module():
    name = "_invented_context_v3_dense_support"
    assert not any(key == name or key.startswith(name+".") for key in sys.modules)
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT/"backend")]
    sys.modules[name] = package
    try:
        yield importlib.import_module(name+".cad_guided_two_plane_context_v3")
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name+"."):
                del sys.modules[key]


def _sha(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _case(m):
    points = np.empty((ROWS, 3), dtype=np.float64)
    z = np.arange(900, dtype=np.float64)
    # For each sheet, i=floor(local_row/900), j=local_row%900 are unique.
    # Sheets are disjoint: first x=0,y>=2; second x>=3,y=0.
    for i in range(500):
        first = slice(i*900, (i+1)*900)
        second = slice(SUPPORT_ROWS+i*900, SUPPORT_ROWS+(i+1)*900)
        points[first, 0], points[first, 1], points[first, 2] = 0., float(i+2), z
        points[second, 0], points[second, 1], points[second, 2] = float(i+3), 0., z
    triangles = np.empty((2*TRIANGLES_PER_SUPPORT, 3), dtype=np.int64)
    columns = np.arange(899, dtype=np.int64)
    for sheet in range(2):
        for i in range(499):
            a = SUPPORT_ROWS*sheet+900*i+columns
            start = TRIANGLES_PER_SUPPORT*sheet+2*899*i
            first = triangles[start:start+2*899:2]
            second = triangles[start+1:start+2*899:2]
            first[:, 0], first[:, 1], first[:, 2] = a, a+900, a+901
            second[:, 0], second[:, 1], second[:, 2] = a, a+901, a+1
    # Independently specified four-triangle CAD rectangles, not a sampling of
    # scan arrays. Original triangle-order witness wins ties, not source ID.
    cad = np.array((
        ((0., 2., 0.), (0., 501., 0.), (0., 501., 899.)),
        ((0., 2., 0.), (0., 501., 899.), (0., 2., 899.)),
        ((3., 0., 0.), (502., 0., 0.), (502., 0., 899.)),
        ((3., 0., 0.), (502., 0., 899.), (3., 0., 899.))), dtype=np.float64)
    faces = np.array((17, 17, 29, 29), dtype=np.int64)
    local = np.array((14, 5, 18, 9), dtype=np.int64)
    scans, cads = [], []
    for i, normal in enumerate(((1., 0., 0.), (0., 1., 0.))):
        raw = np.arange(i*SUPPORT_ROWS, (i+1)*SUPPORT_ROWS, dtype=np.int64)
        scan_tri = np.arange(i*TRIANGLES_PER_SUPPORT, (i+1)*TRIANGLES_PER_SUPPORT, dtype=np.int64)
        cad_tri = np.array((2*i, 2*i+1), dtype=np.int64)
        scans.append(m.ContextPlaneV1(31+i, normal, (0., 0., 0.), 0., 0., True,
            scan_tri, _sha(scan_tri), raw, _sha(raw)))
        cads.append(m.ContextPlaneV1((17, 29)[i], normal, (0., 0., 0.), 0., 0., True,
            cad_tri, _sha(cad_tri)))
    return dict(scan_points=points, scan_triangles=triangles, cad_triangles=cad,
        cad_face_ids=faces, cad_source_triangle_ids=local, scan_planes=tuple(scans), cad_planes=tuple(cads),
        ordered_matches=(m.OrderedPlanePairV1((31, 32), (17, 29)),),
        scan_origin_mm=(0., 0., 0.), cad_origin_mm=(0., 0., 0.),
        unit_assumption="INVENTED_DENSE_GRID_MM_NOT_CALIBRATION", source_frame="INVENTED_DENSE_SCAN",
        target_frame="INVENTED_DENSE_CAD")


def test_single_dense_support_context_complete_cumulative_capacity(context_module):
    m = context_module
    args = _case(m)
    assert args["scan_points"].shape == (900_000, 3)
    assert args["scan_triangles"].shape == (1_794_404, 3)
    assert [len(p.raw_vertex_rows) for p in args["scan_planes"]] == [450_000, 450_000]
    # Source row formulas, not floating uniqueness sorting, establish the
    # complete Cartesian grids; check every constructed coordinate in-place.
    for i in range(500):
        a = args["scan_points"][i*900:(i+1)*900]
        b = args["scan_points"][SUPPORT_ROWS+i*900:SUPPORT_ROWS+(i+1)*900]
        assert np.all(a[:, 0] == 0.) and np.all(a[:, 1] == i+2)
        assert np.all(b[:, 0] == i+3) and np.all(b[:, 1] == 0.)
        assert np.array_equal(a[:, 2], np.arange(900)) and np.array_equal(b[:, 2], np.arange(900))
    result = m.propose_two_plane_context_v3(**args)
    assert result["status"] == "COMPLETE", result["reasons"]
    assert result["selected_policy_evaluation_complete"] is True
    assert result["accepted_registration"] is False and result["finite_support_match_verified"] is False
    assert result["production_use_allowed"] is False and result["observed_bend_count"] is None
    assert result["search_exhaustive"] is False and result["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert result["input_counts"] == dict(scan_raw_rows=900_000, scan_triangles=1_794_404, cad_triangles=4)
    for name in ("scan_points", "scan_triangles", "cad_triangles", "cad_face_ids", "cad_source_triangle_ids"):
        assert result["input_digests"][name+"_sha256"] == _sha(args[name])
    stats = result["statistics"]
    assert result["scheduled_pairs"] == [dict(scan_plane_ids=[31, 32], cad_plane_ids=[17, 29])]
    assert stats["signed_families_planned"] == stats["signed_families_attempted"] == stats["signed_families_completed"] == 4
    assert stats["candidate_slots_planned"] == stats["slot_records_completed"] == 32
    assert stats["proposed_slot_count"] == 32 and stats["absent_midpoint_slots"] == stats["rejected_slot_count"] == 0
    assert [f["normal_signs"] for f in result["families"]] == [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    assert all(tuple(s["slot"] for s in f["slots"]) == SLOTS for f in result["families"])
    indexed = result["scan_extrema_indexes"]
    assert len(indexed) == stats["extrema_indexes"] == stats["extrema_completed_indexes"] == 3
    assert [row["support_row_count"] for row in indexed] == [900_000, 450_000, 450_000]
    assert [row["node_slots"] for row in indexed] == [8192, 4096, 4096]
    assert all(row["leaf_rows"] == 256 and row["sort_axis"] == 2 for row in indexed)
    assert stats["extrema_indexed_rows"] == 1_800_000
    assert all(row["point_buffer_sha256"] == _sha(args["scan_points"]) for row in indexed)

    scan_projections = [p for p in result["context_projections"] if p["role"] == "scan"]
    assert len(scan_projections) == 3
    # Analytic staged oracle: exact integer j times+1, other products zero,
    # then explicit binary64 additions of zero. Every row yields j in[0,899].
    # Smallest original raw IDs at j=0/j=899 are known without implementation.
    for record in scan_projections:
        assert record["projection_direction"] == [0., 0., 1.]
        assert record["projected_coordinate_origin_mm"] == [0., 0., 0.]
        assert record["projected_constant_mm"] == 0.
        assert record["interval_mm"] == [0., 899.]
        if record["scope"] == "WHOLE_SUPPLIED_BUFFER":
            low, high = 0, 899
        else:
            support = next(p for p in args["scan_planes"] if p.raw_vertex_rows_sha256 == record["support_rows_sha256"])
            low = 0 if support.plane_id == 31 else 450_000
            high = low+899
        assert record["minimum_witness"] == dict(raw_vertex_row=low)
        assert record["maximum_witness"] == dict(raw_vertex_row=high)
    for record in (p for p in result["context_projections"] if p["role"] == "cad"):
        direction = record["projection_direction"]
        assert direction[:2] == [0., 0.] and direction[2] in (-1., 1.)
        positive = direction[2] == 1.
        assert record["interval_mm"] == ([0., 899.] if positive else [-899., 0.])
        if record["scope"] == "WHOLE_SUPPLIED_BUFFER":
            triangle = 0
        else:
            plane = next(p for p in args["cad_planes"] if p.triangle_rows_sha256 == record["support_rows_sha256"])
            triangle = 0 if plane.plane_id == 17 else 2
        for name, slot in (("minimum_witness", 0 if positive else 2), ("maximum_witness", 2 if positive else 0)):
            assert record[name] == dict(triangle_row=triangle, vertex_slot=slot,
                face_id=(17 if triangle == 0 else 29), source_triangle_id=(14 if triangle == 0 else 18))

    work = stats["extrema_work"]
    evaluated = work["query_rows_evaluated"]
    assert work["queries"] == stats["extrema_queries"] == 3
    assert work["row_visits"] == 3_600_000+evaluated
    assert work["index_visits"] == 9_000_000+evaluated
    assert work["coordinate_visits"] == 34_200_000+9*evaluated
    assert work["sort_work"] == 35_100_000 and work["hash_bytes"] == 14_400_000
    assert work["owned_bytes"] == 15_579_648
    assert stats["scan_row_visits"] == 4_500_000+evaluated
    assert stats["index_visits"] == 23_360_860+evaluated
    assert stats["index_sort_work"] == 153_530_676
    assert stats["owned_array_bytes"] == 86_221_376
    assert stats["hash_bytes"] == 100_621_312 and stats["cad_vertex_visits"] == 60
    assert work["node_visits"] == 8189+sum(p["extrema_query_work"]["node_visits"] for p in scan_projections)
    assert 0 < evaluated <= MEMBERSHIPS+21  # Worst full leaf evaluation plus at most7seeds/index.
    assert stats["temporary_bytes"] == 0
    assert stats["accounted_working_peak_bytes"] <= 268_435_456
    assert stats["accounted_working_peak_bytes"] >= stats["owned_array_bytes"]+work["owned_bytes"]+131_072
    assert result["budgets"]["max_scan_triangles"] == 2_000_000
    assert result["budgets"]["max_extrema_indexed_rows"] == 4_000_000
    assert result["budgets"]["max_index_visits"] == 64_000_000

    summary = dict(record="CONTEXT_V3_DENSE_SUPPORT_CAPACITY", raw_rows=ROWS, distinct_xyz_rows=ROWS,
        scan_triangles=1_794_404, supports=[450_000, 450_000], indexed_memberships=MEMBERSHIPS,
        ordered_matches=1, signed_families=4, slots=32,
        full_query_reference_rows=MEMBERSHIPS, raw_sha256=result["input_digests"]["scan_points_sha256"],
        statistics=stats, indexes=indexed,
        scan_extrema=[dict(scope=p["scope"], support_rows=p["support_row_count"], interval=p["interval_mm"],
            minimum=p["minimum_witness"], maximum=p["maximum_witness"]) for p in scan_projections],
        rss_scope="PROCESS_RSS_MUST_COME_FROM_SUPERVISOR_NOT_MODELED_PAYLOAD_COUNTERS",
        evidence_scope="INVENTED_DENSE_SUPPORT_CAPACITY_NOT_FULL_WORKFLOW_SPEEDUP_OR_PHYSICAL_ACCURACY")
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
    line = "CONTEXT_V3_DENSE_SUPPORT_CAPACITY_JSON="+encoded
    assert len((line+"\n").encode()) <= 8192
    print(line)
