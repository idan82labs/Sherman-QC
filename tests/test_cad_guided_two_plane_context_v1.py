"""Independent invented two-plane seed contracts; no native/data/integration.

All expected poses and finite geometry are constructed here, independently of
the helper. Exact/permutation fixtures use a fixed1e-8mm action ceiling. No truth
pose is passed to inference. These are proposal tests, not support, registration,
scanner calibration, observed-count or downstream measurement qualification.
Root alone freezes/runs this source suite under60wall/45workerCPU/1GiB/1MiB.
"""
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]/"backend/cad_guided_two_plane_context_v1.py"
SLOTS = ("WHOLE_MIN", "WHOLE_MAX", "FACE0_MIN", "FACE0_MAX", "FACE1_MIN", "FACE1_MAX",
    "WHOLE_CONTEXT_MIDPOINT", "ASSIGNED_CONTEXT_INTERSECTION_MIDPOINT")
POSES = (
    (np.eye(3), (0., 0., 0.)),
    (np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), (3., -5., 7.)),
    (np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), (-8., 4., 11.)),
    (np.diag([-1., 1., -1.]), (6., 9., -2.)),
)
PROBES = np.array([[0., 0., 0.], [4., -3., 2.], [-5., 7., 11.], [9., 1., 30.]])


@pytest.fixture
def kernel(monkeypatch):
    spec = importlib.util.spec_from_file_location("_invented_two_plane_context_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _quad(which, interval):
    low, high = interval
    if which == 0:
        return np.array([[0., -7., low], [0., 7., low], [0., 7., high], [0., -7., high]])
    return np.array([[-5., 0., low], [5., 0., low], [5., 0., high], [-5., 0., high]])


def _case(kernel, pose=0, *, scan_interval=(0., 30.), cad_interval=(0., 30.), weak_caps=False):
    rotation, translation = POSES[pose]
    translation = np.asarray(translation)
    scan = np.vstack([_quad(i, scan_interval) for i in range(2)])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    # Construct CAD independently from its own full intervals, not scan rows.
    cad_points = np.vstack([_quad(i, cad_interval) for i in range(2)])@rotation.T+translation
    cad = np.asarray(cad_points[triangles], dtype=np.float64)
    faces = np.array([101, 101, 202, 202], dtype=np.int64)
    local_ids = np.array([7, 11, 7, 11], dtype=np.int64)
    if weak_caps:
        extra = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                          [0., 0., 30.], [2., 0., 30.], [0., 2., 30.]])
        scan = np.vstack((scan, extra))
        triangles = np.vstack((triangles, [[8, 9, 10], [11, 12, 13]])).astype(np.int64)
    scan_planes, cad_planes = [], []
    anchors = ((2., 4., 6.), (-3., 5., -4.))
    for i, (normal, offset) in enumerate((((1., 0., 0.), -2.), ((0., 1., 0.), -5.))):
        tri_rows = np.array([2*i, 2*i+1], dtype=np.int64)
        raw_rows = np.arange(4*i, 4*i+4, dtype=np.int64)
        scan_planes.append(kernel.ContextPlaneV1(10+10*i, normal, anchors[i], offset, 0., True,
            tri_rows, _sha(tri_rows), raw_rows, _sha(raw_rows)))
        cad_planes.append(kernel.ContextPlaneV1(101+101*i, tuple(float(x) for x in rotation@normal),
            tuple(float(x) for x in rotation@anchors[i]+translation), offset, 0., True,
            tri_rows.copy(), _sha(tri_rows)))
    arguments = dict(scan_points=scan, scan_triangles=triangles, cad_triangles=cad,
        cad_face_ids=faces, cad_source_triangle_ids=local_ids, scan_planes=tuple(scan_planes),
        cad_planes=tuple(cad_planes), ordered_matches=(kernel.OrderedPlanePairV1((10, 20), (101, 202)),),
        scan_origin_mm=(0., 0., 0.), cad_origin_mm=(0., 0., 0.),
        unit_assumption="INVENTED_COMMON_MM_NOT_CALIBRATED", source_frame="INVENTED_SCAN", target_frame="INVENTED_CAD")
    return arguments, rotation.copy(), translation.copy()


def _invoke(kernel, arguments):
    return kernel.propose_two_plane_context_v1(**arguments)


def _safety(result):
    assert result["accepted_registration"] is False
    assert result["finite_support_match_verified"] is False
    assert result["production_use_allowed"] is False and result["observed_bend_count"] is None
    assert result["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert result["axial_context_status"] == "AXIAL_CONTEXT_NOT_CERTIFIED"
    assert result["search_exhaustive"] is False
    assert result["scheduler_qualification"] == "NOT_IMPLEMENTED_CALLER_SCHEDULE_NOT_AUTHENTICATED"
    assert "NO_OFFSET_ERROR_OR_POSE_ENCLOSURE" in result["translation_uncertainty"]
    json.dumps(result, allow_nan=False)


def _errors(result, rotation, translation):
    truth = PROBES@rotation.T+translation
    return [float(np.max(np.linalg.norm(PROBES@np.asarray(row["rotation"]).T+
        row["translation_mm"]-truth, axis=1))) for row in result["candidates"]]


def _complete(result, assignments=1):
    _safety(result)
    assert result["status"] == "COMPLETE" and result["selected_policy_evaluation_complete"] is True
    stats = result["statistics"]
    assert stats["signed_families_planned"] == stats["signed_families_attempted"] == stats["signed_families_completed"] == 4*assignments
    assert stats["candidate_slots_planned"] == stats["slot_records_completed"] == 32*assignments
    assert sum(len(row["slots"]) for row in result["families"]) == 32*assignments
    assert all(tuple(slot["slot"] for slot in row["slots"]) == SLOTS for row in result["families"])
    assert stats["proposed_slot_count"]+stats["absent_midpoint_slots"]+stats["rejected_slot_count"] == 32*assignments
    slots = {(family["signed_family_id"], slot["slot"]): slot for family in result["families"] for slot in family["slots"]}
    origins = []
    for index, row in enumerate(result["candidates"]):
        r = np.asarray(row["rotation"])
        assert row["candidate_index"] == index and row["accepted_registration"] is False
        np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-10, rtol=0.)
        assert abs(np.linalg.det(r)-1.) <= 1e-10
        for witness in row["contributing_slots"]:
            identity = (witness["signed_family_id"], witness["slot"])
            assert slots[identity]["candidate_index"] == index and slots[identity]["status"] == "PROPOSED_SEED_ONLY"
            origins.append(identity)
    assert len(origins) == len(set(origins)) == stats["proposed_slot_count"]


def _refused(result):
    _safety(result)
    assert result["status"] == "UNAVAILABLE" and result["selected_policy_evaluation_complete"] is False
    assert result["usable_proposals"] is False and result["reasons"]
    for key in ("plane_bindings", "scheduled_pairs", "families", "candidates", "context_projections"):
        assert result[key] == []
    assert result["input_digests"] == {}
    assert len(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()) <= 4096


@pytest.mark.parametrize("pose", range(4))
@pytest.mark.parametrize("variant", ("original", "order_and_sign", "origins"))
def test_exact_independent_pose_present_under_sign_order_and_origin_changes(kernel, pose, variant):
    args, rotation, translation = _case(kernel, pose)
    if variant == "order_and_sign":
        args["scan_planes"] = tuple(replace(p, normal=tuple(-x for x in p.normal),
            offset_from_anchor_mm=-p.offset_from_anchor_mm) for p in args["scan_planes"])[::-1]
        args["cad_planes"] = args["cad_planes"][::-1]
        args["ordered_matches"] = (kernel.OrderedPlanePairV1((20, 10), (202, 101)),)
    elif variant == "origins":
        args.update(scan_origin_mm=(5., -8., 7.), cad_origin_mm=(-3., 2., -3.))
    result = _invoke(kernel, args)
    _complete(result)
    assert result["usable_proposals"] is True and min(_errors(result, rotation, translation)) <= 1e-8
    assert result["input_counts"] == dict(scan_raw_rows=8, scan_triangles=4, cad_triangles=4)
    for name in ("scan_points", "scan_triangles", "cad_triangles", "cad_face_ids", "cad_source_triangle_ids"):
        assert result["input_digests"][name+"_sha256"] == _sha(args[name])


def test_anchored_axial_origin_regression_votes_ten_not_zero(kernel):
    args, rotation, translation = _case(kernel)
    args.update(scan_origin_mm=(0., 0., 7.), cad_origin_mm=(0., 0., -3.))
    result = _invoke(kernel, args); _complete(result)
    family, = [row for row in result["families"] if row["normal_signs"] == [1, 1]]
    assert family["rotation"] == np.eye(3).tolist()
    assert family["translation_family"]["target_axial_direction"] == [0., 0., 1.]
    assert all(row["lambda_mm"] == 10. for row in family["slots"])
    assert min(_errors(result, rotation, translation)) == 0.


def test_fraction_plane_constraints_hold_for_every_proper_signed_seed(kernel):
    args, _, _ = _case(kernel, 1)
    result = _invoke(kernel, args); _complete(result)
    scan = {p.plane_id: p for p in args["scan_planes"]}
    cad = {p.plane_id: p for p in args["cad_planes"]}
    F = Fraction.from_float
    for family in result["families"]:
        for slot in family["slots"]:
            if slot["status"] != "PROPOSED_SEED_ONLY": continue
            candidate = result["candidates"][slot["candidate_index"]]
            r, t = candidate["rotation"], candidate["translation_mm"]
            for pair in family["correspondences"]:
                target, source = cad[pair["cad_plane_id"]], scan[pair["scan_plane_id"]]
                for row in source.raw_vertex_rows:
                    point = args["scan_points"][row]
                    y = [sum(F(float(r[i][j]))*F(float(point[j])) for j in range(3))+F(float(t[i])) for i in range(3)]
                    residual = sum(F(float(target.normal[i]))*(y[i]-F(float(target.anchor_mm[i]))) for i in range(3))-F(float(target.offset_from_anchor_mm))
                    assert residual == 0


@pytest.mark.parametrize("scale", (.01, 1., 100.))
def test_unit_rescaling_of_declared_geometry_preserves_exact_pose_action(kernel, scale):
    args, rotation, translation = _case(kernel, 2)
    for name in ("scan_points", "cad_triangles"): args[name] = args[name]*scale
    for name in ("scan_planes", "cad_planes"):
        args[name] = tuple(replace(p, anchor_mm=tuple(scale*x for x in p.anchor_mm),
            offset_from_anchor_mm=scale*p.offset_from_anchor_mm) for p in args[name])
    result = _invoke(kernel, args); _complete(result)
    assert min(_errors(result, rotation, translation*scale)) <= 1e-8


def test_whole_raw_context_retains_weak_caps_without_inventing_a_third_plane(kernel):
    args, rotation, translation = _case(kernel, scan_interval=(1., 29.), weak_caps=True)
    result = _invoke(kernel, args); _complete(result)
    assert len(args["scan_planes"]) == 2 and min(_errors(result, rotation, translation)) <= 1e-8
    family, = [row for row in result["families"] if row["normal_signs"] == [1, 1]]
    assert [row["lambda_mm"] for row in family["slots"]] == [0., 0., -1., 1., -1., 1., 0., 0.]
    whole = result["context_projections"][family["context_intervals"][0]["scan_projection_id"]]
    assert whole["row_visits"] == 14 and whole["interval_mm"] == [0., 30.]
    assert whole["minimum_witness"]["raw_vertex_row"] >= 8 and whole["maximum_witness"]["raw_vertex_row"] >= 8


def test_partial_scan_keeps_axial_continuum_and_midpoint_is_not_unique(kernel):
    args, _, _ = _case(kernel, scan_interval=(10., 20.))
    result = _invoke(kernel, args); _complete(result)
    family, = [row for row in result["families"] if row["normal_signs"] == [1, 1]]
    assert all(row["heuristic_lambda_interval_mm"] == [-10., 10.] for row in family["context_intervals"])
    assert {row["lambda_mm"] for row in family["slots"]} == {-10., 0., 10.}
    assert family["translation_family"]["axial_translation_resolved"] is False


def test_asymmetric_partial_context_can_miss_true_pose_without_certifying_exclusion(kernel):
    args, rotation, translation = _case(kernel, scan_interval=(5., 28.))
    result = _invoke(kernel, args); _complete(result)
    assert min(_errors(result, rotation, translation)) > 1e-8
    family, = [row for row in result["families"] if row["normal_signs"] == [1, 1]]
    assert {row["lambda_mm"] for row in family["slots"]} == {-5., 2., -1.5}
    assert result["usable_proposals"] is True  # Candidate seeds, not a truth-coverage claim.


def test_remote_outlier_retains_endpoint_votes_but_marks_empty_midpoint_slots(kernel):
    args, _, _ = _case(kernel, scan_interval=(0., 40.))
    result = _invoke(kernel, args); _complete(result)
    family, = [row for row in result["families"] if row["normal_signs"] == [1, 1]]
    assert [row["status"] for row in family["slots"]][-2:] == ["ABSENT_EMPTY_HEURISTIC_INTERVAL"]*2
    assert all(not row["interval_nonempty"] for row in family["context_intervals"])
    assert result["statistics"]["absent_midpoint_slots"] == 8


def test_exact_pose_dedup_retains_all_slots_and_projection_cache_work(kernel):
    args, _, _ = _case(kernel)
    result = _invoke(kernel, args); _complete(result)
    assert len(result["candidates"]) == 4
    assert all(len(row["contributing_slots"]) == 8 for row in result["candidates"])
    stats = result["statistics"]
    assert stats["exact_pose_duplicates"] == 28
    # Source axial direction is invariant across target signs (R^T u=e3).
    # Three source contexts and3target contexts in each of two directions.
    assert stats["projection_cache_entries"] == stats["projection_cache_misses"] == 9
    assert stats["projection_cache_hits"] == 15
    assert stats["scan_row_visits"] == 8+(8+4+4)
    assert stats["cad_vertex_visits"] == 12+2*(12+6+6)


def test_normal_cone_central_seed_is_not_an_enclosure_of_feasible_true_pose(kernel):
    args, rotation, translation = _case(kernel)
    delta = .01
    p = args["scan_planes"][0]
    args["scan_planes"] = (replace(p, normal=(float(np.cos(delta)), float(np.sin(delta)), 0.),
        anchor_mm=(0., 0., 0.), offset_from_anchor_mm=0., normal_angle_bound_radians=delta), args["scan_planes"][1])
    result = _invoke(kernel, args); _complete(result)
    assert result["usable_proposals"] is True and min(_errors(result, rotation, translation)) > .01
    assert result["translation_uncertainty"] == "UNAVAILABLE_NO_OFFSET_ERROR_OR_POSE_ENCLOSURE_MODEL"


@pytest.mark.parametrize("which", ("scan", "cad"))
@pytest.mark.parametrize("mode", ("parallel", "near_parallel", "cone_margin"))
def test_rank_two_and_each_side_cone_independence_are_checked(kernel, which, mode):
    args, _, _ = _case(kernel)
    theta = {"parallel": 0., "near_parallel": .02, "cone_margin": .08}[mode]
    for name in ("scan_planes", "cad_planes"):
        planes = list(args[name])
        planes[1] = replace(planes[1], normal=(float(np.cos(theta)), float(np.sin(theta)), 0.))
        if name == which+"_planes" and mode == "cone_margin":
            planes = [replace(p, normal_angle_bound_radians=.04) for p in planes]
        args[name] = tuple(planes)
    result = _invoke(kernel, args); _complete(result)
    assert result["candidates"] == [] and result["statistics"]["rejected_slot_count"] == 32
    for family in result["families"]:
        rank = family["source_rank" if which == "scan" else "target_rank"]
        if mode == "cone_margin":
            assert rank["normal_condition_ratio"] > .02 and rank["normal_independence_margin"] < 0.
            assert "TWO_ROW_INDEPENDENCE_UNRESOLVED_WITH_CONES" in rank["reasons"]
        else: assert "NOMINAL_TWO_ROW_SYSTEM_ILL_CONDITIONED" in rank["reasons"]


def test_disjoint_included_angle_intervals_are_complete_geometric_pair_refusals(kernel):
    args, _, _ = _case(kernel)
    args["cad_planes"] = (args["cad_planes"][0], replace(args["cad_planes"][1], normal=(.6, .8, 0.)))
    result = _invoke(kernel, args); _complete(result)
    assert result["candidates"] == []
    assert all(row["reasons"] == ["CENTRAL_PAIR_ANGLE_INTERVALS_DISJOINT"] for row in result["families"])
    assert result["statistics"]["projection_cache_misses"] == 0


@pytest.mark.parametrize("which", ("source", "target", "both"))
def test_independently_changed_rigid_frames_preserve_physical_action(kernel, which):
    args, r, t = _case(kernel, 1)
    # Proper rational rotation from unit quaternion (1,2,2,0)/3; this is NOT a
    # permutation-axis fixture. Each frame is changed before helper invocation.
    q = np.array([[1., 8., 4.], [8., 1., -4.], [-4., 4., -7.]])/9.
    h = np.array([13., -17., 29.])
    qs, qt = (q if which in ("source", "both") else np.eye(3)), (q.T if which in ("target", "both") else np.eye(3))
    hs, ht = (h if which in ("source", "both") else np.zeros(3)), (-h if which in ("target", "both") else np.zeros(3))
    args["scan_points"] = args["scan_points"]@qs.T+hs
    args["cad_triangles"] = args["cad_triangles"]@qt.T+ht
    for role, transform, shift in (("scan", qs, hs), ("cad", qt, ht)):
        args[role+"_planes"] = tuple(replace(p,
            normal=tuple(float(x) for x in transform@p.normal),
            anchor_mm=tuple(float(x) for x in transform@p.anchor_mm+shift)) for p in args[role+"_planes"])
    args.update(scan_origin_mm=tuple(float(x) for x in hs), cad_origin_mm=tuple(float(x) for x in ht))
    truth_r, truth_t = qt@r@qs.T, qt@t+ht-qt@r@qs.T@hs
    result = _invoke(kernel, args); _complete(result)
    assert min(_errors(result, truth_r, truth_t)) <= 1e-8


def test_raw_and_triangle_permutations_keep_exact_geometry_and_bound_witnesses(kernel):
    args, r, t = _case(kernel, 2)
    perm = np.array([7, 0, 5, 2, 3, 6, 1, 4]); inverse = np.argsort(perm)
    args["scan_points"] = args["scan_points"][perm]
    args["scan_triangles"] = inverse[args["scan_triangles"]].astype(np.int64)
    args["scan_planes"] = tuple(replace(p, raw_vertex_rows=np.sort(inverse[p.raw_vertex_rows]).astype(np.int64),
        raw_vertex_rows_sha256=_sha(np.sort(inverse[p.raw_vertex_rows]).astype(np.int64))) for p in args["scan_planes"])
    order = np.array([3, 0, 2, 1])
    for name in ("cad_triangles", "cad_face_ids", "cad_source_triangle_ids"): args[name] = args[name][order]
    args["cad_planes"] = tuple(replace(p,
        triangle_rows=np.flatnonzero(args["cad_face_ids"] == p.plane_id).astype(np.int64),
        triangle_rows_sha256=_sha(np.flatnonzero(args["cad_face_ids"] == p.plane_id).astype(np.int64))) for p in args["cad_planes"])
    result = _invoke(kernel, args); _complete(result)
    assert min(_errors(result, r, t)) <= 1e-8
    _projection_oracles(result, args)


def _projection_oracles(result, args):
    """Recompute extrema only from explicit complete original support identities."""
    scan = {_sha(p.raw_vertex_rows): p.raw_vertex_rows for p in args["scan_planes"]}
    cad = {_sha(p.triangle_rows): p.triangle_rows for p in args["cad_planes"]}
    for record in result["context_projections"]:
        role, binding = record["role"], record["support_rows_sha256"]
        name = "scan_points" if role == "scan" else "cad_triangles"
        ids = np.arange(len(args[name])) if binding is None else (scan if role == "scan" else cad)[binding]
        points = args[name][ids].reshape(-1, 3)
        values = (points-np.asarray(record["projected_coordinate_origin_mm"]))@record["projection_direction"]+record["projected_constant_mm"]
        np.testing.assert_allclose(record["interval_mm"], [min(values), max(values)], atol=1e-10, rtol=0.)
        assert record["buffer_sha256"] == _sha(args[name]) and record["row_visits"] == len(points)
        for which, extremum in (("minimum_witness", min(values)), ("maximum_witness", max(values))):
            witness = record[which]
            if role == "scan":
                assert witness["raw_vertex_row"] in ids
                point = args[name][witness["raw_vertex_row"]]
            else:
                row = witness["triangle_row"]
                assert row in ids and 0 <= witness["vertex_slot"] < 3
                assert witness["face_id"] == args["cad_face_ids"][row]
                assert witness["source_triangle_id"] == args["cad_source_triangle_ids"][row]
                point = args[name][row, witness["vertex_slot"]]
            value = (point-record["projected_coordinate_origin_mm"])@record["projection_direction"]+record["projected_constant_mm"]
            assert abs(value-extremum) <= 1e-10


@pytest.mark.parametrize("chunk", (3, 6, 65536))
def test_complete_extrema_and_witnesses_do_not_depend_on_chunk_boundaries(kernel, chunk):
    args, _, _ = _case(kernel, 3, scan_interval=(1., 29.), weak_caps=True)
    args["budgets"] = replace(kernel.TwoPlaneContextBudgetsV1(), chunk_rows=chunk)
    result = _invoke(kernel, args); _complete(result)
    _projection_oracles(result, args)
    assert result["statistics"]["scan_row_visits"] == 14+14+4+4
    assert result["statistics"]["cad_vertex_visits"] == 12+2*(12+6+6)


def test_disconnected_repeated_patch_gaps_remain_bound_but_do_not_certify_placement(kernel):
    args, _, _ = _case(kernel)
    # Each source support is two disconnected patches; CAD has a third repeat.
    # Full-face extrema hide gaps, so slots are explicitly NOT finite matches.
    vertices, tris, cad_tris, faces, local, source_planes, target_planes = [], [], [], [], [], [], []
    for i in range(2):
        start_raw, start_tri, start_cad = len(vertices), len(tris), len(cad_tris)
        for interval in ((0., 2.), (10., 12.)):
            base = len(vertices); vertices.extend(_quad(i, interval).tolist())
            tris.extend([[base, base+1, base+2], [base, base+2, base+3]])
        for j, interval in enumerate(((0., 2.), (10., 12.), (20., 22.))):
            quad = _quad(i, interval)
            cad_tris.extend(quad[[[0, 1, 2], [0, 2, 3]]].tolist())
            faces.extend([101+101*i]*2); local.extend([2*j, 2*j+1])
        sr = np.arange(start_raw, len(vertices), dtype=np.int64)
        st = np.arange(start_tri, len(tris), dtype=np.int64)
        ct = np.arange(start_cad, len(cad_tris), dtype=np.int64)
        source_planes.append(replace(args["scan_planes"][i], triangle_rows=st, triangle_rows_sha256=_sha(st),
            raw_vertex_rows=sr, raw_vertex_rows_sha256=_sha(sr)))
        target_planes.append(replace(args["cad_planes"][i], triangle_rows=ct, triangle_rows_sha256=_sha(ct)))
    args.update(scan_points=np.asarray(vertices, dtype=np.float64), scan_triangles=np.asarray(tris, dtype=np.int64),
        cad_triangles=np.asarray(cad_tris, dtype=np.float64), cad_face_ids=np.asarray(faces, dtype=np.int64),
        cad_source_triangle_ids=np.asarray(local, dtype=np.int64), scan_planes=tuple(source_planes), cad_planes=tuple(target_planes))
    result = _invoke(kernel, args); _complete(result); _projection_oracles(result, args)
    family, = [row for row in result["families"] if row["normal_signs"] == [1, 1]]
    assert {row["lambda_mm"] for row in family["slots"]} == {0., 5., 10.}
    assert [p["raw_row_count"] for p in result["plane_bindings"] if p["role"] == "scan"] == [8, 8]
    assert [p["triangle_count"] for p in result["plane_bindings"] if p["role"] == "cad"] == [6, 6]
    assert result["finite_support_match_verified"] is False


def test_distinct_submicron_votes_are_not_tolerance_merged(kernel):
    args, _, _ = _case(kernel, scan_interval=(0., 30.-2**-20))
    result = _invoke(kernel, args); _complete(result)
    family, = [row for row in result["families"] if row["normal_signs"] == [1, 1]]
    assert {row["lambda_mm"] for row in family["slots"]} == {0., 2**-21, 2**-20}
    assert len({row["candidate_index"] for row in family["slots"]}) == 3


@pytest.mark.parametrize("variant", ("stale_raw_digest", "stale_tri_digest", "raw_union_omission", "raw_union_extra",
    "unsorted_raw", "negative_raw", "incomplete_face", "cad_raw_alias", "unqualified", "nonunit", "bool_offset", "cone_over_cap"))
def test_plane_admission_does_not_repair_or_launder_malformed_lineage(kernel, variant):
    args, _, _ = _case(kernel)
    p = args["scan_planes"][0]; c = args["cad_planes"][0]
    if variant == "stale_raw_digest": p = replace(p, raw_vertex_rows_sha256="0"*64)
    elif variant == "stale_tri_digest": p = replace(p, triangle_rows_sha256="0"*64)
    elif variant in ("raw_union_omission", "raw_union_extra", "unsorted_raw", "negative_raw"):
        rows = np.array({"raw_union_omission": [0, 1, 2], "raw_union_extra": [0, 1, 2, 3, 4],
            "unsorted_raw": [1, 0, 2, 3], "negative_raw": [-1, 0, 1, 2]}[variant], dtype=np.int64)
        p = replace(p, raw_vertex_rows=rows, raw_vertex_rows_sha256=_sha(rows))
    elif variant == "incomplete_face":
        rows = np.array([0], dtype=np.int64); c = replace(c, triangle_rows=rows, triangle_rows_sha256=_sha(rows))
    elif variant == "cad_raw_alias": c = replace(c, raw_vertex_rows=p.raw_vertex_rows, raw_vertex_rows_sha256=p.raw_vertex_rows_sha256)
    elif variant == "unqualified": p = replace(p, normal_qualified=False)
    elif variant == "nonunit": p = replace(p, normal=(2., 0., 0.))
    elif variant == "bool_offset": p = replace(p, offset_from_anchor_mm=True)
    else: p = replace(p, normal_angle_bound_radians=.175)
    args["scan_planes"] = (p, args["scan_planes"][1]); args["cad_planes"] = (c, args["cad_planes"][1])
    result = _invoke(kernel, args); _refused(result)
    assert result["statistics"]["signed_families_attempted"] == 0


@pytest.mark.parametrize("variant", ("float32", "shape", "array_subclass", "negative_triangle", "large_triangle",
    "nan_scan", "inf_cad", "range", "negative_face", "negative_source", "duplicate_face_source", "different_lengths",
    "empty_scan", "empty_triangles", "origin_list", "origin_numpy_scalar", "unit_blank", "frame_nonascii"))
def test_input_snapshots_refuse_bad_shapes_values_and_explicit_frame_contracts(kernel, variant):
    args, _, _ = _case(kernel)
    if variant == "float32": args["scan_points"] = args["scan_points"].astype(np.float32)
    elif variant == "shape": args["scan_points"] = args["scan_points"].reshape(4, 6)
    elif variant == "array_subclass":
        class ArraySubclass(np.ndarray): pass
        args["scan_points"] = args["scan_points"].view(ArraySubclass)
    elif variant == "negative_triangle": args["scan_triangles"][0, 0] = -1
    elif variant == "large_triangle": args["scan_triangles"][0, 0] = 8
    elif variant == "nan_scan": args["scan_points"][0, 0] = float("nan")
    elif variant == "inf_cad": args["cad_triangles"][0, 0, 0] = float("inf")
    elif variant == "range": args["scan_points"][0, 0] = 1_000_001.
    elif variant == "negative_face": args["cad_face_ids"][0] = -1
    elif variant == "negative_source": args["cad_source_triangle_ids"][0] = -1
    elif variant == "duplicate_face_source": args["cad_source_triangle_ids"][1] = args["cad_source_triangle_ids"][0]
    elif variant == "different_lengths": args["cad_face_ids"] = args["cad_face_ids"][:-1]
    elif variant == "empty_scan": args["scan_points"] = np.empty((0, 3), dtype=np.float64)
    elif variant == "empty_triangles": args["scan_triangles"] = np.empty((0, 3), dtype=np.int64)
    elif variant == "origin_list": args["scan_origin_mm"] = [0., 0., 0.]
    elif variant == "origin_numpy_scalar": args["scan_origin_mm"] = (np.float64(0.), 0., 0.)
    elif variant == "unit_blank": args["unit_assumption"] = " "
    else: args["target_frame"] = "CAD_é"
    result = _invoke(kernel, args); _refused(result)
    assert result["statistics"].get("signed_families_attempted", 0) == 0


@pytest.mark.parametrize("variant", ("duplicate_reversed", "unknown", "repeated", "bool_id", "mutable_table", "untyped_match"))
def test_scheduled_assignment_validation_retains_no_usable_prefix(kernel, variant):
    args, _, _ = _case(kernel)
    match = kernel.OrderedPlanePairV1
    args["ordered_matches"] = {
        "duplicate_reversed": (match((10, 20), (101, 202)), match((20, 10), (202, 101))),
        "unknown": (match((10, 30), (101, 202)),), "repeated": (match((10, 10), (101, 202)),),
        "bool_id": (match((True, 20), (101, 202)),), "mutable_table": [match((10, 20), (101, 202))],
        "untyped_match": ({"scan_plane_ids": (10, 20), "cad_plane_ids": (101, 202)},)}[variant]
    result = _invoke(kernel, args); _refused(result)
    assert result["statistics"]["signed_families_attempted"] == 0


def test_both_target_orders_remain_independent_four_sign_assignments(kernel):
    args, _, _ = _case(kernel)
    args["ordered_matches"] += (kernel.OrderedPlanePairV1((10, 20), (202, 101)),)
    result = _invoke(kernel, args); _complete(result, 2)
    assert len(result["scheduled_pairs"]) == 2
    for assignment in range(2):
        assert {tuple(f["normal_signs"]) for f in result["families"] if f["ordered_assignment_id"] == assignment} == {(-1, -1), (-1, 1), (1, -1), (1, 1)}


@pytest.mark.parametrize("field,limit,passes", (("max_signed_families", 4, True), ("max_signed_families", 3, False),
    ("max_candidate_slots", 32, True), ("max_candidate_slots", 31, False)))
def test_all_sign_and_slot_budget_boundary_is_preflight_not_partial_success(kernel, field, limit, passes):
    args, _, _ = _case(kernel)
    args["budgets"] = replace(kernel.TwoPlaneContextBudgetsV1(), **{field: limit})
    result = _invoke(kernel, args)
    if passes: _complete(result)
    else:
        _refused(result)
        assert result["reasons"] == ["COMPLETE_SIGN_AND_SLOT_PREFLIGHT_BUDGET"]
        assert result["statistics"]["signed_families_attempted"] == result["statistics"]["projection_cache_misses"] == 0


@pytest.mark.parametrize("extra", (False, True))
def test_maximum_256_assignment_schedule_has_exact_1024_family_8192_slot_preflight(kernel, extra):
    # Private scheduling contract only, not a 256-family geometry execution.
    scans = {i: {} for i in range(128)}; cads = {i: {} for i in range(8)}
    matches = tuple(kernel.OrderedPlanePairV1((2*i, 2*i+1), (2*j, 2*j+1)) for i in range(64) for j in range(4))
    cap = kernel._configuration(kernel.TwoPlaneContextBudgetsV1())
    if extra:
        with pytest.raises(kernel._Unavailable, match="ORDERED_MATCH_TABLE_BUDGET"):
            kernel._schedule(matches+(matches[0],), scans, cads, cap)
    else:
        records = kernel._schedule(matches, scans, cads, cap)
        assert len(records) == 256 and 4*len(records) == 1024 and 32*len(records) == 8192


@pytest.mark.parametrize("field", ("max_scan_rows", "max_scan_triangles", "max_cad_triangles", "max_scan_planes",
    "max_cad_planes", "max_scan_pairs", "max_ordered_cad_matches_per_scan_pair", "max_scan_row_visits",
    "max_cad_vertex_visits", "max_index_visits", "max_index_sort_work", "max_hash_bytes",
    "max_projection_cache_entries", "max_working_bytes", "max_report_bytes"))
def test_zero_input_or_work_cap_is_a_whole_refusal_not_a_positive_prefix(kernel, field):
    args, _, _ = _case(kernel)
    args["budgets"] = replace(kernel.TwoPlaneContextBudgetsV1(), **{field: 0})
    result = _invoke(kernel, args); _refused(result)
    assert result["statistics"].get("signed_families_completed", 0) == 0


@pytest.mark.parametrize("field,value", (("chunk_rows", 2), ("chunk_rows", 3.), ("max_scan_rows", True),
    ("max_candidate_slots", 8193), ("max_scan_pairs", -1)))
def test_mutated_frozen_policy_fields_are_revalidated_before_input_ownership(kernel, field, value):
    args, _, _ = _case(kernel)
    budgets = kernel.TwoPlaneContextBudgetsV1(); object.__setattr__(budgets, field, value)
    args["budgets"] = budgets
    result = _invoke(kernel, args); _refused(result)
    assert result["reasons"] == ["INVALID_BUDGET_VALUES"] and result["statistics"] == {}


def test_empty_caller_schedule_is_complete_but_neither_absence_nor_registration(kernel):
    args, _, _ = _case(kernel); args["ordered_matches"] = ()
    result = _invoke(kernel, args); _complete(result, 0)
    assert result["usable_proposals"] is False and result["candidates"] == []


def test_snapshot_shape_is_captured_before_byte_copy_and_never_reinterpreted(kernel, monkeypatch):
    # Explicit snapshot-boundary mock, not a concurrency or native-write claim.
    source = np.arange(18., dtype=np.float64).reshape(6, 3); original = source.copy()
    real = kernel.np.frombuffer
    def reshape_caller_after_capture(blob, *args, **kwargs):
        source.shape = (3, 6)
        return real(blob, *args, **kwargs)
    monkeypatch.setattr(kernel.np, "frombuffer", reshape_caller_after_capture)
    work = kernel._Work(kernel._configuration(kernel.TwoPlaneContextBudgetsV1()))
    owned = work.own(source, (3,), "float64", 6, "SCAN")
    assert source.shape == (3, 6) and owned.shape == (6, 3)
    np.testing.assert_array_equal(owned, original)
    assert not owned.flags.writeable and not np.shares_memory(source, owned)
    with pytest.raises(ValueError): owned.flags.writeable = True


def test_owned_coordinate_and_support_snapshots_are_detached_from_callers(kernel):
    args, _, _ = _case(kernel)
    work = kernel._Work(kernel._configuration(kernel.TwoPlaneContextBudgetsV1())); result = kernel._base()
    names = ("scan_points", "scan_triangles", "cad_triangles", "cad_face_ids", "cad_source_triangle_ids")
    expected = {name: args[name].copy() for name in names}
    context = kernel._inputs(*(args[name] for name in names), work, result)
    planes = kernel._planes(args["scan_planes"], "scan", context, work, result)
    for name in names:
        args[name][...] = -1
        np.testing.assert_array_equal(context[name], expected[name])
        assert not context[name].flags.writeable and result["input_digests"][name+"_sha256"] == _sha(expected[name])
    args["scan_planes"][0].raw_vertex_rows[:] = 7
    np.testing.assert_array_equal(planes[10]["raw_vertex_rows"], [0, 1, 2, 3])
    assert not planes[10]["raw_vertex_rows"].flags.writeable


def test_late_projection_cap_clears_geometry_and_retains_attempted_visit_ledger(kernel):
    args, _, _ = _case(kernel)
    args["budgets"] = replace(kernel.TwoPlaneContextBudgetsV1(), max_projection_cache_entries=1)
    result = _invoke(kernel, args); _refused(result)
    stats = result["statistics"]
    assert result["reasons"] == ["PROJECTION_CACHE_BUDGET"]
    assert result["discarded_signed_families"] == 1 and result["discarded_candidates"] == 0
    assert stats["projection_cache_entries"] == 1 and stats["projection_cache_misses"] == 2
    assert stats["scan_row_visits"] == 16 and stats["cad_vertex_visits"] == 12
    assert stats["signed_families_attempted"] == 1 and stats["signed_families_completed"] == 0


@pytest.mark.parametrize("exception", (RuntimeError, IndexError, FloatingPointError))
def test_late_family_numeric_or_unexpected_failure_invalidates_all_prior_poses(kernel, monkeypatch, exception):
    args, _, _ = _case(kernel); real = kernel._family; calls = []
    def fail_second(*a, **kw):
        calls.append(1)
        if len(calls) == 2: raise exception("private diagnostic must not appear")
        return real(*a, **kw)
    monkeypatch.setattr(kernel, "_family", fail_second)
    result = _invoke(kernel, args); _refused(result)
    assert result["reasons"] == ["CONTEXT_HELPER_EXCEPTION"]
    assert result["discarded_signed_families"] == 2 and result["discarded_candidates"] == 1
    assert result["statistics"]["signed_families_completed"] == 1
    assert result["statistics"]["signed_families_attempted"] == 2
    assert result["statistics"]["slot_records_completed"] == 8
    assert "private diagnostic" not in json.dumps(result)


def test_final_report_failure_clears_complete_internal_success_without_losing_counters(kernel, monkeypatch):
    args, _, _ = _case(kernel); real = kernel._encoded_size
    def refuse_final(value, maximum):
        if type(value) is dict and value.get("schema_version") == "two_plane_context_v1":
            raise kernel._Unavailable("ENCODED_PAYLOAD_BYTE_BUDGET")
        return real(value, maximum)
    monkeypatch.setattr(kernel, "_encoded_size", refuse_final)
    result = _invoke(kernel, args); _refused(result)
    assert result["reasons"] == ["ENCODED_PAYLOAD_BYTE_BUDGET"]
    assert result["discarded_candidates"] == 4 and result["discarded_signed_families"] == 4
    assert result["statistics"]["signed_families_completed"] == 4
    assert result["statistics"]["slot_records_completed"] == 32
