"""Independent finite-plane initialization tests; invented arrays only.

Root-authored while the implementation is independently owned. No OCP, physical
PLY/STEP, Gold, source-run artifact or oracle transform enters the public API.
These fixtures are prerequisite geometry controls, not the connected bend gate.
"""

from copy import deepcopy
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from backend import cad_guided_plane_initialization_v4 as v4
from backend.cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1


RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
                       process_tree_rss_bytes=1024**3)


def _rectangle(origin, first, second, n=5):
    vertices = np.asarray([np.asarray(origin) + u*np.asarray(first) + v*np.asarray(second)
                           for v in np.linspace(0., 1., n)
                           for u in np.linspace(0., 1., n)])
    triangles = []
    for y in range(n-1):
        for x in range(n-1):
            a = y*n+x
            triangles.extend(((a, a+1, a+n+1), (a, a+n+1, a+n)))
    return vertices, np.asarray(triangles, dtype=np.int64)


def _three_patches(n=5):
    # Three nonorthogonal independent normals; finite pieces are intentionally
    # remote and unequal. CAD is a triangulated construction, not a native solid.
    specifications = (
        ((-10., -4., 0.), (6., 0., 0.), (0., 3., 0.)),
        ((20., 0., 12.), (0., 7., 0.), (0., 0., 5.)),
        ((30., 20., 10.), (4., -4., 0.), (0., 3., -3.)),
    )
    patches = [_rectangle(*specification, n=n) for specification in specifications]
    points = np.vstack([patch[0] for patch in patches])
    rows = np.vstack([patch[1]+n*n*index for index, patch in enumerate(patches)])
    triangle_count = 2*(n-1)**2
    faces = np.repeat(np.asarray([11, 22, 33], dtype=np.int64), triangle_count)
    local = np.tile(np.arange(1, triangle_count+1, dtype=np.int64), 3)
    return points, rows, points[rows].copy(), faces, local


def _policy():
    return WholePartRegistrationPolicyV1(coarse_points=75, refine_points=75,
        coarse_iterations=2, refine_iterations=2, huber_delta_mm=.01,
        max_inlier_plane_p95_mm=.05, max_correspondence_distance_mm=.5,
        min_scan_coverage=.95, min_cad_coverage=.30)


POSES = (
    (np.eye(3), np.zeros(3)),
    (np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), np.asarray([30., -20., 10.])),
    (np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), np.asarray([-100., 70., 25.])),
    (np.asarray([[.36, -.48, .8], [.8, .6, 0.], [-.48, .64, .6]]), np.asarray([1000., -2000., 3000.])),
)


def _public(points, rows, triangles, faces, local, **kwargs):
    return v4.prefilter_plane_initializations_v4(points, rows, triangles, faces, local,
        registration_policy=kwargs.pop("registration_policy", _policy()), **kwargs)


def _complete(result):
    assert result["status"] == "COMPLETE", result
    assert result["search_complete"] is True and result["reasons"] == []
    assert result["schema_version"] == "plane_initialization_prefilter_v4"
    assert result["unit_status"] == "PROVISIONAL_ARRAY_MM_UNVERIFIED"
    # Entire external evidence must serialize strictly without private NumPy
    # arrays or nonfinite sentinels; supports stay in internal immutable arrays.
    json.dumps(result, allow_nan=False)
    return result["start_groups"]


def _truth_supported(result, observed, nominal):
    groups = _complete(result)
    assert groups
    correct = [group for group in groups if np.max(np.linalg.norm(
        observed @ np.asarray(group["initialized_rotation"]).T
        + np.asarray(group["initialized_translation_mm"]) - nominal, axis=1)) < 1e-8]
    assert correct, "No admitted initialization has the independently constructed rigid action"
    for group in correct:
        assert group["hypothesis_indices"]
        for index in group["hypothesis_indices"]:
            hypothesis = result["hypotheses"][index]
            assert hypothesis["prefilter_status"] == "PASSED"
            assert hypothesis["start_group_index"] == group["index"]
            assert len(hypothesis["assigned_support"]) == 3
            for support in hypothesis["assigned_support"]:
                assert support["support_count"] == support["query_rows"] == 25
                assert support["finite_unambiguous_inlier_count"] == 25
                assert support["coverage"] == 1.
    return correct


@pytest.mark.parametrize("pose", range(4))
def test_public_raw_mesh_to_finite_assignments_retains_independently_known_pose(pose):
    points, rows, triangles, faces, local = _three_patches()
    rotation, shift = POSES[pose]
    observed = points @ rotation.T + shift
    result = _public(observed, rows, triangles, faces, local)
    _truth_supported(result, observed, points)
    rejected = [hypothesis for hypothesis in result["hypotheses"]
                if hypothesis["prefilter_status"] == "REJECTED"]
    assert rejected, "Finite checks must reject at least one wrong infinite-plane assignment"
    assert all(len(hypothesis["assigned_support"]) == 3 for hypothesis in rejected)
    assert all(support["query_rows"] == support["support_count"] == 25
               for hypothesis in rejected for support in hypothesis["assigned_support"])


def test_public_raw_row_permutation_and_cad_identity_relabel_do_not_supply_pose():
    points, rows, triangles, faces, local = _three_patches()
    rotation, shift = POSES[3]
    observed = points @ rotation.T + shift
    order = np.arange(75)[::-1]
    inverse = np.empty(75, dtype=np.int64)
    inverse[order] = np.arange(75)
    changed_faces = np.select((faces == 11, faces == 22), (701, 3), default=98).astype(np.int64)
    result = _public(observed[order], inverse[rows], triangles[::-1].copy(),
                     changed_faces[::-1].copy(), local[::-1].copy())
    _truth_supported(result, observed[order], points[order])


def test_public_evidence_is_detached_from_mutable_caller_arrays():
    points, rows, triangles, faces, local = _three_patches()
    result = _public(points, rows, triangles, faces, local)
    _truth_supported(result, points, points)
    saved = deepcopy(result)
    points[:] = 100.
    rows[:] = 0
    triangles[:] = -20.
    faces[:] = 0
    local[:] = 0
    assert result == saved


@pytest.mark.parametrize("field,value", (("min_scan_coverage", False),
    ("max_correspondence_distance_mm", float("nan")), ("coarse_points", 0)))
def test_bypass_mutated_registration_policy_cannot_admit_initialization(field, value):
    policy = _policy()
    object.__setattr__(policy, field, value)
    result = _public(*_three_patches(), registration_policy=policy)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["search_complete"] is False and result["reasons"]


@pytest.mark.parametrize("fault", ("bad_raw_index", "late_nan", "duplicate_cad_source_pair"))
def test_public_invalid_input_has_no_usable_earlier_pose_prefix(fault):
    points, rows, triangles, faces, local = _three_patches()
    if fault == "bad_raw_index":
        rows[-1, -1] = len(points)
    elif fault == "late_nan":
        triangles[-1, -1, -1] = np.nan
    else:
        local[-1] = local[-2]
    result = _public(points, rows, triangles, faces, local)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["search_complete"] is False and result["reasons"]


def _private_case(inputs=None, *, policy=None, budgets=None):
    """Actual extracted supports plus deliberately stipulated pose hypotheses.

    This tests only the private finite-screening seam, NOT discovery/pose truth.
    The public controls above separately exercise genuine hypothesis generation.
    """
    policy = _policy() if policy is None else policy
    budgets = v4.PlaneInitializationBudgetsV4() if budgets is None else budgets
    configuration = v4._configuration(policy, v4._patch.CoplanarPatchPolicyV4(),
        v4._patch.CoplanarPatchBudgetsV4(), v4._planes.PlaneHypothesisPolicyV4(),
        v4._planes.PlaneHypothesisBudgetsV4(), budgets, None)
    result = v4._base()
    context = v4._prepare(*(_three_patches() if inputs is None else inputs), configuration, result)
    context["hypotheses"] = [_stipulated(0)]
    return context, result, configuration


def _stipulated(index, *, translation=(0., 0., 0.), first_face=11):
    return dict(hypothesis_index=index, rotation=np.eye(3).tolist(),
        translation_mm=list(translation),
        correspondences=[dict(scan_plane_id=scan, cad_plane_id=cad, normal_sign=1)
                         for scan, cad in enumerate((first_face, 22, 33))],
        test_scope="STIPULATED_PRIVATE_SEAM_NOT_PUBLIC_POSE_GENERATION")


def _private_screen(context, result, configuration):
    try:
        v4._prefilter_prepared(context, result, configuration["policy"],
            configuration["budgets"], configuration["surface_budgets"])
    except v4._ERRORS as error:
        v4._fail_prefilter(result, error)
    return result


@pytest.mark.parametrize("minimum,expected", ((.95, "PASSED"), (1., "REJECTED")))
def test_private_finite_hole_counts_missing_support_instead_of_using_plane_or_bbox(minimum, expected):
    points, rows, triangles, faces, local = _three_patches()
    # Remove the central four cells of the first rectangle from CAD only.
    # The one central scan vertex is .75 mm from the hole's finite boundary;
    # every other first-patch vertex lies on retained triangles/boundaries.
    omit = {2*(4*y+x)+corner for y in (1, 2) for x in (1, 2) for corner in (0, 1)}
    keep = np.asarray([i not in omit for i in range(len(triangles))])
    context, result, configuration = _private_case(
        (points, rows, triangles[keep], faces[keep], local[keep]),
        policy=replace(_policy(), min_scan_coverage=minimum))
    result = _private_screen(context, result, configuration)
    row, = result["hypotheses"]
    assert result["status"] == "COMPLETE" and result["search_complete"] is True
    assert row["prefilter_status"] == expected
    support = row["assigned_support"][0]
    assert support["support_count"] == support["query_rows"] == 25
    assert support["finite_unambiguous_inlier_count"] == 24
    assert support["coverage"] == 24/25
    assert all(s["coverage"] == 1. for s in row["assigned_support"][1:])


def test_private_full_scan_patch_on_small_cad_cap_fails_without_short_circuiting_other_faces():
    points, rows, triangles, faces, local = _three_patches()
    triangles[:32, :, 0] = -7. + .1*(triangles[:32, :, 0]+7.)
    context, result, configuration = _private_case((points, rows, triangles, faces, local))
    result = _private_screen(context, result, configuration)
    row, = result["hypotheses"]
    assert row["prefilter_status"] == "REJECTED" and result["start_groups"] == []
    assert [s["query_rows"] for s in row["assigned_support"]] == [25, 25, 25]
    assert [s["coverage"] for s in row["assigned_support"]] == [.2, 1., 1.]


def test_private_partial_observation_of_larger_face_has_no_reverse_coverage_requirement():
    points, rows, triangles, faces, local = _three_patches()
    triangles[:32, :, 0] = -7. + 10.*(triangles[:32, :, 0]+7.)
    triangles[:32, :, 1] = -2.5 + 10.*(triangles[:32, :, 1]+2.5)
    context, result, configuration = _private_case((points, rows, triangles, faces, local))
    result = _private_screen(context, result, configuration)
    assert result["status"] == "COMPLETE" and len(result["start_groups"]) == 1
    assert result["hypotheses"][0]["prefilter_status"] == "PASSED"
    assert all(s["coverage"] == 1. for s in result["hypotheses"][0]["assigned_support"])


def test_private_same_pose_different_face_assignment_is_screened_before_exact_deduplication():
    points, rows, triangles, faces, local = _three_patches()
    cap = triangles[:32].copy()
    cap[:, :, 0] = -7. + .1*(cap[:, :, 0]+7.)
    inputs = (points, rows, np.vstack((triangles, cap)),
              np.r_[faces, np.full(32, 44, dtype=np.int64)], np.r_[local, np.arange(1, 33)])
    context, result, configuration = _private_case(inputs)
    context["hypotheses"] = [_stipulated(0), _stipulated(1, first_face=44)]
    result = _private_screen(context, result, configuration)
    assert [row["prefilter_status"] for row in result["hypotheses"]] == ["PASSED", "REJECTED"]
    assert len(result["start_groups"]) == 1
    assert result["start_groups"][0]["hypothesis_indices"] == [0]
    assert "start_group_index" not in result["hypotheses"][1]
    assert result["statistics"]["query_rows"] == 150


@pytest.mark.parametrize("cap", (1, 2))
def test_private_exact_duplicates_share_work_but_nearby_starts_never_merge(cap):
    context, result, configuration = _private_case(
        budgets=replace(v4.PlaneInitializationBudgetsV4(), max_surviving_start_groups=cap))
    context["hypotheses"] = [_stipulated(0), _stipulated(1),
                             _stipulated(2, translation=(1e-10, 0., 0.))]
    result = _private_screen(context, result, configuration)
    assert len(result["hypotheses"]) == 3
    assert all(row["prefilter_status"] == "PASSED" for row in result["hypotheses"])
    assert result["statistics"]["query_rows"] == 225
    if cap == 1:
        assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
        assert result["reasons"] == ["SURVIVING_START_GROUP_BUDGET_EXCEEDED"]
        assert not result["search_complete"]
        assert all(row["usable_initialization"] is False and "start_group_index" not in row
                   for row in result["hypotheses"])
    else:
        assert result["status"] == "COMPLETE" and len(result["start_groups"]) == 2
        assert [row["hypothesis_indices"] for row in result["start_groups"]] == [[0, 1], [2]]


def test_private_query_obligation_cap_precedes_every_bvh_build(monkeypatch):
    context, result, configuration = _private_case(
        budgets=replace(v4.PlaneInitializationBudgetsV4(), max_prefilter_query_rows=74))
    monkeypatch.setattr(v4._surface, "build_triangle_surface",
                        lambda *a, **k: pytest.fail("No BVH before whole-query preflight admission"))
    result = _private_screen(context, result, configuration)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["reasons"] == ["PREFILTER_PREFLIGHT_BUDGET_EXCEEDED"]
    assert len(result["hypotheses"]) == 1
    assert result["hypotheses"][0]["prefilter_status"] == "NOT_ATTEMPTED"
    assert len(result["hypotheses"][0]["assigned_support"]) == 3


def test_private_late_query_failure_invalidates_previously_passing_start(monkeypatch):
    context, result, configuration = _private_case()
    context["hypotheses"] = [_stipulated(0), _stipulated(1)]
    original = v4._surface.query_closest_surface
    calls = []
    def query(*args, **kwargs):
        calls.append(None)
        if len(calls) == 4:
            return dict(status="UNAVAILABLE", reason="INVENTED_LATE_FAILURE", diagnostics={})
        return original(*args, **kwargs)
    monkeypatch.setattr(v4._surface, "query_closest_surface", query)
    result = _private_screen(context, result, configuration)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["reasons"] == ["ASSIGNED_FACE_QUERY_UNAVAILABLE:INVENTED_LATE_FAILURE"]
    assert [row["prefilter_status"] for row in result["hypotheses"]] == ["PASSED", "INCOMPLETE"]
    assert all(row["usable_initialization"] is False for row in result["hypotheses"])
    assert result["hypotheses"][1]["assigned_support"][0]["query_rows"] == 0


@pytest.mark.parametrize("kind", ("scan", "cad"))
def test_private_updated_hash_cannot_hide_a_shortened_support_union(kind):
    context, result, configuration = _private_case()
    if kind == "scan":
        patch = context["scan_patches"]["patches"][0]
        patch["raw_vertex_rows"] = patch["raw_vertex_rows"][1:]
        patch["support_count"] -= 1
        patch["raw_vertex_rows_sha256"] = v4._digest(patch["raw_vertex_rows"])
    else:
        patch = context["cad_patches"]["patches"][0]
        patch["triangle_rows"] = patch["triangle_rows"][1:]
        patch["source_triangle_ids"] = patch["source_triangle_ids"][1:]
        patch["triangle_rows_sha256"] = v4._digest(patch["triangle_rows"])
        patch["source_triangle_ids_sha256"] = v4._digest(patch["source_triangle_ids"])
    result = _private_screen(context, result, configuration)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert any("INCOMPLETE" in reason for reason in result["reasons"])


def test_wrapper_global_bvh_refusal_retains_every_known_unattempted_start():
    inputs = _three_patches()
    # Each 32-triangle assigned face needs three BVH nodes and passes. The
    # combined 96-triangle surface needs more nodes and must fail before ICP.
    surface_budget = replace(v4._surface.TriangleSurfaceBudgets(), max_build_nodes=3)
    result = v4.register_whole_part_planes_v4(*inputs, cad_initialization_points=inputs[0],
        registration_policy=_policy(), unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        surface_budgets=surface_budget)
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] == ["FINITE_TRIANGLE_SURFACE_UNAVAILABLE"]
    evidence = result["evidence"]
    groups = evidence["initialization"]["start_groups"]
    assert groups and evidence["initialization"]["status"] == "COMPLETE"
    assert evidence["candidate_count"] == len(groups)
    assert evidence["attempted_candidate_count"] == 0
    assert not evidence["candidate_evaluation_complete"]
    assert all(row["execution_status"] == "NOT_ATTEMPTED"
               and row["reasons"] == ["NOT_ATTEMPTED_AFTER_FATAL_FAILURE"]
               for row in evidence["candidates"])


@pytest.mark.parametrize("fault", ("unit", "duplicate_rows", "foreign_row", "short_rows", "bad_samples"))
def test_wrapper_frame_row_and_sample_admission_precedes_all_extraction(monkeypatch, fault):
    inputs = _three_patches()
    kwargs = dict(cad_initialization_points=inputs[0], registration_policy=_policy(),
                  unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")
    if fault == "unit":
        kwargs["unit_assumption"] = "VERIFIED_BY_CAD"
    elif fault == "duplicate_rows":
        kwargs["registration_raw_rows"] = np.zeros(36, dtype=np.int64)
    elif fault == "foreign_row":
        kwargs["registration_raw_rows"] = np.arange(36, dtype=np.int64)+50
    elif fault == "short_rows":
        kwargs["registration_raw_rows"] = np.arange(35, dtype=np.int64)
    else:
        kwargs["cad_initialization_points"] = np.full((75, 3), np.nan)
    monkeypatch.setattr(v4, "_initialize", lambda *a, **k: pytest.fail("Invalid input reached extraction"))
    result = v4.register_whole_part_planes_v4(*inputs, **kwargs)
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] and result["evidence"]["candidates"] == []


@pytest.mark.parametrize("name", tuple(field.name for field in fields(v4.PlaneInitializationBudgetsV4)))
def test_bypass_raised_initialization_hard_cap_is_rejected_before_extraction(monkeypatch, name):
    budget = v4.PlaneInitializationBudgetsV4()
    object.__setattr__(budget, name, getattr(budget, name)+1)
    monkeypatch.setattr(v4, "_prepare", lambda *a, **k: pytest.fail("Raised cap reached input preparation"))
    result = _public(*_three_patches(), budgets=budget)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["reasons"] == ["INVALID_INITIALIZATION_BUDGET"]


def test_detailed_report_overflow_returns_only_whole_refusal_never_a_pose_prefix():
    result = _public(*_three_patches(), budgets=replace(v4.PlaneInitializationBudgetsV4(), max_report_bytes=100))
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["transform_scan_to_cad"] is None and result["hypotheses"] == []
    assert result["detailed_report_discarded_whole"] is True
    assert result["reasons"] == ["INITIALIZATION_REPORT_BUDGET_EXCEEDED"]
    assert result["discarded_hypothesis_records"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("pose", range(4))
def test_public_wrapper_all_scan_rows_on_cad_vertices_refuses_face_only_observability(pose):
    points, rows, triangles, faces, local = _three_patches(n=7)
    rotation, shift = POSES[pose]
    observed = points @ rotation.T + shift
    result = v4.register_whole_part_planes_v4(observed, rows, triangles, faces, local,
        cad_initialization_points=points, registration_policy=replace(_policy(), coarse_points=147, refine_points=147),
        unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] == ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
    evidence = result["evidence"]
    assert evidence["candidate_count"] > 0
    assert evidence["attempted_candidate_count"] == evidence["candidate_count"]
    assert evidence["candidate_numerical_failure_count"] == 0
    assert evidence["candidate_evaluation_complete"] is True
    assert all("LOCAL_POSE_UNOBSERVABLE" in row["reasons"]
               and row["informative_scan_count"] == 0 for row in evidence["candidates"])
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    assert evidence["initial_assignments_are_final_face_matches"] is False


def test_valid_explicit_rows_keep_full_prefilter_support_even_when_vertex_rank_refuses():
    points, rows, triangles, faces, local = _three_patches(n=7)
    selected = np.concatenate([np.arange(42, dtype=np.int64)+49*index for index in range(3)])
    result = v4.register_whole_part_planes_v4(points, rows, triangles, faces, local,
        cad_initialization_points=points, registration_raw_rows=selected,
        registration_policy=replace(_policy(), coarse_points=126, refine_points=126),
        unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] == ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
    evidence = result["evidence"]
    assert evidence["prefilter_full_scan_raw_rows"] == 147
    assert evidence["registration_input_rows"] == evidence["scan_unique_rows"] == 126
    assert evidence["registration_raw_rows"]["count"] == 126
    assert evidence["registration_raw_rows"]["sha256"] == v4._digest(selected.astype("<i8"))
    assert evidence["coverage_basis"].startswith("EXPLICIT_SELECTED_")
    for hypothesis in evidence["initialization"]["hypotheses"]:
        assert all(row["support_count"] == row["query_rows"] == 49 for row in hypothesis["assigned_support"])


@pytest.mark.parametrize("scenario", ("same_pose", "competing_pose", "unobservable"))
def test_v2_selection_seam_preserves_pose_classes_and_required_refusal(scenario):
    # These are deliberately mocked final diagnostics, not an accepted public
    # registration or a geometric observability proof.
    result = v4._registration_result("PROVISIONAL_COMMON_MM_NOT_CALIBRATED")
    candidates = result["evidence"]["candidates"]
    for index in range(2):
        candidates.append(dict(index=index, scan_coverage=1., cad_coverage=1.,
            inlier_surface_rms_mm=.01, rotation=np.eye(3).tolist(),
            translation=[float(index) if scenario == "competing_pose" else 0., 0., 0.],
            reasons=["LOCAL_POSE_UNOBSERVABLE"] if scenario == "unobservable" else []))
    v4._v2_select(result, _three_patches()[0], _policy())
    if scenario == "same_pose":
        assert result["status"] == "CONDITIONAL_BEST_FIT"
        assert result["evidence"]["distinct_eligible_pose_class_count"] == 1
    elif scenario == "competing_pose":
        assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
        assert result["reasons"] == ["AMBIGUOUS_GENERATED_WHOLE_PART_POSES"]
    else:
        assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
        assert result["reasons"] == ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]


def _independently_tessellated_surface_case():
    dense, scan_rows, _, _, _ = _three_patches(n=7)
    nominal = dense.copy()
    for index in range(3):
        block = slice(49*index, 49*(index+1))
        center = dense[49*index+24]
        nominal[block] = center + .8*(dense[block]-center)
    _, _, cad_triangles, faces, local = _three_patches(n=2)
    cad_samples = _three_patches(n=9)[0]
    # CAD is the original complete rectangular support with two triangles.
    # Scan tessellation is independently inset. No scan row is a CAD vertex;
    # seven rows/face are on the diagonal, the other42 are strictly FACE-interior.
    return nominal, scan_rows, cad_triangles, faces, local, cad_samples


@pytest.mark.parametrize("pose", range(4))
def test_real_wrapper_independent_interior_scan_recovers_pose_through_v2_gates(pose):
    nominal, rows, triangles, faces, local, samples = _independently_tessellated_surface_case()
    rotation, shift = POSES[pose]
    observed = nominal @ rotation.T + shift
    result = v4.register_whole_part_planes_v4(observed, rows, triangles, faces, local,
        cad_initialization_points=samples,
        registration_policy=replace(_policy(), coarse_points=147, refine_points=147),
        unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")
    assert result["status"] == "CONDITIONAL_BEST_FIT", result["reasons"]
    matrix = np.asarray(result["transform_scan_to_cad"])
    np.testing.assert_allclose(observed @ matrix[:3, :3].T + matrix[:3, 3], nominal, rtol=0., atol=1e-8)
    evidence = result["evidence"]
    assert evidence["candidate_count"] > 0
    assert evidence["attempted_candidate_count"] == evidence["candidate_count"]
    assert evidence["candidate_numerical_failure_count"] == 0
    assert evidence["candidate_evaluation_complete"] is True
    assert evidence["comparable_distinct_pose_count"] == 0
    best = evidence["candidates"][evidence["selected_candidate_index"]]
    assert best["informative_scan_count"] == 126
    assert best["informative_fit_subset_count"] >= 36
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False


def test_interior_scan_selected_rows_preserve_full_prefilter_and_correct_whole_raw_action():
    nominal, rows, triangles, faces, local, samples = _independently_tessellated_surface_case()
    rotation, shift = POSES[3]
    observed = nominal @ rotation.T + shift
    selected = np.concatenate([np.arange(42, dtype=np.int64)+49*index for index in range(3)])
    result = v4.register_whole_part_planes_v4(observed, rows, triangles, faces, local,
        cad_initialization_points=samples, registration_raw_rows=selected,
        registration_policy=replace(_policy(), coarse_points=126, refine_points=126),
        unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")
    assert result["status"] == "CONDITIONAL_BEST_FIT", result["reasons"]
    matrix = np.asarray(result["transform_scan_to_cad"])
    np.testing.assert_allclose(observed @ matrix[:3, :3].T + matrix[:3, 3], nominal, rtol=0., atol=1e-8)
    evidence = result["evidence"]
    assert evidence["candidate_evaluation_complete"] is True
    assert evidence["candidate_numerical_failure_count"] == 0
    assert evidence["attempted_candidate_count"] == evidence["candidate_count"] > 0
    assert evidence["prefilter_full_scan_raw_rows"] == 147 and evidence["registration_input_rows"] == 126
    assert evidence["registration_raw_rows"]["sha256"] == v4._digest(selected.astype("<i8"))
    assert evidence["coverage_basis"].startswith("EXPLICIT_SELECTED_")
    for hypothesis in evidence["initialization"]["hypotheses"]:
        assert all(s["support_count"] == s["query_rows"] == 49 for s in hypothesis["assigned_support"])
