"""V8 all-generated-screen successor retaining all43 frozen V7 array controls; no native/physical IO.

Frozen positive cases: two independent proper frames; three unequal nonorthogonal
rectangles with independently inset 7x7 scan meshes, complete two-triangle CAD
faces and unrelated 9x9 CAD samples. Scan normal noise is 0.001*outer(e,e),
e=(1,-1,0,0,0,-1,1): its mean and both first in-plane moments vanish. Thus the
nominal TLS planes are unchanged in exact arithmetic, while the mesh is noisy.
The stipulated row error is 0.005 mm, not scanner calibration. Required finite
start action error on clean rows <=1e-7 mm; registered clean action <=0.01 mm;
noisy action <=0.011 mm. These are prospective engineering criteria, not confidence
bounds. The last ceiling is the triangle inequality (0.01+0.001). No failure may
turn either positive into an optional abstention. Admission policy remains the
prior strict small matrix: distance .5, finite P95 .05, coverage .95/.30; workload
147 points and two coarse/two fine iterations. Defaults are separately checked.

All injected hypotheses/queries/diagnostics below are explicitly private contract
controls, never a public pose-recovery claim. The actual positives have no mocks.
First run: root-owned serial supervisor, 60s wall /45s worker CPU /1GiB group RSS
/1MiB stdout. This file does not run that supervisor or enlarge those budgets.
"""

import ast
from copy import deepcopy
from dataclasses import asdict, fields, replace
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from backend import cad_guided_plane_initialization_v8 as v5


RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
                       process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
EPSILON_MM = .005
NOISE_MM = .001
START_ERROR_MM = 1e-7
POSE_ERROR_MM = .01
NOISY_ERROR_MM = .011
SPECS = (
    ((-10., -4., 0.), (6., 0., 0.), (0., 3., 0.)),
    ((20., 0., 12.), (0., 7., 0.), (0., 0., 5.)),
    ((30., 20., 10.), (4., -4., 0.), (0., 3., -3.)),
)
POSES = (
    (np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), np.array([30., -20., 10.])),
    (np.array([[1., 8., 4.], [8., 1., -4.], [-4., 4., -7.]])/9., np.array([37., -21., 9.])),
)


def _rectangle(specification, n, *, inset=False):
    origin, first, second = map(np.asarray, specification)
    coordinates = np.linspace(0., 1., n)
    points = np.array([origin + (.13+.8*u if inset else u)*first
                      + (.11+.8*w if inset else w)*second
                      for w in coordinates for u in coordinates])
    rows = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            rows.extend(((a, a+1, a+n+1), (a, a+n+1, a+n)))
    return points, np.asarray(rows, dtype=np.int64)


def _mesh(n=5, *, inset=False, cad_n=None):
    patches = [_rectangle(spec, n, inset=inset) for spec in SPECS]
    points = np.vstack([patch[0] for patch in patches])
    raw = np.vstack([patch[1]+index*n*n for index, patch in enumerate(patches)])
    cad_n = n if cad_n is None else cad_n
    cad_patches = [_rectangle(spec, cad_n) for spec in SPECS]
    cad = np.vstack([p[r] for p, r in cad_patches])
    count = 2*(cad_n-1)**2
    faces = np.repeat(np.array([11, 22, 33], dtype=np.int64), count)
    local = np.tile(np.arange(count, dtype=np.int64), 3)
    return points, raw, cad, faces, local


def _policy(n=75):
    return v5._v1.WholePartRegistrationPolicyV1(coarse_points=n, refine_points=n,
        coarse_iterations=2, refine_iterations=2, huber_delta_mm=.01,
        max_inlier_plane_p95_mm=.05, max_correspondence_distance_mm=.5,
        min_scan_coverage=.95, min_cad_coverage=.30)


def _prefilter(inputs, **kwargs):
    return v5.prefilter_plane_initializations_v8(*inputs,
        position_error_bounds_mm=kwargs.pop("position_error_bounds_mm", EPSILON_MM),
        registration_policy=kwargs.pop("registration_policy", _policy()),
        unit_assumption=kwargs.pop("unit_assumption", UNIT), **kwargs)


def _register(inputs, samples=None, **kwargs):
    return v5.register_whole_part_planes_v8(*inputs,
        position_error_bounds_mm=kwargs.pop("position_error_bounds_mm", EPSILON_MM),
        cad_initialization_points=inputs[0] if samples is None else samples,
        registration_policy=kwargs.pop("registration_policy", _policy(len(inputs[0]))),
        unit_assumption=kwargs.pop("unit_assumption", UNIT), **kwargs)


def _assert_scope(result):
    assert result["observed_bend_count"] is None
    assert result["production_use_allowed"] is False
    assert result["conformance"] == "INDETERMINATE"
    json.dumps(result, allow_nan=False)


def _assert_complete_prefilter(result):
    assert result["status"] == "COMPLETE", result["reasons"]
    assert result["search_complete"] and result["selected_policy_evaluation_complete"]
    assert result["search_exhaustive"] is False
    assert result["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert result["accepted_registration"] is False
    assert result["statistics"]["hypotheses_evaluated"] == result["statistics"]["scheduled_hypothesis_count"]
    assert result["prefilter_schedule"] == "ALL_GENERATED_PROPOSED_SIGNS_HARD_UNSIGNED_BOUND_NO_RANK_OMISSION"
    eligible = sorted({row["unsigned_assignment_id"] for row in result["hypotheses"] if row["status"] == "PROPOSED"})
    assert result["selected_unsigned_assignment_ids"] == eligible
    stats = result["statistics"]
    assert stats["eligible_unsigned_assignment_count"] == stats["scheduled_unsigned_assignment_count"] == len(eligible)
    assert stats["omitted_unsigned_assignment_count"] == stats["omitted_proposed_hypothesis_count"] == 0
    assert stats["proposed_hypothesis_count"] == stats["scheduled_hypothesis_count"]
    assert all(row["prefilter_scheduled"] == (row["status"] == "PROPOSED") for row in result["hypotheses"])
    _assert_scope(result)


def _action_error(points, nominal, rotation, translation):
    return float(np.max(np.linalg.norm(points@np.asarray(rotation).T+translation-nominal, axis=1)))


@pytest.mark.parametrize("pose_index", range(2))
def test_real_noisy_mesh_search_finite_support_and_registration_required_positive(pose_index):
    nominal, raw, cad, faces, local = _mesh(7, inset=True, cad_n=2)
    e = np.array([1., -1., 0., 0., 0., -1., 1.])
    coordinates = np.linspace(-1., 1., 7)
    assert e.sum() == 0. and abs(e@coordinates) < 1e-15
    noise = NOISE_MM*np.outer(e, e).ravel()
    noisy = nominal.copy()
    for block, specification in enumerate(SPECS):
        normal = np.cross(specification[1], specification[2])
        normal = normal/np.linalg.norm(normal)
        noisy[49*block:49*(block+1)] += noise[:, None]*normal
    rotation, shift = POSES[pose_index]
    np.testing.assert_allclose(rotation@rotation.T, np.eye(3), atol=1e-15, rtol=0.)
    assert np.linalg.det(rotation) == pytest.approx(1., abs=1e-15)
    observed, clean_observed = noisy@rotation.T+shift, nominal@rotation.T+shift
    assert np.max(np.linalg.norm(observed-clean_observed, axis=1)) < EPSILON_MM
    samples = _mesh(9)[0]
    result = _register((observed, raw, cad, faces, local), samples)
    assert result["status"] == "CONDITIONAL_BEST_FIT", result["reasons"]
    evidence = result["evidence"]
    initialization = evidence["initialization"]
    _assert_complete_prefilter(initialization)
    assert initialization["hypothesis_evidence"]["schema_version"] == "plane_search_v6"
    assert any(_action_error(clean_observed, nominal, g["initialized_rotation"],
        g["initialized_translation_mm"]) <= START_ERROR_MM for g in initialization["start_groups"])
    for row in initialization["hypotheses"]:
        if row["prefilter_scheduled"]:
            assert len(row["assigned_support"]) == 3
            assert all(s["support_count"] == s["query_rows"] == 49 for s in row["assigned_support"])
    matrix = np.asarray(result["transform_scan_to_cad"])
    assert _action_error(clean_observed, nominal, matrix[:3, :3], matrix[:3, 3]) <= POSE_ERROR_MM
    assert _action_error(observed, nominal, matrix[:3, :3], matrix[:3, 3]) <= NOISY_ERROR_MM
    assert evidence["candidate_evaluation_complete"] is True
    assert evidence["attempted_candidate_count"] == evidence["candidate_count"] > 0
    assert evidence["candidate_numerical_failure_count"] == 0
    assert evidence["comparable_distinct_pose_count"] == 0
    assert evidence["policy"] == asdict(_policy(147))
    assert evidence["search_exhaustive"] is False and evidence["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert evidence["initial_assignments_are_final_face_matches"] is False
    assert evidence["prefilter_full_scan_raw_rows"] == evidence["registration_input_rows"] == 147
    _assert_scope(result)


def _configuration(*, budgets=None, policy=None):
    return v5._configuration(_policy() if policy is None else policy,
        v5._noisy.NoisyMeshPlanePolicyV5(), v5._noisy.NoisyMeshPlaneBudgetsV5(),
        v5._patch.CoplanarPatchPolicyV4(), v5._patch.CoplanarPatchBudgetsV4(),
        v5._search.PlaneSearchPolicyV6(), v5._search.PlaneSearchBudgetsV6(),
        v5.PlaneInitializationBudgetsV8() if budgets is None else budgets, None)


def _private_context(inputs=None, *, budgets=None, policy=None):
    """Real owned extraction; only later stipulated hypotheses are mocked."""
    inputs = _mesh() if inputs is None else inputs
    configuration, result = _configuration(budgets=budgets, policy=policy), v5._base()
    context = v5._prepare(*inputs, EPSILON_MM, configuration, result, UNIT)
    ids = []
    for block in range(3):
        ids.append(next(key for key, row in context["scan_supports"].items()
            if np.array_equal(row["raw_vertex_rows"], np.arange(25*block, 25*(block+1)))))
    return inputs, context, result, configuration, ids


def _hypothesis(index, ids, *, unsigned=0, shift=0., signs=(1, 1, 1), rejected=False):
    return dict(hypothesis_index=index, unsigned_assignment_id=unsigned,
        status="REJECTED" if rejected else "PROPOSED", reasons=[],
        rotation=np.eye(3).tolist(), translation_mm=[shift, 0., 0.], normal_signs=list(signs),
        correspondences=[dict(scan_plane_id=scan, cad_plane_id=face, normal_sign=sign)
                         for scan, face, sign in zip(ids, (11, 22, 33), signs)],
        test_scope="MOCK_HYPOTHESIS_PRIVATE_FINITE_SCREEN_ONLY")


def _private_run(monkeypatch, inputs, context, template, configuration):
    def prepared(*args):
        args[-2].update(deepcopy(template))
        return context
    monkeypatch.setattr(v5, "_prepare", prepared)
    return v5._initialize(*inputs, EPSILON_MM, configuration, UNIT)[0]


def test_real_owned_supports_are_complete_original_unions_and_detached():
    inputs, context, result, _, _ = _private_context()
    for row in context["scan_supports"].values():
        triangles, raw = row["triangle_rows"], row["raw_vertex_rows"]
        assert np.array_equal(raw, np.unique(context["raw_triangles"][triangles]))
        assert len(raw) == row["analysis"]["raw_support_count"]
        for values in (triangles, raw):
            assert values.flags.writeable is False
            with pytest.raises(ValueError):
                values.setflags(write=True)
    before = tuple(array.copy() for array in (context["scan"], context["raw_triangles"], context["cad"]))
    for array in inputs:
        array[...] = 0
    for actual, expected in zip((context["scan"], context["raw_triangles"], context["cad"]), before):
        assert np.array_equal(actual, expected)
    assert result["input_digests"]["scan_points_sha256"] == v5._digest(context["scan"])


@pytest.mark.parametrize("fault", ("drop_raw", "negative_triangle", "drop_cad_face_row", "improper_pose"))
def test_private_forged_support_is_refused_before_any_finite_query(monkeypatch, fault):
    inputs, context, result, configuration, ids = _private_context()
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i) for i in range(3)]
    if fault == "improper_pose":
        context["hypotheses"][0]["rotation"][0][0] = -1.
    elif fault == "drop_cad_face_row":
        patch = context["cad_patches"]["patches"][0]
        for name in ("triangle_rows", "source_triangle_ids"):
            patch[name] = patch[name][1:]
            patch[name+"_sha256"] = v5._digest(patch[name])
    else:
        patch = context["scan_supports"][ids[0]]
        key = "raw_vertex_rows" if fault == "drop_raw" else "triangle_rows"
        values = patch[key][1:] if fault == "drop_raw" else np.array([-1], dtype=np.int64)
        patch[key] = values
        patch[key+"_sha256"] = v5._digest(values)
        if fault == "drop_raw":
            patch["analysis"]["raw_support_count"] -= 1
    monkeypatch.setattr(v5._old, "_FaceQueries", lambda *a: pytest.fail("Invalid lineage reached finite work"))
    answer = _private_run(monkeypatch, inputs, context, result, configuration)
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    assert answer["selected_policy_evaluation_complete"] is False
    assert answer["statistics"]["query_rows"] == 0
    assert [row["hypothesis_index"] for row in answer["hypotheses"]] == [0, 1, 2]
    assert answer["statistics"]["scheduled_hypothesis_count"] == 3
    assert all(row["prefilter_status"] == "NOT_ATTEMPTED" for row in answer["hypotheses"])


def test_private_actual_finite_hole_preserves_all_three_support_denominators(monkeypatch):
    points, raw, cad, faces, local = _mesh()
    omitted = {2*(4*y+x)+side for y in (1, 2) for x in (1, 2) for side in (0, 1)}
    keep = np.array([index not in omitted for index in range(len(cad))])
    inputs, context, template, configuration, ids = _private_context(
        (points, raw, cad[keep], faces[keep], local[keep]), policy=replace(_policy(), min_scan_coverage=1.))
    context["hypotheses"] = [_hypothesis(0, ids)]
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    _assert_complete_prefilter(result)
    row, = result["hypotheses"]
    assert row["prefilter_status"] == "REJECTED" and result["start_groups"] == []
    assert [s["query_rows"] for s in row["assigned_support"]] == [25, 25, 25]
    assert [s["coverage"] for s in row["assigned_support"]] == [24/25, 1., 1.]
    assert result["face_index_evidence"][0]["triangle_count"] == 24


class _AlwaysInsideMockQuery:
    """Mock finite answers used ONLY to isolate scheduling and charging."""
    def __init__(self, context, faces, budget, surface, result):
        self.result = result

    def __call__(self, face, points):
        stats = self.result["statistics"]
        stats["query_calls"] += 1
        stats["query_rows"] += len(points)
        return dict(distances=np.zeros(len(points)), normal_ambiguous=np.zeros(len(points), dtype=bool),
            closest_point_ambiguous=np.zeros(len(points), dtype=bool), closest_regions=np.full(len(points), "FACE"))


@pytest.mark.parametrize("layout", ("contiguous", "gapped_eligible_ids"))
def test_mock_all_generated_unsigned_ids_screen_every_proposed_sign(monkeypatch, layout):
    inputs, context, template, configuration, ids = _private_context()
    if layout == "contiguous":
        context["hypotheses"] = [_hypothesis(8*u+s, ids, unsigned=u, signs=signs)
            for u in range(40) for s, signs in enumerate(itertools.product((-1, 1), repeat=3))]
        context["hypotheses"].append(_hypothesis(320, ids, unsigned=40, rejected=True))
        selected, expected_refused = list(range(40)), 1
    else:
        # IDs 0..79 remain a complete unsigned ledger: each odd ID retains all
        # eight nominal refusals, leaving gaps only in the eligible-ID set.
        context["hypotheses"] = [_hypothesis(8*u+s, ids, unsigned=u, signs=signs, rejected=bool(u % 2))
            for u in range(80) for s, signs in enumerate(itertools.product((-1, 1), repeat=3))]
        for row in context["hypotheses"]:
            if row["status"] == "REJECTED":
                row.update(rotation=None, translation_mm=None)
        selected, expected_refused = list(range(0, 80, 2)), 320
    monkeypatch.setattr(v5._old, "_FaceQueries", _AlwaysInsideMockQuery)
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    _assert_complete_prefilter(result)
    assert result["selected_unsigned_assignment_ids"] == selected
    for u in selected:
        rows = [row for row in result["hypotheses"] if row["unsigned_assignment_id"] == u]
        assert len(rows) == 8 and {tuple(row["normal_signs"]) for row in rows} == set(itertools.product((-1, 1), repeat=3))
        assert all(row["prefilter_scheduled"] is True and row["prefilter_status"] == "PASSED" for row in rows)
    refused = [row for row in result["hypotheses"] if row["status"] == "REJECTED"]
    assert len(refused) == expected_refused
    assert all(row["prefilter_scheduled"] is False and row["assigned_support"] == []
        and row["prefilter_status"] == "SEARCH_NOMINAL_WITNESS_REFUSAL" for row in refused)
    stats = result["statistics"]
    assert stats["scheduled_hypothesis_count"] == stats["hypotheses_evaluated"] == 320
    assert stats["eligible_unsigned_assignment_count"] == stats["scheduled_unsigned_assignment_count"] == 40
    assert stats["omitted_unsigned_assignment_count"] == stats["omitted_proposed_hypothesis_count"] == 0
    assert stats["search_rejected_hypothesis_count"] == expected_refused
    assert stats["planned_query_rows"] == stats["query_rows"] == 320*3*25
    assert len(result["all_passing_start_groups"]) == 1
    assert len(result["all_passing_start_groups"][0]["hypothesis_indices"]) == 320


def test_mock_exact_64_groups_schedule_48_strata_without_approximate_merge(monkeypatch):
    inputs, context, template, configuration, ids = _private_context()
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i//2, shift=i*1e-9) for i in range(64)]
    monkeypatch.setattr(v5._old, "_FaceQueries", _AlwaysInsideMockQuery)
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    _assert_complete_prefilter(result)
    assert len(result["all_passing_start_groups"]) == 64
    assert [g["exact_group_index"] for g in result["start_groups"]] == [i*63//47 for i in range(48)]
    assert result["statistics"]["omitted_passing_start_group_count"] == 16
    assert all(g["hypothesis_indices"] == [i] for i, g in enumerate(result["all_passing_start_groups"]))


@pytest.mark.parametrize("field,maximum", (("max_prefilter_query_rows", 149), ("max_prefilter_query_calls", 5)))
def test_repeated_rows_are_preflighted_for_every_pose_before_surface_build(monkeypatch, field, maximum):
    budget = replace(v5.PlaneInitializationBudgetsV8(), **{field: maximum})
    inputs, context, template, configuration, ids = _private_context(budgets=budget)
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i) for i in range(2)]
    monkeypatch.setattr(v5._old, "_FaceQueries", lambda *a: pytest.fail("Whole obligation did not preflight"))
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["reasons"] == ["PREFILTER_PREFLIGHT_BUDGET_EXCEEDED"]
    assert result["statistics"]["planned_query_rows"] == 150
    assert result["statistics"]["planned_query_calls"] == 6
    assert result["statistics"]["query_rows"] == 0


def test_late_mock_query_failure_invalidates_earlier_exact_group(monkeypatch):
    inputs, context, template, configuration, ids = _private_context()
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i) for i in range(2)]
    class LateFailure(_AlwaysInsideMockQuery):
        def __call__(self, face, points):
            if self.result["statistics"]["query_calls"] == 3:
                raise RuntimeError("MOCK_LATE_FINITE_QUERY_FAILURE")
            return super().__call__(face, points)
    monkeypatch.setattr(v5._old, "_FaceQueries", LateFailure)
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["selected_policy_evaluation_complete"] is False
    group, = result["all_passing_start_groups"]
    assert group["refinement_disposition"] == "INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE"
    assert group["initialized_rotation"] is None and group["initialized_translation_mm"] is None
    assert group["attempted_initialized_rotation"] == np.eye(3).tolist()
    assert result["statistics"]["hypotheses_evaluated"] == 1
    assert result["statistics"]["exact_passing_start_group_count"] == 1
    assert result["statistics"]["exact_passing_start_group_count_scope"] == "ATTEMPTED_PREFILTER_BEFORE_FATAL_NOT_SURVIVING"
    assert result["statistics"]["omitted_passing_start_group_count"] is None
    assert result["hypotheses"][1]["assigned_support"][0]["attempted_query_rows"] == 25
    assert all(row["usable_initialization"] is False for row in result["hypotheses"])


@pytest.mark.parametrize("name", tuple(field.name for field in fields(v5.PlaneInitializationBudgetsV8)))
def test_bypass_raised_literal_cap_refuses_before_extraction(monkeypatch, name):
    budget = v5.PlaneInitializationBudgetsV8()
    object.__setattr__(budget, name, getattr(budget, name)+1)
    monkeypatch.setattr(v5, "_prepare", lambda *a: pytest.fail("Bypass cap reached geometry"))
    result = _prefilter(_mesh(), budgets=budget)
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["reasons"] == ["INVALID_V8_INITIALIZATION_BUDGET"]


@pytest.mark.parametrize("fault", ("unit", "negative_triangle", "duplicate_local", "nan_epsilon", "negative_selection"))
def test_public_input_failures_publish_no_usable_registration(fault):
    inputs = list(_mesh())
    kwargs = {}
    if fault == "unit":
        kwargs["unit_assumption"] = "VERIFIED_BY_CAD"
    elif fault == "negative_triangle":
        inputs[1][-1, -1] = -1
    elif fault == "duplicate_local":
        inputs[4][-1] = inputs[4][-2]
    elif fault == "nan_epsilon":
        kwargs["position_error_bounds_mm"] = float("nan")
    else:
        selection = np.arange(36, dtype=np.int64)
        selection[-1] = -1
        kwargs["registration_raw_rows"] = selection
    result = _register(inputs, **kwargs)
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] and result["evidence"]["candidate_evaluation_complete"] is False
    _assert_scope(result)


def test_owned_registration_selection_rechecks_lower_bound_after_copy_mutation(monkeypatch):
    selection = np.arange(36, dtype=np.int64)
    original, events = v5._patch._immutable, []
    def mutate_at_snapshot(value, dtype):
        if value is selection:
            selection[-1] = -1
            events.append("MUTATED_AFTER_EARLY_ADMISSION")
        return original(value, dtype)
    monkeypatch.setattr(v5._patch, "_immutable", mutate_at_snapshot)
    monkeypatch.setattr(v5, "_initialize", lambda *a: pytest.fail("Owned negative index reached extraction"))
    result = _register(_mesh(), registration_raw_rows=selection)
    assert events == ["MUTATED_AFTER_EARLY_ADMISSION"]
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["evidence"]["candidates"] == []
    assert result["reasons"] == ["INVALID_OWNED_REGISTRATION_RAW_ROW_SELECTION"]


def test_real_single_plane_finishes_as_insufficient_initialization_not_observed_absence():
    points, raw, cad, faces, local = _mesh(7, inset=True, cad_n=2)
    result = _register((points[:49], raw[:72], cad[:2], faces[:2], local[:2]), _mesh(9)[0][:81])
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] == ["NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"]
    initialization = result["evidence"]["initialization"]
    _assert_complete_prefilter(initialization)
    assert initialization["start_groups"] == []
    assert initialization["hypothesis_evidence"]["reasons"] == ["FEWER_THAN_THREE_PLANES_IN_DECLARED_SEARCH"]
    assert result["evidence"]["candidate_count"] == 0
    _assert_scope(result)


def test_valid_owned_global_row_selection_records_exact_basis_and_digest():
    points = _mesh()[0]
    rows = np.concatenate([np.arange(12, dtype=np.int64)+25*i for i in range(3)])
    evidence = {}
    selected = v5._registration_rows(dict(scan=points), rows, evidence)
    assert np.array_equal(selected, np.unique(points[rows], axis=0))
    assert evidence["registration_input_rows"] == 36 and evidence["prefilter_full_scan_raw_rows"] == 75
    assert evidence["registration_raw_rows"]["sha256"] == v5._digest(rows)
    assert evidence["coverage_basis"].startswith("EXPLICIT_SELECTED_")


def test_default_registration_gates_are_snapshotted_without_v5_relaxation():
    default = v5._v1.WholePartRegistrationPolicyV1()
    configuration = _configuration(policy=default)
    assert asdict(configuration["policy"]) == asdict(default)
    assert configuration["policy"] is not default
    assert type(configuration["search_policy"]) is v5._search.PlaneSearchPolicyV6
    assert type(configuration["search_budgets"]) is v5._search.PlaneSearchBudgetsV6
    bounds = configuration["budgets"]
    assert (bounds.max_prefilter_unsigned_assignments, bounds.max_refined_start_groups,
        bounds.max_prefilter_query_rows, bounds.max_prefilter_query_calls) == (1024, 48, 64_000_000, 32768)
    assert (default.max_correspondence_distance_mm, default.max_inlier_plane_p95_mm,
            default.min_scan_coverage, default.min_cad_coverage, default.min_observability_ratio) == (2., .5, .75, .25, .005)
    assert v5._v2._refine is v5._old._v2._refine
    assert v5._v2._diagnose is v5._old._v2._diagnose


@pytest.mark.parametrize("failure", ("numerical", "fatal_surface", "competing_pose"))
def test_mock_selected_candidate_failure_cannot_publish_an_earlier_pass(monkeypatch, failure):
    inputs = _mesh()
    initialization = v5._base()
    initialization.update(status="COMPLETE", search_complete=True, selected_policy_evaluation_complete=True,
        input_digests={}, start_groups=[dict(index=i, initialized_rotation=np.eye(3).tolist(),
            initialized_translation_mm=[float(i) if failure == "competing_pose" else 0., 0., 0.],
            hypothesis_indices=[i]) for i in range(3)])
    monkeypatch.setattr(v5, "_initialize", lambda *a: (deepcopy(initialization), dict(cad=inputs[2], faces=inputs[3])))
    calls = []
    def mocked_refine(points, query, rotation, translation, **kwargs):
        calls.append(kwargs["coarse"])
        if len(calls) == 3 and failure != "competing_pose":
            if failure == "numerical":
                raise np.linalg.LinAlgError("Deliberate helper-only numerical failure")
            raise v5._v2._SurfaceFailure("MOCK_SURFACE_BUDGET_FAILURE")
        return rotation, translation, 0, "MOCK_COMPLETE", {}
    monkeypatch.setattr(v5._v2, "_refine", mocked_refine)
    monkeypatch.setattr(v5._v2, "_diagnose", lambda *a: dict(scan_coverage=1., cad_coverage=1.,
        inlier_surface_rms_mm=0., reasons=[]))
    result = _register(inputs)
    evidence = result["evidence"]
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert evidence["candidate_count"] == 3
    assert evidence["candidate_evaluation_complete"] == (failure == "competing_pose")
    assert "selected_candidate_index" not in evidence
    if failure == "numerical":
        assert evidence["candidate_numerical_failure_count"] == 1
        assert evidence["attempted_candidate_count"] == 3
        assert result["reasons"] == ["INCOMPLETE_SELECTED_CANDIDATE_EVALUATION"]
    elif failure == "fatal_surface":
        assert evidence["attempted_candidate_count"] == 2
        assert evidence["candidates"][2]["reasons"] == ["NOT_ATTEMPTED_AFTER_FATAL_FAILURE"]
        assert result["reasons"] == ["MOCK_SURFACE_BUDGET_FAILURE"]
    else:
        assert evidence["attempted_candidate_count"] == 3
        assert evidence["candidate_numerical_failure_count"] == 0
        assert evidence["comparable_distinct_pose_count"] == 2
        assert result["reasons"] == ["AMBIGUOUS_GENERATED_WHOLE_PART_POSES"]
    _assert_scope(result)


def test_report_overflow_returns_whole_refusal_envelope():
    result = _prefilter(_mesh(), budgets=replace(v5.PlaneInitializationBudgetsV8(), max_report_bytes=100))
    assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
    assert result["hypotheses"] == [] and result["transform_scan_to_cad"] is None
    assert result["detailed_report_discarded_whole"] is True
    assert result["reasons"] == ["INITIALIZATION_REPORT_BUDGET_EXCEEDED"]
    assert result["discarded_hypothesis_records"] > 0
    _assert_scope(result)

def test_v8_compact_patch_evidence_decodes_complete_actual_geometry():
    result = _prefilter(_mesh())
    _assert_complete_prefilter(result)
    compact = result["patch_evidence"]["scan"]
    assert compact["format"] == "noisy_mesh_metadata_schema_v6"
    decoded = v5._compact.decode_metadata_v7(compact)
    assert decoded["schema_version"] == "noisy_mesh_planes_v7"
    assert decoded["status"] == "COMPLETE"
    assert decoded["statistics"]["core_evaluation_complete"] is True
    assert decoded["input_digests"]["points_sha256"] == result["input_digests"]["scan_points_sha256"]
    assert decoded["proposals"] and decoded["cores"]


def test_v8_actual_registration_outer_refusal_preserves_inner_discarded_hypotheses():
    result = _register(_mesh(), budgets=replace(v5.PlaneInitializationBudgetsV8(), max_report_bytes=100))
    assert result["schema_version"] == "initialization_report_refusal_envelope_v6"
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["discarded_hypothesis_records"] > 0
    assert result["first_refusal"]["discarded_hypothesis_records"] == result["discarded_hypothesis_records"]
    assert result["first_refusal"]["source_schema_version"] == "noisy_plane_initialization_prefilter_v8"
    assert result["first_refusal"]["encoded_bytes_examined"] == 101
    assert result["encoded_bytes_examined"] == 101
    repeated = v5._bounded_report(result, 100)
    assert repeated == result and repeated is not result
    _assert_scope(result)


@pytest.mark.parametrize("bound", (2, 1))
def test_hard_unsigned_bound_exact_and_over_retains_complete_ledger_before_support_work(monkeypatch, bound):
    budget = replace(v5.PlaneInitializationBudgetsV8(), max_prefilter_unsigned_assignments=bound)
    inputs, context, template, configuration, ids = _private_context(budgets=budget)
    context["hypotheses"] = [_hypothesis(8*u+s, ids, unsigned=u, signs=signs, rejected=(u == 1))
        for u in range(3) for s, signs in enumerate(itertools.product((-1, 1), repeat=3))]
    monkeypatch.setattr(v5._old, "_FaceQueries", _AlwaysInsideMockQuery)
    if bound == 1:
        monkeypatch.setattr(v5, "_bound_scan", lambda *a: pytest.fail("Unsigned cap reached support validation"))
        monkeypatch.setattr(v5._old, "_bound_support", lambda *a, **k: pytest.fail("Unsigned cap reached CAD validation"))
        monkeypatch.setattr(v5._old, "_FaceQueries", lambda *a: pytest.fail("Unsigned cap reached face allocation"))
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    assert result["selected_unsigned_assignment_ids"] == [0, 2]
    assert [row["hypothesis_index"] for row in result["hypotheses"]] == list(range(24))
    stats = result["statistics"]
    assert stats["eligible_unsigned_assignment_count"] == stats["scheduled_unsigned_assignment_count"] == 2
    assert stats["scheduled_hypothesis_count"] == 16 and stats["search_rejected_hypothesis_count"] == 8
    assert stats["omitted_unsigned_assignment_count"] == stats["omitted_proposed_hypothesis_count"] == 0
    if bound == 2:
        _assert_complete_prefilter(result)
        assert stats["hypotheses_evaluated"] == 16
        assert stats["planned_query_rows"] == stats["query_rows"] == 16*75
    else:
        assert result["status"] == "UNAVAILABLE" and result["start_groups"] == []
        assert result["reasons"] == ["ALL_PROPOSED_UNSIGNED_PREFLIGHT_BUDGET_EXCEEDED"]
        assert result["selected_policy_evaluation_complete"] is False
        assert stats["planned_query_rows"] == stats["query_rows"] == stats["query_calls"] == 0
        assert stats["hypotheses_evaluated"] == 0
        assert all(row["assigned_support"] == [] and row["usable_initialization"] is False
                   for row in result["hypotheses"])
        assert all(row["prefilter_status"] == ("NOT_ATTEMPTED" if row["status"] == "PROPOSED"
            else "SEARCH_NOMINAL_WITNESS_REFUSAL") for row in result["hypotheses"])


@pytest.mark.parametrize("has_proposal", (False, True))
def test_zero_hard_unsigned_bound_is_not_a_zero_search_quota(monkeypatch, has_proposal):
    budget = replace(v5.PlaneInitializationBudgetsV8(), max_prefilter_unsigned_assignments=0)
    inputs, context, template, configuration, ids = _private_context(budgets=budget)
    context["hypotheses"] = [_hypothesis(s, ids, signs=signs, rejected=not has_proposal)
        for s, signs in enumerate(itertools.product((-1, 1), repeat=3))]
    monkeypatch.setattr(v5._old, "_FaceQueries", _AlwaysInsideMockQuery)
    monkeypatch.setattr(v5, "_bound_scan", lambda *a: pytest.fail("Zero obligation/cap reached support validation"))
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    assert len(result["hypotheses"]) == 8 and result["start_groups"] == []
    assert result["statistics"]["query_rows"] == result["statistics"]["query_calls"] == 0
    if has_proposal:
        assert result["status"] == "UNAVAILABLE"
        assert result["reasons"] == ["ALL_PROPOSED_UNSIGNED_PREFLIGHT_BUDGET_EXCEEDED"]
        assert result["statistics"]["scheduled_hypothesis_count"] == 8
        assert not result["selected_policy_evaluation_complete"]
    else:
        _assert_complete_prefilter(result)
        assert result["selected_unsigned_assignment_ids"] == []
        assert result["reasons"] == ["NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"]
    _assert_scope(result)


@pytest.mark.parametrize("field,maximum", (("max_prefilter_query_rows", 150), ("max_prefilter_query_calls", 6)))
def test_repeated_full_row_obligation_exact_work_bound_completes(monkeypatch, field, maximum):
    budget = replace(v5.PlaneInitializationBudgetsV8(), **{field: maximum})
    inputs, context, template, configuration, ids = _private_context(budgets=budget)
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i) for i in range(2)]
    monkeypatch.setattr(v5._old, "_FaceQueries", _AlwaysInsideMockQuery)
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    _assert_complete_prefilter(result)
    assert result["statistics"]["planned_query_rows"] == result["statistics"]["query_rows"] == 150
    assert result["statistics"]["planned_query_calls"] == result["statistics"]["query_calls"] == 6
    assert all([support["query_rows"] for support in row["assigned_support"]] == [25, 25, 25]
               for row in result["hypotheses"])


def test_early_geometric_refusal_does_not_skip_later_supports_or_proposals(monkeypatch):
    inputs, context, template, configuration, ids = _private_context()
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i) for i in range(2)]
    class FirstSupportOutside(_AlwaysInsideMockQuery):
        def __call__(self, face, points):
            answer = super().__call__(face, points)
            if self.result["statistics"]["query_calls"] == 1:
                answer["distances"][:] = 2*configuration["policy"].max_correspondence_distance_mm
            return answer
    monkeypatch.setattr(v5._old, "_FaceQueries", FirstSupportOutside)
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    _assert_complete_prefilter(result)
    assert [row["prefilter_status"] for row in result["hypotheses"]] == ["REJECTED", "PASSED"]
    assert [support["status"] for support in result["hypotheses"][0]["assigned_support"]] == ["REJECTED", "PASSED", "PASSED"]
    assert all([support["query_rows"] for support in row["assigned_support"]] == [25, 25, 25]
               for row in result["hypotheses"])
    assert result["statistics"]["hypotheses_evaluated"] == 2
    assert result["statistics"]["query_rows"] == 150 and result["statistics"]["query_calls"] == 6
    assert result["all_passing_start_groups"][0]["hypothesis_indices"] == [1]


def test_late_failure_beyond_old_32_quota_invalidates_every_prior_exact_group(monkeypatch):
    inputs, context, template, configuration, ids = _private_context()
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i, shift=i*1e-9) for i in range(40)]
    class BeyondOldQuotaFailure(_AlwaysInsideMockQuery):
        def __call__(self, face, points):
            if self.result["statistics"]["query_calls"] == 99:
                raise RuntimeError("MOCK_FAILURE_AFTER_33_COMPLETE_FULL_SUPPORTS")
            return super().__call__(face, points)
    monkeypatch.setattr(v5._old, "_FaceQueries", BeyondOldQuotaFailure)
    result = _private_run(monkeypatch, inputs, context, template, configuration)
    assert result["status"] == "UNAVAILABLE" and not result["selected_policy_evaluation_complete"]
    assert result["start_groups"] == [] and len(result["hypotheses"]) == 40
    assert result["selected_unsigned_assignment_ids"] == list(range(40))
    assert result["statistics"]["scheduled_hypothesis_count"] == 40
    assert result["statistics"]["hypotheses_evaluated"] == 33
    assert result["statistics"]["planned_query_rows"] == 3000
    assert result["statistics"]["query_rows"] == 2475 and result["statistics"]["query_calls"] == 99
    assert result["statistics"]["exact_passing_start_group_count"] == len(result["all_passing_start_groups"]) == 33
    assert result["statistics"]["omitted_passing_start_group_count"] is None
    assert all(group["initialized_rotation"] is group["initialized_translation_mm"] is None
        and group["refinement_disposition"] == "INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE"
        for group in result["all_passing_start_groups"])
    assert all(row["usable_initialization"] is False for row in result["hypotheses"])
    assert result["hypotheses"][33]["assigned_support"][0]["attempted_query_rows"] == 25
    assert all(row["prefilter_status"] == "NOT_ATTEMPTED" for row in result["hypotheses"][34:])


def test_v7_case_bodies_are_preserved_except_declared_outer_screen_assertions():
    old = Path(__file__).with_name("test_cad_guided_plane_initialization_v7.py").read_text()
    replacements = {"cad_guided_plane_initialization_v7": "cad_guided_plane_initialization_v8",
        "prefilter_plane_initializations_v7": "prefilter_plane_initializations_v8",
        "register_whole_part_planes_v7": "register_whole_part_planes_v8",
        "PlaneInitializationBudgetsV7": "PlaneInitializationBudgetsV8",
        "INVALID_V7_INITIALIZATION": "INVALID_V8_INITIALIZATION",
        "noisy_plane_initialization_prefilter_v7": "noisy_plane_initialization_prefilter_v8",
        "def test_v7_": "def test_v8_"}
    for before, after in replacements.items(): old = old.replace(before, after)
    current = {node.name: node for node in ast.parse(Path(__file__).read_text()).body
               if isinstance(node, ast.FunctionDef)}
    allowed_changes = {"_assert_complete_prefilter",
        "test_mock_unsigned_strata_screen_all_signs_and_keep_omitted_provenance",
        "test_default_registration_gates_are_snapshotted_without_v5_relaxation"}
    for node in ast.parse(old).body:
        if isinstance(node, ast.FunctionDef) and node.name not in allowed_changes:
            assert ast.dump(current[node.name], include_attributes=False) == ast.dump(node, include_attributes=False), node.name
