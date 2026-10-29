"""Independent interval-aware registration consumer controls; invented arrays only.

Prospective 68 cases, frozen only by the root's exact-source review and receipt.
No native input or public pose-recovery matrix is implied. Mocked correspondences,
scores and failures are explicitly decision/arithmetic/ledger seams, never
evidence of recovered physical geometry.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
UNITS = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
BOOLEAN_FIELDS = (
    "normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous",
    "normal_ambiguity_uncertain", "closest_point_ambiguity_uncertain", "region_uncertain",
)


@pytest.fixture
def registration():
    """Load only the reviewed pure source namespace, never backend startup."""
    package_name = "independent_registration_v3_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "backend")]
    sys.modules[package_name] = package
    name = package_name + ".cad_guided_global_registration_v3"
    try:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "backend/cad_guided_global_registration_v3.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


def _axis_pairs():
    """36 unique centrally symmetric rows, not a supplied inference pose."""
    return np.vstack([np.vstack((np.eye(3), -np.eye(3))) * radius
                      for radius in range(1, 7)])


def _box_panels():
    """Independent analytic plane rows and outward axis normals for rank seams."""
    lengths = (4., 2.5, 1.5)
    points, normals = [], []
    for axis in range(3):
        others = [value for value in range(3) if value != axis]
        for side in (0., lengths[axis]):
            for u, v in itertools.product((.2, .5, .8), repeat=2):
                point = np.zeros(3)
                point[axis] = side
                point[others[0]], point[others[1]] = lengths[others[0]] * u, lengths[others[1]] * v
                normal = np.zeros(3)
                normal[axis] = -1. if side == 0. else 1.
                points.append(point)
                normals.append(normal)
    return np.asarray(points), np.asarray(normals)


def _query(count, *, lower=.125, upper=.125, targets=None, normals=None):
    """Declared artificial query contract, not output computed by the candidate.

    Tests modify individual fields deliberately. Constant exact dyadic distances
    avoid borrowing the implementation's nominal-distance or enclosure arithmetic
    as a decision oracle.
    """
    low = np.broadcast_to(np.asarray(lower, dtype=float), (count,)).copy()
    high = np.broadcast_to(np.asarray(upper, dtype=float), (count,)).copy()
    result = dict(status="COMPLETE", reason="CONDITIONAL_LOCAL_FINITE_TRIANGLE_QUERY_COMPLETE",
        closest_points=np.zeros((count, 3)) if targets is None else np.asarray(targets, dtype=float).copy(),
        distances=(low + high) / 2., triangle_indices=np.zeros(count, dtype=np.int64),
        source_face_ids=(0,) * count,
        oriented_normals=np.tile([0., 0., 1.], (count, 1)) if normals is None else np.asarray(normals, dtype=float).copy(),
        closest_regions=np.full(count, "FACE", dtype="U6"), near_tie_counts=np.ones(count, dtype=np.int64),
        distance_lower=low, distance_upper=high,
        triangle_distance_lower=low.copy(), triangle_distance_upper=high.copy(),
        numerical_distance_guard=(high - low) / 2.,
        closest_point_error=np.zeros(count), closest_point_restoration_error=np.zeros(count),
        distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
        closest_point_error_scope="SELECTED_TRIANGLE_OPTIMIZER_NOT_GLOBAL_UNIQUE_WITNESS",
        normal_region_status="ENGINEERING_NOT_INTERVAL_CERTIFIED",
        tie_semantics="POSSIBLE_COMPETITORS_AT_FIXED_GLOBAL_UPPER_PLUS_TAU",
        diagnostics=dict(node_visits=2*count, triangle_box_pairs=2*count, triangle_pairs=2*count,
            certificate_pairs=8*count, normal_pairs=0, work_units=28*count, passes_completed=2,
            maximum_pair_batch=count, maximum_certificate_batch=count,
            tie_batches_completed=int(count > 0), maximum_tie_storage_rows=count,
            interval_width_rejections=0, certificate_failures=0, proof_pruned_triangle_pairs=0))
    result.update({field: np.zeros(count, dtype=bool) for field in BOOLEAN_FIELDS})
    return result


def _policy(registration, **changes):
    module = importlib.import_module(
        ".cad_guided_global_registration_v1", registration.__package__)
    return replace(module.WholePartRegistrationPolicyV1(
        coarse_points=1024, refine_points=1024, coarse_iterations=20, refine_iterations=20,
        huber_delta_mm=.01, max_correspondence_distance_mm=.5,
        max_inlier_plane_p95_mm=.05, min_scan_coverage=.95, min_cad_coverage=.30,
        pose_class_distance_mm=.05, ambiguity_plane_rms_margin_mm=.005), **changes)


def test_complete_query_contract_is_accepted_without_mutation(registration):
    query = _query(3)
    original = deepcopy(query)
    assert registration._validate_query_result(query, 3) is query
    for key, value in query.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, original[key])
        else:
            assert value == original[key]


@pytest.mark.parametrize("fault", (
    "missing_lower", "distance_rows", "nonfinite_point", "inverted_interval", "negative_lower",
    "integer_flag", "floating_index", "negative_index", "zero_ties", "face_rows", "bad_region",
    "nonunit_normal", "negative_point_error", "restoration_exceeds_total", "wrong_distance_model",
    "negative_guard", "unavailable",
))
def test_malformed_query_contract_cannot_enter_registration(registration, fault):
    query = _query(3)
    if fault == "missing_lower":
        del query["distance_lower"]
    elif fault == "distance_rows":
        query["distances"] = query["distances"][:2]
    elif fault == "nonfinite_point":
        query["closest_points"][2, 0] = np.nan
    elif fault == "inverted_interval":
        query["distance_lower"][2] = .25
    elif fault == "negative_lower":
        query["distance_lower"][2] = -.125
    elif fault == "integer_flag":
        query["closest_point_ambiguity_uncertain"] = np.zeros(3, dtype=np.int64)
    elif fault == "floating_index":
        query["triangle_indices"] = np.zeros(3, dtype=float)
    elif fault == "negative_index":
        query["triangle_indices"][2] = -1
    elif fault == "zero_ties":
        query["near_tie_counts"][2] = 0
    elif fault == "face_rows":
        query["source_face_ids"] = (0, 0)
    elif fault == "bad_region":
        query["closest_regions"][2] = "VOID"
    elif fault == "nonunit_normal":
        query["oriented_normals"][2] = [0., 0., 2.]
    elif fault == "negative_point_error":
        query["closest_point_error"][2] = -.125
    elif fault == "restoration_exceeds_total":
        query["closest_point_restoration_error"][2] = .125
    elif fault == "wrong_distance_model":
        query["distance_model"] = "NOMINAL_DISTANCE_WITHOUT_INTERVALS"
    elif fault == "negative_guard":
        query["numerical_distance_guard"][2] = -.125
    else:
        query["status"] = "UNAVAILABLE"
    with pytest.raises(registration._SurfaceFailure):
        registration._validate_query_result(query, 3)


def test_support_masks_have_inclusive_upper_and_strict_outside_boundaries(registration):
    above = math.nextafter(.5, math.inf)
    query = _query(6, lower=[0., .5, .5, above, .375, 0.],
                    upper=[.5, .5, above, .75, .625, .375])
    masks = registration._support_masks(query, .5)
    assert masks["definite_support"].tolist() == [True, True, False, False, False, True]
    assert masks["definite_outside"].tolist() == [False, False, False, True, False, False]
    assert masks["unresolved"].tolist() == [False, False, True, False, True, False]
    assert np.all(sum(masks[key].astype(int) for key in
                      ("definite_support", "definite_outside", "unresolved")) == 1)
    query["normal_ambiguous"][:] = True
    ambiguous_masks = registration._support_masks(query, .5)
    for key in ("definite_support", "definite_outside", "unresolved"):
        np.testing.assert_array_equal(ambiguous_masks[key], masks[key])


@pytest.mark.parametrize("flag", (
    "normal_ambiguous", "closest_point_ambiguous", "normal_ambiguity_uncertain",
    "closest_point_ambiguity_uncertain",
))
def test_new_uncertainty_cannot_hide_behind_false_legacy_flags(registration, flag):
    query = _query(3)
    query[flag][1] = True
    assert registration._unambiguous(query).tolist() == [True, False, True]


@pytest.mark.parametrize("flag", ("face_identity_ambiguous", "region_uncertain"))
def test_identity_or_region_uncertainty_does_not_delete_finite_support(registration, flag):
    query = _query(3)
    query[flag][:] = True
    assert registration._unambiguous(query).tolist() == [True, True, True]
    assert registration._support_masks(query, .5)["definite_support"].tolist() == [True] * 3


def test_chosen_rows_cannot_promote_a_straddling_nominal_inlier_or_refill(registration):
    query = _query(60, lower=.125, upper=.125)
    query["normal_ambiguity_uncertain"][:20] = True
    # Nominal .25 is below the gate, but the interval does not prove support.
    query["distance_lower"][20:30] = 0.
    query["distance_upper"][20:30] = .625
    query["distances"][20:30] = .25
    query["triangle_distance_lower"][20:30] = 0.
    query["triangle_distance_upper"][20:30] = .625
    query["numerical_distance_guard"][20:30] = .375
    retained = registration._chosen(query, .5, .85)
    assert retained.tolist() == list(range(30, 60))


def _surface_index(registration):
    surface = importlib.import_module(
        ".cad_guided_triangle_surface_v3", registration.__package__)
    triangle = np.array([[[0., 0., 0.], [4., 0., 0.], [0., 4., 0.]]])
    built = surface.build_triangle_surface(triangle, ["owned-face"])
    assert built["status"] == "COMPLETE", built
    return surface, built["index"]


def test_actual_finite_query_adapter_preserves_interval_and_face_binding(registration):
    surface, index = _surface_index(registration)
    query = registration._Queries(index, registration.SurfaceRegistrationBudgetsV3(),
                                  surface.TriangleSurfaceBudgets())
    points = np.array([[1., 1., .125], [1., 1., .75]])
    answer = query(points)
    assert answer["source_face_ids"] == ("owned-face", "owned-face")
    expected = np.array([.125, .75])  # Exact finite interior projection, no fitted oracle.
    assert np.all(answer["distance_lower"] <= expected)
    assert np.all(answer["distance_upper"] >= expected)
    masks = registration._support_masks(answer, .5)
    assert masks["definite_support"].tolist() == [True, False]
    assert masks["definite_outside"].tolist() == [False, True]
    assert masks["unresolved"].tolist() == [False, False]
    assert query.statistics["query_calls"] == 1 and query.statistics["query_rows"] == 2
    assert query.statistics["certificate_pairs"] == 16


@pytest.mark.parametrize("fault", ("out_of_range_triangle", "forged_face"))
def test_query_validation_binds_selected_triangle_and_face_to_index(registration, fault):
    _, index = _surface_index(registration)
    query = _query(3)
    query["source_face_ids"] = ("owned-face",) * 3
    if fault == "out_of_range_triangle":
        query["triangle_indices"][2] = 1
    else:
        query["source_face_ids"] = ("owned-face", "owned-face", "invented-other-face")
    with pytest.raises(registration._SurfaceFailure):
        registration._validate_query_result(query, 3, index=index)


def test_cumulative_certificate_budget_failure_preserves_charged_work(registration):
    surface, index = _surface_index(registration)
    budget = replace(registration.SurfaceRegistrationBudgetsV3(), max_certificate_pairs=15)
    query = registration._Queries(index, budget, surface.TriangleSurfaceBudgets(max_pair_batch=1))
    point = np.array([[1., 1., .125]])
    assert query(point)["status"] == "COMPLETE"
    assert query.statistics["certificate_pairs"] == 8
    with pytest.raises(registration._SurfaceFailure,
                       match="^FINITE_SURFACE_QUERY_UNAVAILABLE:QUERY_BUDGET_EXCEEDED_CERTIFICATE_PAIRS$"):
        query(point)
    # The second query charges its three completed second-pass witnesses and
    # refuses the eighth certificate, preserving fifteen completed charges.
    assert query.statistics["query_calls"] == 2
    assert query.statistics["query_rows"] == 2
    assert query.statistics["certificate_pairs"] == 15
    assert query.statistics["work_units"] == 55


@pytest.mark.parametrize("field,value", (("max_query_calls", -1),
    ("max_certificate_pairs", True), ("max_work_units", 3.5)))
def test_bypassed_frozen_execution_budget_is_revalidated(registration, field, value):
    surface, index = _surface_index(registration)
    budget = registration.SurfaceRegistrationBudgetsV3()
    object.__setattr__(budget, field, value)
    with pytest.raises((ValueError, registration._SurfaceFailure)):
        registration._Queries(index, budget, surface.TriangleSurfaceBudgets())


def _fraction(value):
    return Fraction(*float(value).as_integer_ratio())


def _exact_linear_p95(values):
    ordered = sorted(_fraction(value) for value in values)
    rank = (len(ordered)-1) * _fraction(.95)
    index = rank.numerator // rank.denominator
    fraction = rank-index
    return ordered[index] if not fraction else ordered[index] + fraction*(ordered[index+1]-ordered[index])


@pytest.mark.parametrize("lower,actual,upper", (
    ([.125], [.25], [.375]),
    ([0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]),
    ([.25, 0., .375, .125, .5], [.375, .25, .5, .25, .5], [.5, .5, .625, .375, .75]),
    ([0., .125, .25, .375, .5, .625, .75], [.125, .25, .375, .5, .625, .75, .875],
     [.25, .375, .5, .625, .75, .875, 1.]),
))
def test_residual_summaries_enclose_independent_exact_quantile_and_rms(registration, lower, actual, upper):
    result = registration._metric_bounds(np.array(lower), np.array(upper))
    p95 = _exact_linear_p95(actual)
    mean_square = sum((_fraction(value)**2 for value in actual), Fraction(0))/len(actual)
    assert _fraction(result["p95_lower"]) <= p95 <= _fraction(result["p95_upper"])
    assert _fraction(result["rms_lower"])**2 <= mean_square <= _fraction(result["rms_upper"])**2


def test_empty_residual_summary_is_unknown_not_zero_error(registration):
    assert registration._metric_bounds(np.empty(0), np.empty(0)) == dict(
        p95_lower=None, p95_upper=None, rms_lower=None, rms_upper=None)


@pytest.mark.parametrize("count,total", ((1, 3), (95, 100)))
def test_coverage_bounds_enclose_exact_integer_denominators(registration, count, total):
    low = registration._fraction_bound(count, total, upper=False)
    high = registration._fraction_bound(count, total, upper=True)
    assert _fraction(low) <= Fraction(count, total) <= _fraction(high)


def _fixed_query(query):
    """Mock only the exact query/diagnosis interface, never a public pose fit."""
    calls = []
    def callback(points):
        assert len(points) == len(query["distances"])
        calls.append(points.copy())
        return deepcopy(query)
    return callback, calls


def _rank_recorder(monkeypatch, registration, ratio=.1):
    calls = []
    def rank(points, normals, weights=None):
        calls.append((points.copy(), normals.copy(), None if weights is None else weights.copy()))
        return ratio, [1., .8, .6, .4, .2, ratio]
    monkeypatch.setattr(registration._v2, "_rank", rank)
    return calls


@pytest.mark.parametrize("kind", ("uncertain_face", "declared_edge"))
def test_region_uncertainty_excludes_rank_rows_not_finite_support(registration, monkeypatch, kind):
    points, normals = _box_panels()
    query = _query(len(points), lower=2.**-7, upper=2.**-6, targets=points, normals=normals)
    if kind == "uncertain_face":
        query["region_uncertain"][:9] = True
    else:
        query["closest_regions"][:9] = "EDGE"
    callback, calls = _fixed_query(query)
    ranks = _rank_recorder(monkeypatch, registration)
    result = registration._diagnose(points, points, points, callback, np.eye(3), np.zeros(3),
                                    _policy(registration, trim_fraction=1.))
    assert len(calls) == 2 and len(ranks) == 3
    assert [len(row[0]) for row in ranks] == [45, 45, 45]
    assert result["inlier_count"] == result["possible_inlier_count"] == 54
    assert result["scan_coverage_lower"] == result["scan_coverage_upper"] == 1.
    assert result["informative_scan_count"] == result["informative_fit_subset_count"] == 45
    assert result["reasons"] == []


def test_local_interval_acceptance_does_not_substitute_rounded_world_residuals(registration, monkeypatch):
    points, normals = _box_panels()
    query = _query(len(points), lower=2.**-7, upper=2.**-6,
                   targets=points+[0., 0., 2.], normals=normals)
    # Deliberately enlarged materialization error in a labeled interface mock.
    query["closest_point_error"][:] = 2.
    query["closest_point_restoration_error"][:] = 2.
    callback, _ = _fixed_query(query)
    _rank_recorder(monkeypatch, registration)
    result = registration._diagnose(points, points, points, callback, np.eye(3), np.zeros(3),
                                    _policy(registration, trim_fraction=1.))
    assert result["reasons"] == []
    assert result["inlier_surface_p95_mm"] == result["inlier_surface_p95_upper_mm"]
    assert 2.**-6 <= result["inlier_surface_p95_mm"] <= 2.**-6+1e-12
    assert result["inlier_plane_p95_mm"] > 1.
    assert result["maximum_closest_point_restoration_error_mm"] == 2.
    assert result["residual_interval_scope"] == "FIXED_DEFINITE_UNAMBIGUOUS_SUPPORT_ONLY"
    assert result["optimizer_objective"] == "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"


def test_possible_support_uses_full_denominator_and_optimistic_zero_padding(registration, monkeypatch):
    points, normals = _box_panels()
    query = _query(54, lower=2.**-6, upper=2.**-6, targets=points, normals=normals)
    for name in ("distance_lower", "triangle_distance_lower"):
        query[name][36:] = .25
    for name in ("distance_upper", "triangle_distance_upper"):
        query[name][36:] = .75
    query["distances"][36:] = .5
    query["numerical_distance_guard"][36:] = .25
    callback, _ = _fixed_query(query)
    _rank_recorder(monkeypatch, registration, ratio=0.)
    result = registration._diagnose(points, points, points, callback, np.eye(3), np.zeros(3),
                                    _policy(registration, trim_fraction=1.))
    assert result["inlier_count"] == result["fixed_residual_denominator"] == 36
    assert result["possible_inlier_count"] == result["possible_residual_denominator"] == 54
    assert _fraction(result["scan_coverage_lower"]) <= Fraction(36, 54)
    assert result["scan_coverage_upper"] == 1.
    assert result["threshold_unresolved_scan_rows"] == 18
    assert result["possible_eligible"] is True and result["possible_eligibility_reasons"] == []
    assert "INSUFFICIENT_FINITE_SURFACE_SUPPORT" in result["reasons"]
    assert "LOCAL_POSE_UNOBSERVABLE" in result["reasons"]
    padded_squared_mean = Fraction(36, 54)*Fraction(1, 64)**2
    assert _fraction(result["possible_inlier_surface_rms_lower_mm"])**2 <= padded_squared_mean
    assert _fraction(result["possible_inlier_surface_p95_lower_mm"]) <= Fraction(1, 64)
    assert result["possible_residual_scope"] == "OPTIMISTIC_ZERO_PADDED_LOWER_BOUND_NOT_OBSERVATIONS_OR_ACCEPTANCE"


def test_unobservable_pose_without_any_potential_rank_rows_is_not_possible_eligible(registration, monkeypatch):
    points, normals = _box_panels()
    callback, _ = _fixed_query(_query(54, lower=2.**-7, upper=2.**-6, targets=points, normals=normals))
    _rank_recorder(monkeypatch, registration, ratio=0.)
    result = registration._diagnose(points, points, points, callback, np.eye(3), np.zeros(3),
                                    _policy(registration, trim_fraction=1.))
    assert result["possible_eligible"] is False
    assert result["possible_eligibility_reasons"] == ["LOCAL_POSE_UNOBSERVABLE_WITHOUT_UNRESOLVED_SUPPORT"]


@pytest.mark.parametrize("coarse", (True, False))
def test_optimizer_keeps_materialized_fixed_target_surrogate_labeled(registration, coarse):
    points = _axis_pairs()
    callback, calls = _fixed_query(_query(36, targets=points+[.125, 0., 0.]))
    policy = _policy(registration, trim_fraction=1., coarse_iterations=1, refine_iterations=1)
    # Exact manual affine target fixture; no pose is injected into public inference.
    rotation, translation, count, state, last = registration._refine(
        points, callback, np.eye(3), np.zeros(3), diagonal=12., policy=policy, coarse=coarse)
    assert len(calls) == count == 1 and state == "ITERATION_LIMIT"
    assert np.allclose(rotation, np.eye(3), rtol=0, atol=2e-14)
    assert np.allclose(translation, [.125, 0., 0.], rtol=0, atol=2e-14)
    assert last["optimizer_objective"] == "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"
    if not coarse:
        assert last["fixed_set_huber_after"] <= last["fixed_set_huber_before"]


@pytest.mark.parametrize("fault", (
    "work_mismatch", "missing_certificate", "unfinished_pass", "boolean_counter", "extra_counter",
))
def test_invalid_complete_query_telemetry_cannot_bypass_consumer(registration, fault):
    query = _query(3)
    stats = query["diagnostics"]
    if fault == "work_mismatch":
        stats["work_units"] += 1
    elif fault == "missing_certificate":
        # Work remains algebraically consistent, but one witness is missing.
        stats["certificate_pairs"] -= 1
        stats["work_units"] -= 1
    elif fault == "unfinished_pass":
        stats["passes_completed"] = 1
    elif fault == "boolean_counter":
        stats["maximum_pair_batch"] = True
    else:
        stats["unbound_extra_counter"] = 0
    with pytest.raises(registration._SurfaceFailure):
        registration._validate_query_result(query, 3)


@pytest.mark.parametrize("field", ("max_query_calls", "max_query_rows"))
def test_zero_dispatch_budget_refuses_before_finite_kernel(registration, monkeypatch, field):
    surface, index = _surface_index(registration)
    budget = replace(registration.SurfaceRegistrationBudgetsV3(), **{field: 0})
    query = registration._Queries(index, budget, surface.TriangleSurfaceBudgets())
    def forbidden(*args, **kwargs):
        raise AssertionError("The finite kernel must not run beyond dispatch admission.")
    monkeypatch.setattr(surface, "query_closest_surface", forbidden)
    with pytest.raises(registration._SurfaceFailure, match="^REGISTRATION_QUERY_BUDGET_EXCEEDED$"):
        query(np.array([[1., 1., .125]]))
    assert all(value == 0 for value in query.statistics.values())


def _candidate(index, *, possible_only=False, displacement=.25):
    """Independent declared selection-interface scores; NOT geometric inference.

    A possible-only row has insufficient definite support, but its full possible
    coverage and optimistic residual can meet the strict constructed policy.
    These scores are not produced by the candidate's diagnosis implementation.
    """
    lower, upper = (.75, 1.) if possible_only else (1., 1.)
    return dict(index=index, execution_status="ATTEMPTED",
        numerical_consumer_status="COMPLETE_V3_INTERVAL_SUPPORT_POLICY",
        numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION",
        optimizer_objective="MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS",
        distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
        rotation=np.eye(3).tolist(), translation=[displacement, 0., 0.],
        reasons=["INSUFFICIENT_FINITE_SURFACE_SUPPORT"] if possible_only else [],
        possible_eligible=True, possible_eligibility_reasons=[],
        scan_coverage=lower, scan_coverage_lower=lower, scan_coverage_upper=upper, cad_coverage=1.,
        inlier_surface_rms_lower_mm=2.**-7, inlier_surface_rms_upper_mm=2.**-6,
        inlier_surface_rms_mm=2.**-6, possible_inlier_surface_rms_lower_mm=0.)


def _selection(candidates):
    # Deliberate stale success checks that every new decision revokes it first.
    return dict(status="CONDITIONAL_BEST_FIT", transform_scan_to_cad=np.eye(4).tolist(),
        reasons=[], evidence=dict(candidates=candidates, candidate_search_complete=True,
            selected_candidate_index=123456))


def test_possible_only_distinct_competitor_blocks_definite_winner(registration):
    winner = _candidate(0, displacement=0.)
    possible = _candidate(1, possible_only=True)
    result = _selection([winner, possible])
    registration._select(result, _axis_pairs(), _policy(registration))
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] == ["AMBIGUOUS_POSSIBLE_GENERATED_WHOLE_PART_POSES"]
    evidence = result["evidence"]
    assert evidence["eligible_candidate_count"] == 1
    assert evidence["possible_eligible_candidate_count"] == 2
    assert evidence["possible_comparable_candidate_indices"] == [1]
    assert evidence["comparable_distinct_pose_count"] == 1
    assert "selected_candidate_index" not in evidence
    assert evidence["candidates"][1]["reasons"] == ["INSUFFICIENT_FINITE_SURFACE_SUPPORT"]


@pytest.mark.parametrize("separation", ("same_action_class", "coverage", "reverse_coverage", "residual"))
def test_noncompetitive_possible_row_does_not_invent_pose_ambiguity(registration, separation):
    winner = _candidate(0, displacement=0.)
    possible = _candidate(1, possible_only=True)
    if separation == "same_action_class":
        possible["translation"] = [.025, 0., 0.]
    elif separation == "coverage":
        # Still >= .95 possible admission, but below 1.0 - .03 comparison margin.
        possible["scan_coverage_upper"] = .96
    elif separation == "reverse_coverage":
        possible["cad_coverage"] = .95
    else:
        # Lower residual remains below the .05 admission ceiling but outside
        # the winner's upper RMS + .005 comparator margin.
        possible["possible_inlier_surface_rms_lower_mm"] = .025
        possible["inlier_surface_rms_lower_mm"] = .03125
        possible["inlier_surface_rms_upper_mm"] = .04
        possible["inlier_surface_rms_mm"] = .04
    result = _selection([winner, possible])
    registration._select(result, _axis_pairs(), _policy(registration))
    assert result["status"] == "CONDITIONAL_BEST_FIT" and result["reasons"] == []
    np.testing.assert_array_equal(result["transform_scan_to_cad"], np.eye(4))
    evidence = result["evidence"]
    assert evidence["selected_candidate_index"] == 0
    assert evidence["possible_eligible_candidate_count"] == 2
    assert evidence["possible_comparable_candidate_indices"] == []
    assert evidence["candidate_comparison"] == "POSSIBLE_COVERAGE_UPPER_AND_RESIDUAL_LOWER_VS_DEFINITE_WINNER"
    assert evidence["pose_class_status"] == "UNCHANGED_V2_ENGINEERING_ACTION_THRESHOLD_NOT_CERTIFIED_GLOBAL_UNIQUENESS"


def test_possible_eligibility_alone_never_publishes_a_pose(registration):
    result = _selection([_candidate(0, possible_only=True)])
    registration._select(result, _axis_pairs(), _policy(registration))
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["reasons"] == ["NO_DEFINITELY_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
    assert result["evidence"]["eligible_candidate_count"] == 0
    assert result["evidence"]["possible_eligible_candidate_count"] == 1
    assert "selected_candidate_index" not in result["evidence"]


@pytest.mark.parametrize("fault", (
    "unfinished_search", "unattempted", "missing_consumer_stamp", "numerical_failure",
    "improper_pose", "missing_possible_residual",
))
def test_incomplete_or_invalid_late_candidate_clears_prior_selection(registration, fault):
    rows = [_candidate(0, displacement=0.), _candidate(1, possible_only=True)]
    result = _selection(rows)
    if fault == "unfinished_search":
        result["evidence"]["candidate_search_complete"] = False
    elif fault == "unattempted":
        rows[1]["execution_status"] = "NOT_ATTEMPTED"
    elif fault == "missing_consumer_stamp":
        del rows[1]["numerical_consumer_status"]
    elif fault == "numerical_failure":
        rows[1]["reasons"] = ["CANDIDATE_NUMERICAL_FAILURE"]
    elif fault == "improper_pose":
        rows[1]["rotation"][2][2] = -1.
    else:
        rows[1]["possible_inlier_surface_rms_lower_mm"] = None
    retained = deepcopy(rows)
    with pytest.raises(registration._SurfaceFailure):
        registration._select(result, _axis_pairs(), _policy(registration))
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert "selected_candidate_index" not in result["evidence"]
    assert result["evidence"]["candidates"] == retained


@pytest.mark.parametrize("ordering", ("definite_coverage_first", "residual_upper_not_lower"))
def test_definite_class_representative_ordering_uses_conservative_scores(registration, ordering):
    first, second = _candidate(0, displacement=0.), _candidate(1, displacement=0.)
    # Same action class isolates representative selection from pose ambiguity.
    # Arbitrary human labels are not permitted ranking inputs.
    first["external_label"], second["external_label"] = "preferred", "unpreferred"
    if ordering == "definite_coverage_first":
        first.update(scan_coverage=.96, scan_coverage_lower=.96,
            inlier_surface_rms_lower_mm=0., inlier_surface_rms_upper_mm=2.**-8,
            inlier_surface_rms_mm=2.**-8)
    else:
        first.update(inlier_surface_rms_lower_mm=0., inlier_surface_rms_upper_mm=.03125,
            inlier_surface_rms_mm=.03125)
    result = _selection([first, second])
    registration._select(result, _axis_pairs(), _policy(registration))
    assert result["status"] == "CONDITIONAL_BEST_FIT"
    assert result["evidence"]["selected_candidate_index"] == 1
    assert result["evidence"]["distinct_eligible_pose_class_count"] == 1
