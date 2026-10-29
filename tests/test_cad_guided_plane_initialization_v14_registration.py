"""Actual (unmocked) V14 registration on the invented two-sheet fixture.

Exercises noisy-plane extraction, CAD patches, plane search, schedule V1,
context V4, finite prefilter on surface V4, coarse/fine refinement with the
V4 registration consumer, conditional context support and publication. No
native CAD, files, physical inputs or measurement. A supported context on this
fixture is wiring evidence with a known 2 mm axial ambiguity of the fixture's
whole-part extrema, not registration accuracy or physical evidence.
"""
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixture():
    spec = importlib.util.spec_from_file_location("_integration_fixture_v4",
        ROOT/"tests/test_cad_guided_context_v4_engine_integration_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def init():
    prefix = "backend"
    saved = {name: value for name, value in sys.modules.items() if name == prefix or name.startswith(prefix+".")}
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROOT/"backend")]
    sys.modules[prefix] = package
    try:
        yield importlib.import_module(prefix+".cad_guided_plane_initialization_v14")
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_actual_v14_registration_publishes_a_supported_context_on_the_two_sheet_fixture(fixture, init):
    arrays, nominal = fixture._arrays()
    observed, scan_triangles, cad, faces, local = arrays
    cad_points = np.unique(cad.reshape(-1, 3), axis=0)
    policy = fixture._policy(types.SimpleNamespace(init=init))
    answer, roles = init.register_measurement_contexts_v14(observed, scan_triangles, cad, faces, local,
        position_error_bounds_mm=.025, cad_initialization_points=cad_points, registration_policy=policy,
        unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        search_budgets=init._search.PlaneSearchBudgetsV6(max_bank_planes=2), two_plane_context_mode="v4")
    assert answer["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS", answer["reasons"]
    assert answer["transform_scan_to_cad"] is None
    evidence = answer["evidence"]
    assert evidence["registration_implementation"] == init.REGISTRATION_IMPLEMENTATION
    assert evidence["initialization_implementation"] == init.INITIALIZATION_IMPLEMENTATION
    assert evidence["initialization"]["two_plane_context"]["schema_version"] == "two_plane_context_v4"
    assert evidence["accepted_registration"] is False and evidence["numerical_consumer_complete"] is True
    indices = evidence["measurement_context_indices"]
    assert indices and all(evidence["candidates"][i]["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"
        for i in indices)
    for candidate in evidence["candidates"]:
        assert candidate["execution_status"] == "ATTEMPTED"
        assert candidate["numerical_consumer_status"] == "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
    supported = evidence["candidates"][indices[0]]
    rotation, translation = np.asarray(supported["rotation"]), np.asarray(supported["translation"])
    error = float(np.max(np.linalg.norm(observed@rotation.T+translation-nominal, axis=1)))
    # The fixture's CAD sheets are longer than the scan sheets, so the
    # whole-part extrema context is axially offset by construction (2 mm).
    assert error <= 2.5
    assert supported["conditional_context_support"]["reasons"] == []
    assert evidence["query_resources"]["query_calls"] > 0
    assert roles is not None and type(roles).__name__ == "PreparedContextSurfaceRolesV1"


def _synthetic_context_result(init, poses, supports):
    """Minimal evidence shape consumed by the schedule: candidates with poses and context support."""
    candidates = []
    for index, ((rotation, translation), support) in enumerate(zip(poses, supports)):
        candidates.append(dict(index=index, execution_status="ATTEMPTED", rotation=rotation.tolist(),
            translation=list(translation), conditional_context_support=dict(support)))
    indices = [i for i, row in enumerate(candidates) if row["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"]
    return dict(status="CONDITIONAL_MEASUREMENT_CONTEXTS", evidence=dict(candidates=candidates,
        measurement_context_indices=indices))


def _support(status, p95, coverage):
    return dict(status=status, finite_distance_metrics=dict(p95_upper=p95, p95_lower=0., rms_lower=0., rms_upper=p95),
        geometric_scan_coverage_lower=coverage, reasons=[] if status == "SUPPORTED_CONTEXT" else ["X"])


def _policy_for_schedule(init):
    return init._v1.WholePartRegistrationPolicyV1(coarse_points=50, refine_points=50, coarse_iterations=1,
        refine_iterations=1, max_correspondence_distance_mm=2., huber_delta_mm=.25, max_inlier_plane_p95_mm=1.,
        min_scan_coverage=.75, min_cad_coverage=.25, pose_class_distance_mm=.1)


def test_measurement_context_schedule_dedups_pose_classes_and_caps_by_ranked_evidence(init):
    refine = np.array([(float(i % 7), float(i // 7), float(i % 3)) for i in range(50)])
    eye = np.eye(3)
    poses = [(eye, (0., 0., 0.)), (eye, (0., 0., .05)), (eye, (5., 0., 0.)), (eye, (0., 5., 0.)),
             (eye, (0., 0., 5.)), (eye, (5., 5., 0.))]
    supports = [_support("SUPPORTED_CONTEXT", .9, .9), _support("SUPPORTED_CONTEXT", .3, .95),
                _support("REJECTED_CONTEXT", 1.4, .9), _support("SUPPORTED_CONTEXT", .3, .95),
                _support("SUPPORTED_CONTEXT", .6, .99), _support("SUPPORTED_CONTEXT", .3, .90)]
    result = _synthetic_context_result(init, poses, supports)
    init._schedule_measurement_contexts(result, refine, _policy_for_schedule(init), 2)
    evidence = result["evidence"]
    schedule = evidence["measurement_context_schedule"]
    # Ranking: p95 .3 (indices 1, 3, 5: coverage .95, .95, .90 -> 1, 3, 5), then .6 (4), then .9 (0).
    # Candidate 1 (translation z .05) and candidate 0 (z 0) are one pose class at .1 mm; 1 ranks
    # first and is kept, so 0 becomes its duplicate. Cap 2 keeps 1 and 3; 5 and 4 are omitted.
    assert evidence["measurement_context_indices"] == [1, 3]
    statuses = [row["conditional_context_support"]["status"] for row in evidence["candidates"]]
    assert statuses == ["SUPPORTED_CONTEXT_DUPLICATE_POSE_CLASS", "SUPPORTED_CONTEXT", "REJECTED_CONTEXT",
        "SUPPORTED_CONTEXT", "SUPPORTED_CONTEXT_OMITTED_BY_MEASUREMENT_BUDGET",
        "SUPPORTED_CONTEXT_OMITTED_BY_MEASUREMENT_BUDGET"]
    assert evidence["candidates"][0]["conditional_context_support"]["duplicate_of_context_index"] == 1
    assert [row["disposition"] for row in schedule["dispositions"]] == ["SCHEDULED", "SCHEDULED",
        "OMITTED_BY_MEASUREMENT_BUDGET", "OMITTED_BY_MEASUREMENT_BUDGET", "DUPLICATE_POSE_CLASS"]
    assert schedule["supported_context_count"] == 5 and schedule["scheduled_context_count"] == 2
    assert schedule["max_measurement_contexts"] == 2 and schedule["pose_class_distance_mm"] == .1
    # The retained measurement consumer recomputes the list from the SUPPORTED_CONTEXT status.
    recomputed = [i for i, row in enumerate(evidence["candidates"])
        if row["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"]
    assert recomputed == evidence["measurement_context_indices"]
    # Without a cap only the duplicate collapses; the ledger stays complete.
    result = _synthetic_context_result(init, poses, supports)
    init._schedule_measurement_contexts(result, refine, _policy_for_schedule(init), None)
    assert result["evidence"]["measurement_context_indices"] == [1, 3, 4, 5]
    for bad in (0, 49, 2.0, True):
        result = _synthetic_context_result(init, poses, supports)
        with pytest.raises(ValueError, match="INVALID_MEASUREMENT_CONTEXT_CAP"):
            init._schedule_measurement_contexts(result, refine, _policy_for_schedule(init), bad)


def test_measurement_context_schedule_never_loses_the_only_supported_context(init):
    refine = np.array([(float(i % 7), float(i // 7), 0.) for i in range(50)])
    result = _synthetic_context_result(init, [(np.eye(3), (0., 0., 0.))], [_support("SUPPORTED_CONTEXT", .8, .8)])
    init._schedule_measurement_contexts(result, refine, _policy_for_schedule(init), 1)
    assert result["evidence"]["measurement_context_indices"] == [0]
    assert result["evidence"]["measurement_context_schedule"]["dispositions"] == [dict(index=0, disposition="SCHEDULED")]


@pytest.mark.parametrize("gate,expected", ((.5, "REJECTED_CONTEXT"), (1., "SUPPORTED_CONTEXT")))
def test_context_support_gate_admits_p95_between_half_and_one_millimetre_only_at_one(init, gate, expected):
    """Narrow adversarial evidence for the provisional 1.0 mm admission gate.

    A mocked complete query places every row at a definite 0.75 mm distance
    (upper bound); coverage and reverse coverage pass, so the p95 gate alone
    decides. This is the V14 `_context_support` consumer, not registration.
    """
    rows = 60
    policy = init._v1.WholePartRegistrationPolicyV1(coarse_points=50, refine_points=50, coarse_iterations=1,
        refine_iterations=1, max_correspondence_distance_mm=2., huber_delta_mm=.25,
        max_inlier_plane_p95_mm=gate, min_scan_coverage=.75, min_cad_coverage=.25)
    scan = np.array([(float(i % 8), float(i // 8), 0.) for i in range(rows)])
    def query_surface(points):
        assert points.shape == (rows, 3)
        low = np.full(rows, .7); high = np.full(rows, .75)
        return dict(status="COMPLETE", reason="CONDITIONAL_LOCAL_FINITE_TRIANGLE_QUERY_COMPLETE",
            closest_points=points.copy(), distances=(low+high)/2., triangle_indices=np.zeros(rows, dtype=np.int64),
            source_face_ids=(0,)*rows, oriented_normals=np.tile([0., 0., 1.], (rows, 1)),
            closest_regions=np.full(rows, "FACE", dtype="U6"), near_tie_counts=np.ones(rows, dtype=np.int64),
            distance_lower=low, distance_upper=high, triangle_distance_lower=low.copy(), triangle_distance_upper=high.copy(),
            numerical_distance_guard=(high-low)/2., closest_point_error=np.zeros(rows),
            closest_point_restoration_error=np.zeros(rows),
            normal_ambiguous=np.zeros(rows, dtype=bool), closest_point_ambiguous=np.zeros(rows, dtype=bool),
            face_identity_ambiguous=np.zeros(rows, dtype=bool), normal_ambiguity_uncertain=np.zeros(rows, dtype=bool),
            closest_point_ambiguity_uncertain=np.zeros(rows, dtype=bool), region_uncertain=np.zeros(rows, dtype=bool),
            distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
            closest_point_error_scope="SELECTED_TRIANGLE_OPTIMIZER_NOT_GLOBAL_UNIQUE_WITNESS",
            normal_region_status="ENGINEERING_NOT_INTERVAL_CERTIFIED",
            tie_semantics="POSSIBLE_COMPETITORS_AT_FIXED_GLOBAL_UPPER_PLUS_TAU",
            initial_upper_bound="NEAREST_REPRESENTED_VERTEX_OUTWARD_ROUNDED_PRUNING_THRESHOLD_ONLY",
            interval_width_policy="WIDE_COMPETITIVE_INTERVAL_MARKS_ROW_UNCERTAIN_NOT_WHOLE_QUERY_REFUSAL",
            diagnostics=dict(node_visits=2*rows, triangle_box_pairs=2*rows, triangle_pairs=2*rows,
                certificate_pairs=8*rows, normal_pairs=0, work_units=44*rows, passes_completed=2,
                maximum_pair_batch=rows, maximum_certificate_batch=rows, tie_batches_completed=1,
                maximum_tie_storage_rows=rows, interval_width_uncertain_pairs=0, certificate_failures=0,
                proof_pruned_triangle_pairs=0, vertex_lookup_rows=rows))
    support = init._context_support(scan, query_surface, np.eye(3), np.zeros(3), policy, .5)
    assert support["status"] == expected
    assert support["finite_distance_metrics"]["p95_upper"] >= .75
    assert support["geometric_scan_coverage_lower"] == 1.
    if expected == "REJECTED_CONTEXT":
        assert support["reasons"] == ["CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE"]
    else:
        assert support["reasons"] == []
    # Coverage gates are untouched by the gate value: reverse coverage below .25 always rejects.
    rejected = init._context_support(scan, query_surface, np.eye(3), np.zeros(3), policy, .2)
    assert "INSUFFICIENT_CONDITIONAL_CONTEXT_PROXIMITY" in rejected["reasons"]


def test_actual_v14_registration_honours_the_measurement_context_cap(fixture, init):
    arrays, nominal = fixture._arrays()
    observed, scan_triangles, cad, faces, local = arrays
    cad_points = np.unique(cad.reshape(-1, 3), axis=0)
    policy = fixture._policy(types.SimpleNamespace(init=init))
    answer, _ = init.register_measurement_contexts_v14(observed, scan_triangles, cad, faces, local,
        position_error_bounds_mm=.025, cad_initialization_points=cad_points, registration_policy=policy,
        unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        search_budgets=init._search.PlaneSearchBudgetsV6(max_bank_planes=2), two_plane_context_mode="v4",
        max_measurement_contexts=1)
    assert answer["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS", answer["reasons"]
    evidence = answer["evidence"]
    assert len(evidence["measurement_context_indices"]) == 1
    schedule = evidence["measurement_context_schedule"]
    assert schedule["max_measurement_contexts"] == 1 and schedule["scheduled_context_count"] == 1
    assert evidence["measurement_context_schedule_policy"] == init.MEASUREMENT_CONTEXT_SCHEDULE
    candidates = evidence["candidates"]
    assert all(row["conditional_context_support"]["status"] in ("SUPPORTED_CONTEXT", "REJECTED_CONTEXT",
        init.DUPLICATE_CONTEXT, init.OMITTED_CONTEXT) for row in candidates)
    # Every supported-before-schedule candidate has exactly one disposition; the
    # two-sheet fixture yields a single SUPPORTED context (see the first test),
    # so the cap/dedup branches themselves are pinned by the synthetic test above.
    supported_before = sum(row["conditional_context_support"].get("supported_before_schedule") is True for row in candidates)
    assert schedule["supported_context_count"] == supported_before == len(schedule["dispositions"]) >= 1
    assert schedule["scheduled_context_count"] == min(1, schedule["supported_context_count"])
