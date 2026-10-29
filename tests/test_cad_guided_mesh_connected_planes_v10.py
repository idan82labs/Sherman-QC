"""Prospective V10 regression of the unchanged four invented V9/V8 native cases.

Only V10 dispatch/labels, rejection-only complete-face box accounting and fresh
supervision change. A completed box exclusion is not a completed finite query.
Original CAD/analytic scan, noise, epsilon=.005, policies and numeric
oracles stay fixed; no nominal abstention passes. Native work needs root's
fresh300s wall/240s worker CPU/4GiB tree RSS/1MiB stdout gate. Four complete
<=16MiB artifacts, <=64MiB total, precede matrix success. Outer limits are exactly
the failed V9 gate's limits,
not widened. No terminal predecessor or internal work allowance is changed.
No native work, physical data, or old native tests run at ordinary collection.
The separate physical-config diagnostic is deliberately not part of this file.
"""

import ast
import builtins
import copy
from dataclasses import asdict
from functools import lru_cache
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
V8_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_planes_v8.py")
V9_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_planes_v9.py")
V4_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_planes_v4.py")
CONNECTED_CASES = (
    ("v1_nominal_oblique", "v1", False, 1),
    ("v1_missing_arc_oblique", "v1", True, 0),
    ("r0_common_wrapper_nominal_oblique", "r0", False, 1),
    ("v1_bounded_3d_noise_nominal_oblique", "v1", False, 1),
)
CRITERIA = dict(radius_error_mm=2e-5, absolute_sweep_error_deg=2e-5,
                generator_endpoint_error_mm=.02, pose_action_error_mm=.20)
NOISY_CRITERIA = dict(radius_error_mm=.01, absolute_sweep_error_deg=.1,
                      generator_endpoint_error_mm=.05, pose_action_error_mm=.20)
POSITION_ERROR_BOUND_MM = .005
NOISE_MAX_EUCLIDEAN_MM = .001
NOISE_ROUNDING_RESERVE_FACTOR = .999
RESOURCE_LIMITS = dict(wall_seconds=300, cpu_seconds=240,
    process_tree_rss_bytes=4*1024**3, stdout_bytes=1024**2)
ARTIFACT_LIMITS = dict(per_case_json_bytes=16*1024**2, total_case_json_bytes=64*1024**2)
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_CONNECTED_PLANES_V10"
ARTIFACT_DIRECTORY_ENV = "SHERMAN_MESH_CONNECTED_PLANES_V10_ARTIFACT_DIRECTORY"
NO_START_REASON = "NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"
CERTIFICATE_CAPS = dict(initialization=1_024_000_000, surface_query=128_000_000,
                        registration=8_000_000_000)
SUPPORT_POLICY = "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
DISTANCE_MODEL = "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
SURROGATE = "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"
CONSUMER_COMPLETE = "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
BOX_POLICY = "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
BOX_SCOPE = "OWNED_COMPLETE_V3_FACE_INDEX_ROOT_ALL_TRIANGLE_VERTICES"
BOX_REASON = "ASSIGNED_WHOLE_FACE_BOX_COVERAGE_UPPER_BELOW_MINIMUM"
POSSIBLE_GEOMETRIC_REFUSALS = frozenset({
    "INSUFFICIENT_POSSIBLE_FINITE_SURFACE_SUPPORT",
    "POSSIBLE_RESIDUAL_LOWER_BOUND_OUTSIDE_GATE",
    "LOCAL_POSE_UNOBSERVABLE_WITHOUT_UNRESOLVED_SUPPORT",
})


@lru_cache(maxsize=None)
def _load(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("_v10_"+path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v8():
    # Imported lazily as a private helper module; none of its tests are re-exported.
    return _load(V8_SOURCE)


def _v4():
    return _v8()._v4()

def _original():
    return _v8()._original()

def _policies(branch):
    from backend.cad_guided_mesh_offline_v10 import MeshWorkflowPolicyV10
    from backend.cad_guided_plane_initialization_v10 import PlaneInitializationBudgetsV10
    from backend.cad_guided_triangle_surface_v3 import TriangleSurfaceBudgets
    from backend.cad_guided_global_registration_v3 import SurfaceRegistrationBudgetsV3
    policies = _v8()._policies(branch)
    return {**policies, "workflow_policy": MeshWorkflowPolicyV10(**asdict(policies["workflow_policy"])),
        "initialization_budgets": PlaneInitializationBudgetsV10(
            max_certificate_pairs=CERTIFICATE_CAPS["initialization"]),
        "surface_budgets": TriangleSurfaceBudgets(
            max_certificate_pairs=CERTIFICATE_CAPS["surface_query"]),
        "registration_budgets": SurfaceRegistrationBudgetsV3(
            max_certificate_pairs=CERTIFICATE_CAPS["registration"])}

def _noisy_rows(raw):
    return _v8()._noisy_rows(raw)

def _adapted_noisy_nominal_assertions():
    return _v8()._adapted_noisy_nominal_assertions()

def _adapted_missing_arc_assertions():
    original = _v4()._missing_arc_assertions
    return types.FunctionType(original.__code__, {**original.__globals__, "_execution_complete": _execution_complete})


def _support_disposition_complete(support):
    assert support["status"] in ("PASSED", "REJECTED") and support["support_count"] > 0
    if support["box_exclusion_decision"] == "BOUND_EXCLUDED":
        assert support["status"] == "REJECTED" and support["reasons"] == [BOX_REASON]
        assert support["query_rows"] == support["attempted_query_rows"] == 0
        assert support["finite_query_disposition"] == "NOT_REQUIRED_BOUND_EXCLUDED"
        assert support["coverage"] is None
    else:
        assert support["box_exclusion_decision"] == "NEEDS_SURFACE_QUERY"
        assert support["query_rows"] == support["attempted_query_rows"] == support["support_count"]
        assert support["finite_query_disposition"] == "COMPLETE"
        assert support["reasons"] == ([] if support["status"] == "PASSED"
            else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"])


def _box_numerical_completion(initialization):
    """Full report contract; geometry/immutable-index binding also has focused tests.

    Report digests retain the bound face identity, not a new proof that a caller
    supplied the correct physical CAD. Only complete strict lower-bound exclusion
    replaces a finite query; uncertainty or an inconclusive box never does.
    """
    assert initialization["box_screening_complete"] is True
    assert initialization["box_screening_policy"] == BOX_POLICY
    chunk = min(initialization["budgets"]["chunk_rows"], initialization["surface_budgets"]["max_query_rows"])
    assert type(chunk) is int and chunk > 0
    minimum = initialization["registration_policy"]["min_scan_coverage"]
    assert type(minimum) in (int, float) and math.isfinite(minimum) and 0 <= minimum <= 1
    face_bindings = {}
    for face in initialization["face_index_evidence"]:
        identity, binding = face["cad_plane_id"], face["box_binding"]
        assert identity not in face_bindings
        assert face["result"]["status"] == "COMPLETE"
        assert type(face["triangle_count"]) is int and face["triangle_count"] > 0
        assert binding["scope"] == BOX_SCOPE and binding["finite_surface_support_admitted"] is False
        for key in ("index_triangles_sha256", "lower_sha256", "upper_sha256"):
            assert type(binding[key]) is str and len(binding[key]) == 64
            assert all(char in "0123456789abcdef" for char in binding[key])
        lower, upper = np.asarray(binding["lower_mm"], dtype="<f8"), np.asarray(binding["upper_mm"], dtype="<f8")
        assert lower.shape == upper.shape == (3,)
        assert np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower <= upper)
        assert binding["lower_sha256"] == hashlib.sha256(lower.tobytes()).hexdigest()
        assert binding["upper_sha256"] == hashlib.sha256(upper.tobytes()).hexdigest()
        face_bindings[identity] = face["triangle_rows_sha256"]
    planned_rows = planned_calls = query_rows = query_calls = 0
    excluded_rows = excluded_calls = support_count = excluded_count = 0
    names = ("definite_distance_inlier_rows", "definite_distance_outside_rows",
        "threshold_unresolved_rows", "uncertain_region_rows", "finite_unambiguous_inlier_count",
        "ambiguous_rows", "boundary_rows")
    for row in initialization["hypotheses"]:
        for support in row["assigned_support"]:
            count = support["support_count"]
            assert type(count) is int and count > 0
            support_count += 1
            calls = (count+chunk-1)//chunk
            planned_rows += count
            planned_calls += calls
            assert support["cad_plane_id"] in face_bindings
            assert support["cad_triangle_rows_sha256"] == face_bindings[support["cad_plane_id"]]
            assert support["box_screening_status"] == "COMPLETE"
            for key in ("attempted_box_screen_rows", "completed_box_screen_rows"):
                assert type(support[key]) is int and support[key] == count
            outside, possible = support["definitely_box_outside_count"], support["possible_box_inlier_count_upper"]
            assert type(outside) is int and 0 <= outside <= count
            assert type(possible) is int and possible == count-outside
            upper = support["possible_box_coverage_upper"]
            assert type(upper) is float and math.isfinite(upper) and 0 <= upper <= 1
            assert Fraction.from_float(upper) >= Fraction(possible, count)
            assert (support["box_exclusion_decision"] == "BOUND_EXCLUDED") == (upper < minimum)
            _support_disposition_complete(support)
            assert all(type(support[name]) is int and 0 <= support[name] <= count for name in names)
            for name in ("maximum_distance_interval_width_mm", "maximum_optimizer_error_mm",
                         "maximum_restoration_error_mm"):
                assert type(support[name]) in (int, float) and math.isfinite(support[name]) and support[name] >= 0
            if support["box_exclusion_decision"] == "BOUND_EXCLUDED":
                assert all(support[name] == 0 for name in names)
                excluded_count += 1
                excluded_rows += count
                excluded_calls += calls
            else:
                assert sum(support[name] for name in names[:3]) == count
                assert support["finite_unambiguous_inlier_count"] <= support["definite_distance_inlier_rows"]
                assert type(support["coverage"]) in (int, float) and math.isfinite(support["coverage"])
                assert 0 <= support["coverage"] <= support["finite_unambiguous_inlier_count"]/count
                assert (support["status"] == "PASSED") == (support["coverage"] >= minimum)
                query_rows += count
                query_calls += calls
    stats = initialization["statistics"]
    expected = dict(planned_query_rows=planned_rows, planned_query_calls=planned_calls,
        planned_box_screen_rows=planned_rows, planned_box_screen_calls=planned_calls,
        box_screen_rows=planned_rows, box_screen_calls=planned_calls,
        completed_box_screen_rows=planned_rows, completed_box_screen_calls=planned_calls,
        query_rows=query_rows, query_calls=query_calls,
        completed_query_rows=query_rows, completed_query_calls=query_calls,
        assigned_support_count=support_count, completed_box_support_count=support_count,
        bound_excluded_support_count=excluded_count, bound_excluded_support_rows=excluded_rows,
        bound_excluded_query_calls=excluded_calls,
        completed_finite_query_support_count=support_count-excluded_count)
    for key, value in expected.items():
        assert type(stats[key]) is int and stats[key] == value
    assert query_rows+excluded_rows == planned_rows and query_calls+excluded_calls == planned_calls
    assert type(stats["work_units"]) is int and stats["work_units"] >= planned_rows


def _numerical_completion(registration, *, allow_prefilter_abstention):
    """V9 interval checks plus V10 box/finite partition; geometric checks follow."""
    evidence, initialization = registration["evidence"], registration["evidence"]["initialization"]
    assert registration["schema_version"] == "noisy_plane_initialized_finite_registration_v10"
    assert evidence["schema"] == "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V10"
    assert initialization["schema_version"] == "noisy_plane_initialization_prefilter_v10"
    assert initialization["numerical_consumer_complete"] is True
    assert initialization["numerical_support_policy"] == evidence["numerical_support_policy"] == SUPPORT_POLICY
    assert evidence["distance_model"] == DISTANCE_MODEL and evidence["optimizer_objective"] == SURROGATE
    _box_numerical_completion(initialization)
    candidates = evidence["candidates"]
    for row in candidates:
        assert row["numerical_consumer_status"] == CONSUMER_COMPLETE
        assert row["numerical_support_policy"] == SUPPORT_POLICY
        assert row["distance_model"] == DISTANCE_MODEL and row["optimizer_objective"] == SURROGATE
        assert type(row["possible_eligible"]) is bool
        assert type(row["possible_eligibility_reasons"]) is list
        assert all(type(reason) is str for reason in row["possible_eligibility_reasons"])
        assert row["possible_eligible"] == (not row["possible_eligibility_reasons"])
        if not row["reasons"]:
            assert row["possible_eligible"] is True
    if not initialization["start_groups"]:
        # V10 records no completed refinement when the fully completed prefilter
        # produces no starts. This is not permission to accept incomplete queries.
        assert allow_prefilter_abstention and candidates == []
        assert evidence["numerical_consumer_complete"] is False
        assert evidence["candidate_search_complete"] is False
        assert evidence["candidate_evaluation_complete"] is False
        assert "selected_candidate_index" not in evidence
    else:
        assert evidence["numerical_consumer_complete"] is True
    if registration["status"] == "CONDITIONAL_BEST_FIT":
        selected = evidence["selected_candidate_index"]
        assert type(selected) is int and 0 <= selected < len(candidates)
        candidate = candidates[selected]
        assert candidate["index"] == selected and candidate["execution_status"] == "ATTEMPTED"
        assert candidate["reasons"] == [] and candidate["possible_eligible"] is True
        rotation, translation = np.asarray(candidate["rotation"]), np.asarray(candidate["translation"])
        assert rotation.shape == (3, 3) and translation.shape == (3,)
        assert rotation.dtype.kind in "fiu" and translation.dtype.kind in "fiu"
        assert np.isfinite(rotation).all() and np.isfinite(translation).all()
        assert np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-8)
        assert abs(float(np.linalg.det(rotation))-1.) <= 1e-8
        expected = np.eye(4)
        expected[:3, :3], expected[:3, 3] = rotation, translation
        assert np.array_equal(np.asarray(registration["transform_scan_to_cad"]), expected)
    else:
        assert "selected_candidate_index" not in evidence


def _execution_complete(registration, *, allow_prefilter_abstention=False):
    _numerical_completion(registration, allow_prefilter_abstention=allow_prefilter_abstention)
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    assert initialization["status"] == "COMPLETE" and initialization["search_complete"] is True
    assert initialization["selected_policy_evaluation_complete"] is True
    assert initialization["search_exhaustive"] is False
    assert initialization["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    search = initialization["hypothesis_evidence"]
    assert search["status"] == "COMPLETE" and search["selected_policy_evaluation_complete"] is True
    assert search["search_exhaustive"] is False and search["accepted_registration"] is False
    ss = search["statistics"]
    assert ss["signed_assignments_attempted"] == ss["signed_assignments_evaluated"] == ss["signed_assignments_planned"]
    assert ss["scan_triplets_scheduled"] == ss["scan_triplets_evaluated"]
    hypotheses, stats = initialization["hypotheses"], initialization["statistics"]
    assert stats["hypothesis_count"] == len(hypotheses) == ss["signed_assignments_evaluated"]
    assert [row["hypothesis_index"] for row in hypotheses] == list(range(len(hypotheses)))
    selected = initialization["selected_unsigned_assignment_ids"]
    eligible = sorted({row["unsigned_assignment_id"] for row in hypotheses if row["status"] == "PROPOSED"})
    assert selected == eligible and len(eligible) <= 1024
    assert initialization["prefilter_schedule"] == "ALL_GENERATED_PROPOSED_SIGNS_HARD_UNSIGNED_BOUND_NO_RANK_OMISSION"
    assert stats["eligible_unsigned_assignment_count"] == stats["scheduled_unsigned_assignment_count"] == len(eligible)
    assert stats["omitted_unsigned_assignment_count"] == stats["omitted_proposed_hypothesis_count"] == 0
    scheduled = [row for row in hypotheses if row["prefilter_scheduled"]]
    omitted = [row for row in hypotheses if row["status"] == "PROPOSED" and not row["prefilter_scheduled"]]
    refused = [row for row in hypotheses if row["status"] == "REJECTED"]
    assert stats["scheduled_hypothesis_count"] == stats["hypotheses_evaluated"] == len(scheduled)
    assert stats["omitted_proposed_hypothesis_count"] == len(omitted)
    assert stats["search_rejected_hypothesis_count"] == len(refused)
    assert stats["proposed_hypothesis_count"] == len(scheduled)+len(omitted)
    assert omitted == [] and stats["proposed_hypothesis_count"] == len(scheduled)
    assert len(hypotheses) == len(scheduled)+len(omitted)+len(refused)
    assert stats["passing_hypothesis_count"]+stats["rejected_hypothesis_count"] == len(scheduled)
    for row in hypotheses:
        assert row["prefilter_scheduled"] == (row["status"] == "PROPOSED" and row["unsigned_assignment_id"] in selected)
        if row["prefilter_scheduled"]:
            assert row["prefilter_status"] in ("PASSED", "REJECTED") and len(row["assigned_support"]) == 3
            assert row["prefilter_reasons"] == ([] if row["prefilter_status"] == "PASSED"
                else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
            for support in row["assigned_support"]:
                _support_disposition_complete(support)
        else:
            assert row["assigned_support"] == []
            assert row["prefilter_status"] == ("SEARCH_NOMINAL_WITNESS_REFUSAL" if row["status"] == "REJECTED"
                else "NOT_SCHEDULED_BY_DECLARED_UNSIGNED_QUOTA")
    groups, all_groups = initialization["start_groups"], initialization["all_passing_start_groups"]
    assert stats["exact_passing_start_group_count"] == len(all_groups)
    assert stats["surviving_start_group_count"] == len(groups)
    assert stats["omitted_passing_start_group_count"] == len(all_groups)-len(groups)
    assert len(groups) <= 48
    selected_group_ids = [i*(len(all_groups)-1)//(min(48, len(all_groups))-1)
        for i in range(min(48, len(all_groups)))] if len(all_groups) > 1 else list(range(len(all_groups)))
    assert [group["exact_group_index"] for group in groups] == selected_group_ids
    origins = []
    for index, group in enumerate(all_groups):
        assert group["exact_group_index"] == index
        assert group["refinement_disposition"] == ("SCHEDULED_FOR_V3_REFINEMENT" if index in selected_group_ids
            else "NOT_SCHEDULED_BY_DECLARED_GROUP_QUOTA")
        assert group["hypothesis_indices"] and all(type(i) is int and 0 <= i < len(hypotheses)
            and hypotheses[i]["prefilter_status"] == "PASSED"
            and hypotheses[i]["exact_start_group_index"] == index for i in group["hypothesis_indices"])
        origins.extend(group["hypothesis_indices"])
    assert sorted(origins) == [row["hypothesis_index"] for row in hypotheses if row["prefilter_status"] == "PASSED"]
    for group in groups:
        assert {key: value for key, value in group.items() if key != "index"} == all_groups[group["exact_group_index"]]
    candidates = evidence["candidates"]
    assert evidence["candidate_numerical_failure_count"] == 0
    if not groups:
        assert allow_prefilter_abstention and scheduled
        assert all(row["prefilter_status"] == "REJECTED" for row in scheduled)
        assert not all_groups and stats["passing_hypothesis_count"] == 0
        assert initialization["reasons"] == registration["reasons"] == [NO_START_REASON]
        assert registration["status"] == "UNAVAILABLE" and candidates == []
        assert evidence["candidate_count"] == evidence["attempted_candidate_count"] == 0
        return "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE"
    assert initialization["reasons"] == []
    assert evidence["candidate_search_complete"] is True and evidence["candidate_evaluation_complete"] is True
    assert evidence["candidate_count"] == evidence["attempted_candidate_count"] == len(groups) == len(candidates)
    for index, (row, group) in enumerate(zip(candidates, groups)):
        assert row["index"] == index and row["execution_status"] == "ATTEMPTED"
        for key in ("hypothesis_indices", "initialized_rotation", "initialized_translation_mm"):
            assert row[key] == group[key]
        assert all(key in row for key in ("rotation", "translation", "scan_coverage", "inlier_surface_p95_mm",
                                         "observability_ratio", "weighted_fit_observability_ratio"))
        assert all(reason in _v4()._GEOMETRIC_CANDIDATE_REFUSALS for reason in row["reasons"])
    return "ALL_SELECTED_PLANE_START_GROUPS_FULLY_EVALUATED"


def _missing_arc_assertions(result):
    disposition = _execution_complete(result["registration"], allow_prefilter_abstention=True)
    registration = result["registration"]
    evidence = registration["evidence"]
    if disposition == "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE":
        assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
        assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
        assert result["phase"] == "WHOLE_PART_REGISTRATION" and result["measurements"] is None
        assert registration["transform_scan_to_cad"] is None
        assert result["reasons"] == ["PROVISIONAL_REGISTRATION_UNAVAILABLE", NO_START_REASON]
        # Excluded uncertain/ambiguous rows alone cannot establish missing-arc
        # geometry. Every rejected proposal needs a definite outside-distance
        # witness that fails even its optimistic possible coverage. V10 permits
        # the strict, fully screened enclosing-box proof for unqueried supports.
        minimum = evidence["initialization"]["registration_policy"]["min_scan_coverage"]
        for row in evidence["initialization"]["hypotheses"]:
            if row["prefilter_scheduled"]:
                assert any(
                    support["possible_box_coverage_upper"] < minimum
                    if support["box_exclusion_decision"] == "BOUND_EXCLUDED"
                    else float(np.nextafter(
                        (support["support_count"]-support["definite_distance_outside_rows"])
                        / support["support_count"], np.inf)) < minimum
                    for support in row["assigned_support"])
        return disposition
    if registration["status"] == "UNAVAILABLE":
        assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
        assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
        assert result["phase"] == "WHOLE_PART_REGISTRATION" and result["measurements"] is None
        assert registration["transform_scan_to_cad"] is None
        assert result["reasons"] == ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]
        candidates = evidence["candidates"]
        if registration["reasons"] == ["NO_DEFINITELY_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]:
            assert candidates
            for row in candidates:
                assert row["reasons"] and row["possible_eligible"] is False
                assert row["possible_eligibility_reasons"] and all(
                    reason in POSSIBLE_GEOMETRIC_REFUSALS for reason in row["possible_eligibility_reasons"])
        else:
            assert registration["reasons"] == ["AMBIGUOUS_POSSIBLE_GENERATED_WHOLE_PART_POSES"]
            competing = evidence["possible_comparable_candidate_indices"]
            assert competing and evidence["comparable_distinct_pose_count"] >= 1
            assert evidence["eligible_candidate_count"] >= 2
            best = evidence["best_generated_candidate_index"]
            assert type(best) is int and 0 <= best < len(candidates) and not candidates[best]["reasons"]
            assert len(competing) == len(set(competing))
            assert all(type(index) is int and 0 <= index < len(candidates) and index != best
                and candidates[index]["reasons"] == [] and candidates[index]["possible_eligible"] is True
                for index in competing)
        return "COMPLETED_GEOMETRIC_REGISTRATION_ABSTENTION_NOT_OBSERVED_ABSENCE"
    # Original local window, finite-support, rigid-pose and reason allowlists.
    return _adapted_missing_arc_assertions()(result)

def _common_assertions(result, cad_blob, scan_blob, *, missing_arc):
    _v4()._legacy_common_assertions(result, cad_blob, scan_blob)
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V10"
    assert result["registration_surface"] == "FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD"
    assert result["registration_initialization"] == "PLANE_ONLY_FINITE_ASSIGNED_FACE_PREFILTER_V10"
    assert result["position_error_model"] == dict(uniform_euclidean_bound_mm=POSITION_ERROR_BOUND_MM,
        status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        scope="PLANE_INITIALIZATION_ONLY_NOT_MEASUREMENT_UNCERTAINTY")
    binding, topology = result["triangle_sample_binding"], result["scan_triangle_binding"]
    assert binding["status"] == "CONSISTENT_IMMUTABLE_TRIANGLE_SAMPLE_SNAPSHOT"
    assert binding["initialization_sample_count"] == 32768
    assert topology["status"] == "CONSISTENT_IMMUTABLE_SCAN_TRIANGLE_SNAPSHOT"
    assert topology["source_sha256"] == hashlib.sha256(scan_blob).hexdigest()
    assert topology["raw_points_sha256"] == result["input_preparation"]["source_bindings"]["raw_xyz_sha256"]
    assert topology["triangle_raw_rows_sha256"] == result["scan_topology"]["triangle_raw_rows_sha256"]
    evidence = result["registration"]["evidence"]
    assert evidence["schema"] == "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V10"
    assert evidence["initialization"]["hypothesis_evidence"]["schema_version"] == "plane_search_v6"
    assert evidence["registration_raw_rows"] == dict(selection="ALL_RAW_ROWS", count=topology["raw_vertex_count"], sha256=None)
    assert evidence["prefilter_full_scan_raw_rows"] == evidence["registration_input_rows"] == topology["raw_vertex_count"]
    initialized = evidence["initialization"]["input_digests"]
    assert initialized["scan_points_sha256"] == topology["raw_points_sha256"]
    assert initialized["scan_triangle_raw_rows_sha256"] == topology["triangle_raw_rows_sha256"]
    assert initialized["cad_triangles_sha256"] == binding["oriented_triangle_xyz_sha256"]
    epsilon_digest = hashlib.sha256(np.full(topology["raw_vertex_count"], POSITION_ERROR_BOUND_MM, dtype="<f8").tobytes()).hexdigest()
    assert initialized["position_error_bounds_mm_sha256"] == epsilon_digest
    assert binding["oriented_triangle_xyz_sha256"] == result["cad_surface_sampling"]["digests"]["oriented_triangle_xyz_sha256"]
    assert binding["initialization_points_sha256"] == result["cad_surface_sampling"]["digests"]["sample_points_sha256"]
    if result["registration"]["status"] == "CONDITIONAL_BEST_FIT":
        final = result["registration_array_binding"]
        assert final["status"] == "CONSISTENT_REGISTRATION_ARRAY_BINDING"
        assert final["registration_selection"] == "ALL_RAW_ROWS"
        assert final["position_error_bounds_mm_sha256"] == epsilon_digest
        assert final["cad_sampler_points_sha256"] == binding["initialization_points_sha256"]
    return _execution_complete(result["registration"], allow_prefilter_abstention=missing_arc)


def _initialization_action_errors(result, raw, cad_frame):
    return _v8()._initialization_action_errors(result, raw, cad_frame)

def _source_digests():
    # Fixture custody subset only: root's harness freezes the full dependency closure.
    paths = (Path(__file__).resolve(), V9_SOURCE, V8_SOURCE, V4_SOURCE, _v4().SURFACE_SOURCE,
        _v4().DENSE_SOURCE, _v4().ORIGINAL_SOURCE, ROOT/"backend/cad_guided_mesh_offline_v10.py",
        ROOT/"docs/research/2026-09-10_mesh_connected_planes_v10_protocol.md")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

def _write_artifact(directory, index, record):
    # Reuse exact exclusive/bounded writer; constants are independently compared.
    return _v8()._write_artifact(directory, index, record)

def _receipt(records, before, after, directory):
    artifacts = all(row.get("artifact", {}).get("status") == "COMPLETE_CASE_RECORD_RETAINED" for row in records)
    return dict(protocol="CAD_GUIDED_MESH_CONNECTED_PLANES_V10", cases=records, expected_case_count=4,
        matrix_passed=len(records) == 4 and before == after and artifacts and all(row["passed"] for row in records),
        criteria=CRITERIA, added_noisy_case_criteria=NOISY_CRITERIA, position_error_bound_mm=POSITION_ERROR_BOUND_MM,
        external_resource_limits=RESOURCE_LIMITS, artifact_limits=ARTIFACT_LIMITS, artifact_directory=str(directory),
        fixture_source_sha256_before=before, fixture_source_sha256_after=after,
        predecessor_receipt="mesh_connected_planes_v4-epppmtsj",
        policy_scope="FIXED_INVENTED_ERROR_BOUND_NOT_SCANNER_CALIBRATION_OR_PHYSICAL_POLICY",
        search_scope="SELECTED_POLICY_COMPLETE_NOT_EXHAUSTIVE_OR_GLOBALLY_UNIQUE",
        r0_scope="NEW_COMMON_WRAPPER_EXTRACTOR_CONTROL_NOT_FROZEN_HISTORICAL_PIPELINE",
        evidence_class="INVENTED_NATIVE_CONNECTED_NOT_PHYSICAL_GENERALIZATION")


def run_invented_mesh_workflow_planes_v10_checks(*, artifact_directory):
    from backend.cad_guided_mesh_offline_v10 import inspect_cad_guided_mesh_v10
    before = _source_digests()
    try:
        directory = _v4()._new_artifact_directory(artifact_directory)
    except Exception as error:
        records = [dict(case=name, passed=False, phase="artifact_setup", expected_predictions=count,
            error_type=type(error).__name__, error=str(error)[:1000]) for name, _, _, count in CONNECTED_CASES]
        return _receipt(records, before, _source_digests(), artifact_directory)
    original = _original()
    cad_blob, fixture_error = None, None
    try:
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
    except Exception as error:
        fixture_error = error
    records = []
    for index, (name, branch, missing_arc, expected) in enumerate(CONNECTED_CASES):
        record = dict(case=name, passed=False, phase="fixture", expected_predictions=expected,
            qualification_scope="CASE_ASSERTIONS_ONLY_MATRIX_REQUIRES_FINAL_SOURCE_AND_ARTIFACT_MANIFEST",
            production_use_allowed=False, observed_absence_claim=False)
        try:
            if fixture_error is not None:
                raise fixture_error
            raw, faces, cad_frame = original._analytic_mesh_scan(missing_arc)
            if index == 3:
                raw, delta = _noisy_rows(raw)
                assert np.max(np.linalg.norm(delta, axis=1)) <= NOISE_MAX_EUCLIDEAN_MM
                record["noise"] = dict(model="FIXED_ORIGINAL_ROW_ID_3D_MODULAR_PATTERN",
                    rounding_reserve_factor=NOISE_ROUNDING_RESERVE_FACTOR,
                    displacement_basis="REPRESENTED_NOISY_RAW_MINUS_CLEAN_RAW",
                    maximum_euclidean_mm=float(np.max(np.linalg.norm(delta, axis=1))),
                    delta_sha256=hashlib.sha256(np.asarray(delta, dtype="<f8").tobytes()).hexdigest())
            scan_blob = original._ply_bytes(raw, faces)
            record.update(scan_rows=len(raw), mesh_faces=len(faces), cad_sha256=hashlib.sha256(cad_blob).hexdigest(),
                scan_sha256=hashlib.sha256(scan_blob).hexdigest(), phase="actual_byte_to_v10_workflow")
            result = inspect_cad_guided_mesh_v10(scan_bytes=scan_blob, cad_bytes=cad_blob,
                position_error_bound_mm=POSITION_ERROR_BOUND_MM, allow_ocp=True, **_policies(branch))
            record["workflow"] = result
            record["phase"] = "post_inference_evaluation"
            record["post_inference_candidate_action_errors"] = (_v4()._surface()._candidate_action_errors(result, raw, cad_frame)
                if isinstance(result.get("registration"), dict) else [])
            record["post_inference_initialization_action_errors"] = _initialization_action_errors(result, raw, cad_frame)
            record["phase"] = "frozen_assertions"
            _common_assertions(result, cad_blob, scan_blob, missing_arc=missing_arc)
            assert result["input_preparation"]["source_bindings"]["raw_xyz_sha256"] == hashlib.sha256(np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
            assert result["scan_triangle_binding"]["triangle_raw_rows_sha256"] == hashlib.sha256(np.asarray(faces, dtype="<i8").tobytes()).hexdigest()
            assert result["input_preparation"]["coverage"]["raw_vertex_count"] == len(raw)
            assert len(result["predictions"]) == expected
            if missing_arc:
                record["sentinel_disposition"] = _missing_arc_assertions(result)
            elif index == 3:
                record["errors"] = _adapted_noisy_nominal_assertions()(result, raw, cad_frame)
            else:
                record["errors"] = original._nominal_assertions(result, raw, cad_frame)
            record.update(passed=True, phase="complete")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:1000])
        compact = _v4()._compact_case(record)
        try:
            compact["artifact"] = _write_artifact(directory, index, record)
        except Exception as error:
            compact.update(passed=False, phase="artifact_publication", error_type=type(error).__name__, error=str(error)[:1000],
                artifact=dict(relative_path=f"case_{index:02d}.json", status="FAILED_OR_PARTIAL_NOT_QUALIFIED"))
        records.append(compact)
    after = _source_digests()
    for record in records:
        if before != after:
            record.update(passed=False, source_consistency_failure="FIXTURE_OR_PROTOCOL_SOURCE_CHANGED")
        if sum(row.get("artifact", {}).get("bytes", 0) for row in records) > ARTIFACT_LIMITS["total_case_json_bytes"]:
            record.update(passed=False, artifact_budget_failure="TOTAL_CASE_ARTIFACT_BUDGET_EXCEEDED")
    return _receipt(records, before, after, directory)


@pytest.mark.parametrize("branch", ("v1", "r0"))
def test_source_policies_and_oracles_preserve_frozen_predecessor(monkeypatch, branch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    old, original = _v8(), _original()
    before = copy.deepcopy((old.CONNECTED_CASES, old.CRITERIA, old.NOISY_CRITERIA, original.CRITERIA))
    assert _original() is old._original() and _v4() is old._v4()
    assert CONNECTED_CASES == old.CONNECTED_CASES and CRITERIA == old.CRITERIA
    assert NOISY_CRITERIA == old.NOISY_CRITERIA
    policies, previous = _policies(branch), old._policies(branch)
    assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV10"
    assert {key: asdict(policies[key]) for key in previous} == {
        key: asdict(value) for key, value in previous.items()}
    adapted = _adapted_noisy_nominal_assertions()
    assert adapted.__code__ is original._nominal_assertions.__code__
    assert {key for key in adapted.__globals__
        if adapted.__globals__[key] is not original._nominal_assertions.__globals__[key]} == {"CRITERIA"}
    sentinel = _adapted_missing_arc_assertions()
    assert sentinel.__code__ is _v4()._missing_arc_assertions.__code__
    assert {key for key in sentinel.__globals__
        if sentinel.__globals__[key] is not _v4()._missing_arc_assertions.__globals__[key]} == {"_execution_complete"}
    assert (old.CONNECTED_CASES, old.CRITERIA, old.NOISY_CRITERIA, original.CRITERIA) == before

def test_source_error_bound_noise_and_resource_constants_are_fixed():
    assert POSITION_ERROR_BOUND_MM == .005 and NOISE_MAX_EUCLIDEAN_MM == .001
    assert NOISE_ROUNDING_RESERVE_FACTOR == .999
    raw = np.arange(300., dtype=float).reshape(100, 3)
    before = raw.copy()
    first, delta = _noisy_rows(raw)
    second, again = _noisy_rows(raw)
    assert np.array_equal(raw, before) and np.array_equal(first, second) and np.array_equal(delta, again)
    assert np.all(np.linalg.norm(delta, axis=1) <= .001) and np.all(np.std(delta, axis=0) > 0.)
    assert not np.array_equal(first, raw)
    assert np.array_equal(delta, first-raw)
    assert NOISY_CRITERIA == dict(radius_error_mm=.01, absolute_sweep_error_deg=.1,
        generator_endpoint_error_mm=.05, pose_action_error_mm=.20)
    assert RESOURCE_LIMITS == dict(wall_seconds=300, cpu_seconds=240, process_tree_rss_bytes=4*1024**3, stdout_bytes=1024**2)
    assert ARTIFACT_LIMITS == dict(per_case_json_bytes=16*1024**2, total_case_json_bytes=64*1024**2)
    assert NATIVE_OPT_IN != _v4().NATIVE_OPT_IN and ARTIFACT_DIRECTORY_ENV != _v4().ARTIFACT_DIRECTORY_ENV


def test_full_analytic_nominal_noise_uses_represented_delta_with_fixed_reserve(monkeypatch):
    # Pure analytic fixture only: no STEP writer, OCP, native shape or workflow.
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    raw, _, _ = _original()._analytic_mesh_scan(False)
    assert raw.shape == (2752, 3)
    clean = raw.copy()
    noisy, actual = _noisy_rows(raw)
    assert np.array_equal(raw, clean) and np.array_equal(actual, noisy-clean)
    assert np.max(np.linalg.norm(actual, axis=1)) <= NOISE_MAX_EUCLIDEAN_MM
    i = 1550
    endpoint_direction = np.array([((17*i+3) % 19-9)/9.,
        ((7*i+5) % 23-11)/11., ((11*i+2) % 29-14)/14.])
    np.testing.assert_array_equal(endpoint_direction, [-1., 1., -1.])
    assert np.dot(endpoint_direction, endpoint_direction) == 3.  # Full unreserved modular amplitude.
    intended = NOISE_ROUNDING_RESERVE_FACTOR*NOISE_MAX_EUCLIDEAN_MM/math.sqrt(3.)*endpoint_direction
    assert not np.array_equal(actual[i], intended)
    assert hashlib.sha256(np.asarray(actual, dtype="<f8").tobytes()).hexdigest() == hashlib.sha256(
        np.asarray(noisy-clean, dtype="<f8").tobytes()).hexdigest()


def test_source_actual_worker_has_no_runtime_pose_or_geometry_patch():
    tree = ast.parse(Path(__file__).read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "run_invented_mesh_workflow_planes_v10_checks")
    forbidden = {"setattr", "monkeypatch", "_refine", "_starts", "_oracle_pose"}
    assert not any(isinstance(node, ast.Name) and node.id in forbidden or isinstance(node, ast.Attribute)
        and node.attr in forbidden for node in ast.walk(worker))
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    call = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "inspect_cad_guided_mesh_v10")
    assert [key.arg for key in call.keywords] == ["scan_bytes", "cad_bytes", "position_error_bound_mm", "allow_ocp", None]
    for node in calls:
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("_nominal_assertions", "_candidate_action_errors"):
            assert node.lineno > call.lineno


def _mock_completed_sentinel(branch="registration"):
    """Injected V10 report assertions only; never native/geometry execution."""
    result = _v8()._mock_completed_sentinel(branch)
    registration, evidence = result["registration"], result["registration"]["evidence"]
    initialization = evidence["initialization"]
    registration["schema_version"] = "noisy_plane_initialized_finite_registration_v10"
    initialization.update(schema_version="noisy_plane_initialization_prefilter_v10",
        numerical_consumer_complete=True, numerical_support_policy=SUPPORT_POLICY,
        box_screening_complete=True, box_screening_policy=BOX_POLICY,
        registration_policy={"min_scan_coverage": .95},
        budgets={"chunk_rows": 65536}, surface_budgets={"max_query_rows": 200000})
    rows, calls, supports = 0, 0, 0
    initialization["face_index_evidence"] = []
    for row in initialization["hypotheses"]:
        for support in row["assigned_support"]:
            count = support["support_count"]
            supports += 1
            face = supports
            digest = hashlib.sha256(str(face).encode()).hexdigest()
            lower, upper = np.zeros(3), np.ones(3)
            initialization["face_index_evidence"].append(dict(cad_plane_id=face,
                triangle_rows_sha256=digest, triangle_count=1,
                result={"status": "COMPLETE"}, box_binding=dict(scope=BOX_SCOPE,
                    index_triangles_sha256=digest, lower_mm=lower.tolist(), upper_mm=upper.tolist(),
                    lower_sha256=hashlib.sha256(lower.astype("<f8").tobytes()).hexdigest(),
                    upper_sha256=hashlib.sha256(upper.astype("<f8").tobytes()).hexdigest(),
                    finite_surface_support_admitted=False)))
            support.update(cad_plane_id=face, cad_triangle_rows_sha256=digest,
                attempted_box_screen_rows=count, completed_box_screen_rows=count,
                definitely_box_outside_count=0, possible_box_inlier_count_upper=count,
                possible_box_coverage_upper=1., box_screening_status="COMPLETE",
                box_exclusion_decision="NEEDS_SURFACE_QUERY", finite_query_disposition="COMPLETE")
            passed = support["status"] == "PASSED"
            support.update(finite_unambiguous_inlier_count=count if passed else 0,
                definite_distance_inlier_rows=count if passed else 0,
                definite_distance_outside_rows=0 if passed else count,
                threshold_unresolved_rows=0, uncertain_region_rows=0,
                ambiguous_rows=0, boundary_rows=0, coverage=1. if passed else 0.,
                maximum_distance_interval_width_mm=0., maximum_optimizer_error_mm=0.,
                maximum_restoration_error_mm=0.)
            rows += count
            calls += (count+65535)//65536
    initialization["statistics"].update(planned_query_rows=rows, query_rows=rows,
        planned_query_calls=calls, query_calls=calls,
        planned_box_screen_rows=rows, planned_box_screen_calls=calls,
        box_screen_rows=rows, box_screen_calls=calls,
        completed_box_screen_rows=rows, completed_box_screen_calls=calls,
        completed_query_rows=rows, completed_query_calls=calls,
        assigned_support_count=supports, completed_box_support_count=supports,
        completed_finite_query_support_count=supports, bound_excluded_support_count=0,
        bound_excluded_support_rows=0, bound_excluded_query_calls=0, work_units=rows)
    for groups in (initialization["start_groups"], initialization["all_passing_start_groups"]):
        for group in groups:
            group["refinement_disposition"] = "SCHEDULED_FOR_V3_REFINEMENT"
    evidence.update(schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V10",
        numerical_consumer_complete=bool(evidence["candidates"]), numerical_support_policy=SUPPORT_POLICY,
        distance_model=DISTANCE_MODEL, optimizer_objective=SURROGATE)
    for row in evidence["candidates"]:
        eligible = not row["reasons"]
        row.update(numerical_consumer_status=CONSUMER_COMPLETE, numerical_support_policy=SUPPORT_POLICY,
            distance_model=DISTANCE_MODEL, optimizer_objective=SURROGATE, possible_eligible=eligible,
            possible_eligibility_reasons=[] if eligible else ["INSUFFICIENT_POSSIBLE_FINITE_SURFACE_SUPPORT"],
            rotation=np.eye(3).tolist(), translation=[0., 0., 0.])
    if branch == "registration":
        registration["reasons"] = ["NO_DEFINITELY_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
        result["reasons"] = ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]
    elif branch.startswith("local_"):
        evidence["selected_candidate_index"] = 0
    return result

@pytest.mark.parametrize("branch", ("prefilter", "registration", "local_row", "local_finite"))
def test_adapted_original_sentinel_accepts_only_completed_declared_branches(branch):
    original = _v4()._missing_arc_assertions
    original_globals = dict(original.__globals__)
    result = _mock_completed_sentinel(branch)
    assert _missing_arc_assertions(result).endswith("NOT_OBSERVED_ABSENCE")
    assert all(original.__globals__[key] is value for key, value in original_globals.items())


@pytest.mark.parametrize("fault", ("search_incomplete", "signed_incomplete", "partial_support", "candidate_numeric",
    "candidate_incomplete", "candidate_unattempted", "binding_downgrade", "local_budget", "empty_native", "unknown_reason",
    "support_resource_reason"))
def test_scheduled_completion_never_turns_failures_into_missing_arc_success(fault):
    result = _mock_completed_sentinel("local_row")
    evidence = result["registration"]["evidence"]
    if fault == "search_incomplete":
        evidence["initialization"]["hypothesis_evidence"]["selected_policy_evaluation_complete"] = False
    elif fault == "signed_incomplete":
        evidence["initialization"]["hypothesis_evidence"]["statistics"]["signed_assignments_evaluated"] -= 1
    elif fault == "partial_support":
        evidence["initialization"]["hypotheses"][0]["assigned_support"][0]["query_rows"] -= 1
    elif fault == "candidate_numeric":
        evidence["candidate_numerical_failure_count"] = 1
    elif fault == "candidate_incomplete":
        evidence["candidate_evaluation_complete"] = False
    elif fault == "candidate_unattempted":
        evidence["candidates"][0]["execution_status"] = "NOT_ATTEMPTED"
    elif fault == "binding_downgrade":
        result["registration"].update(status="UNAVAILABLE", transform_scan_to_cad=None,
            reasons=["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"])
        result.update(phase="WHOLE_PART_REGISTRATION", measurements=None, reasons=["WORKFLOW_EXCEPTION:ValueError"])
    elif fault == "local_budget":
        result["measurements"]["features"][0]["reasons"] = ["LOCAL_OR_TOTAL_SUPPORT_BUDGET_NO_THINNING"]
    elif fault == "empty_native":
        result["native_features"]["features"] = []
    elif fault == "support_resource_reason":
        evidence["initialization"]["hypotheses"][0]["assigned_support"][0]["reasons"] = ["QUERY_BUDGET_EXCEEDED"]
    else:
        result["measurements"]["features"][0]["reasons"] = ["UNKNOWN_FAILURE"]
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


def test_zero_start_sentinel_requires_nonempty_geometric_finite_work():
    result = _mock_completed_sentinel("prefilter")
    initialization = result["registration"]["evidence"]["initialization"]
    initialization.update(hypotheses=[], selected_unsigned_assignment_ids=[])
    for key in ("hypothesis_count", "hypotheses_evaluated", "proposed_hypothesis_count", "scheduled_hypothesis_count",
                "search_rejected_hypothesis_count", "rejected_hypothesis_count",
                "eligible_unsigned_assignment_count", "scheduled_unsigned_assignment_count",
                "planned_query_rows", "query_rows", "planned_query_calls", "query_calls",
                "planned_box_screen_rows", "planned_box_screen_calls", "box_screen_rows", "box_screen_calls",
                "completed_box_screen_rows", "completed_box_screen_calls",
                "completed_query_rows", "completed_query_calls", "assigned_support_count",
                "completed_box_support_count", "completed_finite_query_support_count", "work_units"):
        initialization["statistics"][key] = 0
    search = initialization["hypothesis_evidence"]["statistics"]
    search.update(signed_assignments_attempted=0, signed_assignments_evaluated=0, signed_assignments_planned=0)
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


def test_group_origins_cover_every_passed_hypothesis_even_when_counts_are_coherent():
    result = _mock_completed_sentinel("registration")
    initialization = result["registration"]["evidence"]["initialization"]
    initialization["hypotheses"][1] = dict(copy.deepcopy(initialization["hypotheses"][0]), hypothesis_index=1)
    initialization["statistics"].update(proposed_hypothesis_count=2, scheduled_hypothesis_count=2,
        hypotheses_evaluated=2, passing_hypothesis_count=2, search_rejected_hypothesis_count=6)
    for key in ("planned_query_rows", "query_rows", "planned_query_calls", "query_calls",
                "planned_box_screen_rows", "planned_box_screen_calls", "box_screen_rows", "box_screen_calls",
                "completed_box_screen_rows", "completed_box_screen_calls", "completed_query_rows",
                "completed_query_calls", "assigned_support_count", "completed_box_support_count",
                "completed_finite_query_support_count", "work_units"):
        initialization["statistics"][key] *= 2
    # The second passed sign coherently belongs to exact group0, but its origin
    # is lost from both the all-group and selected-group ledger. Cardinalities,
    # scheduled/refused totals and candidate/group equality alone still agree.
    assert initialization["hypotheses"][1]["exact_start_group_index"] == 0
    assert initialization["all_passing_start_groups"][0]["hypothesis_indices"] == [0]
    with pytest.raises(AssertionError):
        _execution_complete(result["registration"])


def test_coherently_counted_omitted_proposal_cannot_pass_all_generated_completion():
    result = _mock_completed_sentinel("registration")
    initialization = result["registration"]["evidence"]["initialization"]
    initialization["hypotheses"][1] = dict(copy.deepcopy(initialization["hypotheses"][0]),
        hypothesis_index=1, unsigned_assignment_id=1, prefilter_scheduled=False,
        prefilter_status="NOT_SCHEDULED_BY_DECLARED_UNSIGNED_QUOTA", assigned_support=[])
    initialization["hypotheses"][1].pop("exact_start_group_index", None)
    initialization["statistics"].update(proposed_hypothesis_count=2, omitted_proposed_hypothesis_count=1,
        search_rejected_hypothesis_count=6, eligible_unsigned_assignment_count=2,
        scheduled_unsigned_assignment_count=1, omitted_unsigned_assignment_count=1)
    # All old scheduled/omitted/refused and exact-group totals remain coherent.
    # The new all-generated obligation alone forbids leaving this sign unqueried.
    assert initialization["selected_unsigned_assignment_ids"] == [0]
    assert initialization["statistics"]["scheduled_hypothesis_count"] == 1
    with pytest.raises(AssertionError):
        _execution_complete(result["registration"])



def test_new_certificate_budgets_preserve_every_predecessor_work_field(monkeypatch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    from backend.cad_guided_plane_initialization_v8 import PlaneInitializationBudgetsV8
    from backend.cad_guided_triangle_surface_v2 import TriangleSurfaceBudgets
    from backend.cad_guided_global_registration_v2 import SurfaceRegistrationBudgetsV2
    policies = _policies("v1")
    old = dict(initialization_budgets=PlaneInitializationBudgetsV8(),
        surface_budgets=TriangleSurfaceBudgets(), registration_budgets=SurfaceRegistrationBudgetsV2())
    caps = dict(initialization_budgets=CERTIFICATE_CAPS["initialization"],
        surface_budgets=CERTIFICATE_CAPS["surface_query"],
        registration_budgets=CERTIFICATE_CAPS["registration"])
    for name, previous in old.items():
        current = asdict(policies[name])
        assert current.pop("max_certificate_pairs") == caps[name]
        assert current == asdict(previous)
    assert CERTIFICATE_CAPS == dict(initialization=1_024_000_000,
        surface_query=128_000_000, registration=8_000_000_000)


def test_native_worker_and_geometric_completion_preserve_v8_ast():
    def functions(text):
        return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    old = functions(V8_SOURCE.read_text())
    current = functions(Path(__file__).read_text())
    worker_text = ast.unparse(old["run_invented_mesh_workflow_planes_v8_checks"])
    for before, after in (
        ("cad_guided_mesh_offline_v8", "cad_guided_mesh_offline_v10"),
        ("inspect_cad_guided_mesh_v8", "inspect_cad_guided_mesh_v10"),
        ("mesh_workflow_planes_v8", "mesh_workflow_planes_v10"),
        ("actual_byte_to_v8_workflow", "actual_byte_to_v10_workflow"),
    ):
        worker_text = worker_text.replace(before, after)
    expected = ast.parse(worker_text).body[0]
    assert ast.dump(current[expected.name], include_attributes=False) == ast.dump(expected, include_attributes=False)
    # V9's complete geometric/group/candidate ledger stays exact except these
    # TWO explicit support-query accounting statements. The replacements require
    # either complete finite work or a complete strict box rejection witness.
    previous = functions(V9_SOURCE.read_text())
    target_rows = ast.parse('''assert all(s["status"] in ("PASSED", "REJECTED")
        and s["query_rows"] == s["attempted_query_rows"] == s["support_count"] > 0
        for s in row["assigned_support"])''').body[0]
    target_reasons = ast.parse('''assert all(s["reasons"] == ([] if s["status"] == "PASSED"
        else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"]) for s in row["assigned_support"])''').body[0]
    replacement = ast.parse('''for support in row["assigned_support"]:
        _support_disposition_complete(support)''').body[0]
    class SupportAccountingOnly(ast.NodeTransformer):
        changes = 0
        def visit_Assert(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(target_rows, include_attributes=False):
                self.changes += 1
                return copy.deepcopy(replacement)
            if ast.dump(node, include_attributes=False) == ast.dump(target_reasons, include_attributes=False):
                self.changes += 1
                return None
            return self.generic_visit(node)
    normalization = SupportAccountingOnly()
    expected_completion = normalization.visit(previous["_execution_complete"])
    assert normalization.changes == 2
    assert ast.dump(current["_execution_complete"], include_attributes=False) == ast.dump(expected_completion, include_attributes=False)
    constants = lambda text: {node.targets[0].id: node.value for node in ast.parse(text).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    prior_constants, current_constants = constants(V9_SOURCE.read_text()), constants(Path(__file__).read_text())
    for name in ("CONNECTED_CASES", "CRITERIA", "NOISY_CRITERIA", "POSITION_ERROR_BOUND_MM",
                 "NOISE_MAX_EUCLIDEAN_MM", "NOISE_ROUNDING_RESERVE_FACTOR", "RESOURCE_LIMITS",
                 "ARTIFACT_LIMITS", "CERTIFICATE_CAPS"):
        assert ast.dump(current_constants[name], include_attributes=False) == ast.dump(prior_constants[name], include_attributes=False)
    for name in ("NATIVE_OPT_IN", "ARTIFACT_DIRECTORY_ENV"):
        assert ast.literal_eval(current_constants[name]) != ast.literal_eval(prior_constants[name])
    for name, expression in (
        ("_original", "return _v8()._original()"),
        ("_noisy_rows", "return _v8()._noisy_rows(raw)"),
        ("_adapted_noisy_nominal_assertions", "return _v8()._adapted_noisy_nominal_assertions()"),
        ("_initialization_action_errors", "return _v8()._initialization_action_errors(result, raw, cad_frame)"),
        ("_write_artifact", "return _v8()._write_artifact(directory, index, record)"),
    ):
        assert len(current[name].body) == 1
        assert ast.dump(current[name].body[0], include_attributes=False) == ast.dump(
            ast.parse(expression).body[0], include_attributes=False)


@pytest.mark.parametrize("fault", (
    "initialization_schema", "registration_schema", "initialization_numerical",
    "registration_numerical", "initialization_policy", "candidate_policy",
    "candidate_numerical", "support_partition", "planned_rows", "selected_bool", "selected_pose",
))
def test_v10_incomplete_numerical_or_pose_evidence_cannot_pass_sentinel(fault):
    result = _mock_completed_sentinel("local_row")
    registration, evidence = result["registration"], result["registration"]["evidence"]
    initialization = evidence["initialization"]
    if fault == "initialization_schema":
        initialization["schema_version"] = "LEGACY"
    elif fault == "registration_schema":
        registration["schema_version"] = "LEGACY"
    elif fault == "initialization_numerical":
        initialization["numerical_consumer_complete"] = False
    elif fault == "registration_numerical":
        evidence["numerical_consumer_complete"] = False
    elif fault == "initialization_policy":
        initialization["numerical_support_policy"] = "ROUNDED_DISTANCE"
    elif fault == "candidate_policy":
        evidence["candidates"][0]["numerical_support_policy"] = "ROUNDED_DISTANCE"
    elif fault == "candidate_numerical":
        evidence["candidates"][0]["numerical_consumer_status"] = "INCOMPLETE"
    elif fault == "support_partition":
        initialization["hypotheses"][0]["assigned_support"][0]["threshold_unresolved_rows"] = 1
    elif fault == "planned_rows":
        initialization["statistics"]["planned_query_rows"] += 1
    elif fault == "selected_bool":
        evidence["selected_candidate_index"] = False
    else:
        registration["transform_scan_to_cad"][0][3] += .01
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


@pytest.mark.parametrize("branch", ("prefilter", "registration"))
def test_uncertainty_only_refusal_is_not_geometric_missing_arc(branch):
    result = _mock_completed_sentinel(branch)
    evidence = result["registration"]["evidence"]
    if branch == "prefilter":
        for row in evidence["initialization"]["hypotheses"]:
            for support in row["assigned_support"]:
                support["definite_distance_outside_rows"] = 0
                support["threshold_unresolved_rows"] = support["support_count"]
                support["uncertain_region_rows"] = support["support_count"]
    else:
        evidence["candidates"][0].update(possible_eligible=True, possible_eligibility_reasons=[])
    # The completed numerical ledger itself remains coherent.
    _execution_complete(result["registration"], allow_prefilter_abstention=True)
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


def test_nominal_abstention_still_fails_original_positive_oracle():
    result = _mock_completed_sentinel("registration")
    with pytest.raises(AssertionError):
        _original()._nominal_assertions(result, np.zeros((2, 3)), np.zeros((2, 3)))


def test_four_artifact_slots_remain_exclusive_with_the_same_byte_limits(tmp_path):
    assert ARTIFACT_LIMITS == _v8().ARTIFACT_LIMITS
    directory = _v4()._new_artifact_directory(tmp_path/"ARTIFACTS")
    for index, (case, _, _, _) in enumerate(CONNECTED_CASES):
        record = dict(case=case, passed=False, reasons=["SOURCE_HELPER_ONLY"])
        artifact = _write_artifact(directory, index, record)
        assert artifact["relative_path"] == f"case_{index:02d}.json"
        assert json.loads((directory/artifact["relative_path"]).read_bytes()) == record
        with pytest.raises(FileExistsError):
            _write_artifact(directory, index, {"passed": True})


@pytest.mark.parametrize("fault", ("case_missing", "case_failed", "artifact_missing", "source_drift"))
def test_manifest_never_promotes_a_partial_or_failed_matrix(fault):
    records = [dict(case=case, passed=True, artifact={"status": "COMPLETE_CASE_RECORD_RETAINED"})
               for case, _, _, _ in CONNECTED_CASES]
    before, after = {"fixture": "unchanged"}, {"fixture": "unchanged"}
    if fault == "case_missing":
        records.pop()
    elif fault == "case_failed":
        records[-1]["passed"] = False
    elif fault == "artifact_missing":
        records[-1].pop("artifact")
    else:
        after["fixture"] = "changed"
    assert _receipt(records, before, after, "SOURCE_ONLY_NO_FILES")["matrix_passed"] is False


def _mock_box_excluded_sentinel(*, mixed=False):
    """Report-only witness fixture, not supplied geometry or an inference pose."""
    result = _mock_completed_sentinel("prefilter")
    initialization = result["registration"]["evidence"]["initialization"]
    supports = initialization["hypotheses"][0]["assigned_support"]
    chosen = supports[:1] if mixed else supports
    for support in chosen:
        count = support["support_count"]
        support.update(definitely_box_outside_count=count, possible_box_inlier_count_upper=0,
            possible_box_coverage_upper=float(np.nextafter(0., np.inf)),
            box_exclusion_decision="BOUND_EXCLUDED", finite_query_disposition="NOT_REQUIRED_BOUND_EXCLUDED",
            query_rows=0, attempted_query_rows=0, coverage=None, reasons=[BOX_REASON],
            definite_distance_inlier_rows=0, definite_distance_outside_rows=0,
            threshold_unresolved_rows=0, uncertain_region_rows=0, finite_unambiguous_inlier_count=0,
            ambiguous_rows=0, boundary_rows=0)
    excluded_rows = sum(s["support_count"] for s in chosen)
    stats = initialization["statistics"]
    stats.update(bound_excluded_support_count=len(chosen), bound_excluded_support_rows=excluded_rows,
        bound_excluded_query_calls=len(chosen), completed_finite_query_support_count=3-len(chosen),
        query_rows=75-excluded_rows, completed_query_rows=75-excluded_rows,
        query_calls=3-len(chosen), completed_query_calls=3-len(chosen))
    return result


@pytest.mark.parametrize("mixed", (False, True))
def test_complete_strict_box_refusal_is_geometric_but_not_completed_finite_query(mixed):
    result = _mock_box_excluded_sentinel(mixed=mixed)
    assert _missing_arc_assertions(result) == "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE"
    initialization = result["registration"]["evidence"]["initialization"]
    stats = initialization["statistics"]
    assert stats["completed_box_screen_rows"] == stats["planned_query_rows"] == 75
    assert stats["completed_query_rows"] == (50 if mixed else 0)
    assert stats["bound_excluded_support_rows"] == (25 if mixed else 75)
    assert stats["completed_query_rows"]+stats["bound_excluded_support_rows"] == 75
    assert result["observed_bend_count"] is None and result["predictions"] == []


@pytest.mark.parametrize("fault", ("flag", "policy", "binding_absent", "partial_box", "unsound_fraction",
    "invented_query_completion", "inconclusive_exclusion", "excluded_finite_count", "wrong_reason", "binding_digest"))
def test_box_exclusion_cannot_mask_incomplete_unbound_or_inconclusive_work(fault):
    result = _mock_box_excluded_sentinel()
    initialization = result["registration"]["evidence"]["initialization"]
    support = initialization["hypotheses"][0]["assigned_support"][0]
    if fault == "flag":
        initialization["box_screening_complete"] = False
    elif fault == "policy":
        initialization["box_screening_policy"] = "BOX_SUPPORT_ADMISSION"
    elif fault == "binding_absent":
        initialization["face_index_evidence"] = []
    elif fault == "partial_box":
        support["completed_box_screen_rows"] -= 1
    elif fault == "unsound_fraction":
        support.update(definitely_box_outside_count=0, possible_box_inlier_count_upper=25,
            possible_box_coverage_upper=0.)
    elif fault == "invented_query_completion":
        initialization["statistics"]["completed_query_rows"] = 75
    elif fault == "inconclusive_exclusion":
        support["possible_box_coverage_upper"] = 1.
    elif fault == "excluded_finite_count":
        support["definite_distance_outside_rows"] = 1
    elif fault == "wrong_reason":
        support["reasons"] = ["QUERY_BUDGET_EXCEEDED"]
    else:
        support["cad_triangle_rows_sha256"] = "f"*64
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


def test_fixed_invented_native_planes_v10_workflow_matrix():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Root must opt in under frozen serial300/240-second4-GiB protocol")
    directory = os.environ.get(ARTIFACT_DIRECTORY_ENV)
    assert directory, "Root must supply its fresh retained-job ARTIFACTS child"
    receipt = run_invented_mesh_workflow_planes_v10_checks(artifact_directory=directory)
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) <= RESOURCE_LIMITS["stdout_bytes"]
    print(encoded, flush=True)
    assert len(receipt["cases"]) == 4 and receipt["matrix_passed"] is True
    assert receipt["fixture_source_sha256_before"] == receipt["fixture_source_sha256_after"]
    assert all(row["passed"] and row.get("artifact", {}).get("status") == "COMPLETE_CASE_RECORD_RETAINED"
               for row in receipt["cases"]), receipt["cases"]
