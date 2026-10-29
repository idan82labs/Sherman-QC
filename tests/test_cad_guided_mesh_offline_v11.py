"""Prospective V11 orchestration/publication controls with explicit pose stubs.

The frozen V10/V2 fixture supplies invented byte loaders, native extraction,
measurement and a STIPULATED transform; no CAD parsing or native call occurs.
Its pure CAD sampler is given three invented triangles so mocked triplet rows
have three distinct CAD IDs. Pair rows always have exactly TWO supports. Full
joint slot/group/origin ledgers below are declared mock data, not recovered
geometry, actual .05/free-circle success, physical evidence or observed counts.

42 frozen V10 refusal parameterizations are called without changing their
assertions; additional controls cover the V11 shared ledger and source origins.
Root owns review/freeze and the60wall/45CPU/1GiB/1MiB source gate; no subprocess,
physical input, score, native fixture or runtime smoke is permitted here.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


LEGACY = Path(__file__).with_name("test_cad_guided_mesh_offline_v10.py")
LEGACY_SHA = "0bba4eb137eede10d5a2724598285cddda6ebf0da9544bc4d01b18666240b9c2"
MOCK_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_offline_v2.py")
MOCK_SHA = "10fdfeac837d32a4f61eff53e329fbfeb0b742fcf7d9834024819e6241bb9cfc"
BOX_POLICY = "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
                       process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)


@pytest.fixture(scope="module")
def legacy():
    blob = LEGACY.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == LEGACY_SHA
    assert hashlib.sha256(MOCK_SOURCE.read_bytes()).hexdigest() == MOCK_SHA
    module = ModuleType("_v11_reviewed_v10_workflow_controls")
    module.__file__ = str(LEGACY)
    exec(compile(blob, str(LEGACY), "exec"), module.__dict__)
    return module


def _mock_origins(old, enabled):
    """Complete declared slot list, deliberately NOT actual support inference."""
    def row(index, lane, passed, unsigned=0):
        count = 3 if lane == "TRIPLET" else 2
        matches = [dict(scan_plane_id=i, cad_plane_id=face, normal_sign=1)
                   for i, face in enumerate((2, 5, 9)[:count])]
        return dict(hypothesis_index=index, unsigned_assignment_id=unsigned,
            proposal_lane=lane, status="PROPOSED" if passed else "REJECTED",
            prefilter_status="PASSED" if passed else "SEARCH_NOMINAL_WITNESS_REFUSAL",
            exact_start_group_index=0 if passed else None,
            correspondences=deepcopy(matches),
            assigned_support=[dict(match, status="PASSED") for match in matches] if passed else [],
            rotation=old.ROTATION.tolist() if passed else None,
            translation_mm=old.TRANSLATION.tolist() if passed else None,
            test_scope="STIPULATED_ORCHESTRATION_SUPPORT_NOT_GEOMETRY")
    hypotheses = [row(0, "TRIPLET", True), row(1, "TRIPLET", False)]
    if enabled:
        for i in range(64):
            hypotheses.append(row(2+i, "TWO_PLANE_CONTEXT", i < 2, unsigned=1+i//32))
    origins = [0, 2, 3] if enabled else [0]
    group = dict(exact_group_index=0, initialized_rotation=old.ROTATION.tolist(),
        initialized_translation_mm=old.TRANSLATION.tolist(), hypothesis_indices=origins,
        refinement_disposition="SCHEDULED_FOR_V3_REFINEMENT")
    return hypotheses, group


@pytest.fixture
def plumbing(legacy, monkeypatch):
    old, previous_workflow, state, _ = legacy.plumbing.__wrapped__(monkeypatch)
    package = previous_workflow.__package__
    workflow = importlib.import_module(package+".cad_guided_mesh_offline_v11")
    initialize = importlib.import_module(package+".cad_guided_plane_initialization_v11")
    previous_initialize = importlib.import_module(package+".cad_guided_plane_initialization_v10")
    old_registration_stub = previous_initialize.register_whole_part_planes_v10
    # This is a new invented mock sampler fixture, not a modification of V10.
    state["cad"] = old.surface.prepare_cad_triangle_surface_v2(
        np.array([[[0.,0.,0.],[8.,0.,0.],[0.,5.,0.]],
                  [[0.,0.,2.],[8.,0.,2.],[0.,5.,2.]],
                  [[3.,0.,0.],[3.,8.,0.],[3.,0.,5.]]]),
        source_face_indices=np.array([2,5,9]), source_triangle_indices=np.array([1,1,1]), sample_count=36)
    assert state["cad"]["status"] == "AVAILABLE"
    def register(*args, **kwargs):
        damage = state.pop("numerical_fault", None)
        try:
            result = old_registration_stub(*args, **kwargs)
        finally:
            if damage is not None:
                state["numerical_fault"] = damage
        state["events"][-1] = "register_v11"
        state["register_v11_args"] = (*args, kwargs)
        enabled = kwargs["two_plane_enabled"]
        result["schema_version"] = "noisy_plane_initialized_finite_registration_v11"
        evidence = result["evidence"]
        evidence["schema"] = "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V11"
        initialization = evidence["initialization"]
        initialization["schema_version"] = "noisy_plane_initialization_prefilter_v11"
        hypotheses, group = _mock_origins(old, enabled)
        initialization.update(hypotheses=hypotheses, start_groups=[dict(deepcopy(group), index=0)],
            all_passing_start_groups=[deepcopy(group)],
            joint_proposal_accounting=dict(triplet_records=2, triplet_unsigned_reservation=1,
                ordered_pair_matches=2 if enabled else 0, pair_slot_reservation=64 if enabled else 0,
                total_hypothesis_records=len(hypotheses), pair_lane_enabled=enabled),
            two_plane_schedule=dict(status="COMPLETE" if enabled else "DISABLED_BY_EXPLICIT_POLICY",
                selected_policy_evaluation_complete=True,
                scheduled_matches=[dict(scan_plane_ids=[0,1], cad_plane_ids=face)
                    for face in ([2,5], [5,2])] if enabled else []))
        digest_keys = (("scan_points", "scan_points"), ("scan_triangles", "scan_triangle_raw_rows"),
            ("cad_triangles", "cad_triangles"), ("cad_face_ids", "cad_face_ids"),
            ("cad_source_triangle_ids", "cad_source_triangle_ids"))
        initialization["two_plane_context"] = dict(status="COMPLETE",
            selected_policy_evaluation_complete=True, accepted_registration=False,
            finite_support_match_verified=False, statistics=dict(slot_records_completed=64),
            input_digests={target+"_sha256": initialization["input_digests"][source+"_sha256"]
                for target, source in digest_keys}) if enabled else None
        evidence["candidates"][0].update(deepcopy(group))
        if damage is not None:
            damage(result)
        return result
    monkeypatch.setattr(initialize, "register_whole_part_planes_v11", register)
    def run(**overrides):
        policy = old.policies()
        policy["workflow_policy"] = workflow.MeshWorkflowPolicyV11(**asdict(policy["workflow_policy"]))
        values = dict(scan_bytes=old.SCAN_BYTES, cad_bytes=old.CAD_BYTES,
            position_error_bound_mm=.05, allow_ocp=True, **policy)
        values.update(overrides)
        return workflow.inspect_cad_guided_mesh_v11(**values)
    return old, workflow, state, run


@pytest.mark.parametrize("enabled", (False, True))
def test_declared_point_zero_five_full_raw_inverse_and_real_two_support_shape(plumbing, enabled):
    old, _, state, run = plumbing
    answer = run(two_plane_enabled=enabled)
    assert answer["status"] == "PARTIAL" and answer["phase"] == "COMPLETE", answer["reasons"]
    assert state["read_bytes"] is old.CAD_BYTES
    assert state["extract_args"][0] is state["sample_args"][0] is state["shape"]
    points, triangles, cad, faces, local, kwargs = state["register_v11_args"]
    assert points is state["scan"]["raw_points"] and len(points) == 72
    assert triangles is state["topology"]["triangle_raw_rows"]
    assert cad is state["cad"]["triangles"] and faces is state["cad"]["source_face_indices"]
    assert local is state["cad"]["source_triangle_indices"]
    assert kwargs["position_error_bounds_mm"] == .05 and kwargs["two_plane_enabled"] is enabled
    assert kwargs["registration_raw_rows"] is state["scan"]["engine_to_raw_row_indices"]
    aligned, _, measurement = state["measure_args"]
    np.testing.assert_array_equal(aligned, points@old.ROTATION.T+old.TRANSLATION)
    assert len(aligned) == 72 and "position_error_bounds_mm" not in measurement
    assert answer["position_error_model"]["uniform_euclidean_bound_mm"] == .05
    assert measurement["scan_sha256"] == hashlib.sha256(old.SCAN_BYTES).hexdigest()
    assert answer["measurements"]["features"][0]["raw_support"]["arc_rows"] == [1, 5, 69]
    for index, key in enumerate(old.ENDPOINT_FIELDS):
        np.testing.assert_array_equal(answer["predictions"][0][key], [index+.25, 2., 3.])
    initialization = answer["registration"]["evidence"]["initialization"]
    assert initialization["joint_proposal_accounting"]["pair_lane_enabled"] is enabled
    assert len(initialization["hypotheses"]) == (66 if enabled else 2)
    if enabled:
        assert all(len(initialization["hypotheses"][i]["assigned_support"]) == 2 for i in (2,3))
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
    assert state["events"] == ["prepare", "read", "extract:v1", "sample", "register_v11", "measure"]


@pytest.mark.parametrize("fault", ("epsilon_digest", "scan_byte_digest", "incomplete", "refusal", "late_publication"))
def test_frozen_bad_binding_or_late_output_oracles(plumbing, legacy, fault):
    legacy.test_mock_bad_binding_refusal_or_late_output_cannot_publish(plumbing, fault)


@pytest.mark.parametrize("epsilon", (None, False, -.1, float("nan"), float("inf"), "0.005"))
def test_frozen_invalid_error_declaration_oracles(plumbing, legacy, epsilon):
    legacy.test_mock_invalid_position_declaration_stops_before_loader(plumbing, epsilon)


@pytest.mark.parametrize("fault", ("registration_schema", "evidence_schema", "initialization_schema",
    "initialization_complete", "registration_complete", "initialization_policy", "registration_policy",
    "distance_model", "optimizer_objective", "selected_negative", "selected_boolean",
    "selected_reasons", "selected_numerical_status"))
def test_frozen_numerical_consumer_oracles(plumbing, legacy, fault):
    legacy.test_missing_or_mismatched_numerical_consumer_contract_never_reaches_measurement(plumbing, fault)


@pytest.mark.parametrize("fault", ("candidate_boolean_index", "candidate_numpy_index",
    "published_translation", "published_rotation", "selected_translation", "selected_rotation",
    "selected_missing_rotation", "selected_nan_translation", "selected_boolean_rotation",
    "selected_wrong_rotation_shape", "selected_reflection", "selected_legacy_translation_alias",
    "selected_missing_policy", "selected_wrong_policy", "selected_missing_model", "selected_wrong_model",
    "selected_missing_objective", "selected_wrong_objective"))
def test_frozen_selected_transform_oracles(plumbing, legacy, fault):
    legacy.test_selected_pose_and_its_numerical_labels_bind_the_published_transform(plumbing, fault)


def test_all_certificate_limits_are_snapshotted_into_v11_types(plumbing):
    _, workflow, state, run = plumbing
    initialize = importlib.import_module(workflow.__package__+".cad_guided_plane_initialization_v11")
    init = initialize.PlaneInitializationBudgetsV11(max_certificate_pairs=123)
    registration = initialize._v3.SurfaceRegistrationBudgetsV3(max_certificate_pairs=456)
    surface = initialize._surface.TriangleSurfaceBudgets(max_certificate_pairs=17)
    answer = run(initialization_budgets=init, registration_budgets=registration, surface_budgets=surface)
    assert answer["status"] == "PARTIAL"
    kwargs = state["register_v11_args"][-1]
    assert type(kwargs["budgets"]) is initialize.PlaneInitializationBudgetsV11
    assert kwargs["budgets"].max_certificate_pairs == 123 and kwargs["budgets"] is not init
    assert kwargs["registration_budgets"].max_certificate_pairs == 456 and kwargs["registration_budgets"] is not registration
    assert kwargs["surface_budgets"].max_certificate_pairs == 17 and kwargs["surface_budgets"] is not surface


@pytest.mark.parametrize("enabled", (None, 0, 1, "true", np.bool_(True)))
def test_non_boolean_lane_flag_refuses_before_loader(plumbing, enabled):
    _, _, state, run = plumbing
    answer = run(two_plane_enabled=enabled)
    assert answer["reasons"] == ["INVALID_TWO_PLANE_LANE_POLICY"]
    assert state["events"] == [] and answer["predictions"] == []


JOINT_FAULTS = ("missing_joint", "boolean_count", "wrong_total", "wrong_slot_reservation",
    "disabled_report_for_enabled_call", "incomplete_schedule", "wrong_schedule_count",
    "incomplete_helper", "accepted_helper", "finite_verified_helper", "helper_slot_count", "helper_digest",
    "missing_origin", "duplicate_origin", "boolean_origin", "wrong_group_index", "subset_candidate_origins",
    "subset_all_three_origin_lists", "wrong_group_pose", "unscheduled_group", "wrong_reciprocal_group",
    "pair_third_support", "pair_support_id", "pair_support_sign", "pair_boolean_support_id",
    "pair_boolean_support_sign", "pair_support_refused",
    "wrong_origin_pose", "rejected_origin")


@pytest.mark.parametrize("fault", JOINT_FAULTS)
def test_incomplete_joint_slot_and_selected_origin_ledger_never_measures(plumbing, fault):
    _, _, state, run = plumbing
    def damage(registration):
        evidence = registration["evidence"]
        initial, candidate = evidence["initialization"], evidence["candidates"][0]
        joint, group, all_group = initial["joint_proposal_accounting"], initial["start_groups"][0], initial["all_passing_start_groups"][0]
        origin = initial["hypotheses"][2]
        if fault == "missing_joint": initial.pop("joint_proposal_accounting")
        elif fault == "boolean_count": joint["triplet_unsigned_reservation"] = True
        elif fault == "wrong_total": joint["total_hypothesis_records"] -= 1
        elif fault == "wrong_slot_reservation": joint["pair_slot_reservation"] -= 1
        elif fault == "disabled_report_for_enabled_call":
            joint.update(pair_lane_enabled=False, ordered_pair_matches=0, pair_slot_reservation=0,
                total_hypothesis_records=2)
            initial["hypotheses"] = initial["hypotheses"][:2]
            initial["two_plane_schedule"].update(status="DISABLED_BY_EXPLICIT_POLICY", scheduled_matches=[])
            for row in (candidate, group, all_group): row["hypothesis_indices"] = [0]
        elif fault == "incomplete_schedule": initial["two_plane_schedule"]["selected_policy_evaluation_complete"] = False
        elif fault == "wrong_schedule_count": initial["two_plane_schedule"]["scheduled_matches"].pop()
        elif fault == "incomplete_helper": initial["two_plane_context"]["selected_policy_evaluation_complete"] = False
        elif fault == "accepted_helper": initial["two_plane_context"]["accepted_registration"] = True
        elif fault == "finite_verified_helper": initial["two_plane_context"]["finite_support_match_verified"] = True
        elif fault == "helper_slot_count": initial["two_plane_context"]["statistics"]["slot_records_completed"] -= 1
        elif fault == "helper_digest": initial["two_plane_context"]["input_digests"]["scan_points_sha256"] = "0"*64
        elif fault == "missing_origin": candidate["hypothesis_indices"] = []
        elif fault == "duplicate_origin": candidate["hypothesis_indices"] = [0,2,2,3]
        elif fault == "boolean_origin": candidate["hypothesis_indices"][0] = False
        elif fault == "wrong_group_index": candidate["exact_group_index"] = 1
        elif fault == "subset_candidate_origins": candidate["hypothesis_indices"] = [0,2]
        elif fault == "subset_all_three_origin_lists":
            for row in (candidate, group, all_group): row["hypothesis_indices"] = [0,2]
        elif fault == "wrong_group_pose": group["initialized_translation_mm"][0] += 1.
        elif fault == "unscheduled_group": all_group["refinement_disposition"] = "NOT_SCHEDULED"
        elif fault == "wrong_reciprocal_group": origin["exact_start_group_index"] = 1
        elif fault == "pair_third_support": origin["assigned_support"].append(deepcopy(origin["assigned_support"][0]))
        elif fault == "pair_support_id": origin["assigned_support"][0]["cad_plane_id"] = 9
        elif fault == "pair_support_sign": origin["assigned_support"][0]["normal_sign"] = -1
        elif fault == "pair_boolean_support_id": origin["assigned_support"][0]["scan_plane_id"] = False
        elif fault == "pair_boolean_support_sign": origin["assigned_support"][0]["normal_sign"] = True
        elif fault == "pair_support_refused": origin["assigned_support"][0]["status"] = "REJECTED"
        elif fault == "wrong_origin_pose": origin["translation_mm"][0] += 1.
        else: origin["prefilter_status"] = "REJECTED"
    state["numerical_fault"] = damage
    answer = run()
    assert answer["status"] == "UNAVAILABLE" and answer["predictions"] == []
    assert answer["measurements"] is None and "measure" not in state["events"]
    assert answer["registration"]["status"] == "UNAVAILABLE"
    assert answer["registration"]["transform_scan_to_cad"] is None


@pytest.mark.parametrize("field,value", (("box_screening_complete", None),
    ("box_screening_complete", False), ("box_screening_complete", 1),
    ("box_screening_policy", None), ("box_screening_policy", "BOXES_ADMIT_SUPPORT")))
def test_pair_origin_does_not_bypass_original_box_consumer_contract(plumbing, field, value):
    _, _, state, run = plumbing
    def damage(registration):
        registration["evidence"]["initialization"][field] = value
    state["numerical_fault"] = damage
    answer = run()
    assert answer["status"] == "UNAVAILABLE" and answer["predictions"] == []
    assert "measure" not in state["events"]


def test_native_unavailability_after_valid_pair_registration_keeps_correct_phase(plumbing):
    _, _, state, run = plumbing
    state["native"].update(status="UNAVAILABLE", features=())
    answer = run()
    assert answer["status"] == "UNAVAILABLE" and answer["phase"] == "NATIVE_FEATURES"
    assert answer["registration"]["status"] == "CONDITIONAL_BEST_FIT"
    assert answer["reasons"] == ["NATIVE_FEATURES_UNAVAILABLE"]
    assert "measure" not in state["events"] and answer["observed_bend_count"] is None
