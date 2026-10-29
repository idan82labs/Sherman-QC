"""Draft thin-part nominal diagnostic at the exact physical-run CONFIG.

Two original V1 nominal cases (clean and unchanged bounded-3D-noise)
use actual V11 byte dispatch. No missing-arc case, thickened cap, alternate pose,
additional samples, changed error ceiling or supplied answer is permitted.
A completed diagnostic record is NOT positive geometry: nominal abstention is
failed output coverage, never a passing safety sentinel. Both ledgers persist.

The source-only copy of CONFIG is compared structurally with the named frozen
runner without importing or invoking that physical-data program. V11 types,
explicit certificate-pair caps and the default-only pair lane are additive.
The frozen V10 post-inference box checker is reused directly, without changing
its function globals. All version/lane/slot/phase checks are explicit below.
Free-circle evaluation
changes exactly the original oracle's model/role strings; geometry is unchanged.
The original2752 rows cannot exercise8192 effective refinement rows. Its.8mm
third-normal cap can lose noisy-plane admission at stipulated epsilon.05mm;
this is a coverage diagnostic, not a qualified scanner uncertainty or F1 run.

Prospective outer limit300s wall/240s worker CPU/4GiB sampled tree RSS/1MiB
stdout; two <=16MiB complete artifacts, <=32MiB total. Root must independently
freeze/review a new serial supervisor/protocol before enabling the native test.
Import/ordinary collection performs no native work or physical/Gold reads.
"""

import ast
import builtins
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SOURCE = ROOT/"scripts/run_gold11_mesh_development_v2.py"
CONNECTED_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_planes_v9.py")
PREDECESSOR_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_physical_config_constructed_v9.py")
BOX_TEST_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_planes_v10.py")
INITIALIZER_SOURCE = ROOT/"backend/cad_guided_plane_initialization_v11.py"
ORIGINAL_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_native_v1.py")
CASES = (("v1_physical_config_clean_nominal", False),
         ("v1_physical_config_bounded_3d_noise_nominal", True))
CRITERIA = dict(radius_error_mm=2e-5, absolute_sweep_error_deg=2e-5,
                generator_endpoint_error_mm=.02, pose_action_error_mm=.20)
NOISY_CRITERIA = dict(radius_error_mm=.01, absolute_sweep_error_deg=.1,
                      generator_endpoint_error_mm=.05, pose_action_error_mm=.20)
RESOURCE_LIMITS = dict(wall_seconds=300, cpu_seconds=240,
    process_tree_rss_bytes=4*1024**3, stdout_bytes=1024**2)
ARTIFACT_LIMITS = dict(per_case_json_bytes=16*1024**2, total_case_json_bytes=32*1024**2)
CERTIFICATE_CAPS = dict(initialization_budgets=1_024_000_000,
    surface_budgets=128_000_000, registration_budgets=8_000_000_000)
PAIR_DEFAULTS = dict(
    scheduler=dict(max_scan_planes=128, max_cad_planes=2048, max_scan_pair_records=8128,
        max_selected_scan_pairs=64, max_scan_selection_tests=520192, max_cad_pair_records=2096128,
        max_cad_sort_work=44018688, max_window_pair_tests=134152192,
        max_compatible_layout_bytes=134217728, max_compatible_layouts=64,
        max_working_bytes=268435456, max_report_bytes=33554432, chunk_pair_rows=65536),
    context=dict(max_scan_rows=1_000_000, max_scan_triangles=2_000_000, max_cad_triangles=500_000,
        max_scan_planes=128, max_cad_planes=2048, max_scan_pairs=64,
        max_ordered_cad_matches_per_scan_pair=4, max_signed_families=1024, max_candidate_slots=8192,
        max_scan_row_visits=64_000_000, max_cad_vertex_visits=128_000_000,
        max_index_visits=64_000_000, max_index_sort_work=256_000_000,
        max_hash_bytes=1_073_741_824, max_projection_cache_entries=6144,
        max_working_bytes=268_435_456, max_report_bytes=33_554_432, chunk_rows=65_536))
SUPPORT_POLICY = "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
DISTANCE_MODEL = "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
SURROGATE = "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"
CONSUMER_COMPLETE = "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
NO_START_REASON = "NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"
CONTEXT_SLOTS = ("WHOLE_MIN", "WHOLE_MAX", "FACE0_MIN", "FACE0_MAX", "FACE1_MIN", "FACE1_MAX",
    "WHOLE_CONTEXT_MIDPOINT", "ASSIGNED_CONTEXT_INTERSECTION_MIDPOINT")
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V11"
ARTIFACT_DIRECTORY_ENV = "SHERMAN_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V11_ARTIFACT_DIRECTORY"
COVERAGE_LIMITATIONS = dict(original_raw_rows=2752, requested_coarse_rows=2048,
    requested_refine_rows=8192, maximum_effective_refine_rows=2752,
    effective_refine_8192_exercised=False, third_normal_cap_width_mm=.8,
    physical_scanner_bound_established=False, physical_or_unseen_generalization=False,
    scope="ORIGINAL_THIN_PART_ONLY_NOT_DENSE_OR_MULTI_BEND_QUALIFICATION")

# Exact source expression from the frozen runner; never import that program.
CONFIG = dict(
    workflow=dict(max_registration_scan_rows=50000, cad_sample_count=32768,
                  cad_linear_deflection_mm=.1, cad_angular_deflection_radians=.2,
                  unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED"),
    registration=dict(coarse_points=2048, refine_points=8192, coarse_iterations=30,
                      refine_iterations=30, trim_fraction=.85, huber_delta_mm=.25,
                      coarse_distance_fraction=.15, max_correspondence_distance_mm=2.,
                      max_inlier_plane_p95_mm=.5, min_scan_coverage=.75,
                      min_cad_coverage=.25, min_observability_ratio=.005,
                      pose_class_distance_mm=.1, ambiguity_plane_rms_margin_mm=.05,
                      ambiguity_coverage_margin=.03, max_rotation_step_radians=.15,
                      max_translation_step_mm=5.),
    measurement=dict(flange_window_mm=[2.5, 12.5], search_margin_mm=3.,
                     max_fit_residual_mm=.05, plane_exclusion_mm=.075,
                     max_tangency_error_mm=.1, fit_model="free_circle"),
    support=dict(max_axial_gap_mm=4., max_angular_gap_deg=5., min_arc_sweep_deg=20.,
                 min_local_points=3),
    # JSON-stable lists are restored to the exact tuple required by the policy
    # only inside the guarded worker; freeze never imports backend classes.
    position_error_bound_mm=.05,
    noisy_policy=dict(tile_triangles=[128, 1024], halo_rings=1, min_support_rows=3,
        max_normal_angle_radians=math.pi/18., min_inplane_covariance_ratio=1e-8),
    noisy_budgets=dict(max_raw_rows=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_core_records=65_536, max_core_triangle_rows=4096,
        max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
        max_full_support_row_tests=64_000_000, max_cone_row_tests=16_000_000,
        max_closure_triangle_tests=32_000_000, max_closure_edge_tests=96_000_000,
        max_fit_evaluations=65_552, max_closure_schedules=16,
        max_retained_array_bytes=536_870_912, max_metadata_bytes=67_108_864,
        chunk_rows=65_536),
    cad_patch_policy=dict(max_pair_normal_sine=1e-8, max_pair_anchored_distance_mm=1e-8,
        max_full_support_distance_mm=1e-8, min_support_vertices=12,
        min_second_extent_mm=.25, min_inplane_covariance_ratio=1e-8),
    cad_patch_budgets=dict(max_vertices=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_components=65_536, max_output_patches=4096,
        max_support_memberships=6_000_000, max_full_vertex_tests=12_000_000,
        chunk_triangles=65_536),
    search_policy=dict(min_scan_support_count=3, max_scan_support_rms_mm=.1,
        min_normal_condition_ratio=.02, max_normal_alignment_extra_radians=.02,
        max_intersection_backward_residual_mm=1e-8, pair_angle_numeric_guard_radians=1e-10),
    search_budgets=dict(max_scan_planes=65_552, max_cad_planes=2048, max_bank_planes=128,
        max_scan_triplets=256, max_scan_partner_tests=1_048_576, max_cad_pair_records=2_096_128,
        max_pair_join_tests=32_000_000, max_cad_pair_samples_per_triplet=12,
        max_unsigned_per_scan_triplet=4, max_compatible_rows_per_triplet=49_104,
        max_unsigned_assignments=1024, max_signed_assignments=8192,
        max_report_bytes=67_108_864, chunk_pair_rows=65_536),
    initialization_budgets=dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
        max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
        max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
        chunk_rows=65_536, max_node_visits=512_000_000, max_triangle_pairs=256_000_000,
        max_normal_pairs=32_000_000, max_work_units=4_000_000_000,
        max_face_build_nodes=131_071, max_face_build_triangle_visits=10_000_000,
        max_refined_start_groups=48, max_report_bytes=100_663_296),
    surface_budgets=dict(max_triangles=500_000, max_query_rows=200_000, leaf_size=16,
        max_build_nodes=131_071, max_build_triangle_visits=10_000_000,
        max_node_visits=32_000_000, max_triangle_pairs=32_000_000,
        max_normal_pairs=32_000_000, max_work_units=512_000_000,
        max_pair_batch=65_536, max_ties_per_query=32),
    registration_budgets=dict(max_query_calls=4096, max_query_rows=64_000_000,
        max_node_visits=2_000_000_000, max_triangle_pairs=2_000_000_000,
        max_normal_pairs=500_000_000, max_work_units=20_000_000_000),
)


@lru_cache(maxsize=1)
def _connected():
    spec = importlib.util.spec_from_file_location("_physical_config_v11_helpers", CONNECTED_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original():
    return _connected()._original()


@lru_cache(maxsize=1)
def _box_evidence():
    spec = importlib.util.spec_from_file_location("_physical_v11_frozen_box_checks", BOX_TEST_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle_pose():
    return _original()._oracle_pose()


def _noisy_rows(raw):
    return _connected()._noisy_rows(raw)


def _policies():
    from backend.cad_guided_mesh_offline_v11 import MeshWorkflowPolicyV11
    from backend.cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1
    from backend.cad_guided_native_features import NativeGuidedPolicy
    from backend.cad_guided_instances import SupportPolicy
    from backend.cad_guided_noisy_mesh_planes_v5 import NoisyMeshPlanePolicyV5, NoisyMeshPlaneBudgetsV5
    from backend.cad_guided_mesh_plane_proposals_v4 import CoplanarPatchPolicyV4, CoplanarPatchBudgetsV4
    from backend.cad_guided_plane_search_v6 import PlaneSearchPolicyV6, PlaneSearchBudgetsV6
    from backend.cad_guided_plane_initialization_v11 import PlaneInitializationBudgetsV11
    from backend.cad_guided_triangle_surface_v3 import TriangleSurfaceBudgets
    from backend.cad_guided_global_registration_v3 import SurfaceRegistrationBudgetsV3

    config = copy.deepcopy(CONFIG)
    noisy = config["noisy_policy"]
    noisy["tile_triangles"] = tuple(noisy["tile_triangles"])
    return dict(
        workflow_policy=MeshWorkflowPolicyV11(**config["workflow"], native_adapter_version="v1"),
        registration_policy=WholePartRegistrationPolicyV1(**config["registration"]),
        measurement_policy=NativeGuidedPolicy(**config["measurement"]),
        instance_policy=SupportPolicy(**config["support"]),
        noisy_policy=NoisyMeshPlanePolicyV5(**noisy),
        noisy_budgets=NoisyMeshPlaneBudgetsV5(**config["noisy_budgets"]),
        cad_patch_policy=CoplanarPatchPolicyV4(**config["cad_patch_policy"]),
        cad_patch_budgets=CoplanarPatchBudgetsV4(**config["cad_patch_budgets"]),
        search_policy=PlaneSearchPolicyV6(**config["search_policy"]),
        search_budgets=PlaneSearchBudgetsV6(**config["search_budgets"]),
        initialization_budgets=PlaneInitializationBudgetsV11(**config["initialization_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]),
        surface_budgets=TriangleSurfaceBudgets(**config["surface_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["surface_budgets"]),
        registration_budgets=SurfaceRegistrationBudgetsV3(**config["registration_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["registration_budgets"]))


def _policy_snapshot(policies):
    return json.loads(json.dumps({key: asdict(value) for key, value in policies.items()},
                                sort_keys=True, allow_nan=False))


def _joint_completion(initialization):
    """Independent count/identity ledger, not a scheduler or pose generator."""
    joint, schedule = initialization["joint_proposal_accounting"], initialization["two_plane_schedule"]
    hypotheses = initialization["hypotheses"]
    for key in ("triplet_records", "triplet_unsigned_reservation", "ordered_pair_matches",
                "pair_slot_reservation", "total_hypothesis_records"):
        assert type(joint[key]) is int and joint[key] >= 0
    assert joint["pair_lane_enabled"] is True
    matches = schedule["scheduled_matches"]
    assert schedule["schema_version"] == "two_plane_schedule_v1"
    assert schedule["status"] == "COMPLETE" and schedule["selected_policy_evaluation_complete"] is True
    assert schedule["search_exhaustive"] is False and schedule["accepted_registration"] is False
    assert schedule["budgets"] == PAIR_DEFAULTS["scheduler"]
    assert schedule["observed_bend_count"] is None and schedule["production_use_allowed"] is False
    h3, u3, matches_count = joint["triplet_records"], joint["triplet_unsigned_reservation"], len(matches)
    assert matches_count == joint["ordered_pair_matches"] <= 256
    assert joint["pair_slot_reservation"] == 32*matches_count
    assert joint["total_hypothesis_records"] == h3+32*matches_count == len(hypotheses) <= 8192
    assert u3+matches_count <= 1024
    assert all(row["proposal_lane"] == "TRIPLET" for row in hypotheses[:h3])
    assert all(row["proposal_lane"] == "TWO_PLANE_CONTEXT" for row in hypotheses[h3:])
    assert u3 == max((row["unsigned_assignment_id"] for row in hypotheses[:h3]), default=-1)+1
    reservation = schedule["joint_budget"]
    assert reservation["retained_triplet_hypothesis_count"] == h3
    assert reservation["retained_triplet_unsigned_count"] == u3
    assert reservation["joint_hypothesis_records_reserved"] == h3+32*matches_count
    assert reservation["joint_unsigned_assignments_reserved"] == u3+matches_count
    assert [row["ordered_match_index"] for row in matches] == list(range(matches_count))
    assert matches_count % 2 == 0
    for a, b in zip(matches[::2], matches[1::2]):
        assert a["scan_pair_id"] == b["scan_pair_id"]
        assert a["scan_plane_ids"] == b["scan_plane_ids"]
        assert a["cad_plane_ids"] == b["cad_plane_ids"][::-1]
        assert a["canonical_cad_pair_index"] == b["canonical_cad_pair_index"]
        assert all(row["reserved_hypothesis_records"] == 32 and row["reserved_unsigned_assignments"] == 1
                   for row in (a, b))
    helper = initialization["two_plane_context"]
    if not matches:
        assert helper["status"] == "NOT_CALLED_EMPTY_DECLARED_SCHEDULE"
        assert helper["usable_proposals"] is False and helper["accepted_registration"] is False
        return joint
    assert helper["status"] == "COMPLETE" and helper["selected_policy_evaluation_complete"] is True
    assert helper["accepted_registration"] is False and helper["finite_support_match_verified"] is False
    assert helper["search_exhaustive"] is False and helper["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert helper["budgets"] == PAIR_DEFAULTS["context"]
    expected = {a+"_sha256": initialization["input_digests"][b+"_sha256"] for a, b in (
        ("scan_points", "scan_points"), ("scan_triangles", "scan_triangle_raw_rows"),
        ("cad_triangles", "cad_triangles"), ("cad_face_ids", "cad_face_ids"),
        ("cad_source_triangle_ids", "cad_source_triangle_ids"))}
    assert helper["input_digests"] == expected
    stats = helper["statistics"]
    assert stats["signed_families_completed"] == stats["signed_families_planned"] == 4*matches_count
    assert stats["slot_records_completed"] == stats["candidate_slots_planned"] == 32*matches_count
    assert len(helper["families"]) == 4*matches_count
    signatures, proposed_origins, slot_rows = set(), {}, []
    for fi, family in enumerate(helper["families"]):
        assert family["signed_family_id"] == fi
        ai = family["ordered_assignment_id"]
        assert type(ai) is int and 0 <= ai < matches_count
        signs = tuple(family["normal_signs"])
        assert len(signs) == 2 and all(type(sign) is int and sign in (-1, 1) for sign in signs)
        assert (ai, signs) not in signatures
        signatures.add((ai, signs))
        match = matches[ai]
        source, target = match["scan_plane_ids"], match["cad_plane_ids"]
        if source[0] > source[1]:
            source, target = source[::-1], target[::-1]
        correspondence = [dict(scan_plane_id=a, cad_plane_id=b, normal_sign=sign)
                          for a, b, sign in zip(source, target, signs)]
        assert family["correspondences"] == correspondence
        assert tuple(slot["slot"] for slot in family["slots"]) == CONTEXT_SLOTS
        for slot in family["slots"]:
            row = hypotheses[h3+len(slot_rows)]
            slot_rows.append(row)
            assert row["signed_family_id"] == fi and row["axial_slot"] == slot["slot"]
            assert row["unsigned_assignment_id"] == u3+ai and row["correspondences"] == correspondence
            assert row["context_slot_disposition"] == slot["status"]
            assert row["context_candidate_index"] == slot["candidate_index"]
            if slot["status"] == "PROPOSED_SEED_ONLY":
                ci = slot["candidate_index"]
                assert type(ci) is int and 0 <= ci < len(helper["candidates"])
                candidate = helper["candidates"][ci]
                assert family["status"] == "AVAILABLE"
                assert row["status"] == "PROPOSED" and candidate["candidate_index"] == ci
                assert row["rotation"] == candidate["rotation"]
                assert row["translation_mm"] == candidate["translation_mm"]
                proposed_origins.setdefault(ci, []).append(dict(signed_family_id=fi, slot=slot["slot"]))
            else:
                assert slot["status"] in ("PAIR_REJECTED", "ABSENT_EMPTY_HEURISTIC_INTERVAL")
                assert (slot["status"] == "PAIR_REJECTED") == (family["status"] != "AVAILABLE")
                assert row["status"] == ("REJECTED" if slot["status"] == "PAIR_REJECTED" else "ABSENT_CONTEXT_SLOT")
                assert row["rotation"] is None and row["translation_mm"] is None
                assert slot["candidate_index"] is None and slot["lambda_mm"] is None
    assert len(signatures) == 4*matches_count and len(slot_rows) == 32*matches_count
    assert set(proposed_origins) == set(range(len(helper["candidates"])))
    for ci, candidate in enumerate(helper["candidates"]):
        assert candidate["contributing_slots"] == proposed_origins[ci]
        assert candidate["accepted_registration"] is False
    return joint


def _free_circle_nominal_assertions(result, raw, cad_frame):
    assert result["status"] == "PARTIAL", result.get("reasons")
    assert result["registration"]["status"] == "CONDITIONAL_BEST_FIT"
    assert len(result["predictions"]) == 1
    transform = np.asarray(result["registration"]["transform_scan_to_cad"])
    pose_error = float(np.linalg.norm(raw @ transform[:3, :3].T + transform[:3, 3] - cad_frame, axis=1).max())
    assert pose_error <= CRITERIA["pose_action_error_mm"], pose_error
    measured = result["measurements"]
    assert measured["raw_input_rows"] == len(raw)
    assert measured["observed_bend_count"] is None
    rows = [row for row in measured["features"] if row["status"] == "MEASURED_CONDITIONAL"]
    assert len(rows) == 1
    row, value = rows[0], rows[0]["measurements"]
    radius_error = abs(value["measured_radius_mm"] - 5.)
    angle_error = abs(abs(value["measured_sweep_deg"]) - 90.)
    assert radius_error <= CRITERIA["radius_error_mm"], radius_error
    assert angle_error <= CRITERIA["absolute_sweep_error_deg"], angle_error
    assert value["fit_model"] == "free_circle"
    assert value["tangency_check_role"] == "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK"
    support = row["raw_support"]
    assert isinstance(support["arc_rows"], list)
    assert support["arc_rows_sha256"] == hashlib.sha256(np.asarray(support["arc_rows"], dtype="<i8").tobytes()).hexdigest()
    for ids, digest in zip(support["flange_rows"], support["flange_rows_sha256"]):
        assert digest == hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    prediction = result["predictions"][0]
    rotation, translation = _oracle_pose()
    expected = np.array([[5 / math.sqrt(2), 5 / math.sqrt(2), z] for z in (1., 29.)]) @ rotation.T + translation
    actual = np.asarray([prediction["line_start"], prediction["line_end"]])
    endpoint_error = min(float(np.linalg.norm(actual - expected, axis=1).max()),
                         float(np.linalg.norm(actual[::-1] - expected, axis=1).max()))
    assert endpoint_error <= CRITERIA["generator_endpoint_error_mm"], endpoint_error
    assert prediction["line_convention"] == "MEASURED_INNER_ARC_MID_ANGLE_GENERATOR"
    assert prediction["physical_trim_endpoints"] is None
    assert prediction["manufacturing_event_identity"] == "UNKNOWN"
    # Both axis endpoints must undergo the same CAD-to-original-scan inverse.
    axis_actual = np.asarray([prediction["axis_support_line_start"], prediction["axis_support_line_end"]])
    axis_expected = np.array([[0., 0., z] for z in (1., 29.)]) @ rotation.T + translation
    axis_error = min(float(np.linalg.norm(axis_actual - axis_expected, axis=1).max()),
                     float(np.linalg.norm(axis_actual[::-1] - axis_expected, axis=1).max()))
    assert axis_error <= CRITERIA["generator_endpoint_error_mm"], axis_error
    return dict(pose_action_error_mm=pose_error, radius_error_mm=radius_error,
                absolute_sweep_error_deg=angle_error, generator_endpoint_error_mm=endpoint_error,
                axis_endpoint_error_mm=axis_error)


def _nominal_oracle(noisy):
    return types.FunctionType(_free_circle_nominal_assertions.__code__,
        {**_free_circle_nominal_assertions.__globals__, "CRITERIA": NOISY_CRITERIA if noisy else CRITERIA})


def _binding_assertions(result, raw, faces, cad_blob, scan_blob):
    _connected()._v4()._legacy_common_assertions(result, cad_blob, scan_blob)
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V11"
    assert result["registration_initialization"] == "TRIPLET_AND_TWO_PLANE_CONTEXT_FINITE_PREFILTER_V11"
    assert result["policies"]["two_plane_enabled"] is True
    assert result["position_error_model"]["uniform_euclidean_bound_mm"] == CONFIG["position_error_bound_mm"]
    assert result["position_error_model"]["status"] == "STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION"
    source = result["input_preparation"]["source_bindings"]
    raw_sha = hashlib.sha256(np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
    faces_sha = hashlib.sha256(np.asarray(faces, dtype="<i8").tobytes()).hexdigest()
    epsilon_sha = hashlib.sha256(np.full(len(raw), CONFIG["position_error_bound_mm"], dtype="<f8").tobytes()).hexdigest()
    assert source["raw_xyz_sha256"] == raw_sha
    assert result["input_preparation"]["coverage"]["raw_vertex_count"] == len(raw) == 2752
    topology, binding = result["scan_triangle_binding"], result["triangle_sample_binding"]
    assert topology["status"] == "CONSISTENT_IMMUTABLE_SCAN_TRIANGLE_SNAPSHOT"
    assert topology["source_sha256"] == hashlib.sha256(scan_blob).hexdigest()
    assert topology["raw_points_sha256"] == raw_sha and topology["triangle_raw_rows_sha256"] == faces_sha
    assert result["scan_topology"]["triangle_raw_rows_sha256"] == faces_sha
    assert binding["status"] == "CONSISTENT_IMMUTABLE_TRIANGLE_SAMPLE_SNAPSHOT"
    assert binding["initialization_sample_count"] == CONFIG["workflow"]["cad_sample_count"]
    evidence = result["registration"]["evidence"]
    initialized = evidence["initialization"]["input_digests"]
    assert initialized["scan_points_sha256"] == raw_sha
    assert initialized["scan_triangle_raw_rows_sha256"] == faces_sha
    assert initialized["position_error_bounds_mm_sha256"] == epsilon_sha
    assert initialized["cad_triangles_sha256"] == binding["oriented_triangle_xyz_sha256"]
    assert evidence["registration_raw_rows"] == dict(selection="ALL_RAW_ROWS", count=len(raw), sha256=None)
    if result["registration"]["status"] == "CONDITIONAL_BEST_FIT":
        final = result["registration_array_binding"]
        assert final["status"] == "CONSISTENT_REGISTRATION_ARRAY_BINDING"
        assert final["registration_selection"] == "ALL_RAW_ROWS"
        assert final["position_error_bounds_mm_sha256"] == epsilon_sha
        assert final["cad_sampler_points_sha256"] == binding["initialization_points_sha256"]


def _numerical_completion(registration, *, allow_prefilter_abstention):
    """V9 interval checks plus V11 box/finite partition; geometric checks follow."""
    evidence, initialization = registration["evidence"], registration["evidence"]["initialization"]
    assert registration["schema_version"] == "noisy_plane_initialized_finite_registration_v11"
    assert evidence["schema"] == "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V11"
    assert initialization["schema_version"] == "noisy_plane_initialization_prefilter_v11"
    assert initialization["numerical_consumer_complete"] is True
    assert initialization["numerical_support_policy"] == evidence["numerical_support_policy"] == SUPPORT_POLICY
    assert evidence["distance_model"] == DISTANCE_MODEL and evidence["optimizer_objective"] == SURROGATE
    _box_evidence()._box_numerical_completion(initialization)
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
        # V11 records no completed refinement when the fully completed prefilter
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
    joint = _joint_completion(initialization)
    assert stats["hypothesis_count"] == len(hypotheses) == joint["total_hypothesis_records"]
    assert ss["signed_assignments_evaluated"] == joint["triplet_records"]
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
    absent = [row for row in hypotheses if row["status"] == "ABSENT_CONTEXT_SLOT"]
    assert stats["absent_context_slot_count"] == len(absent)
    assert stats["scheduled_hypothesis_count"] == stats["hypotheses_evaluated"] == len(scheduled)
    assert stats["omitted_proposed_hypothesis_count"] == len(omitted)
    assert stats["search_rejected_hypothesis_count"] == len(refused)
    assert stats["proposed_hypothesis_count"] == len(scheduled)+len(omitted)
    assert omitted == [] and stats["proposed_hypothesis_count"] == len(scheduled)
    assert len(hypotheses) == len(scheduled)+len(omitted)+len(refused)+len(absent)
    assert stats["passing_hypothesis_count"]+stats["rejected_hypothesis_count"] == len(scheduled)
    for row in hypotheses:
        assert row["prefilter_scheduled"] == (row["status"] == "PROPOSED" and row["unsigned_assignment_id"] in selected)
        if row["prefilter_scheduled"]:
            assert row["prefilter_status"] in ("PASSED", "REJECTED")
            assert len(row["assigned_support"]) == {"TRIPLET": 3, "TWO_PLANE_CONTEXT": 2}[row["proposal_lane"]]
            assert [(s["scan_plane_id"], s["cad_plane_id"], s["normal_sign"]) for s in row["assigned_support"]] == [
                (s["scan_plane_id"], s["cad_plane_id"], s["normal_sign"]) for s in row["correspondences"]]
            assert row["prefilter_reasons"] == ([] if row["prefilter_status"] == "PASSED"
                else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
            for support in row["assigned_support"]:
                _box_evidence()._support_disposition_complete(support)
        else:
            assert row["assigned_support"] == []
            assert row["prefilter_status"] == ("SEARCH_NOMINAL_WITNESS_REFUSAL" if row["status"] == "REJECTED"
                else "AXIAL_SLOT_ABSENT_NOT_GEOMETRIC_EXCLUSION" if row["status"] == "ABSENT_CONTEXT_SLOT"
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
        assert allow_prefilter_abstention
        assert all(row["prefilter_status"] == "REJECTED" for row in scheduled)
        assert not all_groups and stats["passing_hypothesis_count"] == 0
        assert initialization["reasons"] == registration["reasons"] == [NO_START_REASON]
        assert registration["status"] == "UNAVAILABLE" and candidates == []
        assert evidence["candidate_count"] == evidence["attempted_candidate_count"] == 0
        return ("PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE" if scheduled
                else "EMPTY_GENERATED_SCHEDULE_COMPLETE_NOT_POSITIVE_OUTPUT")
    assert initialization["reasons"] == []
    assert evidence["candidate_search_complete"] is True and evidence["candidate_evaluation_complete"] is True
    assert evidence["candidate_count"] == evidence["attempted_candidate_count"] == len(groups) == len(candidates)
    for index, (row, group) in enumerate(zip(candidates, groups)):
        assert row["index"] == index and row["execution_status"] == "ATTEMPTED"
        for key in ("hypothesis_indices", "initialized_rotation", "initialized_translation_mm"):
            assert row[key] == group[key]
        assert all(key in row for key in ("rotation", "translation", "scan_coverage", "inlier_surface_p95_mm",
                                         "observability_ratio", "weighted_fit_observability_ratio"))
        assert all(reason in _connected()._v4()._GEOMETRIC_CANDIDATE_REFUSALS for reason in row["reasons"])
    return "ALL_SELECTED_PLANE_START_GROUPS_FULLY_EVALUATED"


def _terminal_return(result):
    # This checks a retained return contract, NOT successful pipeline execution.
    json.dumps(result, allow_nan=False)
    assert type(result) is dict and result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V11"
    assert result["status"] in ("UNAVAILABLE", "PARTIAL")
    assert type(result["phase"]) is str and result["phase"]
    assert type(result["reasons"]) is list and all(type(reason) is str for reason in result["reasons"])
    assert type(result["predictions"]) is list and all(type(row) is dict for row in result["predictions"])
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    assert result["conformance"] == "INDETERMINATE" and result["uncertainty_status"] == "UNCALIBRATED"
    return len(result["predictions"])


def _stage_summary(result):
    # Missing/null stages are unknown, not a claim they completed or never ran.
    stages = {}
    for key in ("input_preparation", "scan_topology", "native_features",
                "cad_surface_sampling", "registration", "measurements"):
        row = result.get(key)
        stages[key] = ({name: row[name] for name in ("status", "phase", "reasons",
            "computation_status", "detailed_report_discarded_whole") if name in row}
            if type(row) is dict else {"evidence": "MISSING_OR_NULL_UNKNOWN"})
    registration = result.get("registration")
    evidence = registration.get("evidence", {}) if type(registration) is dict else {}
    if type(evidence) is dict:
        stages["registration_evidence"] = {key: evidence[key] for key in (
            "candidate_search_complete", "candidate_evaluation_complete", "numerical_consumer_complete",
            "candidate_count", "attempted_candidate_count", "candidate_numerical_failure_count",
            "coarse_scan_rows", "refined_scan_rows") if key in evidence}
        initialization = evidence.get("initialization")
        if type(initialization) is dict:
            stages["initialization"] = {key: initialization[key] for key in (
                "status", "reasons", "search_complete", "selected_policy_evaluation_complete",
                "numerical_consumer_complete", "box_screening_complete", "statistics",
                "joint_proposal_accounting") if key in initialization}
            for name in ("two_plane_schedule", "two_plane_context"):
                row = initialization.get(name)
                stages[name] = ({key: row[key] for key in ("status", "reasons", "statistics", "joint_budget",
                    "selected_policy_evaluation_complete", "usable_schedule", "usable_proposals") if key in row}
                    if type(row) is dict else {"evidence": "MISSING_OR_NULL_UNKNOWN"})
    return stages


def _two_plane_origin_attribution(registration):
    """Only after output qualification: distinguish origin attribution from accuracy."""
    evidence = registration["evidence"]
    candidate = evidence["candidates"][evidence["selected_candidate_index"]]
    hypotheses = evidence["initialization"]["hypotheses"]
    origins = candidate["hypothesis_indices"]
    assert origins and all(type(index) is int and 0 <= index < len(hypotheses) for index in origins)
    pair_origins = [hypotheses[index] for index in origins
                    if hypotheses[index]["proposal_lane"] == "TWO_PLANE_CONTEXT"]
    for row in pair_origins:
        assert row["prefilter_status"] == "PASSED" and len(row["assigned_support"]) == 2
        assert all(support["status"] == "PASSED" for support in row["assigned_support"])
    return bool(pair_origins)


def _record_result(record, result, raw, faces, cad_frame, cad_blob, scan_blob, *, noisy):
    record["two_plane_origin_recovery_passed"] = False
    record["workflow"] = result  # Full unfiltered result, including every failed ledger.
    record["inference_returned"] = True
    try:
        count = _terminal_return(result)
        record.update(diagnostic_record_complete=True, returned_prediction_count=count,
            reported_workflow_phase=result["phase"], reported_workflow_status=result["status"],
            reported_workflow_reasons=result["reasons"], stages=_stage_summary(result),
            output_coverage=("FAILED_NOMINAL_NO_PREDICTIONS" if count == 0 else
                "ONE_PREDICTION_GEOMETRY_NOT_YET_QUALIFIED" if count == 1 else
                "FAILED_NOMINAL_PREDICTION_COUNT"))
    except Exception as error:
        record.update(diagnostic_error_type=type(error).__name__, diagnostic_error=str(error)[:1000])
        return
    try:
        # Deliberately not the missing-arc assertion: a completed no-start result
        # may explain this nominal failure but can never qualify its geometry.
        record["numerical_disposition"] = _execution_complete(
            result["registration"], allow_prefilter_abstention=True)
        record["scheduled_numerical_work_complete"] = True
    except Exception as error:
        record.update(numerical_error_type=type(error).__name__, numerical_error=str(error)[:1000])
    if count != 1:
        record["geometry_oracle_status"] = "FAILED_REQUIRED_ONE_NOMINAL_OUTPUT"
        return
    try:
        _binding_assertions(result, raw, faces, cad_blob, scan_blob)
        _execution_complete(result["registration"], allow_prefilter_abstention=False)
        assert record["scheduled_numerical_work_complete"] is True
        assert result["phase"] == "COMPLETE" and result["measurements"]["computation_status"] == "COMPLETE"
        record["errors"] = _nominal_oracle(noisy)(result, raw, cad_frame)
        record.update(positive_geometry_passed=True, output_coverage="ONE_MEASURED_OUTPUT_ORIGINAL_GEOMETRY_PASSED",
                      geometry_oracle_status="ORIGINAL_CEILINGS_PASSED_FREE_CIRCLE_LABELS")
    except Exception as error:
        record.update(geometry_oracle_status="FAILED_NOMINAL_GEOMETRY_OR_CONTRACT",
            geometry_error_type=type(error).__name__, geometry_error=str(error)[:1000])
    if record["positive_geometry_passed"]:
        try:
            record["two_plane_origin_recovery_passed"] = _two_plane_origin_attribution(result["registration"])
        except Exception as error:
            record.update(mechanism_attribution_error_type=type(error).__name__,
                          mechanism_attribution_error=str(error)[:1000])


def _source_digests():
    # Fixture/config custody subset only; the supervisor must freeze full closure.
    connected = _connected()
    paths = (Path(__file__).resolve(), CONFIG_SOURCE, CONNECTED_SOURCE, connected.V8_SOURCE,
        connected.V4_SOURCE, connected._v4().SURFACE_SOURCE, connected._v4().DENSE_SOURCE,
        ORIGINAL_SOURCE, PREDECESSOR_SOURCE, BOX_TEST_SOURCE, INITIALIZER_SOURCE,
        ROOT/"backend/cad_guided_two_plane_schedule_v1.py", ROOT/"backend/cad_guided_two_plane_context_v1.py",
        ROOT/"backend/cad_guided_mesh_offline_v11.py",
        ROOT/"backend/cad_guided_measurement.py")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _write_artifact(directory, index, record):
    # Original exclusive writer, unchanged16MiB per-case bound; total is two slots.
    return _connected()._write_artifact(directory, index, record)


def _receipt(records, before, after, directory):
    exact_cases = [row["case"] for row in records] == [case for case, _ in CASES]
    retained = all(row.get("artifact", {}).get("status") == "COMPLETE_CASE_RECORD_RETAINED" for row in records)
    total = sum(row.get("artifact", {}).get("bytes", 0) for row in records)
    custody = exact_cases and retained and before == after and total <= ARTIFACT_LIMITS["total_case_json_bytes"]
    complete = custody and all(row["diagnostic_record_complete"] for row in records)
    positive = complete and all(row["positive_geometry_passed"] for row in records)
    pair_recovery = positive and all(row.get("two_plane_origin_recovery_passed") is True for row in records)
    return dict(protocol="CAD_GUIDED_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V11",
        expected_case_count=2, cases=records, diagnostic_completed=complete,
        positive_geometry_matrix_passed=positive,
        two_plane_origin_recovery_matrix_passed=pair_recovery,
        interpretation="DIAGNOSTIC_COMPLETION_IS_NOT_POSITIVE_OUTPUT_OR_PHYSICAL_QUALIFICATION",
        config=CONFIG, certificate_caps=CERTIFICATE_CAPS, criteria=CRITERIA,
        two_plane_defaults=PAIR_DEFAULTS, two_plane_enabled=True,
        noisy_criteria=NOISY_CRITERIA, coverage_limitations=COVERAGE_LIMITATIONS,
        external_resource_limits=RESOURCE_LIMITS, artifact_limits=ARTIFACT_LIMITS,
        artifact_directory=str(directory), fixture_source_sha256_before=before,
        fixture_source_sha256_after=after, observed_bend_count=None, production_use_allowed=False)


def run_invented_physical_config_constructed_v11_diagnostic(*, artifact_directory):
    from backend.cad_guided_mesh_offline_v11 import inspect_cad_guided_mesh_v11

    before = _source_digests()
    records = [dict(case=case, noisy=noisy, expected_predictions=1, diagnostic_record_complete=False,
        inference_returned=False, returned_prediction_count=None, positive_geometry_passed=False,
        scheduled_numerical_work_complete=False, output_coverage="NOT_EXECUTED",
        geometry_oracle_status="NOT_EXECUTED", phase="fixture",
        coverage_limitations=COVERAGE_LIMITATIONS, observed_bend_count=None,
        production_use_allowed=False) for case, noisy in CASES]
    try:
        directory = _connected()._v4()._new_artifact_directory(artifact_directory)
    except Exception as error:
        for record in records:
            record.update(phase="artifact_setup", error_type=type(error).__name__, error=str(error)[:1000])
        return _receipt(records, before, _source_digests(), artifact_directory)
    original = _original()
    cad_blob, fixture_error = None, None
    try:
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
    except Exception as error:
        fixture_error = error
    compact = []
    for index, record in enumerate(records):
        try:
            if fixture_error is not None:
                raise fixture_error
            raw, faces, cad_frame = original._analytic_mesh_scan(False)
            assert len(raw) == 2752
            if record["noisy"]:
                raw, delta = _noisy_rows(raw)
                assert np.max(np.linalg.norm(delta, axis=1)) <= .001
                record["noise"] = dict(model="FIXED_ORIGINAL_ROW_ID_3D_MODULAR_PATTERN",
                    rounding_reserve_factor=.999, maximum_euclidean_mm=float(np.max(np.linalg.norm(delta, axis=1))),
                    delta_sha256=hashlib.sha256(np.asarray(delta, dtype="<f8").tobytes()).hexdigest())
            scan_blob = original._ply_bytes(raw, faces)
            policies = _policies()
            record.update(scan_rows=len(raw), mesh_faces=len(faces), policy_snapshot=_policy_snapshot(policies),
                two_plane_defaults=copy.deepcopy(PAIR_DEFAULTS), two_plane_enabled=True,
                cad_sha256=hashlib.sha256(cad_blob).hexdigest(), scan_sha256=hashlib.sha256(scan_blob).hexdigest(),
                phase="actual_byte_to_v11_physical_config_workflow")
            result = inspect_cad_guided_mesh_v11(scan_bytes=scan_blob, cad_bytes=cad_blob,
                position_error_bound_mm=CONFIG["position_error_bound_mm"], allow_ocp=True, two_plane_enabled=True, **policies)
            record["phase"] = "post_inference_evaluation"
            _record_result(record, result, raw, faces, cad_frame, cad_blob, scan_blob, noisy=record["noisy"])
            record["phase"] = "terminal_return_recorded"
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:1000])
        summary = {key: value for key, value in record.items() if key not in ("workflow", "policy_snapshot")}
        try:
            summary["artifact"] = _write_artifact(directory, index, record)
        except Exception as error:
            summary.update(phase="artifact_publication", error_type=type(error).__name__, error=str(error)[:1000],
                artifact=dict(relative_path=f"case_{index:02d}.json", status="FAILED_OR_PARTIAL_NOT_QUALIFIED"))
        compact.append(summary)
    return _receipt(compact, before, _source_digests(), directory)


def _ast_config(path):
    assignments = [node.value for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CONFIG"
                                              for target in node.targets)]
    assert len(assignments) == 1
    return ast.dump(assignments[0], include_attributes=False)


def test_config_is_exact_frozen_physical_expression_without_runner_import():
    assert _ast_config(Path(__file__)) == _ast_config(CONFIG_SOURCE)
    assert _ast_config(Path(__file__)) == _ast_config(PREDECESSOR_SOURCE)
    assert CONFIG["position_error_bound_mm"] == .05
    assert CONFIG["measurement"]["fit_model"] == "free_circle"
    assert CONFIG["registration"]["coarse_points"] == 2048 and CONFIG["registration"]["refine_points"] == 8192
    tree = ast.parse(Path(__file__).read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and "run_gold11" in ast.unparse(node)
                   for node in ast.walk(tree))


def test_policy_snapshot_adds_only_v11_types_certificate_caps_and_fixed_native_v1(monkeypatch):
    real_import = builtins.__import__
    def guard(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    before = copy.deepcopy(CONFIG)
    policies = _policies()
    snapshot = _policy_snapshot(policies)
    mapping = dict(workflow_policy="workflow", registration_policy="registration",
        measurement_policy="measurement", instance_policy="support")
    for key, value in snapshot.items():
        expected = copy.deepcopy(CONFIG[mapping.get(key, key)])
        if key == "workflow_policy":
            expected["native_adapter_version"] = "v1"
        if key in CERTIFICATE_CAPS:
            expected["max_certificate_pairs"] = CERTIFICATE_CAPS[key]
        assert value == expected
    assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV11"
    assert type(policies["initialization_budgets"]).__name__ == "PlaneInitializationBudgetsV11"
    assert type(policies["registration_budgets"]).__name__ == "SurfaceRegistrationBudgetsV3"
    assert CONFIG == before


def test_free_circle_oracle_preserves_original_ast_except_two_explicit_model_labels():
    def function(path, name):
        return next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    original = function(ORIGINAL_SOURCE, "_nominal_assertions")
    expected = copy.deepcopy(original)
    expected.name = "_free_circle_nominal_assertions"
    replacements = {"measured_flange_tangent": "free_circle",
                    "ENFORCED_CONSTRAINT_NOT_INDEPENDENT_VALIDATION": "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK"}
    changed = []
    for node in ast.walk(expected):
        if isinstance(node, ast.Constant) and type(node.value) is str and node.value in replacements:
            changed.append(node.value)
            node.value = replacements[node.value]
    assert set(changed) == set(replacements) and len(changed) == 2
    actual = function(Path(__file__), "_free_circle_nominal_assertions")
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def test_original_helpers_noise_and_clean_noisy_error_ceilings_remain_unchanged(monkeypatch):
    real_import = builtins.__import__
    def guard(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    connected, original = _connected(), _original()
    before = copy.deepcopy((original.CRITERIA, connected.CRITERIA, connected.NOISY_CRITERIA))
    assert original is connected._original()
    assert CRITERIA == original.CRITERIA == connected.CRITERIA
    assert NOISY_CRITERIA == connected.NOISY_CRITERIA
    raw, faces, frame = original._analytic_mesh_scan(False)
    saved = (raw.copy(), faces.copy(), frame.copy())
    actual, delta = _noisy_rows(raw)
    expected, expected_delta = connected._noisy_rows(raw)
    assert raw.shape == (2752, 3) and np.array_equal(actual, expected) and np.array_equal(delta, expected_delta)
    assert np.max(np.linalg.norm(delta, axis=1)) <= .001
    assert all(np.array_equal(a, b) for a, b in zip((raw, faces, frame), saved))
    for noisy in (False, True):
        oracle = _nominal_oracle(noisy)
        assert oracle.__code__ is _free_circle_nominal_assertions.__code__
        changed = {key for key in oracle.__globals__
            if oracle.__globals__[key] is not _free_circle_nominal_assertions.__globals__[key]}
        assert changed == ({"CRITERIA"} if noisy else set())
    assert (original.CRITERIA, connected.CRITERIA, connected.NOISY_CRITERIA) == before


def test_actual_worker_uses_original_nominal_geometry_and_unpatched_public_dispatch():
    tree = ast.parse(Path(__file__).read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "run_invented_physical_config_constructed_v11_diagnostic")
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    workflow = [node for node in calls if isinstance(node.func, ast.Name)
                and node.func.id == "inspect_cad_guided_mesh_v11"]
    assert len(workflow) == 1
    assert [key.arg for key in workflow[0].keywords] == [
        "scan_bytes", "cad_bytes", "position_error_bound_mm", "allow_ocp", "two_plane_enabled", None]
    scan = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "_analytic_mesh_scan"]
    assert len(scan) == 1 and ast.unparse(scan[0].args[0]) == "False"
    forbidden = {"setattr", "monkeypatch", "_refine", "_starts", "_oracle_pose", "_missing_arc_assertions"}
    assert not any(isinstance(node, ast.Name) and node.id in forbidden or isinstance(node, ast.Attribute)
                   and node.attr in forbidden for node in ast.walk(worker))
    evaluate = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_record_result")
    assert evaluate.lineno > workflow[0].lineno
    original_helpers = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "original"}
    assert original_helpers == {"_owned_step_bytes", "_asymmetric_native_strip", "_analytic_mesh_scan", "_ply_bytes"}


@pytest.mark.parametrize("noisy", (False, True))
def test_nominal_abstention_cannot_pass_the_free_circle_geometry_oracle(noisy):
    with pytest.raises(AssertionError):
        _nominal_oracle(noisy)({"status": "UNAVAILABLE"}, np.zeros((2, 3)), np.zeros((2, 3)))


def _source_only_refusal():
    return dict(schema="CAD_GUIDED_MESH_OFFLINE_V11", status="UNAVAILABLE", phase="WHOLE_PART_REGISTRATION",
        reasons=["SOURCE_ONLY_INJECTED_REFUSAL"], predictions=[], registration=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED")


def test_returned_nominal_refusal_is_complete_record_but_failed_output_not_sentinel():
    result = _source_only_refusal()
    record = dict(diagnostic_record_complete=False, positive_geometry_passed=False,
                  scheduled_numerical_work_complete=False)
    _record_result(record, result, None, None, None, b"source-only", b"source-only", noisy=False)
    assert record["workflow"] is result and record["diagnostic_record_complete"] is True
    assert record["returned_prediction_count"] == 0
    assert record["output_coverage"] == "FAILED_NOMINAL_NO_PREDICTIONS"
    assert record["positive_geometry_passed"] is False and record["scheduled_numerical_work_complete"] is False
    assert record["stages"]["registration"] == {"evidence": "MISSING_OR_NULL_UNKNOWN"}
    assert "refined_scan_rows" not in record["stages"]["registration_evidence"]
    assert "sentinel_disposition" not in record


@pytest.mark.parametrize("fault", ("count_claim", "production", "nonfinite", "malformed_predictions", "wrong_schema"))
def test_malformed_or_promoted_return_cannot_complete_diagnostic_contract(fault):
    result = _source_only_refusal()
    if fault == "count_claim":
        result["observed_bend_count"] = 0
    elif fault == "production":
        result["production_use_allowed"] = True
    elif fault == "nonfinite":
        result["diagnostic"] = float("nan")
    elif fault == "malformed_predictions":
        result["predictions"] = {}
    else:
        result["schema"] = "LEGACY"
    with pytest.raises((AssertionError, ValueError)):
        _terminal_return(result)


@pytest.mark.parametrize("fault", ("none", "case_missing", "case_duplicate", "artifact_missing", "source_drift", "oversize"))
def test_complete_negative_diagnostic_never_claims_positive_matrix(fault):
    records = [dict(case=case, diagnostic_record_complete=True, positive_geometry_passed=False,
        artifact={"status": "COMPLETE_CASE_RECORD_RETAINED", "bytes": 10}) for case, _ in CASES]
    before, after = {"source": "unchanged"}, {"source": "unchanged"}
    if fault == "case_missing":
        records.pop()
    elif fault == "case_duplicate":
        records[1]["case"] = records[0]["case"]
    elif fault == "artifact_missing":
        records[1].pop("artifact")
    elif fault == "source_drift":
        after["source"] = "changed"
    elif fault == "oversize":
        records[1]["artifact"]["bytes"] = ARTIFACT_LIMITS["total_case_json_bytes"]
    receipt = _receipt(records, before, after, "SOURCE_ONLY_NO_FILES")
    assert receipt["diagnostic_completed"] is (fault == "none")
    assert receipt["positive_geometry_matrix_passed"] is False
    assert receipt["coverage_limitations"]["effective_refine_8192_exercised"] is False


def test_two_artifact_slots_retain_failures_exclusively_under_original_per_case_cap(tmp_path):
    assert ARTIFACT_LIMITS["per_case_json_bytes"] == _connected().ARTIFACT_LIMITS["per_case_json_bytes"]
    assert ARTIFACT_LIMITS["total_case_json_bytes"] == 2*ARTIFACT_LIMITS["per_case_json_bytes"]
    directory = _connected()._v4()._new_artifact_directory(tmp_path/"ARTIFACTS")
    for index, (case, _) in enumerate(CASES):
        record = dict(case=case, positive_geometry_passed=False, diagnostic_record_complete=True)
        artifact = _write_artifact(directory, index, record)
        assert json.loads((directory/artifact["relative_path"]).read_bytes()) == record
        with pytest.raises(FileExistsError):
            _write_artifact(directory, index, {"positive_geometry_passed": True})


def test_constructed_row_capacity_and_new_outer_limits_are_explicit():
    assert COVERAGE_LIMITATIONS == dict(original_raw_rows=2752, requested_coarse_rows=2048,
        requested_refine_rows=8192, maximum_effective_refine_rows=2752, effective_refine_8192_exercised=False,
        third_normal_cap_width_mm=.8, physical_scanner_bound_established=False,
        physical_or_unseen_generalization=False,
        scope="ORIGINAL_THIN_PART_ONLY_NOT_DENSE_OR_MULTI_BEND_QUALIFICATION")
    assert RESOURCE_LIMITS == dict(wall_seconds=300, cpu_seconds=240,
        process_tree_rss_bytes=4*1024**3, stdout_bytes=1024**2)
    assert [noisy for _, noisy in CASES] == [False, True]


def test_pair_defaults_are_explicit_fixed_constructor_values_not_runtime_overrides(monkeypatch):
    real_import = builtins.__import__
    def guard(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    from backend.cad_guided_two_plane_schedule_v1 import TwoPlaneScheduleBudgetsV1
    from backend.cad_guided_two_plane_context_v1 import TwoPlaneContextBudgetsV1
    assert PAIR_DEFAULTS == dict(scheduler=asdict(TwoPlaneScheduleBudgetsV1()),
                                 context=asdict(TwoPlaneContextBudgetsV1()))
    tree = ast.parse(INITIALIZER_SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_two_plane_records")
    for name in ("schedule_two_plane_pairs_v1", "propose_two_plane_context_v1"):
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == name]
        assert len(calls) == 1
        assert "budgets" not in [key.arg for key in calls[0].keywords]
    own = ast.parse(Path(__file__).read_text())
    remaps = [node for node in ast.walk(own) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "FunctionType"]
    assert len(remaps) == 1
    assert ast.unparse(remaps[0].args[0]) == "_free_circle_nominal_assertions.__code__"
    oracle = next(node for node in own.body if isinstance(node, ast.FunctionDef) and node.name == "_nominal_oracle")
    assert remaps[0] in list(ast.walk(oracle))  # Only criteria adaptation, never version masquerading.


def _mock_joint_ledger(*, proposed=False):
    """Invented REPORT-only identity fixture, deliberately not geometric evidence."""
    digests = {name+"_sha256": hashlib.sha256(name.encode()).hexdigest() for name in (
        "scan_points", "scan_triangle_raw_rows", "cad_triangles", "cad_face_ids", "cad_source_triangle_ids")}
    matches = [dict(ordered_match_index=i, scan_pair_id=0, scan_plane_ids=[1, 2],
        cad_plane_ids=target, canonical_cad_pair_index=0, reserved_hypothesis_records=32,
        reserved_unsigned_assignments=1) for i, target in enumerate(([10, 11], [11, 10]))]
    hypotheses, families, contributions = [], [], []
    for ai, match in enumerate(matches):
        for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            fi = len(families)
            correspondence = [dict(scan_plane_id=a, cad_plane_id=b, normal_sign=sign)
                for a, b, sign in zip(match["scan_plane_ids"], match["cad_plane_ids"], signs)]
            slots = []
            for name in CONTEXT_SLOTS:
                disposition = "PROPOSED_SEED_ONLY" if proposed else "PAIR_REJECTED"
                slots.append(dict(slot=name, status=disposition, candidate_index=0 if proposed else None,
                                  lambda_mm=0. if proposed else None))
                if proposed:
                    contributions.append(dict(signed_family_id=fi, slot=name))
                hypotheses.append(dict(hypothesis_index=len(hypotheses), proposal_lane="TWO_PLANE_CONTEXT",
                    signed_family_id=fi, axial_slot=name, unsigned_assignment_id=ai,
                    correspondences=copy.deepcopy(correspondence), context_slot_disposition=disposition,
                    context_candidate_index=0 if proposed else None, status="PROPOSED" if proposed else "REJECTED",
                    rotation=np.eye(3).tolist() if proposed else None,
                    translation_mm=[0., 0., 0.] if proposed else None))
            families.append(dict(signed_family_id=fi, ordered_assignment_id=ai, normal_signs=list(signs),
                correspondences=correspondence, slots=slots, status="AVAILABLE" if proposed else "REJECTED"))
    helper = dict(status="COMPLETE", selected_policy_evaluation_complete=True,
        accepted_registration=False, finite_support_match_verified=False,
        search_exhaustive=False, global_pose_uniqueness="NOT_ESTABLISHED",
        budgets=copy.deepcopy(PAIR_DEFAULTS["context"]),
        input_digests={a+"_sha256": digests[b+"_sha256"] for a, b in (
            ("scan_points", "scan_points"), ("scan_triangles", "scan_triangle_raw_rows"),
            ("cad_triangles", "cad_triangles"), ("cad_face_ids", "cad_face_ids"),
            ("cad_source_triangle_ids", "cad_source_triangle_ids"))},
        statistics=dict(signed_families_completed=8, signed_families_planned=8,
            slot_records_completed=64, candidate_slots_planned=64), families=families,
        candidates=([dict(candidate_index=0, rotation=np.eye(3).tolist(), translation_mm=[0., 0., 0.],
            contributing_slots=contributions, accepted_registration=False)] if proposed else []))
    return dict(input_digests=digests, hypotheses=hypotheses, two_plane_context=helper,
        joint_proposal_accounting=dict(triplet_records=0, triplet_unsigned_reservation=0,
            ordered_pair_matches=2, pair_slot_reservation=64, total_hypothesis_records=64, pair_lane_enabled=True),
        two_plane_schedule=dict(schema_version="two_plane_schedule_v1", status="COMPLETE",
            selected_policy_evaluation_complete=True, accepted_registration=False, search_exhaustive=False,
            observed_bend_count=None, production_use_allowed=False, budgets=copy.deepcopy(PAIR_DEFAULTS["scheduler"]),
            scheduled_matches=matches, joint_budget=dict(retained_triplet_hypothesis_count=0,
                retained_triplet_unsigned_count=0, joint_hypothesis_records_reserved=64,
                joint_unsigned_assignments_reserved=2)))


@pytest.mark.parametrize("proposed", (False, True))
def test_joint_report_identity_control_retains_all_64_slots_without_geometry_claim(proposed):
    result = _mock_joint_ledger(proposed=proposed)
    joint = _joint_completion(result)
    assert joint["total_hypothesis_records"] == 64
    assert result["two_plane_context"]["accepted_registration"] is False
    assert len(result["hypotheses"]) == 64


@pytest.mark.parametrize("fault", ("disabled", "schedule_schema", "schedule_incomplete", "one_order",
    "missing_slot", "duplicate_sign", "missing_origin", "stale_buffers", "missing_hypothesis",
    "third_correspondence", "false_count", "helper_budget", "scheduler_budget", "absent_with_pose"))
def test_joint_lane_report_failures_never_qualify_complete_numerical_work(fault):
    result = _mock_joint_ledger(proposed=True)
    schedule, helper = result["two_plane_schedule"], result["two_plane_context"]
    if fault == "disabled": result["joint_proposal_accounting"]["pair_lane_enabled"] = False
    elif fault == "schedule_schema": schedule["schema_version"] = "LEGACY"
    elif fault == "schedule_incomplete": schedule["selected_policy_evaluation_complete"] = False
    elif fault == "one_order": schedule["scheduled_matches"].pop()
    elif fault == "missing_slot": helper["families"][0]["slots"].pop()
    elif fault == "duplicate_sign": helper["families"][1]["normal_signs"] = [-1, -1]
    elif fault == "missing_origin": helper["candidates"][0]["contributing_slots"].pop()
    elif fault == "stale_buffers": helper["input_digests"]["scan_points_sha256"] = "0"*64
    elif fault == "missing_hypothesis": result["hypotheses"].pop()
    elif fault == "third_correspondence": helper["families"][0]["correspondences"].append(dict(scan_plane_id=3))
    elif fault == "false_count": schedule["observed_bend_count"] = 0
    elif fault == "helper_budget": helper["budgets"]["max_candidate_slots"] += 1
    elif fault == "scheduler_budget": schedule["budgets"]["max_selected_scan_pairs"] += 1
    elif fault == "absent_with_pose":
        slot, row = helper["families"][0]["slots"][0], result["hypotheses"][0]
        slot["status"] = row["context_slot_disposition"] = "ABSENT_EMPTY_HEURISTIC_INTERVAL"
        row["status"] = "ABSENT_CONTEXT_SLOT"
    with pytest.raises((AssertionError, KeyError, IndexError)):
        _joint_completion(result)


@pytest.mark.parametrize("fault", ("none", "numerical_first", "numerical_second", "phase", "measurement_incomplete"))
def test_recording_path_cannot_promote_late_or_incomplete_stage_with_mocked_geometry(monkeypatch, fault):
    result = _source_only_refusal()
    result.update(status="PARTIAL", phase="COMPLETE", predictions=[{}], registration={},
                  measurements={"computation_status": "COMPLETE"})
    record = dict(diagnostic_record_complete=False, positive_geometry_passed=False,
                  scheduled_numerical_work_complete=False)
    calls = []
    def numerical(*args, **kwargs):
        calls.append(kwargs["allow_prefilter_abstention"])
        if (fault == "numerical_first" and len(calls) == 1) or (fault == "numerical_second" and len(calls) == 2):
            raise AssertionError("Explicit report-only numerical failure")
        return "REPORT_ONLY_COMPLETE"
    monkeypatch.setitem(globals(), "_execution_complete", numerical)
    monkeypatch.setitem(globals(), "_binding_assertions", lambda *a: None)
    monkeypatch.setitem(globals(), "_nominal_oracle", lambda noisy: lambda *a: {})
    if fault == "phase": result["phase"] = "FULL_RAW_LOCAL_MEASUREMENT"
    if fault == "measurement_incomplete": result["measurements"]["computation_status"] = "UNAVAILABLE"
    _record_result(record, result, None, None, None, b"invented", b"invented", noisy=False)
    assert record["diagnostic_record_complete"] is True
    assert record["positive_geometry_passed"] is (fault == "none")
    assert record["workflow"] is result


def _mock_v11_execution(branch):
    """Explicit report-only V11 extension of frozen V10 count/box mocks.

    No function globals, inference, pose calculation or raw geometry is patched.
    This tests the recorder's empty-pair branch; native success requires actual
    V11 dispatch, independent measurement/oracles and all returned pair records.
    """
    result = _box_evidence()._mock_completed_sentinel(branch)
    registration = result["registration"]
    evidence, initialization = registration["evidence"], registration["evidence"]["initialization"]
    registration["schema_version"] = "noisy_plane_initialized_finite_registration_v11"
    evidence["schema"] = "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V11"
    initialization["schema_version"] = "noisy_plane_initialization_prefilter_v11"
    rows = initialization["hypotheses"]
    for row in rows:
        row["proposal_lane"] = "TRIPLET"
        for index, support in enumerate(row["assigned_support"]):
            support.update(scan_plane_id=index, normal_sign=1)
        row["correspondences"] = [{key: support[key] for key in ("scan_plane_id", "cad_plane_id", "normal_sign")}
                                  for support in row["assigned_support"]]
    h3, u3 = len(rows), max(row["unsigned_assignment_id"] for row in rows)+1
    empty = _mock_joint_ledger()
    schedule = empty["two_plane_schedule"]
    schedule["scheduled_matches"] = []
    schedule["joint_budget"] = dict(retained_triplet_hypothesis_count=h3, retained_triplet_unsigned_count=u3,
        joint_hypothesis_records_reserved=h3, joint_unsigned_assignments_reserved=u3)
    initialization.update(two_plane_schedule=schedule,
        two_plane_context=dict(status="NOT_CALLED_EMPTY_DECLARED_SCHEDULE", usable_proposals=False,
            accepted_registration=False), joint_proposal_accounting=dict(triplet_records=h3,
            triplet_unsigned_reservation=u3, ordered_pair_matches=0, pair_slot_reservation=0,
            total_hypothesis_records=h3, pair_lane_enabled=True))
    initialization["statistics"]["absent_context_slot_count"] = 0
    return registration


@pytest.mark.parametrize("branch", ("prefilter", "registration", "local_row"))
def test_explicit_v11_full_stage_checker_handles_completed_report_branches(branch):
    registration = _mock_v11_execution(branch)
    disposition = _execution_complete(registration, allow_prefilter_abstention=True)
    assert disposition == ("PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE" if branch == "prefilter"
                            else "ALL_SELECTED_PLANE_START_GROUPS_FULLY_EVALUATED")
    if branch == "prefilter":
        with pytest.raises(AssertionError):
            _execution_complete(registration, allow_prefilter_abstention=False)


@pytest.mark.parametrize("fault", ("old_schema", "numerical_incomplete", "partial_box", "dropped_group_origin",
    "candidate_incomplete", "lost_support", "wrong_selected_pose"))
def test_v11_full_stage_checker_rejects_incomplete_or_unbound_report(fault):
    registration = _mock_v11_execution("local_row")
    evidence, initialization = registration["evidence"], registration["evidence"]["initialization"]
    if fault == "old_schema": registration["schema_version"] = "noisy_plane_initialized_finite_registration_v9"
    elif fault == "numerical_incomplete": evidence["numerical_consumer_complete"] = False
    elif fault == "partial_box": initialization["statistics"]["completed_box_screen_rows"] -= 1
    elif fault == "dropped_group_origin": initialization["all_passing_start_groups"][0]["hypothesis_indices"] = []
    elif fault == "candidate_incomplete": evidence["candidates"][0]["execution_status"] = "NOT_ATTEMPTED"
    elif fault == "lost_support": initialization["hypotheses"][0]["assigned_support"].pop()
    elif fault == "wrong_selected_pose": registration["transform_scan_to_cad"][0][3] += 1.
    with pytest.raises((AssertionError, KeyError, IndexError)):
        _execution_complete(registration, allow_prefilter_abstention=True)


@pytest.mark.parametrize("lane", ("TRIPLET", "TWO_PLANE_CONTEXT"))
def test_origin_attribution_is_separate_from_output_correctness(lane):
    # Explicit identity-only recorder fixture, not a registration/measurement.
    support = [dict(status="PASSED") for _ in range(3 if lane == "TRIPLET" else 2)]
    registration = dict(evidence=dict(selected_candidate_index=0,
        candidates=[dict(hypothesis_indices=[0])], initialization=dict(hypotheses=[
            dict(proposal_lane=lane, prefilter_status="PASSED", assigned_support=support)])))
    assert _two_plane_origin_attribution(registration) is (lane == "TWO_PLANE_CONTEXT")
    records = [dict(case=case, diagnostic_record_complete=True, positive_geometry_passed=True,
        two_plane_origin_recovery_passed=(lane == "TWO_PLANE_CONTEXT"),
        artifact=dict(status="COMPLETE_CASE_RECORD_RETAINED", bytes=10)) for case, _ in CASES]
    receipt = _receipt(records, {"source": "same"}, {"source": "same"}, "INVENTED_NO_FILES")
    assert receipt["positive_geometry_matrix_passed"] is True
    assert receipt["two_plane_origin_recovery_matrix_passed"] is (lane == "TWO_PLANE_CONTEXT")


def test_native_diagnostic_records_two_nominal_outcomes_not_positive_qualification():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Separate root-frozen native diagnostic; source-only collection is not permission")
    directory = os.environ.get(ARTIFACT_DIRECTORY_ENV)
    assert directory, "Root must supply a fresh retained-job ARTIFACTS child"
    receipt = run_invented_physical_config_constructed_v11_diagnostic(artifact_directory=directory)
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) <= RESOURCE_LIMITS["stdout_bytes"]
    print(encoded, flush=True)
    # This one test qualifies recording ONLY. Read the separate positive-geometry
    # ledger even if pytest is green; a nominal refusal is never positive output.
    assert receipt["diagnostic_completed"] is True
    assert len(receipt["cases"]) == 2
    assert type(receipt["positive_geometry_matrix_passed"]) is bool
