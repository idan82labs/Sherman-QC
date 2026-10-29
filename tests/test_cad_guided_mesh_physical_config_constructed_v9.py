"""Draft thin-part nominal diagnostic at the exact physical-run CONFIG.

Two original V1 nominal cases (clean and unchanged bounded-3D-noise)
use actual V9 byte dispatch. No missing-arc case, thickened cap, alternate pose,
additional samples, changed error ceiling or supplied answer is permitted.
A completed diagnostic record is NOT positive geometry: nominal abstention is
failed output coverage, never a passing safety sentinel. Both ledgers persist.

The source-only copy of CONFIG is compared structurally with the named frozen
runner without importing or invoking that physical-data program. Only the V9
types and explicit certificate-pair caps are additive. Free-circle evaluation
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
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V9"
ARTIFACT_DIRECTORY_ENV = "SHERMAN_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V9_ARTIFACT_DIRECTORY"
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
    spec = importlib.util.spec_from_file_location("_physical_config_v9_helpers", CONNECTED_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original():
    return _connected()._original()


def _oracle_pose():
    return _original()._oracle_pose()


def _noisy_rows(raw):
    return _connected()._noisy_rows(raw)


def _policies():
    from backend.cad_guided_mesh_offline_v9 import MeshWorkflowPolicyV9
    from backend.cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1
    from backend.cad_guided_native_features import NativeGuidedPolicy
    from backend.cad_guided_instances import SupportPolicy
    from backend.cad_guided_noisy_mesh_planes_v5 import NoisyMeshPlanePolicyV5, NoisyMeshPlaneBudgetsV5
    from backend.cad_guided_mesh_plane_proposals_v4 import CoplanarPatchPolicyV4, CoplanarPatchBudgetsV4
    from backend.cad_guided_plane_search_v6 import PlaneSearchPolicyV6, PlaneSearchBudgetsV6
    from backend.cad_guided_plane_initialization_v9 import PlaneInitializationBudgetsV9
    from backend.cad_guided_triangle_surface_v3 import TriangleSurfaceBudgets
    from backend.cad_guided_global_registration_v3 import SurfaceRegistrationBudgetsV3

    config = copy.deepcopy(CONFIG)
    noisy = config["noisy_policy"]
    noisy["tile_triangles"] = tuple(noisy["tile_triangles"])
    return dict(
        workflow_policy=MeshWorkflowPolicyV9(**config["workflow"], native_adapter_version="v1"),
        registration_policy=WholePartRegistrationPolicyV1(**config["registration"]),
        measurement_policy=NativeGuidedPolicy(**config["measurement"]),
        instance_policy=SupportPolicy(**config["support"]),
        noisy_policy=NoisyMeshPlanePolicyV5(**noisy),
        noisy_budgets=NoisyMeshPlaneBudgetsV5(**config["noisy_budgets"]),
        cad_patch_policy=CoplanarPatchPolicyV4(**config["cad_patch_policy"]),
        cad_patch_budgets=CoplanarPatchBudgetsV4(**config["cad_patch_budgets"]),
        search_policy=PlaneSearchPolicyV6(**config["search_policy"]),
        search_budgets=PlaneSearchBudgetsV6(**config["search_budgets"]),
        initialization_budgets=PlaneInitializationBudgetsV9(**config["initialization_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]),
        surface_budgets=TriangleSurfaceBudgets(**config["surface_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["surface_budgets"]),
        registration_budgets=SurfaceRegistrationBudgetsV3(**config["registration_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["registration_budgets"]))


def _policy_snapshot(policies):
    return json.loads(json.dumps({key: asdict(value) for key, value in policies.items()},
                                sort_keys=True, allow_nan=False))


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
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V9"
    assert result["registration_initialization"] == "PLANE_ONLY_FINITE_ASSIGNED_FACE_PREFILTER_V9"
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


def _terminal_return(result):
    # This checks a retained return contract, NOT successful pipeline execution.
    json.dumps(result, allow_nan=False)
    assert type(result) is dict and result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V9"
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
                "numerical_consumer_complete", "statistics") if key in initialization}
    return stages


def _record_result(record, result, raw, faces, cad_frame, cad_blob, scan_blob, *, noisy):
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
        record["numerical_disposition"] = _connected()._execution_complete(
            result["registration"], allow_prefilter_abstention=True)
        record["scheduled_numerical_work_complete"] = True
    except Exception as error:
        record.update(numerical_error_type=type(error).__name__, numerical_error=str(error)[:1000])
    if count != 1:
        record["geometry_oracle_status"] = "FAILED_REQUIRED_ONE_NOMINAL_OUTPUT"
        return
    try:
        _binding_assertions(result, raw, faces, cad_blob, scan_blob)
        _connected()._execution_complete(result["registration"], allow_prefilter_abstention=False)
        record["errors"] = _nominal_oracle(noisy)(result, raw, cad_frame)
        record.update(positive_geometry_passed=True, output_coverage="ONE_MEASURED_OUTPUT_ORIGINAL_GEOMETRY_PASSED",
                      geometry_oracle_status="ORIGINAL_CEILINGS_PASSED_FREE_CIRCLE_LABELS")
    except Exception as error:
        record.update(geometry_oracle_status="FAILED_NOMINAL_GEOMETRY_OR_CONTRACT",
            geometry_error_type=type(error).__name__, geometry_error=str(error)[:1000])


def _source_digests():
    # Fixture/config custody subset only; the supervisor must freeze full closure.
    connected = _connected()
    paths = (Path(__file__).resolve(), CONFIG_SOURCE, CONNECTED_SOURCE, connected.V8_SOURCE,
        connected.V4_SOURCE, connected._v4().SURFACE_SOURCE, connected._v4().DENSE_SOURCE,
        ORIGINAL_SOURCE, ROOT/"backend/cad_guided_mesh_offline_v9.py",
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
    return dict(protocol="CAD_GUIDED_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V9",
        expected_case_count=2, cases=records, diagnostic_completed=complete,
        positive_geometry_matrix_passed=positive,
        interpretation="DIAGNOSTIC_COMPLETION_IS_NOT_POSITIVE_OUTPUT_OR_PHYSICAL_QUALIFICATION",
        config=CONFIG, certificate_caps=CERTIFICATE_CAPS, criteria=CRITERIA,
        noisy_criteria=NOISY_CRITERIA, coverage_limitations=COVERAGE_LIMITATIONS,
        external_resource_limits=RESOURCE_LIMITS, artifact_limits=ARTIFACT_LIMITS,
        artifact_directory=str(directory), fixture_source_sha256_before=before,
        fixture_source_sha256_after=after, observed_bend_count=None, production_use_allowed=False)


def run_invented_physical_config_constructed_v9_diagnostic(*, artifact_directory):
    from backend.cad_guided_mesh_offline_v9 import inspect_cad_guided_mesh_v9

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
                cad_sha256=hashlib.sha256(cad_blob).hexdigest(), scan_sha256=hashlib.sha256(scan_blob).hexdigest(),
                phase="actual_byte_to_v9_physical_config_workflow")
            result = inspect_cad_guided_mesh_v9(scan_bytes=scan_blob, cad_bytes=cad_blob,
                position_error_bound_mm=CONFIG["position_error_bound_mm"], allow_ocp=True, **policies)
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
    assert CONFIG["position_error_bound_mm"] == .05
    assert CONFIG["measurement"]["fit_model"] == "free_circle"
    assert CONFIG["registration"]["coarse_points"] == 2048 and CONFIG["registration"]["refine_points"] == 8192
    tree = ast.parse(Path(__file__).read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and "run_gold11" in ast.unparse(node)
                   for node in ast.walk(tree))


def test_policy_snapshot_adds_only_v9_types_certificate_caps_and_fixed_native_v1(monkeypatch):
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
    assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV9"
    assert type(policies["initialization_budgets"]).__name__ == "PlaneInitializationBudgetsV9"
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
                  and node.name == "run_invented_physical_config_constructed_v9_diagnostic")
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    workflow = [node for node in calls if isinstance(node.func, ast.Name)
                and node.func.id == "inspect_cad_guided_mesh_v9"]
    assert len(workflow) == 1
    assert [key.arg for key in workflow[0].keywords] == [
        "scan_bytes", "cad_bytes", "position_error_bound_mm", "allow_ocp", None]
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
    return dict(schema="CAD_GUIDED_MESH_OFFLINE_V9", status="UNAVAILABLE", phase="WHOLE_PART_REGISTRATION",
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


def test_native_diagnostic_records_two_nominal_outcomes_not_positive_qualification():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Separate root-frozen native diagnostic; source-only collection is not permission")
    directory = os.environ.get(ARTIFACT_DIRECTORY_ENV)
    assert directory, "Root must supply a fresh retained-job ARTIFACTS child"
    receipt = run_invented_physical_config_constructed_v9_diagnostic(artifact_directory=directory)
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) <= RESOURCE_LIMITS["stdout_bytes"]
    print(encoded, flush=True)
    # This one test qualifies recording ONLY. Read the separate positive-geometry
    # ledger even if pytest is green; a nominal refusal is never positive output.
    assert receipt["diagnostic_completed"] is True
    assert len(receipt["cases"]) == 2
    assert type(receipt["positive_geometry_matrix_passed"]) is bool
