"""Unqualified additive V13 byte workflow with exact-support-cover measurement.

Explicit connected-only policy. V12 registration/input/feature custody is reused;
optional composite_v1 scan proposals are explicitly configured and byte-bound.
The measurement projection uses V2. Nested predecessor predictions
are evidence, never concatenated into the top-level prediction collection.
No default product routing, native qualification or physical release is implied.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from . import cad_guided_mesh_offline_v12 as _v12

_v2 = _v12._v2
_stages = _v12._stages
MAX_WORKFLOW_REPORT_BYTES = _v12.MAX_WORKFLOW_REPORT_BYTES
_triangle_scan_binding = _v12._triangle_scan_binding
_registration_selection = _v12._registration_selection
_registration_binding = _v12._registration_binding


@dataclass(frozen=True)
class MeshWorkflowPolicyV13(_v2.MeshWorkflowPolicyV2):
    """Unchanged V2 policy fields, distinct explicit cover-path type."""


def _bounded_workflow(output, maximum=MAX_WORKFLOW_REPORT_BYTES):
    size = 0
    try:
        _v2._require(type(maximum) is int and 0 <= maximum <= MAX_WORKFLOW_REPORT_BYTES,
                     "INVALID_WORKFLOW_REPORT_BUDGET")
        for piece in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(output):
            size += len(piece.encode("utf-8"))
            _v2._require(size <= maximum, "WORKFLOW_REPORT_BUDGET_EXCEEDED")
    except (ValueError, TypeError, OverflowError, MemoryError):
        # No successful prefix or malformed later measurement survives. The
        # small explicit refusal envelope is outside the detailed-report cap.
        return dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status="UNAVAILABLE",
            phase="REPORT_PUBLICATION", reasons=["WORKFLOW_DETAILED_REPORT_UNAVAILABLE"],
            predictions=[], measurements=None, registration=None, observed_bend_count=None,
            production_use_allowed=False, conformance="INDETERMINATE",
            detailed_report_discarded_whole=True, detailed_report_limit_bytes=maximum,
            encoded_bytes_examined=size)
    return output


def inspect_cad_guided_mesh_v13(*, scan_bytes, cad_bytes, position_error_bound_mm, workflow_policy,
        registration_policy, measurement_policy, instance_policy,
        noisy_policy=None, noisy_budgets=None, cad_patch_policy=None, cad_patch_budgets=None,
        search_policy=None, search_budgets=None, initialization_budgets=None,
        surface_budgets=None, registration_budgets=None, allow_ocp=False, two_plane_enabled=True,
        connected_measurement_contexts=True, scan_plane_mode="v7", seeded_plane_budgets=None,
        composite_plane_budgets=None, two_plane_schedule_mode="v1", two_plane_schedule_budgets=None):
    """Byte/native workflow with conditional contexts, never an accepted pose.

    Complete selected-context binding precedes fresh support-cover measurement.
    Position bounds guide initialization and context surface-role exclusions;
    they are not scanner calibration or propagated measurement uncertainty.
    """
    output = dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status="UNAVAILABLE",
        phase="ADMISSION", reasons=[], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        registration_mode="FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE", datum_status="NO_VERIFIED_STABLE_DATUM",
        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
        registration_initialization="TRIPLET_AND_TWO_PLANE_CONTEXT_FINITE_PREFILTER_V12",
        measurement_geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
    if connected_measurement_contexts is not True:
        return {**output, "reasons": ["V13_CONNECTED_CONTEXTS_REQUIRED"]}
    output["connected_measurement_contexts"] = connected_measurement_contexts
    if type(two_plane_enabled) is not bool:
        return {**output, "reasons": ["INVALID_TWO_PLANE_LANE_POLICY"]}
    if type(allow_ocp) is not bool or not allow_ocp:
        return {**output, "reasons": ["EXPLICIT_OCP_OPT_IN_REQUIRED"]}
    if type(workflow_policy) is not MeshWorkflowPolicyV13:
        return {**output, "reasons": ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]}
    if type(cad_bytes) is not bytes or not 1 <= len(cad_bytes) <= 8 * 1024**2:
        return {**output, "reasons": ["CAD_BYTE_BUDGET"]}
    if type(scan_bytes) is not bytes or not 1 <= len(scan_bytes) <= 64 * 1024**2:
        return {**output, "reasons": ["SCAN_BYTE_BUDGET"]}
    if type(position_error_bound_mm) not in (int, float) or not 0 <= position_error_bound_mm <= 1000.:
        return {**output, "reasons": ["EXPLICIT_FINITE_POSITION_ERROR_BOUND_REQUIRED"]}
    position_error_bound_mm = float(position_error_bound_mm)
    output["position_error_model"] = dict(uniform_euclidean_bound_mm=position_error_bound_mm,
        status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        scope="PLANE_INITIALIZATION_AND_CONTEXT_SURFACE_ROLE_POLICY_NOT_MEASUREMENT_UNCERTAINTY")
    output["phase"] = "INPUT_PREPARATION"
    try:
        from .cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from .cad_guided_mesh_topology_v4 import parse_triangle_mesh_ply_v4
        from . import cad_guided_plane_initialization_v12 as initialize
        from .cad_guided_native_features import NativeGuidedPolicy
        from .cad_guided_instances import SupportPolicy

        _v2._require(type(measurement_policy) is NativeGuidedPolicy
                     and type(instance_policy) is SupportPolicy, "VALIDATED_INFERENCE_POLICIES_REQUIRED")
        _v2._require(measurement_policy.fit_model == "free_circle",
                     "CONNECTED_CONTEXT_FREE_CIRCLE_REQUIRED")
        workflow_policy = MeshWorkflowPolicyV13(**asdict(workflow_policy))
        measurement_policy = NativeGuidedPolicy(**asdict(measurement_policy))
        instance_policy = SupportPolicy(**asdict(instance_policy))
        schedule_request = {} if (type(two_plane_schedule_mode) is str and two_plane_schedule_mode == "v1"
            and two_plane_schedule_budgets is None) else dict(two_plane_schedule_mode=two_plane_schedule_mode,
                two_plane_schedule_budgets=two_plane_schedule_budgets)
        config = initialize._configuration(registration_policy,
            initialize._noisy.NoisyMeshPlanePolicyV5() if noisy_policy is None else noisy_policy,
            initialize._noisy.NoisyMeshPlaneBudgetsV5() if noisy_budgets is None else noisy_budgets,
            initialize._patch.CoplanarPatchPolicyV4() if cad_patch_policy is None else cad_patch_policy,
            initialize._patch.CoplanarPatchBudgetsV4() if cad_patch_budgets is None else cad_patch_budgets,
            initialize._search.PlaneSearchPolicyV6() if search_policy is None else search_policy,
            initialize._search.PlaneSearchBudgetsV6() if search_budgets is None else search_budgets,
            initialize.PlaneInitializationBudgetsV12() if initialization_budgets is None else initialization_budgets,
            surface_budgets, two_plane_enabled, scan_plane_mode, seeded_plane_budgets, composite_plane_budgets,
            **schedule_request)
        scan_plane_arguments = {} if config["scan_plane_mode"] == "v7" else dict(
            scan_plane_mode=config["scan_plane_mode"], seeded_plane_budgets=config["seeded_plane_budgets"],
            composite_plane_budgets=config["composite_plane_budgets"])
        schedule_arguments = {} if config.get("two_plane_schedule_mode", "v1") == "v1" else dict(
            two_plane_schedule_mode="v2", two_plane_schedule_budgets=config["two_plane_schedule_budgets"])
        registration_policy = config["policy"]
        registration_budgets = initialize._v3._validated_execution_budgets(
            initialize._v3.SurfaceRegistrationBudgetsV3() if registration_budgets is None else registration_budgets)
        registration_budgets = initialize._v3.SurfaceRegistrationBudgetsV3(**asdict(registration_budgets))
        output["policies"] = dict(workflow=asdict(workflow_policy), registration=asdict(registration_policy),
            measurement=asdict(measurement_policy), support=asdict(instance_policy),
            noisy=asdict(config["noisy_policy"]), noisy_budgets=asdict(config["noisy_budgets"]),
            cad_patch=asdict(config["cad_patch_policy"]), cad_patch_budgets=asdict(config["cad_patch_budgets"]),
            search=asdict(config["search_policy"]), search_budgets=asdict(config["search_budgets"]),
            initialization_budgets=asdict(config["budgets"]), surface_budgets=asdict(config["surface_budgets"]),
            registration_budgets=asdict(registration_budgets), two_plane_enabled=two_plane_enabled,
            stage_policies=_stages.ablation_stage_policies_v1(connected=connected_measurement_contexts))
        if scan_plane_arguments:
            output["policies"]["scan_plane_configuration"] = _v12._scan_plane_configuration_snapshot(
                config["scan_plane_configuration"])[0]
        if schedule_arguments:
            output["policies"]["two_plane_schedule_configuration"] = _v12._two_plane_schedule_configuration_snapshot(
                config["two_plane_schedule_configuration"])[0]
        scan = prepare_mesh_scan_v1(scan_bytes=scan_bytes,
                                    max_engine_rows=workflow_policy.max_registration_scan_rows)
        output["input_preparation"] = {key: scan[key] for key in (
            "status", "reasons", "source_bindings", "coverage", "measurement_cloud_contract",
            "sample_cloud_contract", "units_status", "coordinate_frame_status")}
        if scan["status"] != "AVAILABLE":
            return _bounded_workflow({**output, "reasons": ["MESH_INPUT_UNAVAILABLE", *scan["reasons"]]})
        _v2._scan_binding(scan, scan_bytes, workflow_policy.max_registration_scan_rows)
        topology = parse_triangle_mesh_ply_v4(scan_bytes)
        output["scan_topology"] = {key: value for key, value in topology.items()
                                   if key not in ("points", "triangle_raw_rows")}
        if topology["status"] != "AVAILABLE":
            return _bounded_workflow({**output, "reasons": ["TRIANGLE_SCAN_UNAVAILABLE", *topology["reasons"]]})
        output["scan_triangle_binding"] = _triangle_scan_binding(scan, topology, scan_bytes)
        selection = _registration_selection(scan)
    except Exception as error:
        return _bounded_workflow({**output, "reasons": ["INPUT_PREPARATION_EXCEPTION:"+type(error).__name__]})
    output["cad_sha256"] = hashlib.sha256(cad_bytes).hexdigest()
    try:
        output["phase"] = "NATIVE_IMPORT"
        shape, receipt = _v2._read_step_owned_snapshot(cad_bytes)
        _v2._require(receipt["target_unit"] == "mm"
                     and receipt["physical_calibration"] == "NOT_ESTABLISHED", "INVALID_NATIVE_UNIT_RECEIPT")
        output["cad_units"] = receipt
        output["phase"] = "NATIVE_FEATURES"
        if workflow_policy.native_adapter_version == "r0":
            from .cad_guided_native_features import extract_ocp_guided_features as extract
        else:
            from .cad_guided_native_features_v1 import extract_ocp_guided_features_v1 as extract
        features = extract(shape, policy=measurement_policy, unit_status="DECLARED_MM", allow_ocp=True)
        output["native_features"] = features["metadata"]
        output["phase"] = "CAD_SURFACE_SAMPLING"
        from .cad_guided_cad_surface_v2 import sample_native_cad_surface_v2
        cad = sample_native_cad_surface_v2(shape, allow_ocp=True, native_binding="ocp",
            sample_count=workflow_policy.cad_sample_count,
            linear_deflection_mm=workflow_policy.cad_linear_deflection_mm,
            angular_deflection_radians=workflow_policy.cad_angular_deflection_radians)
        array_keys = {"triangles", "source_face_indices", "source_triangle_indices", "points", "normals",
                      "sample_triangle_rows", "sample_source_face_indices",
                      "sample_source_triangle_indices", "sample_barycentric_weights"}
        output["cad_surface_sampling"] = {key: value for key, value in cad.items() if key not in array_keys}
        if cad["status"] != "AVAILABLE":
            return _bounded_workflow({**output, "reasons": ["CAD_SURFACE_SAMPLING_UNAVAILABLE", *cad["reasons"]]})
        output["triangle_sample_binding"] = _v2._surface_binding(cad, workflow_policy.cad_sample_count)
        output["phase"] = "WHOLE_PART_REGISTRATION"
        registration, owned_surface_roles = initialize.register_measurement_contexts_v12(
            scan["raw_points"], topology["triangle_raw_rows"],
            cad["triangles"], cad["source_face_indices"], cad["source_triangle_indices"],
            position_error_bounds_mm=position_error_bound_mm,
            cad_initialization_points=cad["points"], registration_policy=registration_policy,
            registration_raw_rows=selection, unit_assumption=workflow_policy.unit_assumption,
            noisy_policy=config["noisy_policy"], noisy_budgets=config["noisy_budgets"],
            cad_patch_policy=config["cad_patch_policy"], cad_patch_budgets=config["cad_patch_budgets"],
            search_policy=config["search_policy"], search_budgets=config["search_budgets"],
            budgets=config["budgets"], surface_budgets=config["surface_budgets"],
            registration_budgets=registration_budgets, two_plane_enabled=two_plane_enabled,
            connected_measurement_contexts=connected_measurement_contexts, **scan_plane_arguments, **schedule_arguments)
        output["registration"] = registration
        if registration["status"] != "CONDITIONAL_MEASUREMENT_CONTEXTS":
            return _bounded_workflow({**output, "reasons": ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]})
        try:
            _v2._require(registration["evidence"]["initialization"]["joint_proposal_accounting"]["pair_lane_enabled"]
                is two_plane_enabled, "REGISTRATION_PAIR_LANE_POLICY_MISMATCH")
            indices = registration["evidence"]["measurement_context_indices"]
            _v2._require(type(indices) is list and indices and len(indices) <= 48
                and len(set(indices)) == len(indices), "INVALID_CONTEXT_INDEX_LEDGER")
            scan_binding_arguments = {}
            if schedule_arguments:
                published_budget = registration["evidence"]["initialization"]["budgets"]
                requested_budget = asdict(config["budgets"])
                _v2._require(type(published_budget) is dict and set(published_budget) == set(requested_budget)
                    and all(type(published_budget[key]) is int and published_budget[key] == value
                        for key, value in requested_budget.items()),
                    "V2_SCHEDULE_INITIALIZATION_BUDGET_REQUEST_MISMATCH")
                scan_binding_arguments["expected_two_plane_schedule_configuration"] = (
                    output["policies"]["two_plane_schedule_configuration"])
            if scan_plane_arguments:
                expected = config["scan_plane_configuration"]
                scan_binding_arguments.update(expected_scan_plane_configuration=expected,
                    _prepared_scan_plane_binding=_v12._scan_plane_binding(registration, scan, topology,
                        position_error_bound_mm, expected))
            output["registration_array_binding"] = [
                _registration_binding(registration, scan, topology, cad, selection,
                    position_error_bound_mm, context_index=index, **scan_binding_arguments) for index in indices]
        except Exception:
            output["registration"] = dict(registration, status="UNAVAILABLE", transform_scan_to_cad=None,
                                            reasons=["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"])
            if schedule_arguments or _v12._has_two_plane_schedule_v2_publication(registration["evidence"]):
                attempted = registration["evidence"].get("measurement_context_indices", [])
                output["registration"]["evidence"] = dict(registration["evidence"],
                    attempted_measurement_context_indices=attempted, measurement_context_indices=[],
                    evidence_scope="ATTEMPTED_WORK_ONLY_INVALID_V2_SCHEDULE_BINDING")
            raise
        output["phase"] = "NATIVE_FEATURES"
        if features["status"] not in ("AVAILABLE", "PARTIAL") or not features["features"]:
            return _bounded_workflow({**output, "reasons": ["NATIVE_FEATURES_UNAVAILABLE"]})
        output["phase"] = "FULL_RAW_LOCAL_MEASUREMENT"
        from .cad_guided_support_cover_measurement_v2 import (
            inspect_support_cover_context_windows_v2, validate_support_cover_publication_v2)
        measured = inspect_support_cover_context_windows_v2(scan["raw_points"], features["features"],
            registration=registration, scan_sha256=scan["source_bindings"]["scan_sha256"],
            instance_policy=instance_policy, surface_roles=owned_surface_roles)
        output["support_cover_publication_validation"] = validate_support_cover_publication_v2(
            measured, raw_points=scan["raw_points"], scan_sha256=scan["source_bindings"]["scan_sha256"])
        output.update(status=measured["status"],
            phase="MEASUREMENT_REFUSAL" if measured.get("computation_status") != "COMPLETE" else "COMPLETE",
            measurements=measured, predictions=measured["predictions"], reasons=measured["reasons"],
            prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS")
    except Exception as error:
        output.update(status="UNAVAILABLE", measurements=None, predictions=[],
                      reasons=["WORKFLOW_EXCEPTION:"+type(error).__name__])
    return _bounded_workflow(output)
