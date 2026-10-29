"""Unqualified V11 opt-in byte-to-measurement successor with real pair seeds.

SOURCE-AUTHORED ONLY. V10 byte custody, native feature admission, full-raw
scan measurement and selected-pose publication remain unchanged. V11 adds a
triplet-plus-two-plane initialization lane with explicit joint accounting.
No physical qualification, default activation or production permission.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from . import cad_guided_mesh_offline_v2 as _v2
from . import cad_guided_mesh_offline_v8 as _previous


MAX_WORKFLOW_REPORT_BYTES = 128 * 1024**2


@dataclass(frozen=True)
class MeshWorkflowPolicyV11(_v2.MeshWorkflowPolicyV2):
    """Exactly the unchanged V2 fields/admission, a distinct opt-in type."""


# Frozen pure byte/array custody helpers, no caller-supplied geometry certificates.
_triangle_scan_binding = _previous._triangle_scan_binding
_registration_selection = _previous._registration_selection


def _registration_binding(registration, scan, topology, cad, selection, position_error_bound_mm):
    binding = _previous._registration_binding(
        registration, scan, topology, cad, selection, position_error_bound_mm)
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    _v2._require(registration["schema_version"] == "noisy_plane_initialized_finite_registration_v11"
        and evidence["schema"] == "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V11"
        and initialization["schema_version"] == "noisy_plane_initialization_prefilter_v11"
        and initialization["numerical_consumer_complete"] is True
        and initialization["box_screening_complete"] is True
        and initialization["box_screening_policy"] == "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
        and evidence["numerical_consumer_complete"] is True
        and initialization["numerical_support_policy"] == evidence["numerical_support_policy"]
            == "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
        and evidence["distance_model"] == "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
        and evidence["optimizer_objective"] ==
            "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS",
        "MISSING_BOUND_AWARE_REGISTRATION_CONSUMER_CONTRACT")
    selected = evidence["selected_candidate_index"]
    _v2._require(type(selected) is int and 0 <= selected < len(evidence["candidates"]),
                 "INVALID_BOUND_AWARE_SELECTED_CANDIDATE")
    candidate = evidence["candidates"][selected]
    _v2._require(type(candidate) is dict and type(candidate["index"]) is int
        and candidate["index"] == selected and candidate["execution_status"] == "ATTEMPTED"
        and candidate["reasons"] == []
        and candidate["numerical_consumer_status"] == "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
        and all(candidate[name] == evidence[name] for name in
            ("numerical_support_policy", "distance_model", "optimizer_objective")),
        "UNQUALIFIED_BOUND_AWARE_SELECTED_CANDIDATE")

    joint = initialization["joint_proposal_accounting"]
    schedule = initialization["two_plane_schedule"]
    hypotheses = initialization["hypotheses"]
    _v2._require(type(joint) is dict and type(joint.get("pair_lane_enabled")) is bool
        and all(type(joint.get(key)) is int and 0 <= joint[key] <= maximum for key, maximum in (
            ("triplet_records", 8192), ("triplet_unsigned_reservation", 1024),
            ("ordered_pair_matches", 256), ("pair_slot_reservation", 8192),
            ("total_hypothesis_records", 8192)))
        and type(hypotheses) is list and joint["total_hypothesis_records"] == len(hypotheses)
        and joint["pair_slot_reservation"] == 32*joint["ordered_pair_matches"]
        and joint["triplet_records"]+joint["pair_slot_reservation"] == len(hypotheses),
        "INCOMPLETE_JOINT_PROPOSAL_PUBLICATION")
    _v2._require(type(schedule) is dict and schedule.get("selected_policy_evaluation_complete") is True
        and schedule.get("status") == ("COMPLETE" if joint["pair_lane_enabled"] else "DISABLED_BY_EXPLICIT_POLICY"),
        "INCOMPLETE_TWO_PLANE_SCHEDULE_PUBLICATION")
    if joint["pair_lane_enabled"] and joint["ordered_pair_matches"]:
        helper = initialization["two_plane_context"]
        _v2._require(helper["status"] == "COMPLETE" and helper["selected_policy_evaluation_complete"] is True
            and helper["accepted_registration"] is False
            and helper["finite_support_match_verified"] is False
            and len(schedule["scheduled_matches"]) == joint["ordered_pair_matches"]
            and helper["statistics"]["slot_records_completed"] == joint["pair_slot_reservation"],
            "INCOMPLETE_TWO_PLANE_CONTEXT_PUBLICATION")
        expected_context = {key+"_sha256": initialization["input_digests"][name+"_sha256"]
            for key, name in (("scan_points", "scan_points"),
                ("scan_triangles", "scan_triangle_raw_rows"), ("cad_triangles", "cad_triangles"),
                ("cad_face_ids", "cad_face_ids"), ("cad_source_triangle_ids", "cad_source_triangle_ids"))}
        _v2._require(helper["input_digests"] == expected_context,
                     "PUBLISHED_TWO_PLANE_CONTEXT_BUFFER_MISMATCH")
    selected_groups = initialization["start_groups"]
    all_groups = initialization["all_passing_start_groups"]
    _v2._require(type(selected_groups) is list and len(selected_groups) == len(evidence["candidates"])
        and type(all_groups) is list and selected < len(selected_groups),
        "INCOMPLETE_SELECTED_START_GROUP_LEDGER")
    group = selected_groups[selected]
    gi = candidate["exact_group_index"]
    _v2._require(type(gi) is int and 0 <= gi < len(all_groups)
        and group["index"] == selected and group["exact_group_index"] == gi
        and all_groups[gi]["exact_group_index"] == gi
        and group["refinement_disposition"] == all_groups[gi]["refinement_disposition"]
            == "SCHEDULED_FOR_V3_REFINEMENT"
        and candidate["hypothesis_indices"] == group["hypothesis_indices"] == all_groups[gi]["hypothesis_indices"]
        and all(np.array_equal(candidate[key], group[key]) and np.array_equal(group[key], all_groups[gi][key])
                for key in ("initialized_rotation", "initialized_translation_mm")),
        "SELECTED_CANDIDATE_NOT_COMPLETE_SCHEDULED_GROUP")
    origins = candidate["hypothesis_indices"]
    _v2._require(type(origins) is list and origins and len(set(origins)) == len(origins),
                 "MISSING_SELECTED_START_SUPPORT_ORIGINS")
    for index in origins:
        _v2._require(type(index) is int and 0 <= index < len(hypotheses),
                     "INVALID_SELECTED_START_SUPPORT_ORIGIN")
        origin = hypotheses[index]
        count = {"TRIPLET": 3, "TWO_PLANE_CONTEXT": 2}.get(origin.get("proposal_lane"))
        _v2._require(origin["hypothesis_index"] == index and origin["prefilter_status"] == "PASSED"
            and count is not None and len(origin["assigned_support"]) == count
            and len(origin["correspondences"]) == count
            and all(all(type(a.get(key)) is int and type(b.get(key)) is int and a[key] == b[key]
                    for key in ("scan_plane_id", "cad_plane_id", "normal_sign"))
                for a, b in zip(origin["assigned_support"], origin["correspondences"]))
            and all(row["status"] == "PASSED" for row in origin["assigned_support"])
            and np.array_equal(origin["rotation"], candidate["initialized_rotation"])
            and np.array_equal(origin["translation_mm"], candidate["initialized_translation_mm"]),
            "SELECTED_START_NOT_BOUND_TO_PASSED_LANE_SUPPORT")
    _v2._require(origins == [index for index, row in enumerate(hypotheses)
        if row.get("exact_start_group_index") == gi and row.get("prefilter_status") == "PASSED"],
        "SELECTED_GROUP_ORIGIN_LEDGER_INCOMPLETE")

    # Bind publication to the pose whose diagnostics actually passed, not just
    # any well-formed rigid transform accompanying the same input digests.
    selected_rotation = np.asarray(candidate["rotation"])
    selected_translation = np.asarray(candidate["translation"])
    _v2._require(selected_rotation.shape == (3, 3) and selected_rotation.dtype.kind in "fiu"
        and selected_translation.shape == (3,) and selected_translation.dtype.kind in "fiu",
        "INVALID_SELECTED_CANDIDATE_POSE")
    selected_transform = np.eye(4)
    selected_transform[:3, :3] = selected_rotation
    selected_transform[:3, 3] = selected_translation
    sr, st = _v2._rigid_transform(selected_transform)
    pr, pt = _v2._rigid_transform(registration["transform_scan_to_cad"])
    _v2._require(np.array_equal(sr, pr) and np.array_equal(st, pt),
                 "PUBLISHED_TRANSFORM_NOT_SELECTED_CANDIDATE_POSE")
    return dict(binding, numerical_support_policy=evidence["numerical_support_policy"],
        numerical_consumer_status=candidate["numerical_consumer_status"],
        distance_model=evidence["distance_model"], optimizer_objective=evidence["optimizer_objective"])


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
        return dict(schema="CAD_GUIDED_MESH_OFFLINE_V11", status="UNAVAILABLE",
            phase="REPORT_PUBLICATION", reasons=["WORKFLOW_DETAILED_REPORT_UNAVAILABLE"],
            predictions=[], measurements=None, registration=None, observed_bend_count=None,
            production_use_allowed=False, conformance="INDETERMINATE",
            detailed_report_discarded_whole=True, detailed_report_limit_bytes=maximum,
            encoded_bytes_examined=size)
    return output


def inspect_cad_guided_mesh_v11(*, scan_bytes, cad_bytes, position_error_bound_mm, workflow_policy,
        registration_policy, measurement_policy, instance_policy,
        noisy_policy=None, noisy_budgets=None, cad_patch_policy=None, cad_patch_budgets=None,
        search_policy=None, search_budgets=None, initialization_budgets=None,
        surface_budgets=None, registration_budgets=None, allow_ocp=False, two_plane_enabled=True):
    """Same byte/native/measurement workflow, declared-noise full-raw initialization.

    A successful global fit remains provisional and may absorb deformation.
    Selected-candidate completeness is enforced at publication; it is not a
    global-search or production verdict. Noise bounds are explicit declarations,
    not a scanner error estimate or propagated measurement uncertainty.
    """
    output = dict(schema="CAD_GUIDED_MESH_OFFLINE_V11", status="UNAVAILABLE",
        phase="ADMISSION", reasons=[], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT", datum_status="NO_VERIFIED_STABLE_DATUM",
        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
        registration_initialization="TRIPLET_AND_TWO_PLANE_CONTEXT_FINITE_PREFILTER_V11",
        measurement_geometry_frame="CAD_ALIGNED_COORDINATES_ASSUMED_MM",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
    if type(two_plane_enabled) is not bool:
        return {**output, "reasons": ["INVALID_TWO_PLANE_LANE_POLICY"]}
    if type(allow_ocp) is not bool or not allow_ocp:
        return {**output, "reasons": ["EXPLICIT_OCP_OPT_IN_REQUIRED"]}
    if type(workflow_policy) is not MeshWorkflowPolicyV11:
        return {**output, "reasons": ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]}
    if type(cad_bytes) is not bytes or not 1 <= len(cad_bytes) <= 8 * 1024**2:
        return {**output, "reasons": ["CAD_BYTE_BUDGET"]}
    if type(scan_bytes) is not bytes or not 1 <= len(scan_bytes) <= 64 * 1024**2:
        return {**output, "reasons": ["SCAN_BYTE_BUDGET"]}
    if type(position_error_bound_mm) not in (int, float) or not 0 <= position_error_bound_mm <= 1000.:
        return {**output, "reasons": ["EXPLICIT_FINITE_POSITION_ERROR_BOUND_REQUIRED"]}
    position_error_bound_mm = float(position_error_bound_mm)
    output["position_error_model"] = dict(uniform_euclidean_bound_mm=position_error_bound_mm,
        status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION", scope="PLANE_INITIALIZATION_ONLY_NOT_MEASUREMENT_UNCERTAINTY")
    output["phase"] = "INPUT_PREPARATION"
    try:
        from .cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from .cad_guided_mesh_topology_v4 import parse_triangle_mesh_ply_v4
        from . import cad_guided_plane_initialization_v11 as initialize
        from .cad_guided_native_features import NativeGuidedPolicy
        from .cad_guided_instances import SupportPolicy

        _v2._require(type(measurement_policy) is NativeGuidedPolicy
                     and type(instance_policy) is SupportPolicy, "VALIDATED_INFERENCE_POLICIES_REQUIRED")
        workflow_policy = MeshWorkflowPolicyV11(**asdict(workflow_policy))
        measurement_policy = NativeGuidedPolicy(**asdict(measurement_policy))
        instance_policy = SupportPolicy(**asdict(instance_policy))
        config = initialize._configuration(registration_policy,
            initialize._noisy.NoisyMeshPlanePolicyV5() if noisy_policy is None else noisy_policy,
            initialize._noisy.NoisyMeshPlaneBudgetsV5() if noisy_budgets is None else noisy_budgets,
            initialize._patch.CoplanarPatchPolicyV4() if cad_patch_policy is None else cad_patch_policy,
            initialize._patch.CoplanarPatchBudgetsV4() if cad_patch_budgets is None else cad_patch_budgets,
            initialize._search.PlaneSearchPolicyV6() if search_policy is None else search_policy,
            initialize._search.PlaneSearchBudgetsV6() if search_budgets is None else search_budgets,
            initialize.PlaneInitializationBudgetsV11() if initialization_budgets is None else initialization_budgets,
            surface_budgets, two_plane_enabled)
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
            registration_budgets=asdict(registration_budgets), two_plane_enabled=two_plane_enabled)
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
        registration = initialize.register_whole_part_planes_v11(scan["raw_points"], topology["triangle_raw_rows"],
            cad["triangles"], cad["source_face_indices"], cad["source_triangle_indices"],
            position_error_bounds_mm=position_error_bound_mm,
            cad_initialization_points=cad["points"], registration_policy=registration_policy,
            registration_raw_rows=selection, unit_assumption=workflow_policy.unit_assumption,
            noisy_policy=config["noisy_policy"], noisy_budgets=config["noisy_budgets"],
            cad_patch_policy=config["cad_patch_policy"], cad_patch_budgets=config["cad_patch_budgets"],
            search_policy=config["search_policy"], search_budgets=config["search_budgets"],
            budgets=config["budgets"], surface_budgets=config["surface_budgets"],
            registration_budgets=registration_budgets, two_plane_enabled=two_plane_enabled)
        output["registration"] = registration
        if registration["status"] != "CONDITIONAL_BEST_FIT":
            return _bounded_workflow({**output, "reasons": ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]})
        try:
            _v2._require(registration["evidence"]["initialization"]["joint_proposal_accounting"]["pair_lane_enabled"]
                is two_plane_enabled, "REGISTRATION_PAIR_LANE_POLICY_MISMATCH")
            output["registration_array_binding"] = _registration_binding(registration, scan, topology, cad, selection, position_error_bound_mm)
            rotation, translation = _v2._rigid_transform(registration["transform_scan_to_cad"])
        except Exception:
            output["registration"] = dict(registration, status="UNAVAILABLE", transform_scan_to_cad=None,
                                            reasons=["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"])
            raise
        output["phase"] = "NATIVE_FEATURES"
        if features["status"] not in ("AVAILABLE", "PARTIAL") or not features["features"]:
            return _bounded_workflow({**output, "reasons": ["NATIVE_FEATURES_UNAVAILABLE"]})
        output["phase"] = "FULL_RAW_LOCAL_MEASUREMENT"
        aligned = scan["raw_points"]@rotation.T+translation
        _v2._require(np.isfinite(aligned).all(), "NONFINITE_ALIGNED_RAW_POINTS")
        from .cad_guided_local_measurement_v1 import inspect_full_raw_windows_v1
        measured = inspect_full_raw_windows_v1(aligned, features["features"], instance_policy=instance_policy,
            scan_sha256=scan["source_bindings"]["scan_sha256"], registration_status="CONDITIONAL_BEST_FIT")
        published, predictions = _v2._publish_measurements(measured, len(aligned), rotation,
                                                           translation, output["prediction_frame"])
        output.update(status="PARTIAL" if predictions else "UNAVAILABLE", phase="COMPLETE",
            measurements=published, predictions=predictions, observed_support_instance_count=len(predictions),
            reasons=[] if predictions else ["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"])
    except Exception as error:
        output.update(status="UNAVAILABLE", measurements=None, predictions=[],
                      reasons=["WORKFLOW_EXCEPTION:"+type(error).__name__])
    return _bounded_workflow(output)
