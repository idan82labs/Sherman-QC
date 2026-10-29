"""Opt-in byte-to-measurement plane-initialized successor; no product routing.

Frozen V2 owns STEP/unit import, immutable scan/CAD sample binding, rigid inverse
and measurement publication. This small orchestration inserts the V4 raw triangle
parser and plane-only initialization. No pose, label, expected count or part
identity is accepted. One native shape feeds both feature extraction and CAD
sampling. Full raw points, not registration samples, feed final measurement.
External native CPU/wall/process-tree RSS supervision is mandatory.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from . import cad_guided_mesh_offline_v2 as _v2


MAX_WORKFLOW_REPORT_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class MeshWorkflowPolicyV4(_v2.MeshWorkflowPolicyV2):
    """Exactly the unchanged V2 fields/admission, a distinct opt-in type."""


def _triangle_scan_binding(scan, topology, scan_bytes):
    _v2._require(topology["status"] == "AVAILABLE", "TRIANGLE_SCAN_UNAVAILABLE")
    points = _v2._array(topology["points"], "<f8", (3,), 1000000)
    triangles = _v2._array(topology["triangle_raw_rows"], "<i8", (3,), 2000000)
    _v2._require(topology["vertex_count"] == len(points)
        and topology["triangle_count"] == len(triangles)
        and scan["input_admission"]["face_count"] == len(triangles)
        and scan["input_admission"]["face_index_count"] == 3*len(triangles)
        and np.all(triangles >= 0) and np.all(triangles < len(points))
        and np.all(triangles[:, 0] != triangles[:, 1])
        and np.all(triangles[:, 0] != triangles[:, 2])
        and np.all(triangles[:, 1] != triangles[:, 2]), "INVALID_TRIANGLE_RAW_ROW_BINDING")
    _v2._require(topology["source_sha256"] == hashlib.sha256(scan_bytes).hexdigest()
        == scan["source_bindings"]["scan_sha256"]
        and topology["points_sha256"] == _v2._digest(points)
        == scan["source_bindings"]["raw_xyz_sha256"]
        and topology["triangle_raw_rows_sha256"] == _v2._digest(triangles)
        and np.array_equal(points, scan["raw_points"]), "TRIANGLE_SCAN_BYTE_ROW_BINDING_MISMATCH")
    _v2._require(topology["raw_row_order"] == "ORIGINAL_PLY_VERTEX_ROWS"
        and topology["triangle_row_order"] == "ORIGINAL_PLY_FACE_ROWS",
        "TRIANGLE_SCAN_ROW_ORDER_MISMATCH")
    return dict(status="CONSISTENT_IMMUTABLE_SCAN_TRIANGLE_SNAPSHOT",
        scope="SAME_VALIDATED_BYTES_AND_ORIGINAL_ROW_IDENTITIES_NOT_MATERIAL_TOPOLOGY",
        source_sha256=topology["source_sha256"], raw_points_sha256=topology["points_sha256"],
        triangle_raw_rows_sha256=topology["triangle_raw_rows_sha256"],
        raw_vertex_count=len(points), triangle_count=len(triangles))


def _registration_selection(scan):
    rows = scan["engine_to_raw_row_indices"]
    if len(rows) == len(scan["raw_points"]):
        _v2._require(np.array_equal(rows, np.arange(len(rows))), "NONIDENTITY_FULL_RAW_SELECTION")
        return None
    return rows


def _registration_binding(registration, scan, topology, cad, selection):
    """Bind a successful report to arrays actually sent to the geometry arm."""
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    _v2._require(initialization["status"] == "COMPLETE" and initialization["search_complete"] is True,
                 "INCOMPLETE_INITIALIZATION_CANNOT_REGISTER")
    expected = dict(scan_points_sha256=_v2._digest(scan["raw_points"]),
        scan_triangle_raw_rows_sha256=_v2._digest(topology["triangle_raw_rows"]),
        cad_triangles_sha256=_v2._digest(cad["triangles"]),
        cad_face_ids_sha256=_v2._digest(cad["source_face_indices"]),
        cad_source_triangle_ids_sha256=_v2._digest(cad["source_triangle_indices"]))
    _v2._require(initialization["input_digests"] == expected
        and evidence["input_digests"] == expected, "REGISTRATION_GEOMETRY_BINDING_MISMATCH")
    selected = evidence["registration_raw_rows"]
    _v2._require(evidence["prefilter_full_scan_raw_rows"] == len(scan["raw_points"])
        and evidence["registration_input_rows"] == len(scan["points"])
        and evidence["registration_input_points_sha256"] == _v2._digest(scan["points"])
        and selected["count"] == len(scan["points"]), "REGISTRATION_SCAN_SELECTION_BINDING_MISMATCH")
    if selection is None:
        _v2._require(selected["selection"] == "ALL_RAW_ROWS" and selected["sha256"] is None,
                     "REGISTRATION_FULL_ROW_MODE_MISMATCH")
    else:
        _v2._require(selected["selection"] == "EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING"
            and selected["sha256"] == _v2._digest(selection), "REGISTRATION_SELECTED_ROW_MODE_MISMATCH")
    canonical_cad = np.unique(cad["points"], axis=0)
    canonical_digest = _v2._digest(np.ascontiguousarray(canonical_cad, dtype="<f8"))
    _v2._require(evidence["cad_initialization_points_sha256"] == canonical_digest,
                 "REGISTRATION_CAD_SAMPLE_BINDING_MISMATCH")
    return dict(status="CONSISTENT_REGISTRATION_ARRAY_BINDING",
        raw_scan_points_sha256=expected["scan_points_sha256"],
        raw_triangle_rows_sha256=expected["scan_triangle_raw_rows_sha256"],
        cad_triangles_sha256=expected["cad_triangles_sha256"],
        cad_sampler_points_sha256=_v2._digest(cad["points"]),
        registration_canonical_cad_points_sha256=canonical_digest,
        registration_selection="ALL_RAW_ROWS" if selection is None else "BOUND_V1_RAW_ROW_SELECTION",
        scope="EXACT_IN_PROCESS_ARRAY_BINDING_NOT_CALIBRATION_OR_NATIVE_AUTHORIZATION")


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
        return dict(schema="CAD_GUIDED_MESH_OFFLINE_V4", status="UNAVAILABLE",
            phase="REPORT_PUBLICATION", reasons=["WORKFLOW_DETAILED_REPORT_UNAVAILABLE"],
            predictions=[], measurements=None, registration=None, observed_bend_count=None,
            production_use_allowed=False, conformance="INDETERMINATE",
            detailed_report_discarded_whole=True, detailed_report_limit_bytes=maximum,
            encoded_bytes_examined=size)
    return output


def inspect_cad_guided_mesh_v4(*, scan_bytes, cad_bytes, workflow_policy,
        registration_policy, measurement_policy, instance_policy,
        patch_policy=None, patch_budgets=None, hypothesis_policy=None,
        hypothesis_budgets=None, initialization_budgets=None,
        surface_budgets=None, registration_budgets=None, allow_ocp=False):
    """Same byte/native/measurement workflow, new full-raw plane initialization.

    A successful global fit remains provisional and may absorb deformation.
    Candidate completeness is retained as evidence, not manufactured into a
    production verdict. The separately frozen connected test rejects incomplete
    candidate evaluation even if V2's unchanged best-fit policy returns a pose.
    """
    output = dict(schema="CAD_GUIDED_MESH_OFFLINE_V4", status="UNAVAILABLE",
        phase="ADMISSION", reasons=[], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT", datum_status="NO_VERIFIED_STABLE_DATUM",
        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
        registration_initialization="PLANE_ONLY_FINITE_ASSIGNED_FACE_PREFILTER_V4",
        measurement_geometry_frame="CAD_ALIGNED_COORDINATES_ASSUMED_MM",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
    if type(allow_ocp) is not bool or not allow_ocp:
        return {**output, "reasons": ["EXPLICIT_OCP_OPT_IN_REQUIRED"]}
    if type(workflow_policy) is not MeshWorkflowPolicyV4:
        return {**output, "reasons": ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]}
    if type(cad_bytes) is not bytes or not 1 <= len(cad_bytes) <= 8 * 1024**2:
        return {**output, "reasons": ["CAD_BYTE_BUDGET"]}
    if type(scan_bytes) is not bytes or not 1 <= len(scan_bytes) <= 64 * 1024**2:
        return {**output, "reasons": ["SCAN_BYTE_BUDGET"]}
    output["phase"] = "INPUT_PREPARATION"
    try:
        from .cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from .cad_guided_mesh_topology_v4 import parse_triangle_mesh_ply_v4
        from . import cad_guided_plane_initialization_v4 as initialize
        from .cad_guided_native_features import NativeGuidedPolicy
        from .cad_guided_instances import SupportPolicy

        _v2._require(type(measurement_policy) is NativeGuidedPolicy
                     and type(instance_policy) is SupportPolicy, "VALIDATED_INFERENCE_POLICIES_REQUIRED")
        workflow_policy = MeshWorkflowPolicyV4(**asdict(workflow_policy))
        measurement_policy = NativeGuidedPolicy(**asdict(measurement_policy))
        instance_policy = SupportPolicy(**asdict(instance_policy))
        config = initialize._configuration(registration_policy,
            initialize._patch.CoplanarPatchPolicyV4() if patch_policy is None else patch_policy,
            initialize._patch.CoplanarPatchBudgetsV4() if patch_budgets is None else patch_budgets,
            initialize._planes.PlaneHypothesisPolicyV4() if hypothesis_policy is None else hypothesis_policy,
            initialize._planes.PlaneHypothesisBudgetsV4() if hypothesis_budgets is None else hypothesis_budgets,
            initialize.PlaneInitializationBudgetsV4() if initialization_budgets is None else initialization_budgets,
            surface_budgets)
        registration_policy = config["policy"]
        registration_budgets = initialize._v2._validated_execution_budgets(
            initialize._v2.SurfaceRegistrationBudgetsV2() if registration_budgets is None else registration_budgets)
        registration_budgets = initialize._v2.SurfaceRegistrationBudgetsV2(**asdict(registration_budgets))
        output["policies"] = dict(workflow=asdict(workflow_policy), registration=asdict(registration_policy),
            measurement=asdict(measurement_policy), support=asdict(instance_policy),
            patch=asdict(config["patch_policy"]), patch_budgets=asdict(config["patch_budgets"]),
            hypothesis=asdict(config["hypothesis_policy"]), hypothesis_budgets=asdict(config["hypothesis_budgets"]),
            initialization_budgets=asdict(config["budgets"]), surface_budgets=asdict(config["surface_budgets"]),
            registration_budgets=asdict(registration_budgets))
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
        registration = initialize.register_whole_part_planes_v4(scan["raw_points"], topology["triangle_raw_rows"],
            cad["triangles"], cad["source_face_indices"], cad["source_triangle_indices"],
            cad_initialization_points=cad["points"], registration_policy=registration_policy,
            registration_raw_rows=selection, unit_assumption=workflow_policy.unit_assumption,
            patch_policy=config["patch_policy"], patch_budgets=config["patch_budgets"],
            hypothesis_policy=config["hypothesis_policy"], hypothesis_budgets=config["hypothesis_budgets"],
            budgets=config["budgets"], surface_budgets=config["surface_budgets"],
            registration_budgets=registration_budgets)
        output["registration"] = registration
        if registration["status"] != "CONDITIONAL_BEST_FIT":
            return _bounded_workflow({**output, "reasons": ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]})
        try:
            output["registration_array_binding"] = _registration_binding(registration, scan, topology, cad, selection)
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
