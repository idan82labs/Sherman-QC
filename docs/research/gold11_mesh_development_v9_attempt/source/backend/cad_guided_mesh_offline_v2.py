"""Opt-in finite-triangle successor workflow, not a product/default route.

Source closure explicitly includes the frozen V1 mesh loader, STEP snapshot
importer, CAD sampler arithmetic, registration policy, native feature adapters,
full-raw measurement and instance modules, plus the V2 surface/registration
modules and triangle kernel. No label, case identity, expected count, external
pose or reference-line input exists. Native CPU/wall/RSS supervision remains
the caller's responsibility; these admission limits are not a sandbox.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class MeshWorkflowPolicyV2:
    max_registration_scan_rows: int = 50000
    cad_sample_count: int = 32768
    cad_linear_deflection_mm: float = .1
    cad_angular_deflection_radians: float = .2
    native_adapter_version: str = "v1"
    unit_assumption: str = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"

    def __post_init__(self):
        # Identical mesh/unit/native-choice admission, separately versioned API.
        from .cad_guided_mesh_offline_v1 import MeshWorkflowPolicyV1
        MeshWorkflowPolicyV1(**asdict(self))


def _read_step_owned_snapshot(cad_bytes):
    # Reuse exactly the reviewed V1 byte snapshot/unit transfer sequence.
    from .cad_guided_mesh_offline_v1 import _read_step_owned_snapshot as read
    return read(cad_bytes)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _array(value, dtype, trailing_shape, maximum):
    """Only consume the immutable arrays emitted by the owned source seams."""
    _require(type(value) is np.ndarray and value.dtype == np.dtype(dtype)
             and value.ndim == len(trailing_shape) + 1
             and value.shape[1:] == trailing_shape and 1 <= len(value) <= maximum
             and value.flags.c_contiguous and not value.flags.writeable,
             "INVALID_BOUND_ARRAY")
    base = value
    for _ in range(8):
        if type(base) is not np.ndarray:
            break
        _require(not base.flags.writeable, "MUTABLE_ARRAY_BASE")
        base = base.base
    _require(type(base) is bytes, "IMMUTABLE_BYTE_BACKING_REQUIRED")
    _require(np.isfinite(value).all(), "NONFINITE_BOUND_ARRAY")
    return value


def _scan_binding(scan, scan_bytes, row_limit):
    raw = _array(scan["raw_points"], "<f8", (3,), 1000000)
    points = _array(scan["points"], "<f8", (3,), row_limit)
    rows = _array(scan["engine_to_raw_row_indices"], "<i8", (), row_limit)
    _require(len(rows) == len(points) and np.all(rows >= 0) and np.all(rows < len(raw))
             and len(np.unique(rows)) == len(rows), "INVALID_SCAN_SAMPLE_ROWS")
    receipt = scan["source_bindings"]
    _require(receipt["scan_sha256"] == hashlib.sha256(scan_bytes).hexdigest()
             and receipt["raw_xyz_sha256"] == _digest(raw)
             and receipt["sample_xyz_sha256"] == _digest(points)
             and receipt["engine_to_raw_sha256"] == _digest(rows)
             and np.array_equal(points, raw[rows]), "SCAN_BYTE_ROW_BINDING_MISMATCH")


def _surface_binding(cad, sample_count):
    """Check byte custody and finite-triangle incidence, not CAD accuracy."""
    from .cad_guided_cad_sampling_v1 import _affine_interior_points, _triangle_geometry

    triangles = _array(cad["triangles"], "<f8", (3, 3), 500000)
    faces = _array(cad["source_face_indices"], "<i8", (), len(triangles))
    local_ids = _array(cad["source_triangle_indices"], "<i8", (), len(triangles))
    points = _array(cad["points"], "<f8", (3,), sample_count)
    normals = _array(cad["normals"], "<f8", (3,), sample_count)
    rows = _array(cad["sample_triangle_rows"], "<i8", (), sample_count)
    sample_faces = _array(cad["sample_source_face_indices"], "<i8", (), sample_count)
    sample_ids = _array(cad["sample_source_triangle_indices"], "<i8", (), sample_count)
    weights = _array(cad["sample_barycentric_weights"], "<f8", (3,), sample_count)
    _require(len(faces) == len(local_ids) == len(triangles)
             and all(len(v) == sample_count for v in
                     (points, normals, rows, sample_faces, sample_ids, weights))
             and np.all(rows >= 0) and np.all(rows < len(triangles)),
             "INVALID_TRIANGLE_SAMPLE_ROWS")
    digests = cad["digests"]
    for key, value in (("oriented_triangle_xyz_sha256", triangles),
                       ("source_face_ids_sha256", faces), ("source_triangle_ids_sha256", local_ids),
                       ("sample_points_sha256", points), ("sample_normals_sha256", normals),
                       ("sample_triangle_rows_sha256", rows)):
        _require(digests[key] == _digest(value), "TRIANGLE_SAMPLE_DIGEST_MISMATCH")
    policy_bytes = json.dumps(cad["sampling_policy"], sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
    _require(cad["sampling_policy"]["sample_count"] == sample_count
             and digests["sampling_policy_sha256"] == hashlib.sha256(policy_bytes).hexdigest(),
             "TRIANGLE_SAMPLING_POLICY_BINDING_MISMATCH")
    _require(np.array_equal(sample_faces, faces[rows])
             and np.array_equal(sample_ids, local_ids[rows])
             and np.all(weights > 0) and np.all(weights < 1)
             and np.allclose(weights.sum(axis=1), 1., rtol=0, atol=4e-16),
             "TRIANGLE_SAMPLE_MEMBERSHIP_MISMATCH")
    chosen = triangles[rows]
    _, expected_normals = _triangle_geometry(chosen)
    _require(np.array_equal(points, _affine_interior_points(chosen, weights))
             and np.array_equal(normals, expected_normals), "TRIANGLE_SAMPLE_GEOMETRY_MISMATCH")
    return dict(status="CONSISTENT_IMMUTABLE_TRIANGLE_SAMPLE_SNAPSHOT",
                scope="IN_PROCESS_BYTES_AND_TRIANGLE_INCIDENCE_NOT_NATIVE_AUTHORIZATION",
                source_identity_status="IMPORT_LOCAL_NOT_UNIVERSAL_FACE_IDENTITY",
                oriented_triangle_xyz_sha256=digests["oriented_triangle_xyz_sha256"],
                initialization_points_sha256=digests["sample_points_sha256"],
                triangle_count=len(triangles), initialization_sample_count=sample_count)


def _rigid_transform(value):
    raw = np.asarray(value)
    _require(raw.shape == (4, 4) and raw.dtype.kind in "fiu", "INVALID_REGISTRATION_TRANSFORM")
    matrix = raw.astype(np.float64)
    rotation = matrix[:3, :3]
    # Serialization/SE(3) sanity only; no fit score or metrology gate is added.
    _require(np.isfinite(matrix).all() and np.array_equal(matrix[3], [0., 0., 0., 1.])
             and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
             and abs(float(np.linalg.det(rotation)) - 1.) <= 1e-8,
             "NONRIGID_REGISTRATION_TRANSFORM")
    return rotation, matrix[:3, 3]


def _publish_measurements(measured, raw_count, rotation, translation, prediction_frame):
    _require(measured["status"] in ("PARTIAL", "UNAVAILABLE")
             and measured.get("observed_bend_count") is None,
             "INVALID_MEASUREMENT_STATUS")
    _require(len(measured["features"]) <= 512 and len(measured["finite_support_instances"]) <= 512,
             "MEASUREMENT_PUBLICATION_BUDGET")
    _require(measured["status"] == "PARTIAL" or not measured["finite_support_instances"],
             "UNAVAILABLE_MEASUREMENT_HAS_INSTANCES")
    published = dict(measured, features=[],
        geometry_coordinate_frame="CAD_ALIGNED_COORDINATES_ASSUMED_MM",
        geometry_transform_reference="registration.transform_scan_to_cad",
        raw_support_identity_frame="ORIGINAL_SCAN_VERTEX_ROW_IDENTITIES")
    support_count = 0
    for feature in measured["features"]:
        row = dict(feature)
        support = row.get("raw_support")
        if support is not None:
            _require(len(support["flange_rows"]) == 2, "INVALID_FLANGE_SUPPORT")
            groups = [support["arc_rows"], *support["flange_rows"]]
            for ids in groups:
                _array(ids, "<i8", (), 200000)
                _require(np.all(ids >= 0) and np.all(ids < raw_count)
                         and len(np.unique(ids)) == len(ids), "INVALID_ORIGINAL_RAW_SUPPORT_ROWS")
                support_count += len(ids)
            _require(support_count <= 4000000, "RAW_SUPPORT_PUBLICATION_BUDGET")
            row["raw_support"] = dict(arc_rows=groups[0].tolist(),
                flange_rows=[ids.tolist() for ids in groups[1:]],
                arc_rows_sha256=_digest(groups[0]),
                flange_rows_sha256=[_digest(ids) for ids in groups[1:]],
                digest_encoding="LITTLE_ENDIAN_INT64_ORIGINAL_RAW_ROW_INDICES")
        published["features"].append(row)
    predictions = []
    for instance in measured["finite_support_instances"]:
        row = dict(instance)
        for field in ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end"):
            value = np.asarray(row[field])
            _require(value.shape == (3,) and value.dtype.kind in "fiu"
                     and np.isfinite(value).all(), "INVALID_FINITE_INSTANCE_ENDPOINT")
            inverse = (value.astype(np.float64) - translation) @ rotation
            _require(np.isfinite(inverse).all(), "NONFINITE_INVERSE_ENDPOINT")
            row[field] = inverse.tolist()
        row["reference_frame"] = prediction_frame
        predictions.append(row)
    return published, predictions


def inspect_cad_guided_mesh_v2(*, scan_bytes, cad_bytes, workflow_policy,
                               registration_policy, measurement_policy, instance_policy,
                               surface_budgets=None, registration_budgets=None, allow_ocp=False):
    """One exact STEP snapshot, finite-surface registration, full-raw fitting.

    Both native adapter variants share all downstream policy and arithmetic.
    This remains conditional operator assistance with unknown observed bend count.
    Registration can be recorded even when no native feature is supported.
    """
    output = dict(schema="CAD_GUIDED_MESH_OFFLINE_V2", status="UNAVAILABLE",
        phase="ADMISSION", reasons=[], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT", datum_status="NO_VERIFIED_STABLE_DATUM",
        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
        measurement_geometry_frame="CAD_ALIGNED_COORDINATES_ASSUMED_MM",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
    if type(allow_ocp) is not bool or not allow_ocp:
        return {**output, "reasons": ["EXPLICIT_OCP_OPT_IN_REQUIRED"]}
    if type(workflow_policy) is not MeshWorkflowPolicyV2:
        return {**output, "reasons": ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]}
    if type(cad_bytes) is not bytes or not 1 <= len(cad_bytes) <= 8 * 1024**2:
        return {**output, "reasons": ["CAD_BYTE_BUDGET"]}
    if type(scan_bytes) is not bytes or not 1 <= len(scan_bytes) <= 64 * 1024**2:
        return {**output, "reasons": ["SCAN_BYTE_BUDGET"]}
    output["phase"] = "INPUT_PREPARATION"
    try:
        from .cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from .cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1
        from .cad_guided_global_registration_v2 import (
            register_whole_part_v2, SurfaceRegistrationBudgetsV2, _validated_execution_budgets)
        from .cad_guided_triangle_surface_v2 import _budgets
        from .cad_guided_native_features import NativeGuidedPolicy
        from .cad_guided_instances import SupportPolicy
        _require(type(registration_policy) is WholePartRegistrationPolicyV1
                 and type(measurement_policy) is NativeGuidedPolicy
                 and type(instance_policy) is SupportPolicy, "VALIDATED_INFERENCE_POLICIES_REQUIRED")
        # Frozen objects can be bypass-mutated; re-run the unchanged constructors.
        for policy in (workflow_policy, registration_policy, measurement_policy, instance_policy):
            type(policy)(**asdict(policy))
        surface_budgets = _budgets(surface_budgets)
        registration_budgets = _validated_execution_budgets(
            SurfaceRegistrationBudgetsV2() if registration_budgets is None else registration_budgets)
        output["policies"] = dict(workflow=asdict(workflow_policy), registration=asdict(registration_policy),
            measurement=asdict(measurement_policy), support=asdict(instance_policy),
            surface_budgets=asdict(surface_budgets), registration_budgets=asdict(registration_budgets))
        scan = prepare_mesh_scan_v1(scan_bytes=scan_bytes,
                                    max_engine_rows=workflow_policy.max_registration_scan_rows)
        output["input_preparation"] = {key: scan[key] for key in (
            "status", "reasons", "source_bindings", "coverage", "measurement_cloud_contract",
            "sample_cloud_contract", "units_status", "coordinate_frame_status")}
        if scan["status"] != "AVAILABLE":
            return {**output, "reasons": ["MESH_INPUT_UNAVAILABLE", *scan["reasons"]]}
        _scan_binding(scan, scan_bytes, workflow_policy.max_registration_scan_rows)
    except Exception as error:
        return {**output, "reasons": ["INPUT_PREPARATION_EXCEPTION:" + type(error).__name__]}
    output["cad_sha256"] = hashlib.sha256(cad_bytes).hexdigest()
    try:
        output["phase"] = "NATIVE_IMPORT"
        shape, receipt = _read_step_owned_snapshot(cad_bytes)
        _require(receipt["target_unit"] == "mm"
                 and receipt["physical_calibration"] == "NOT_ESTABLISHED",
                 "INVALID_NATIVE_UNIT_RECEIPT")
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
            return {**output, "reasons": ["CAD_SURFACE_SAMPLING_UNAVAILABLE", *cad["reasons"]]}
        output["triangle_sample_binding"] = _surface_binding(cad, workflow_policy.cad_sample_count)
        output["phase"] = "WHOLE_PART_REGISTRATION"
        registration = register_whole_part_v2(scan["points"], cad["triangles"], cad["source_face_indices"],
            cad_initialization_points=cad["points"], policy=registration_policy,
            unit_assumption=workflow_policy.unit_assumption, surface_budgets=surface_budgets,
            registration_budgets=registration_budgets)
        output["registration"] = registration
        if registration["status"] != "CONDITIONAL_BEST_FIT":
            return {**output, "reasons": ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]}
        try:
            rotation, translation = _rigid_transform(registration["transform_scan_to_cad"])
        except Exception:
            # Keep stage evidence without exporting an invalid/nonfinite pose.
            output["registration"] = dict(registration, status="UNAVAILABLE",
                transform_scan_to_cad=None, reasons=["INVALID_REGISTRATION_TRANSFORM_CONTRACT"])
            raise
        # Correct V1's stale registration phase on this distinct downstream refusal.
        output["phase"] = "NATIVE_FEATURES"
        if features["status"] not in ("AVAILABLE", "PARTIAL") or not features["features"]:
            return {**output, "reasons": ["NATIVE_FEATURES_UNAVAILABLE"]}
        output["phase"] = "FULL_RAW_LOCAL_MEASUREMENT"
        aligned = scan["raw_points"] @ rotation.T + translation
        _require(np.isfinite(aligned).all(), "NONFINITE_ALIGNED_RAW_POINTS")
        from .cad_guided_local_measurement_v1 import inspect_full_raw_windows_v1
        measured = inspect_full_raw_windows_v1(aligned, features["features"],
            instance_policy=instance_policy, scan_sha256=scan["source_bindings"]["scan_sha256"],
            registration_status="CONDITIONAL_BEST_FIT")
        published, predictions = _publish_measurements(measured, len(aligned), rotation,
                                                       translation, output["prediction_frame"])
        output.update(status="PARTIAL" if predictions else "UNAVAILABLE", phase="COMPLETE",
            measurements=published, predictions=predictions, observed_support_instance_count=len(predictions),
            reasons=[] if predictions else ["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"])
    except Exception as error:
        # No successful publication prefix survives malformed later rows/errors.
        output.update(status="UNAVAILABLE", measurements=None, predictions=[],
                      reasons=["WORKFLOW_EXCEPTION:" + type(error).__name__])
    return output
