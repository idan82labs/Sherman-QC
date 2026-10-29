"""Explicit successor offline CAD+mesh workflow; no default product activation.

Only input snapshots and frozen policies enter inference. No case identity,
reference lines, known pose or bend labels are accepted. One native STEP import
feeds both registration surface sampling and native finite-feature extraction.
Caller supervision must bound native CPU/wall/RSS; parser limits alone are not
an execution sandbox. The frozen datum/R0 engine modules remain unchanged.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import tempfile

import numpy as np


@dataclass(frozen=True)
class MeshWorkflowPolicyV1:
    max_registration_scan_rows: int = 50000
    cad_sample_count: int = 32768
    cad_linear_deflection_mm: float = .1
    cad_angular_deflection_radians: float = .2
    native_adapter_version: str = "v1"
    unit_assumption: str = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"

    def __post_init__(self):
        if (type(self.max_registration_scan_rows) is not int
                or not 36 <= self.max_registration_scan_rows <= 200000
                or type(self.cad_sample_count) is not int
                or not 36 <= self.cad_sample_count <= 100000):
            raise ValueError("INVALID_WORKFLOW_ROW_BUDGET")
        for value in (self.cad_linear_deflection_mm, self.cad_angular_deflection_radians):
            if type(value) not in (int, float) or not np.isfinite(value) or not 1e-4 <= value <= 1:
                raise ValueError("INVALID_MESH_APPROXIMATION_POLICY")
        if type(self.native_adapter_version) is not str or self.native_adapter_version not in ("r0", "v1"):
            raise ValueError("INVALID_NATIVE_ADAPTER_VERSION")
        if type(self.unit_assumption) is not str or self.unit_assumption != "PROVISIONAL_COMMON_MM_NOT_CALIBRATED":
            raise ValueError("EXPLICIT_PROVISIONAL_UNITS_REQUIRED")


def _read_step_owned_snapshot(cad_bytes):
    """Explicit native import from exact bytes, never an arbitrary raw path."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TColStd import TColStd_SequenceOfAsciiString

    reader = STEPControl_Reader()
    with tempfile.TemporaryDirectory(prefix="cad-mesh-v1-") as directory:
        path = Path(directory) / "snapshot.step"
        with path.open("xb") as stream:
            stream.write(cad_bytes)
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise ValueError("STEP_READ_FAILED")
        lengths, angles, solid_angles = (TColStd_SequenceOfAsciiString() for _ in range(3))
        reader.FileUnits(lengths, angles, solid_angles)
        declarations = {}
        for key, seq in (("length", lengths), ("plane_angle", angles), ("solid_angle", solid_angles)):
            if not 0 <= seq.Length() <= 64:
                raise ValueError("STEP_UNIT_DECLARATION_BUDGET")
            declarations[key] = [str(seq.Value(i).ToCString()) for i in range(1, seq.Length()+1)]
            if any(len(v) > 128 for v in declarations[key]):
                raise ValueError("STEP_UNIT_DECLARATION_BUDGET")
        if not declarations["length"]:
            raise ValueError("STEP_LENGTH_DECLARATION_ABSENT")
        roots = int(reader.NbRootsForTransfer())
        if not 1 <= roots <= 32:
            raise ValueError("STEP_TRANSFER_ROOT_BUDGET")
        reader.SetSystemLengthUnit(1.)
        if abs(float(reader.SystemLengthUnit())-1.) > 1e-12:
            raise ValueError("STEP_TARGET_UNIT_NOT_CONFIRMED")
        if int(reader.TransferRoots()) != roots:
            raise ValueError("STEP_PARTIAL_ROOT_TRANSFER")
        shape = reader.OneShape()
        if shape.IsNull():
            raise ValueError("NULL_NATIVE_SHAPE")
    return shape, dict(source_unit_names=declarations, target_unit="mm",
                       target_unit_action="SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER",
                       transferred_roots=roots, physical_calibration="NOT_ESTABLISHED")


def inspect_cad_guided_mesh_v1(*, scan_bytes, cad_bytes, workflow_policy,
                               registration_policy, measurement_policy,
                               instance_policy, allow_ocp=False):
    """Return the full provisional workflow plus scan-frame finite predictions.

    `native_adapter_version=r0` is a NEW mesh-compatible common-wrapper control,
    not the historical incumbent and not the untouched frozen R0 pipeline.
    Both variants use identical sampling, registration, full raw fitting and
    fixed measured-surface carrier. Only native-feature extraction changes.
    """
    output = dict(schema="CAD_GUIDED_MESH_OFFLINE_V1", status="UNAVAILABLE",
                  phase="ADMISSION", reasons=[], predictions=[], measurements=None,
                  observed_bend_count=None, production_use_allowed=False,
                  conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED",
                  units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                  registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT",
                  datum_status="NO_VERIFIED_STABLE_DATUM",
                  prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
                  reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
    if type(allow_ocp) is not bool or not allow_ocp:
        output["reasons"] = ["EXPLICIT_OCP_OPT_IN_REQUIRED"]
        return output
    if type(workflow_policy) is not MeshWorkflowPolicyV1:
        output["reasons"] = ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]
        return output
    if type(cad_bytes) is not bytes or not 1 <= len(cad_bytes) <= 8*1024**2:
        output["reasons"] = ["CAD_BYTE_BUDGET"]
        return output
    output["phase"] = "INPUT_PREPARATION"
    try:
        from .cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from .cad_guided_global_registration_v1 import register_whole_part_v1, WholePartRegistrationPolicyV1
        from .cad_guided_native_features import NativeGuidedPolicy
        from .cad_guided_instances import SupportPolicy
    except Exception as error:
        output["reasons"] = ["PREPARATION_IMPORT_EXCEPTION:" + type(error).__name__]
        return output
    if (type(registration_policy) is not WholePartRegistrationPolicyV1
            or type(measurement_policy) is not NativeGuidedPolicy
            or type(instance_policy) is not SupportPolicy):
        output["reasons"] = ["VALIDATED_INFERENCE_POLICIES_REQUIRED"]
        return output
    output["policies"] = dict(workflow=asdict(workflow_policy), registration=asdict(registration_policy),
                              measurement=asdict(measurement_policy), support=asdict(instance_policy))
    try:
        scan = prepare_mesh_scan_v1(scan_bytes=scan_bytes,
                                    max_engine_rows=workflow_policy.max_registration_scan_rows)
        output["input_preparation"] = {key: scan[key] for key in (
            "status", "reasons", "source_bindings", "coverage", "measurement_cloud_contract",
            "sample_cloud_contract", "units_status", "coordinate_frame_status")}
        if scan["status"] != "AVAILABLE":
            output["reasons"] = ["MESH_INPUT_UNAVAILABLE", *scan["reasons"]]
            return output
    except Exception as error:
        output["reasons"] = ["INPUT_PREPARATION_EXCEPTION:" + type(error).__name__]
        return output
    output["cad_sha256"] = hashlib.sha256(cad_bytes).hexdigest()
    try:
        output["phase"] = "NATIVE_IMPORT"
        shape, unit_receipt = _read_step_owned_snapshot(cad_bytes)
        output["cad_units"] = unit_receipt
        output["phase"] = "NATIVE_FEATURES"
        if workflow_policy.native_adapter_version == "r0":
            from .cad_guided_native_features import extract_ocp_guided_features as extract
        else:
            from .cad_guided_native_features_v1 import extract_ocp_guided_features_v1 as extract
        features = extract(shape, policy=measurement_policy, unit_status="DECLARED_MM", allow_ocp=True)
        output["native_features"] = features["metadata"]
        # Registration still executes if feature extraction abstains. Keeping
        # this stage evidence avoids making the control's failures invisible.
        output["phase"] = "CAD_SURFACE_SAMPLING"
        from .cad_guided_cad_sampling_v1 import sample_native_cad_surface_v1
        cad = sample_native_cad_surface_v1(shape, allow_ocp=True, native_binding="ocp",
            sample_count=workflow_policy.cad_sample_count,
            linear_deflection_mm=workflow_policy.cad_linear_deflection_mm,
            angular_deflection_radians=workflow_policy.cad_angular_deflection_radians)
        output["cad_surface_sampling"] = {key: value for key, value in cad.items()
                                          if not isinstance(value, np.ndarray)}
        if cad["status"] != "AVAILABLE":
            output["reasons"] = ["CAD_SURFACE_SAMPLING_UNAVAILABLE"]
            return output
        output["phase"] = "WHOLE_PART_REGISTRATION"
        registration = register_whole_part_v1(scan["points"], cad["points"], cad["normals"],
            policy=registration_policy, unit_assumption=workflow_policy.unit_assumption)
        output["registration"] = registration
        if registration["status"] != "CONDITIONAL_BEST_FIT":
            output["reasons"] = ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]
            return output
        if not features["features"]:
            output["reasons"] = ["NATIVE_FEATURES_UNAVAILABLE"]
            return output
        output["phase"] = "FULL_RAW_LOCAL_MEASUREMENT"
        transform = np.asarray(registration["transform_scan_to_cad"])
        r, t = transform[:3, :3], transform[:3, 3]
        aligned = scan["raw_points"] @ r.T+t
        from .cad_guided_local_measurement_v1 import inspect_full_raw_windows_v1
        measured = inspect_full_raw_windows_v1(aligned, features["features"],
            instance_policy=instance_policy, scan_sha256=scan["source_bindings"]["scan_sha256"],
            registration_status="CONDITIONAL_BEST_FIT")
        # Array-only fitting retains immutable raw-row maps; the workflow boundary
        # publishes bounded explicit lists and digests, not unserializable NumPy
        # objects or silently discarded provenance. Local caps bound these rows.
        published = dict(measured, features=[])
        for feature_row in measured["features"]:
            row = dict(feature_row)
            support = row.get("raw_support")
            if support is not None:
                row["raw_support"] = {
                    "arc_rows": support["arc_rows"].tolist(),
                    "flange_rows": [ids.tolist() for ids in support["flange_rows"]],
                    "arc_rows_sha256": hashlib.sha256(support["arc_rows"].tobytes()).hexdigest(),
                    "flange_rows_sha256": [hashlib.sha256(ids.tobytes()).hexdigest()
                                           for ids in support["flange_rows"]],
                    "digest_encoding": "LITTLE_ENDIAN_INT64_ORIGINAL_RAW_ROW_INDICES",
                }
            published["features"].append(row)
        output["measurements"] = published
        predictions = []
        for instance in measured["finite_support_instances"]:
            row = dict(instance)
            for field in ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end"):
                row[field] = ((np.asarray(row[field])-t) @ r).tolist()
            row["reference_frame"] = output["prediction_frame"]
            predictions.append(row)
        output.update(status="PARTIAL" if predictions else "UNAVAILABLE", phase="COMPLETE",
                      predictions=predictions,
                      observed_support_instance_count=len(predictions),
                      reasons=[] if predictions else ["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"])
    except Exception as error:
        # Native errors cannot turn into a partial transform or nominal answer.
        # Phase/type remain evidence; arbitrary library exception text is not
        # interpreted as a defect and cannot suppress the case denominator.
        output["reasons"] = ["WORKFLOW_EXCEPTION:" + type(error).__name__]
    return output
