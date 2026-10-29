"""Import-safe registered CAD-assisted offline engine boundary.

Only NumPy is imported at module load. Native extraction, registration and
measurement are lazy, explicit dependencies; no legacy pipeline, report model,
application, DB, Open3D, file reader or evaluator is imported here. A caller must
provide an already-retained native shape and bounded scan/datum arrays. Unit,
stable-datum selection and CAD-frame binding remain declarations, not capture
qualification or automatic datum segmentation. Default product routing is not
changed by this module.
"""
from __future__ import annotations

import numpy as np


def _unavailable_registered(reason, registration=None):
    return {
        "schema_version": 1, "scope": "CAD_GUIDED_REGISTERED_DATUM_OFFLINE",
        "status": "UNAVAILABLE", "computation_status": "NOT_RUN",
        "measurement_availability": "UNAVAILABLE", "features": [],
        "expected_feature_count": None, "measured_conditional_feature_count": 0,
        "finite_support_instances": [], "observed_support_instance_count": 0,
        "instance_extraction_status": "UNAVAILABLE", "observed_bend_count": None,
        "line_convention": "CYLINDER_CENTER_AXIS_SUPPORT",
        "endpoint_semantics": "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
        "reference_frame": "CAD_REFERENCE", "unit": "mm",
        "registration_status": "UNAVAILABLE", "registration": registration,
        "conformance": "INDETERMINATE", "uncertainty_status": "UNCALIBRATED",
        "physical_completion_state": "UNKNOWN", "production_use_allowed": False,
        "reasons": [reason],
    }


def _inspect_native(cad_result, aligned_points, *, measurement_policy,
                    instance_policy, unit_status, allow_ocp):
    """The existing native-to-kernel semantics without a legacy report import."""
    if __package__:
        from .cad_guided_native_features import extract_ocp_guided_features
    else:  # Existing backend-directory launcher convention, without retry.
        from cad_guided_native_features import extract_ocp_guided_features

    extraction = extract_ocp_guided_features(
        getattr(cad_result, "native_shape", None), policy=measurement_policy,
        unit_status=unit_status, allow_ocp=allow_ocp,
        native_binding=getattr(cad_result, "native_binding", None))
    if extraction["status"] == "AVAILABLE" and extraction["features"]:
        if __package__:
            from .cad_guided_measurement import inspect_cad_guided_scan
        else:
            from cad_guided_measurement import inspect_cad_guided_scan
        payload = inspect_cad_guided_scan(
            aligned_points, extraction["features"],
            registration_status="DECLARED_ALIGNED", instance_policy=instance_policy)
        payload["computation_status"] = payload["status"]
        measured = payload["measured_conditional_feature_count"]
        expected = payload["expected_feature_count"]
        payload["measurement_availability"] = (
            "UNAVAILABLE" if measured == 0 else "AVAILABLE" if measured == expected else "PARTIAL")
        payload["status"] = (
            "UNAVAILABLE" if measured == 0 else "COMPLETE" if measured == expected else "PARTIAL")
        payload["native_cad_extraction"] = extraction["metadata"]
        return payload
    return {
        "schema_version": 1, "scope": "CAD_GUIDED_OFFLINE_SINGLE_INNER_FACE",
        "status": "UNAVAILABLE", "computation_status": "NOT_RUN",
        "measurement_availability": "UNAVAILABLE",
        "registration_status": "DECLARED_ALIGNED",
        "feature_identity_status": "NATIVE_FEATURES_UNAVAILABLE",
        "features": [], "expected_feature_count": None,
        "measured_conditional_feature_count": 0, "observed_bend_count": None,
        "uncertainty_status": "UNCALIBRATED", "conformance": "INDETERMINATE",
        "physical_completion_state": "UNKNOWN", "production_use_allowed": False,
        "reasons": ["NATIVE_CAD_FEATURES_UNAVAILABLE"],
        "native_cad_extraction": extraction["metadata"],
    }


def inspect_registered_native(
    cad_result,
    scan_points: np.ndarray,
    *,
    cad_datum_points,
    cad_datum_normals,
    scan_datum_row_indices,
    registration_policy,
    measurement_policy,
    instance_policy,
    unit_status: str = "UNVERIFIED",
    allow_ocp: bool = False,
):
    """Return conditional registration/measurement/finite-support evidence.

    No transform or point correspondences are accepted. The pose uses only the
    explicitly selected stable datum rows; measurement uses only the remaining
    rows. Both array sets are declared to share millimetre units, and CAD datum
    positions/normals must be bound to the native CAD reference frame by the
    caller's reviewed acquisition contract. Acceptance is a bounded search result,
    not a global registration, calibrated metrology or physical-count verdict.
    """
    if type(allow_ocp) is not bool:
        raise TypeError("allow_ocp must be a bool")
    if not allow_ocp or getattr(cad_result, "native_binding", None) != "ocp":
        return _unavailable_registered("EXPLICIT_OCP_BINDING_REQUIRED")
    if getattr(cad_result, "success", False) is not True:
        return _unavailable_registered("NATIVE_CAD_IMPORT_UNAVAILABLE")
    if type(unit_status) is not str or unit_status != "DECLARED_MM":
        return _unavailable_registered("EXPLICIT_NATIVE_MILLIMETRE_DECLARATION_REQUIRED")
    if instance_policy is None:
        return _unavailable_registered("EXPLICIT_FINITE_SUPPORT_POLICY_REQUIRED")
    if (not isinstance(scan_points, np.ndarray) or scan_points.ndim != 2
            or scan_points.shape[1] != 3 or scan_points.dtype.kind not in "fiu"
            or not 1 <= len(scan_points) <= 200_000):
        return _unavailable_registered("INVALID_OR_OVER_BUDGET_SCAN")
    points = np.array(scan_points, dtype=np.float64, copy=True)
    if not np.isfinite(points).all() or float(np.max(np.abs(points))) > 1e6:
        return _unavailable_registered("UNSUPPORTED_POINT_NUMERICS")
    indices = np.asarray(scan_datum_row_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or not 36 <= len(indices) <= 5000
            or np.any(indices < 0) or np.any(indices >= len(points))
            or len(np.unique(indices)) != len(indices)):
        return _unavailable_registered("INVALID_DATUM_ROW_SELECTION")
    # Range and uniqueness checks precede casting: uint64 overflow must not wrap
    # into an apparently valid row, and repeated rows are not independent datums.
    indices = indices.astype(np.int64)
    inspect_rows = np.ones(len(points), dtype=bool)
    inspect_rows[indices] = False
    if int(inspect_rows.sum()) < 12:
        return _unavailable_registered("NO_DISJOINT_INSPECTION_ROWS")
    if __package__:
        from .cad_guided_registration import register_datum_clouds
    else:
        from cad_guided_registration import register_datum_clouds
    registration = register_datum_clouds(
        points[indices], cad_datum_points, cad_datum_normals, policy=registration_policy)
    if registration["status"] != "ACCEPTED_BOUNDED":
        return _unavailable_registered("DATUM_REGISTRATION_UNAVAILABLE", registration)
    transform = np.asarray(registration["transform_scan_to_cad"], dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or abs(float(np.linalg.det(transform[:3, :3])) - 1) > 1e-8):
        return _unavailable_registered("REGISTRATION_NOT_PROPER_RIGID", registration)
    aligned = points[inspect_rows] @ transform[:3, :3].T + transform[:3, 3]
    payload = _inspect_native(
        cad_result, aligned, measurement_policy=measurement_policy,
        instance_policy=instance_policy, unit_status=unit_status, allow_ocp=allow_ocp)
    payload.update(
        scope="CAD_GUIDED_REGISTERED_DATUM_OFFLINE",
        registration_status="ACCEPTED_BOUNDED", registration=registration,
        datum_selection_status="EXPLICIT_STABLE_REGION_SELECTION_NOT_AUTOMATIC",
        datum_frame_status="DECLARED_CAD_REFERENCE", units_status=unit_status,
        uncertainty_status="UNCALIBRATED", physical_completion_state="UNKNOWN",
        datum_input_row_count=len(indices), inspection_input_row_count=int(inspect_rows.sum()),
    )
    payload.setdefault("finite_support_instances", [])
    payload.setdefault("observed_support_instance_count", 0)
    payload.setdefault("instance_extraction_status", "UNAVAILABLE")
    payload.setdefault("line_convention", "CYLINDER_CENTER_AXIS_SUPPORT")
    payload.setdefault("endpoint_semantics", "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM")
    payload.setdefault("reference_frame", "CAD_REFERENCE")
    payload.setdefault("unit", "mm")
    return payload
