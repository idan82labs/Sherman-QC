"""V11 single-part recovery candidate: native bend-axis support on the raw scan.

This is a separately versioned research candidate.  It does not edit or replace
the frozen V11 kernel/attempt.  The native STEP topology supplies only a finite
axis proposal; every published endpoint is trimmed from raw scan support.  CAD
nominal radii and sweeps are retained as reference metadata and are never
published as measured values.  No Gold, expected count, case identity, or
scoring data is consumed by this module.

The candidate is deliberately conservative about semantics: a supported axis is
a CAD-targeted, scan-supported contact-line proposal, not a manufacturing-event
count or a verified inner-face measurement.  The caller must keep the result
research-only until an independent audit establishes comparability and
generalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.build_additional_line_label_review import line_scan_support
from scripts.extract_step_topology_bend_axes import extract_step_bend_axes


IMPLEMENTATION = "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V1"
SUPPORT_RADIUS_MM = 3.0


def _finite_line(value: Any) -> np.ndarray:
    line = np.asarray(value, dtype=np.float64)
    if line.shape != (2, 3) or not np.isfinite(line).all():
        raise ValueError("INVALID_AXIS_LINE")
    if float(np.linalg.norm(line[1] - line[0])) <= 1e-9:
        raise ValueError("DEGENERATE_AXIS_LINE")
    return line


def _raw_from_cad(line: np.ndarray, scan_to_cad: np.ndarray) -> np.ndarray:
    transform = np.asarray(scan_to_cad, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("INVALID_SCAN_TO_CAD_TRANSFORM")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or float(np.linalg.det(rotation)) <= 0.0):
        raise ValueError("SCAN_TO_CAD_TRANSFORM_MUST_BE_PROPER_RIGID")
    inverse = np.linalg.inv(transform)
    homogeneous = np.column_stack((line, np.ones(2, dtype=np.float64)))
    return (homogeneous @ inverse.T)[:, :3]


def _duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = _finite_line((left["line_start"], left["line_end"]))
    b = _finite_line((right["line_start"], right["line_end"]))
    a_delta, b_delta = a[1] - a[0], b[1] - b[0]
    a_len, b_len = float(np.linalg.norm(a_delta)), float(np.linalg.norm(b_delta))
    a_axis, b_axis = a_delta / a_len, b_delta / b_len
    angle = float(np.degrees(np.arccos(np.clip(abs(a_axis @ b_axis), -1.0, 1.0))))
    if b_axis @ a_axis < 0.0:
        b_axis = -b_axis
    midpoint_delta = b.mean(axis=0) - a.mean(axis=0)
    transverse = max(
        float(np.linalg.norm(midpoint_delta - a_axis * (midpoint_delta @ a_axis))),
        float(np.linalg.norm(midpoint_delta - b_axis * (midpoint_delta @ b_axis))),
    )
    a_mid = a.mean(axis=0)
    a_interval = sorted(float((point - a_mid) @ a_axis) for point in a)
    b_interval = sorted(float((point - a_mid) @ a_axis) for point in b)
    overlap = max(0.0, min(a_interval[1], b_interval[1]) - max(a_interval[0], b_interval[0]))
    shorter_overlap = overlap / min(a_len, b_len)
    return bool(angle <= 5.0 and transverse <= 4.0 and shorter_overlap >= 0.60)


def recover_scan_supported_axes_v1(
    scan_points: np.ndarray,
    cad_path,
    scan_to_cad: np.ndarray,
    *,
    support_radius_mm: float = SUPPORT_RADIUS_MM,
) -> dict[str, Any]:
    """Return bounded native-axis proposals whose finite spans are scan-supported.

    The line endpoints and support statistics come from ``line_scan_support``
    over the supplied raw points.  The STEP extraction is only a topology
    proposal and cannot by itself create a prediction.  No expected count or
    Gold input is accepted.
    """
    points = np.asarray(scan_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not (1 <= len(points) <= 1_000_000):
        raise ValueError("INVALID_RAW_SCAN_POINTS")
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 1e6:
        raise ValueError("INVALID_RAW_SCAN_COORDINATES")
    if not np.isfinite(float(support_radius_mm)) or not 0.0 < float(support_radius_mm) <= 20.0:
        raise ValueError("INVALID_SCAN_SUPPORT_RADIUS")

    axes = extract_step_bend_axes(cad_path)["bend_axes"]
    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for axis in axes:
        cad_line = _finite_line((axis["line_start"], axis["line_end"]))
        raw_line = _raw_from_cad(cad_line, scan_to_cad)
        support = line_scan_support(
            points,
            raw_line[0].tolist(),
            raw_line[1].tolist(),
            distance_mm=float(support_radius_mm),
        )
        record = {
            "bend_id": str(axis["bend_id"]),
            "cad_axis_start": cad_line[0].tolist(),
            "cad_axis_end": cad_line[1].tolist(),
            "raw_axis_start": raw_line[0].tolist(),
            "raw_axis_end": raw_line[1].tolist(),
            "scan_support": support,
            "accepted": bool(support["frame_integrity_pass"]),
        }
        records.append(record)
        if not record["accepted"]:
            continue

        candidate = {
            "bend_id": record["bend_id"],
            "line_start": support["trimmed_line_start"],
            "line_end": support["trimmed_line_end"],
            "assignment_source": "V11_NATIVE_STEP_AXIS_SCAN_SUPPORT",
            "measurement_method": "RAW_SCAN_FINITE_AXIS_SUPPORT",
            "observability_state": "SCAN_SUPPORTED_CONDITIONAL_CONTACT_LINE",
            "research_only": True,
            # These are reference values, not measured answers.
            "cad_reference": {
                "radii_mm": list(axis["radii_mm"]),
                "angular_sweeps_deg": list(axis["angular_sweeps_deg"]),
                "surface_count": int(axis["surface_count"]),
                "semantics": "CAD_NOMINAL_TOPOLOGY_REFERENCE_NOT_MEASURED_SCAN_GEOMETRY",
            },
            "bend_data": {
                "line_span_mm": float(support["supported_span_mm"]),
                "support_fraction": float(support["support_fraction"]),
                "longest_support_run_fraction": float(support["longest_support_run_fraction"]),
                "median_scan_distance_mm": float(support["median_distance_mm"]),
                "p90_scan_distance_mm": float(support["p90_distance_mm"]),
                "geometry_semantics": "SCAN_SUPPORT_DISTANCE_TO_CAD_TARGETED_AXIS_NOT_BEND_RADIUS",
            },
        }
        duplicate = next((row for row in accepted if _duplicate(candidate, row)), None)
        if duplicate is not None:
            record["accepted"] = False
            record["rejection_reason"] = "DUPLICATE_OF_" + str(duplicate["bend_id"])
            continue
        accepted.append(candidate)

    return {
        "schema": "CAD_GUIDED_V11_TOPOLOGY_RECOVERY_V1",
        "implementation": IMPLEMENTATION,
        "status": "PARTIAL" if accepted else "UNAVAILABLE",
        "research_only": True,
        "gold_or_expected_count_used": False,
        "predictions": accepted,
        "records": records,
        "accepted_count": len(accepted),
        "native_axis_count": len(axes),
        "support_radius_mm": float(support_radius_mm),
        "prediction_geometry_frame": "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        "observed_bend_count": None,
        "observed_bend_count_status": "MANUFACTURING_EVENT_IDENTITY_UNKNOWN",
    }
