from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


COUNTABLE_SKIP_FEATURE_TYPES = {"ROLLED_SECTION", "PROCESS_FEATURE"}
PASSIVE_STATES = {"PASS"}
ACTIVE_FAIL_STATES = {"FAIL", "WARNING"}
ROI_MODE = "bend_center_local_v2"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _normalize_scan_state(scan_state: Optional[str], scan_name: Optional[str]) -> str:
    text = str(scan_state or "").strip().lower()
    if text in {"full", "partial", "unknown"}:
        return text
    name = str(scan_name or "").strip().lower()
    if any(token in name for token in ("full", "final", "complete")):
        return "full"
    if name:
        return "partial"
    return "unknown"


def _independence_class(measurement_method: Optional[str]) -> str:
    method = str(measurement_method or "").strip().lower()
    if method == "signed_distance_gradient":
        return "NON_INDEPENDENT"
    if method in {"surface_classification", "profile_section"}:
        return "PARTIAL"
    return "INDEPENDENT"


def _center_local_roi_mask(
    points: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
    radial_radius: float,
    axial_half_window: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    tangent = np.asarray(line_end, dtype=np.float64) - np.asarray(line_start, dtype=np.float64)
    line_length = float(np.linalg.norm(tangent))
    if line_length <= 1e-6:
        return np.zeros(len(points), dtype=bool), np.zeros(len(points)), np.zeros(len(points)), 0.0, np.zeros(3), np.asarray(line_start, dtype=np.float64)
    tangent = tangent / line_length
    start = np.asarray(line_start, dtype=np.float64)
    center = start + tangent * (line_length * 0.5)
    rel = np.asarray(points, dtype=np.float64) - start
    axial = rel @ tangent
    closest = start + np.outer(axial, tangent)
    radial = np.linalg.norm(np.asarray(points, dtype=np.float64) - closest, axis=1)
    centered_axial = axial - (line_length * 0.5)
    mask = (np.abs(centered_axial) <= axial_half_window) & (radial <= radial_radius)
    effective_length = max(1e-6, axial_half_window * 2.0)
    return mask, centered_axial, radial, effective_length, tangent, center


def _coverage_ratio(roi_point_count: int, line_length: float, radial_radius: float, density_pts_per_cm2: float) -> float:
    density_mm2 = density_pts_per_cm2 / 100.0 if density_pts_per_cm2 > 0 else 0.1
    band_area_mm2 = max(1.0, line_length * max(2.0 * radial_radius, 6.0))
    expected_points = max(25.0, band_area_mm2 * density_mm2)
    return float(min(1.0, roi_point_count / expected_points))


def _side_axis(
    tangent: np.ndarray,
    n1: Optional[np.ndarray],
    n2: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if n1 is None and n2 is None:
        return None
    normal_a = np.asarray(n1 if n1 is not None else np.zeros(3), dtype=np.float64)
    normal_b = np.asarray(n2 if n2 is not None else np.zeros(3), dtype=np.float64)
    bisector = normal_a + normal_b
    bis_norm = float(np.linalg.norm(bisector))
    if bis_norm < 1e-6:
        candidate = normal_a if np.linalg.norm(normal_a) > np.linalg.norm(normal_b) else normal_b
        bisector = candidate
        bis_norm = float(np.linalg.norm(bisector))
        if bis_norm < 1e-6:
            return None
    bisector = bisector / bis_norm
    axis = np.cross(tangent, bisector)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return None
    return axis / axis_norm


def _signal_extent(axial_values: np.ndarray, signal_mask: np.ndarray, line_length: float) -> float:
    if line_length <= 1e-6 or signal_mask.sum() < 3:
        return 0.0
    bin_count = max(4, min(12, int(round(line_length / 10.0))))
    lo = float(np.min(axial_values))
    hi = float(np.max(axial_values))
    if hi - lo < 1e-6:
        return 0.0
    bins = np.linspace(lo, hi, num=bin_count + 1)
    occupied = np.unique(np.digitize(axial_values[signal_mask], bins, right=False))
    occupied = [idx for idx in occupied if 0 < idx <= bin_count]
    return float(min(1.0, len(occupied) / float(bin_count)))


def _match_line_geometry(match: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    cad_bend = getattr(match, "cad_bend", None)
    if cad_bend is None:
        return None, None
    start = getattr(cad_bend, "bend_line_start", None)
    end = getattr(cad_bend, "bend_line_end", None)
    if start is None or end is None:
        return None, None
    return np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)


def _match_normals(match: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    cad_bend = getattr(match, "cad_bend", None)
    if cad_bend is None:
        return None, None
    n1 = getattr(cad_bend, "flange1_normal", None)
    n2 = getattr(cad_bend, "flange2_normal", None)
    n1_arr = None if n1 is None else np.asarray(n1, dtype=np.float64)
    n2_arr = None if n2 is None else np.asarray(n2, dtype=np.float64)
    return n1_arr, n2_arr


def _match_feature_type(match: Any) -> str:
    cad_bend = getattr(match, "cad_bend", None)
    if cad_bend is None:
        return "UNKNOWN"
    return str(getattr(cad_bend, "feature_type", "UNKNOWN") or "UNKNOWN").upper()


def _match_countable(match: Any) -> bool:
    cad_bend = getattr(match, "cad_bend", None)
    if cad_bend is None:
        return True
    return bool(getattr(cad_bend, "countable_in_regression", True))


def _build_bend_result(
    match: Any,
    points: np.ndarray,
    deviations: np.ndarray,
    density_pts_per_cm2: float,
    global_tolerance_mm: float,
    scan_state: str,
) -> Tuple[Dict[str, Any], float]:
    bend_id = str(getattr(getattr(match, "cad_bend", None), "bend_id", "unknown"))
    feature_type = _match_feature_type(match)
    if not _match_countable(match) or feature_type in COUNTABLE_SKIP_FEATURE_TYPES:
        return {
            "status": "NOT_APPLICABLE",
            "consistency_score": None,
            "independence_class": _independence_class(getattr(match, "measurement_method", None)),
            "roi_point_count": 0,
            "roi_coverage_ratio": 0.0,
            "local_abs_mean_mm": None,
            "local_abs_p95_mm": None,
            "local_out_of_tol_ratio": None,
            "side_asymmetry_score": None,
            "signal_extent_along_flange": None,
            "notes": ["Process or rolled feature excluded from bend heatmap validation."],
        }, 0.0

    line_start, line_end = _match_line_geometry(match)
    if line_start is None or line_end is None:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "consistency_score": None,
            "independence_class": _independence_class(getattr(match, "measurement_method", None)),
            "roi_mode": ROI_MODE,
            "axial_half_window_mm": None,
            "radial_radius_mm": None,
            "roi_point_count": 0,
            "roi_coverage_ratio": 0.0,
            "local_abs_mean_mm": None,
            "local_abs_p95_mm": None,
            "local_out_of_tol_ratio": None,
            "side_asymmetry_score": None,
            "signal_extent_along_flange": None,
            "notes": ["CAD bend line unavailable for local heatmap ROI extraction."],
        }, 0.0

    cad_bend = getattr(match, "cad_bend")
    tolerance_mm = max(global_tolerance_mm, _safe_float(getattr(cad_bend, "tolerance_radius", None), global_tolerance_mm), 0.25)
    target_radius = _safe_float(getattr(cad_bend, "target_radius", None), 0.0)
    radial_radius = float(max(4.0, min(10.0, max(2.5 * tolerance_mm, 1.25 * target_radius, 4.0))))
    axial_half_window = 10.0 if scan_state == "full" else 7.0

    mask, axial, _radial, line_length, tangent, center = _center_local_roi_mask(
        points,
        line_start,
        line_end,
        radial_radius,
        axial_half_window,
    )
    roi_count = int(mask.sum())
    coverage_ratio = _coverage_ratio(roi_count, line_length, radial_radius, density_pts_per_cm2)
    independence = _independence_class(getattr(match, "measurement_method", None))
    notes: List[str] = []
    if independence != "INDEPENDENT":
        notes.append(f"Corroboration only: measurement method {getattr(match, 'measurement_method', 'unknown')} is not independent from deviations.")

    if roi_count < 50 or coverage_ratio < 0.35:
        if roi_count < 50:
            notes.append(f"Local ROI has only {roi_count} aligned scan points.")
        if coverage_ratio < 0.35:
            notes.append(f"Local ROI coverage ratio is {coverage_ratio:.2f}.")
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "consistency_score": None,
            "independence_class": independence,
            "roi_mode": ROI_MODE,
            "axial_half_window_mm": round(float(axial_half_window), 3),
            "radial_radius_mm": round(float(radial_radius), 3),
            "roi_point_count": roi_count,
            "roi_coverage_ratio": round(coverage_ratio, 3),
            "local_abs_mean_mm": None,
            "local_abs_p95_mm": None,
            "local_out_of_tol_ratio": None,
            "side_asymmetry_score": None,
            "signal_extent_along_flange": None,
            "notes": notes,
        }, 0.0

    roi_devs = np.asarray(deviations[mask], dtype=np.float64)
    roi_abs = np.abs(roi_devs)
    local_abs_mean = float(np.mean(roi_abs))
    local_abs_p95 = float(np.percentile(roi_abs, 95))
    local_out_ratio = float(np.mean(roi_abs > tolerance_mm))
    signal_threshold = max(tolerance_mm, 0.35)
    local_signal_mask = roi_abs > signal_threshold
    signal_extent = _signal_extent(axial[mask], local_signal_mask, line_length)

    n1, n2 = _match_normals(match)
    side_axis = _side_axis(tangent, n1, n2)
    side_asymmetry_score: Optional[float] = None
    if side_axis is not None:
        signed_side = (np.asarray(points[mask], dtype=np.float64) - center) @ side_axis
        pos = roi_abs[signed_side >= 0.0]
        neg = roi_abs[signed_side < 0.0]
        if len(pos) >= 5 and len(neg) >= 5:
            denom = max(local_abs_mean, 1e-6)
            side_asymmetry_score = float(min(1.0, abs(float(pos.mean()) - float(neg.mean())) / denom))
    if side_asymmetry_score is None:
        side_asymmetry_score = 0.0
        notes.append("Side asymmetry is weak or unavailable for this bend ROI.")

    bend_status = str(getattr(match, "status", "NOT_DETECTED") or "NOT_DETECTED").upper()
    consistency_score: Optional[float] = None
    consistency_status = "INSUFFICIENT_EVIDENCE"
    contradiction_allowed = (
        scan_state == "full" and (
            independence == "INDEPENDENT" or independence == "PARTIAL"
        )
    )
    strong_roi = roi_count >= 150 and coverage_ratio >= 0.60
    moderate_roi = roi_count >= 80 and coverage_ratio >= 0.40
    pass_hot = (
        strong_roi
        and local_abs_p95 >= max(2.0 * tolerance_mm, 0.8)
        and local_out_ratio >= 0.60
        and side_asymmetry_score >= 0.20
    )
    pass_quiet = (
        moderate_roi
        and local_abs_p95 <= max(1.1 * tolerance_mm, 0.45)
        and local_out_ratio <= 0.20
    )
    pass_quiet_partial_full = (
        scan_state == "full"
        and independence == "PARTIAL"
        and moderate_roi
        and local_abs_p95 <= max(1.3 * tolerance_mm, 0.65)
        and local_out_ratio <= 0.12
    )
    fail_hot = (
        moderate_roi
        and (
            local_abs_p95 >= max(1.2 * tolerance_mm, 0.5)
            or local_out_ratio >= 0.20
        )
    )
    fail_warm_full = (
        scan_state == "full"
        and strong_roi
        and local_abs_p95 >= max(1.1 * tolerance_mm, 0.55)
        and local_out_ratio >= 0.10
    )
    fail_quiet = (
        strong_roi
        and local_abs_p95 <= max(1.0 * tolerance_mm, 0.40)
        and local_out_ratio <= 0.10
    )

    if bend_status in PASSIVE_STATES:
        if pass_hot and contradiction_allowed and independence == "INDEPENDENT":
            consistency_status = "CONTRADICTED"
            consistency_score = round(float(min(1.0, max(local_abs_p95 / max(2.0 * tolerance_mm, 1e-6), local_out_ratio))), 3)
            notes.append("Tight bend-center ROI is locally hot around a bend reported in spec.")
        elif pass_quiet:
            consistency_status = "SUPPORTED"
            consistency_score = round(
                float(
                    min(
                        1.0,
                        max(
                            1.0 - (local_abs_p95 / max(max(1.1 * tolerance_mm, 0.45), 1e-6)),
                            1.0 - min(1.0, local_out_ratio / 0.20),
                        ),
                    )
                ),
                3,
            )
        elif pass_quiet_partial_full:
            consistency_status = "SUPPORTED"
            consistency_score = round(
                float(
                    min(
                        1.0,
                        max(
                            1.0 - (local_abs_p95 / max(max(1.3 * tolerance_mm, 0.65), 1e-6)),
                            1.0 - min(1.0, local_out_ratio / 0.12),
                        ),
                    )
                ),
                3,
            )
            notes.append("Full-scan partial-independence ROI is quiet enough to support this in-spec bend.")
        else:
            consistency_status = "INSUFFICIENT_EVIDENCE"
    elif bend_status in ACTIVE_FAIL_STATES:
        if fail_hot:
            fail_support_p95_threshold = max(1.2 * tolerance_mm, 0.5)
            fail_support_out_ratio_threshold = 0.20
            consistency_status = "SUPPORTED"
            consistency_score = round(
                float(
                    min(
                        1.0,
                        max(
                            local_abs_p95 / max(fail_support_p95_threshold, 1e-6),
                            local_out_ratio / max(fail_support_out_ratio_threshold, 1e-6),
                        ),
                    )
                ),
                3,
            )
        elif fail_warm_full:
            consistency_status = "SUPPORTED"
            consistency_score = round(
                float(
                    min(
                        1.0,
                        max(
                            local_abs_p95 / max(max(1.1 * tolerance_mm, 0.55), 1e-6),
                            local_out_ratio / 0.10,
                        ),
                    )
                ),
                3,
            )
            notes.append("Full-scan bend ROI shows moderate local deviation consistent with an out-of-spec bend.")
        elif fail_quiet and contradiction_allowed and independence == "INDEPENDENT":
            consistency_status = "CONTRADICTED"
            consistency_score = round(
                float(
                    min(
                        1.0,
                        max(
                            1.0 - (local_abs_p95 / max(max(1.0 * tolerance_mm, 0.40), 1e-6)),
                            1.0 - min(1.0, local_out_ratio / 0.10),
                        ),
                    )
                ),
                3,
            )
            notes.append("Tight bend-center ROI is quiet around a bend reported out of spec.")
        else:
            consistency_status = "INSUFFICIENT_EVIDENCE"
    else:
        consistency_status = "INSUFFICIENT_EVIDENCE"
        consistency_score = None
        notes.append("v2 keeps NOT_DETECTED bends advisory-only and never contradicts them.")

    result = {
        "status": consistency_status,
        "consistency_score": consistency_score,
        "independence_class": independence,
        "roi_mode": ROI_MODE,
        "axial_half_window_mm": round(float(axial_half_window), 3),
        "radial_radius_mm": round(float(radial_radius), 3),
        "roi_point_count": roi_count,
        "roi_coverage_ratio": round(coverage_ratio, 3),
        "local_abs_mean_mm": round(local_abs_mean, 4),
        "local_abs_p95_mm": round(local_abs_p95, 4),
        "local_out_of_tol_ratio": round(local_out_ratio, 4),
        "side_asymmetry_score": round(float(side_asymmetry_score), 4),
        "signal_extent_along_flange": round(float(signal_extent), 4),
        "notes": notes,
    }
    return result, float(np.sum(roi_abs))


def evaluate_bend_heatmap_consistency(
    report: Any,
    aligned_points: np.ndarray,
    deviations: np.ndarray,
    *,
    scan_state: Optional[str] = None,
    scan_name: Optional[str] = None,
    global_tolerance_mm: float = 1.0,
) -> Dict[str, Any]:
    points = np.asarray(aligned_points, dtype=np.float64)
    devs = np.asarray(deviations, dtype=np.float64)
    if len(points) == 0 or len(points) != len(devs):
        return {
            "matches": {},
            "summary": {
                "status": "NOT_APPLICABLE",
                "inconsistent_bend_count": 0,
                "insufficient_evidence_bend_count": 0,
                "bend_roi_energy_share": None,
                "global_in_tolerance_rate": None,
                "notes": ["Aligned scan points and deviations are unavailable or mismatched."],
            },
        }

    density = _safe_float(getattr(report, "scan_quality", {}).get("density_pts_per_cm2") if getattr(report, "scan_quality", None) else 0.0)
    norm_scan_state = _normalize_scan_state(scan_state, scan_name)
    match_results: Dict[str, Dict[str, Any]] = {}
    bend_roi_energy = 0.0
    inconsistent = 0
    insufficient = 0
    evaluable = 0
    failish_count = 0
    pass_count = 0

    for match in list(getattr(report, "matches", []) or []):
        bend_id = str(getattr(getattr(match, "cad_bend", None), "bend_id", "unknown"))
        result, roi_energy = _build_bend_result(match, points, devs, density, global_tolerance_mm, norm_scan_state)
        match_results[bend_id] = result
        bend_roi_energy += roi_energy
        if result["status"] == "CONTRADICTED" and norm_scan_state == "full" and result.get("independence_class") in {"INDEPENDENT", "PARTIAL"}:
            inconsistent += 1
        if result["status"] == "INSUFFICIENT_EVIDENCE":
            insufficient += 1
        if result["status"] in {"SUPPORTED", "CONTRADICTED"}:
            evaluable += 1
        bend_status = str(getattr(match, "status", "NOT_DETECTED") or "NOT_DETECTED").upper()
        if bend_status in ACTIVE_FAIL_STATES:
            failish_count += 1
        elif bend_status in PASSIVE_STATES:
            pass_count += 1

    abs_devs = np.abs(devs)
    global_in_tol = float(np.mean(abs_devs <= global_tolerance_mm)) if len(abs_devs) else None
    total_energy = float(np.sum(abs_devs)) if len(abs_devs) else 0.0
    roi_energy_share = float(bend_roi_energy / total_energy) if total_energy > 1e-9 else 0.0
    summary_notes: List[str] = []
    if norm_scan_state != "full":
        summary_status = "NOT_APPLICABLE"
        summary_notes.append("Whole-part heatmap agreement is only scored on full scans in v2.")
    else:
        if failish_count > 0:
            if inconsistent >= max(1, failish_count) and (global_in_tol or 0.0) >= 0.97:
                summary_status = "CONTRADICTED"
                summary_notes.append("Reported bend failures are not reflected by the local heatmap evidence despite globally quiet whole-part deviations.")
            elif evaluable > 0 and (roi_energy_share >= 0.06 or (global_in_tol or 0.0) < 0.95):
                summary_status = "SUPPORTED"
            else:
                summary_status = "WEAK_SUPPORT"
        else:
            if evaluable > 0 and (global_in_tol or 0.0) >= 0.97 and inconsistent == 0:
                summary_status = "SUPPORTED"
            elif (global_in_tol or 0.0) < 0.90:
                summary_status = "CONTRADICTED"
                summary_notes.append("Whole-part heatmap remains globally hot despite all bend verdicts being non-failing.")
            else:
                summary_status = "WEAK_SUPPORT"

    return {
        "matches": match_results,
        "summary": {
            "status": summary_status,
            "inconsistent_bend_count": inconsistent,
            "insufficient_evidence_bend_count": insufficient,
            "evaluable_bend_count": evaluable,
            "bend_roi_energy_share": round(roi_energy_share, 4) if norm_scan_state == "full" else None,
            "global_in_tolerance_rate": round(global_in_tol, 4) if global_in_tol is not None and norm_scan_state == "full" else None,
            "notes": summary_notes,
        },
    }


def merge_heatmap_consistency_into_report(report_dict: Dict[str, Any], consistency_bundle: Dict[str, Any]) -> Dict[str, Any]:
    matches_by_id = (consistency_bundle or {}).get("matches") or {}
    for match in report_dict.get("matches") or []:
        bend_id = str(match.get("bend_id") or "")
        if bend_id in matches_by_id:
            match["heatmap_consistency"] = matches_by_id[bend_id]
    summary = report_dict.setdefault("summary", {})
    summary["heatmap_consistency_summary"] = (consistency_bundle or {}).get("summary") or {
        "status": "NOT_APPLICABLE",
        "inconsistent_bend_count": 0,
        "insufficient_evidence_bend_count": 0,
        "bend_roi_energy_share": None,
        "global_in_tolerance_rate": None,
        "notes": ["Heatmap consistency was not evaluated."],
    }
    return report_dict
