#!/usr/bin/env python3
"""Search locally for displaced source-native bend-carrier support.

The search translates each finite CAD axis only in its perpendicular plane.  A
source-thickness-scaled window and deterministic coarse-to-fine grid are fixed
before evaluation.  The objective is finite-bin radial-annulus coverage, then
radial residual, then minimum displacement.  This localizes possible carrier
motion but does not by itself establish curvature or pass/fail disposition.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


HERE = Path(__file__).resolve().parent
EXP18_PATH = HERE / "run_exp_cad_0018_native_carrier_support.py"
SPEC = importlib.util.spec_from_file_location("exp18_carrier_support", EXP18_PATH)
EXP18 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP18)


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    anchors = np.eye(3)
    anchor = anchors[int(np.argmin(np.abs(anchors @ axis)))]
    first = np.cross(axis, anchor)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def shifted_event(event: Mapping[str, Any], shift: np.ndarray) -> dict[str, Any]:
    result = dict(event)
    result["line_start_mm"] = (np.asarray(event["line_start_mm"], dtype=float) + shift).tolist()
    result["line_end_mm"] = (np.asarray(event["line_end_mm"], dtype=float) + shift).tolist()
    return result


def search_local_carrier(
    points: np.ndarray,
    event: Mapping[str, Any],
    *,
    thickness_mm: float,
    max_search_points: int = 40000,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    start = np.asarray(event["line_start_mm"], dtype=float)
    end = np.asarray(event["line_end_mm"], dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    axis = direction / length
    first, second = perpendicular_basis(axis)
    limit_mm = max(6.0, 6.0 * float(thickness_mm))
    coarse_step_mm = max(1.0, float(thickness_mm))
    tolerance_mm = max(0.8, 0.6 * float(thickness_mm))
    relative = points - start
    parameters = relative @ axis
    radial = np.linalg.norm(relative - parameters[:, None] * axis[None, :], axis=1)
    max_radius = max(float(value) for value in event["radii_mm"])
    local_mask = (
        (parameters >= -tolerance_mm)
        & (parameters <= length + tolerance_mm)
        & (radial <= max_radius + limit_mm + tolerance_mm)
    )
    local_points = points[local_mask]
    if not len(local_points):
        raise ValueError("no scan points inside the bounded local search volume")
    sample_step = max(1, int(math.ceil(len(local_points) / max_search_points)))
    search_points = local_points[::sample_step]

    def score(offset_first: float, offset_second: float) -> tuple[tuple[float, float, float], dict[str, Any]]:
        shift = first * offset_first + second * offset_second
        candidate = shifted_event(event, shift)
        metric, _mask = EXP18.evaluate_carrier_support(
            search_points,
            candidate,
            thickness_mm=thickness_mm,
            min_points_per_bin=1,
        )
        residual = metric["median_carrier_radial_residual_mm"]
        residual_score = -float(residual) if residual is not None else -1e9
        displacement = float(np.linalg.norm(shift))
        return (
            float(metric["carrier_coverage_fraction"]),
            residual_score,
            -displacement,
        ), metric

    coarse_values = np.arange(-limit_mm, limit_mm + 0.5 * coarse_step_mm, coarse_step_mm)
    best = None
    for offset_first in coarse_values:
        for offset_second in coarse_values:
            objective, metric = score(float(offset_first), float(offset_second))
            key = (*objective, -abs(float(offset_first)), -abs(float(offset_second)))
            if best is None or key > best[0]:
                best = (key, float(offset_first), float(offset_second), metric)
    assert best is not None
    refinement_step_mm = coarse_step_mm / 4.0
    refinement = np.arange(-coarse_step_mm, coarse_step_mm + 0.5 * refinement_step_mm, refinement_step_mm)
    coarse_first, coarse_second = best[1], best[2]
    for delta_first in refinement:
        for delta_second in refinement:
            offset_first = float(np.clip(coarse_first + delta_first, -limit_mm, limit_mm))
            offset_second = float(np.clip(coarse_second + delta_second, -limit_mm, limit_mm))
            objective, metric = score(offset_first, offset_second)
            key = (*objective, -abs(offset_first), -abs(offset_second))
            if key > best[0]:
                best = (key, offset_first, offset_second, metric)

    shift = first * best[1] + second * best[2]
    candidate = shifted_event(event, shift)
    final_metric, final_mask = EXP18.evaluate_carrier_support(
        points,
        candidate,
        thickness_mm=thickness_mm,
        min_points_per_bin=3,
    )
    nominal_metric, _ = EXP18.evaluate_carrier_support(
        points,
        event,
        thickness_mm=thickness_mm,
        min_points_per_bin=3,
    )
    result = {
        "event_id": str(event["event_id"]),
        "search_limit_mm": limit_mm,
        "coarse_step_mm": coarse_step_mm,
        "refinement_step_mm": refinement_step_mm,
        "local_search_point_count": int(len(local_points)),
        "search_sample_point_count": int(len(search_points)),
        "offset_basis_first": first.tolist(),
        "offset_basis_second": second.tolist(),
        "best_offset_coordinates_mm": [best[1], best[2]],
        "best_shift_vector_mm": shift.tolist(),
        "best_shift_norm_mm": float(np.linalg.norm(shift)),
        "nominal_coverage_fraction": nominal_metric["carrier_coverage_fraction"],
        "best_coverage_fraction": final_metric["carrier_coverage_fraction"],
        "coverage_gain": final_metric["carrier_coverage_fraction"] - nominal_metric["carrier_coverage_fraction"],
        "best_median_radial_residual_mm": final_metric["median_carrier_radial_residual_mm"],
        "best_carrier_point_count": final_metric["carrier_point_count"],
    }
    return result, candidate, final_mask


def render_search(
    path: Path,
    case_token: str,
    points: np.ndarray,
    native_events: list[Mapping[str, Any]],
    shifted_events: list[Mapping[str, Any]],
    support_union: np.ndarray,
) -> None:
    step = max(1, len(points) // 150000)
    scan = points[::step]
    supported = support_union[::step]
    projections = ((0, 2, "front X–Z"), (1, 2, "side Y–Z"), (0, 1, "top X–Y"))
    figure, axes = plt.subplots(1, 3, figsize=(30, 18), dpi=100, facecolor="#0b1016")
    for axis, (horizontal, vertical, title) in zip(axes, projections):
        axis.set_facecolor("#0b1016")
        axis.scatter(scan[~supported, horizontal], scan[~supported, vertical], color="#46d9ff", s=0.8, alpha=0.24, linewidths=0)
        axis.scatter(scan[supported, horizontal], scan[supported, vertical], color="#72ff70", s=2.0, alpha=0.72, linewidths=0)
        for native, shifted in zip(native_events, shifted_events):
            native_start = np.asarray(native["line_start_mm"], dtype=float)
            native_end = np.asarray(native["line_end_mm"], dtype=float)
            shifted_start = np.asarray(shifted["line_start_mm"], dtype=float)
            shifted_end = np.asarray(shifted["line_end_mm"], dtype=float)
            axis.plot([native_start[horizontal], native_end[horizontal]], [native_start[vertical], native_end[vertical]], color="#ff2da8", linewidth=3.0, alpha=0.75)
            axis.plot([shifted_start[horizontal], shifted_end[horizontal]], [shifted_start[vertical], shifted_end[vertical]], color="#ffd23f", linewidth=3.5, alpha=0.95)
        h_min, h_max = np.percentile(scan[:, horizontal], [0.05, 99.95])
        v_min, v_max = np.percentile(scan[:, vertical], [0.05, 99.95])
        h_margin = max((h_max - h_min) * 0.04, 1.0)
        v_margin = max((v_max - v_min) * 0.04, 1.0)
        axis.set_xlim(h_min - h_margin, h_max + h_margin)
        axis.set_ylim(v_min - v_margin, v_max + v_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, color="white", fontsize=19, pad=14)
        axis.set_axis_off()
    figure.suptitle(
        f"{case_token} — bounded local carrier search\n"
        "magenta: nominal CAD axis; yellow: best local axis; green: supporting scan; cyan: remaining scan",
        color="white", fontsize=22, y=0.96,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", action="append", required=True, type=Path)
    parser.add_argument("--registration-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    records = []
    seen_cases: set[str] = set()
    for summary_path in args.candidate_summary:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = summary["extraction"]["events"]
        thickness_mm = float(summary["extraction"]["sheet_thickness_mm"])
        for source_record in summary["records"]:
            case_token = source_record["case_token"]
            if case_token in seen_cases:
                raise ValueError(f"duplicate case token {case_token}")
            seen_cases.add(case_token)
            registration_semantic = args.registration_root / "cases" / case_token / "semantic_result.json"
            registration = json.loads(registration_semantic.read_text(encoding="utf-8"))
            aligned_scan = Path(registration["artifacts"]["aligned_scan"]["path"])
            points = np.asarray(o3d.io.read_point_cloud(str(aligned_scan)).points, dtype=np.float64)
            event_results, shifted_events = [], []
            support_union = np.zeros(len(points), dtype=bool)
            for event in events:
                result, shifted, mask = search_local_carrier(points, event, thickness_mm=thickness_mm)
                event_results.append(result)
                shifted_events.append(shifted)
                support_union |= mask
            shifts = np.asarray([row["best_shift_norm_mm"] for row in event_results], dtype=float)
            gains = np.asarray([row["coverage_gain"] for row in event_results], dtype=float)
            case_root = args.output_root / "cases" / case_token
            render_path = case_root / "local_carrier_search_closeup.png"
            render_search(render_path, case_token, points, events, shifted_events, support_union)
            payload = {
                "schema": "EXP_CAD_0019_LOCAL_CARRIER_SEARCH_CASE_V1",
                "case_token": case_token,
                "source_candidate_summary_sha256": EXP18.sha256_path(summary_path),
                "registration_semantic_sha256": EXP18.sha256_path(registration_semantic),
                "source_sldprt_sha256": summary["source_sldprt_sha256"],
                "sheet_thickness_mm": thickness_mm,
                "event_count": len(event_results),
                "median_best_shift_norm_mm": float(np.median(shifts)),
                "max_best_shift_norm_mm": float(np.max(shifts)),
                "mean_coverage_gain": float(np.mean(gains)),
                "event_results": event_results,
                "render": {"path": str(render_path.resolve()), "sha256": EXP18.sha256_path(render_path)},
                "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
                "evaluator_data_opened": False,
                "labels_or_expected_counts_used": False,
                "is_metric_eligible": False,
            }
            EXP18.atomic_json(case_root / "semantic_result.json", payload)
            records.append({"case_token": case_token, "semantic_sha256": EXP18.sha256_path(case_root / "semantic_result.json"), "render_sha256": EXP18.sha256_path(render_path), "median_best_shift_norm_mm": payload["median_best_shift_norm_mm"], "max_best_shift_norm_mm": payload["max_best_shift_norm_mm"], "mean_coverage_gain": payload["mean_coverage_gain"]})
    summary = {
        "schema": "EXP_CAD_0019_LOCAL_CARRIER_SEARCH_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0019",
        "case_count": len(records),
        "records": records,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE",
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
        "is_metric_eligible": False,
    }
    EXP18.atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
