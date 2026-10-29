#!/usr/bin/env python3
"""Measure scan support for source-native bend carriers without a global CAD mesh.

For each EXP17 event, scan points are projected onto its finite neutral axis.
Points support the carrier when their radial distance matches either native
cylindrical radius within a source-thickness-scaled tolerance.  Coverage is the
fraction of finite axis bins containing carrier points.  This diagnostic is
local, source-only, and intentionally independent of incomplete global mesh
tessellation and evaluator truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_carrier_support(
    points: np.ndarray,
    event: Mapping[str, Any],
    *,
    thickness_mm: float,
    min_points_per_bin: int = 3,
) -> tuple[dict[str, Any], np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    start = np.asarray(event["line_start_mm"], dtype=np.float64)
    end = np.asarray(event["line_end_mm"], dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("scan points must be finite Nx3")
    if not math.isfinite(length) or length <= 0:
        raise ValueError("event axis must have finite positive length")
    axis = direction / length
    relative = points - start
    parameters = relative @ axis
    radial_vectors = relative - parameters[:, None] * axis[None, :]
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radii = np.asarray(event["radii_mm"], dtype=np.float64)
    if radii.shape != (2,) or not np.isfinite(radii).all() or np.min(radii) < 0:
        raise ValueError("event must provide two finite native radii")
    tolerance_mm = max(0.8, 0.6 * float(thickness_mm))
    in_segment = (parameters >= -tolerance_mm) & (parameters <= length + tolerance_mm)
    radial_residual = np.min(np.abs(radial_distances[:, None] - radii[None, :]), axis=1)
    carrier_mask = in_segment & (radial_residual <= tolerance_mm)
    bin_count = max(8, min(48, int(math.ceil(length / max(4.0 * thickness_mm, 4.0)))))
    clipped = np.clip(parameters[carrier_mask], 0.0, np.nextafter(length, 0.0))
    occupied = np.zeros(bin_count, dtype=np.int64)
    if len(clipped):
        indices = np.minimum((clipped / length * bin_count).astype(np.int64), bin_count - 1)
        occupied = np.bincount(indices, minlength=bin_count)
    supported_bins = occupied >= min_points_per_bin
    carrier_residuals = radial_residual[carrier_mask]
    result = {
        "event_id": str(event["event_id"]),
        "span_mm": length,
        "native_radii_mm": radii.tolist(),
        "radial_tolerance_mm": tolerance_mm,
        "bin_count": bin_count,
        "min_points_per_bin": min_points_per_bin,
        "carrier_point_count": int(carrier_mask.sum()),
        "supported_bin_count": int(supported_bins.sum()),
        "carrier_coverage_fraction": float(supported_bins.mean()),
        "median_carrier_radial_residual_mm": float(np.median(carrier_residuals)) if len(carrier_residuals) else None,
        "p90_carrier_radial_residual_mm": float(np.percentile(carrier_residuals, 90)) if len(carrier_residuals) else None,
        "occupied_bin_point_counts": occupied.tolist(),
    }
    return result, carrier_mask


def render_support(
    path: Path,
    case_token: str,
    points: np.ndarray,
    events: list[Mapping[str, Any]],
    carrier_union: np.ndarray,
) -> None:
    step = max(1, len(points) // 150000)
    scan = points[::step]
    supported = carrier_union[::step]
    projections = ((0, 2, "front X–Z"), (1, 2, "side Y–Z"), (0, 1, "top X–Y"))
    figure, axes = plt.subplots(1, 3, figsize=(30, 18), dpi=100, facecolor="#0b1016")
    for axis, (horizontal, vertical, title) in zip(axes, projections):
        axis.set_facecolor("#0b1016")
        axis.scatter(scan[~supported, horizontal], scan[~supported, vertical], color="#46d9ff", s=0.8, alpha=0.25, linewidths=0)
        axis.scatter(scan[supported, horizontal], scan[supported, vertical], color="#72ff70", s=2.0, alpha=0.78, linewidths=0)
        for event in events:
            start = np.asarray(event["line_start_mm"], dtype=np.float64)
            end = np.asarray(event["line_end_mm"], dtype=np.float64)
            axis.plot([start[horizontal], end[horizontal]], [start[vertical], end[vertical]], color="#ff2da8", linewidth=4.0, solid_capstyle="round")
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
        f"{case_token} — local source-native bend carrier support\n"
        "green: scan points matching native cylindrical carriers; magenta: native axes; cyan: remaining scan",
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
        extraction = summary["extraction"]
        events = extraction["events"]
        thickness_mm = float(extraction["sheet_thickness_mm"])
        for source_record in summary["records"]:
            case_token = source_record["case_token"]
            if case_token in seen_cases:
                raise ValueError(f"duplicate case token {case_token}")
            seen_cases.add(case_token)
            registration_semantic = args.registration_root / "cases" / case_token / "semantic_result.json"
            registration = json.loads(registration_semantic.read_text(encoding="utf-8"))
            aligned_scan = Path(registration["artifacts"]["aligned_scan"]["path"])
            if sha256_path(aligned_scan) != registration["artifacts"]["aligned_scan"]["sha256"]:
                raise ValueError(f"aligned scan hash mismatch for {case_token}")
            points = np.asarray(o3d.io.read_point_cloud(str(aligned_scan)).points, dtype=np.float64)
            union = np.zeros(len(points), dtype=bool)
            event_metrics = []
            for event in events:
                metric, mask = evaluate_carrier_support(points, event, thickness_mm=thickness_mm)
                union |= mask
                event_metrics.append(metric)
            coverages = np.asarray([row["carrier_coverage_fraction"] for row in event_metrics], dtype=np.float64)
            case_root = args.output_root / "cases" / case_token
            render_path = case_root / "native_carrier_support_closeup.png"
            render_support(render_path, case_token, points, events, union)
            payload = {
                "schema": "EXP_CAD_0018_NATIVE_CARRIER_SUPPORT_CASE_V1",
                "case_token": case_token,
                "source_candidate_summary_sha256": sha256_path(summary_path),
                "registration_semantic_sha256": sha256_path(registration_semantic),
                "source_sldprt_sha256": summary["source_sldprt_sha256"],
                "sheet_thickness_mm": thickness_mm,
                "event_count": len(event_metrics),
                "mean_carrier_coverage_fraction": float(coverages.mean()),
                "min_carrier_coverage_fraction": float(coverages.min()),
                "events_with_at_least_half_coverage": int(np.sum(coverages >= 0.5)),
                "event_metrics": event_metrics,
                "render": {"path": str(render_path.resolve()), "sha256": sha256_path(render_path)},
                "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
                "evaluator_data_opened": False,
                "labels_or_expected_counts_used": False,
                "is_metric_eligible": False,
            }
            atomic_json(case_root / "semantic_result.json", payload)
            records.append({"case_token": case_token, "semantic_sha256": sha256_path(case_root / "semantic_result.json"), "render_sha256": sha256_path(render_path), "mean_carrier_coverage_fraction": payload["mean_carrier_coverage_fraction"], "min_carrier_coverage_fraction": payload["min_carrier_coverage_fraction"], "events_with_at_least_half_coverage": payload["events_with_at_least_half_coverage"], "event_count": len(event_metrics)})
    output = {
        "schema": "EXP_CAD_0018_NATIVE_CARRIER_SUPPORT_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0018",
        "case_count": len(records),
        "records": records,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE",
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
        "is_metric_eligible": False,
    }
    atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", output)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
