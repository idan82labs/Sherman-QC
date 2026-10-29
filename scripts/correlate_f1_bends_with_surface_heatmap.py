#!/usr/bin/env python3
"""Correlate bend-by-bend F1 changes with registered surface deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def _points(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        points = np.asarray(mesh.vertices, dtype=np.float64)
    return points[np.isfinite(points).all(axis=1)]


def _endpoints(strip: dict, transform: np.ndarray | None = None) -> np.ndarray:
    hint = strip.get("axis_line_hint") or {}
    supported = hint.get("supported_endpoints")
    if isinstance(supported, list) and len(supported) >= 2:
        endpoints = np.asarray(supported[:2], dtype=np.float64)
    else:
        midpoint = np.asarray(hint["midpoint"], dtype=np.float64)
        axis = np.asarray(strip["axis_direction"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        half = axis * float(strip["along_axis_span_mm"]) * 0.5
        endpoints = np.asarray([midpoint - half, midpoint + half])
    if transform is not None:
        endpoints = endpoints @ transform[:3, :3].T + transform[:3, 3]
    return endpoints


def _finite_segment_distance(points: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    delta = endpoints[1] - endpoints[0]
    length_squared = float(np.dot(delta, delta))
    parameter = np.clip((points - endpoints[0]) @ delta / max(length_squared, 1e-12), 0.0, 1.0)
    closest = endpoints[0] + parameter[:, None] * delta
    return np.linalg.norm(points - closest, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good-scan", type=Path, required=True)
    parser.add_argument("--defective-scan", type=Path, required=True)
    parser.add_argument("--registration-summary", type=Path, required=True)
    parser.add_argument("--good-result", type=Path, required=True)
    parser.add_argument("--defective-result", type=Path, required=True)
    parser.add_argument("--manual-score", type=Path, required=True)
    parser.add_argument("--bend-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corridor-mm", type=float, default=12.0)
    args = parser.parse_args()

    good_points = _points(args.good_scan)
    defective_points = _points(args.defective_scan)
    registration = json.loads(args.registration_summary.read_text(encoding="utf-8"))
    transform = np.asarray(registration["transform_defective_to_good"], dtype=np.float64)
    aligned_defective = defective_points @ transform[:3, :3].T + transform[:3, 3]
    surface_distance, _ = cKDTree(good_points).query(aligned_defective, k=1, workers=1)

    good_payload = json.loads(args.good_result.read_text(encoding="utf-8"))
    defective_payload = json.loads(args.defective_result.read_text(encoding="utf-8"))
    good_strips = {str(item["strip_id"]): item for item in good_payload["bend_support_strips"]}
    defective_strips = {str(item["strip_id"]): item for item in defective_payload["bend_support_strips"]}
    score = json.loads(args.manual_score.read_text(encoding="utf-8"))
    good_strip_to_b = {str(item["prediction_id"]): str(item["gold_id"]) for item in score["matches"]}
    comparison = json.loads(args.bend_comparison.read_text(encoding="utf-8"))

    rows = []
    for match in comparison["matched_bends"]:
        good_id = str(match["good_strip_id"])
        defective_id = str(match["defective_strip_id"])
        bend_id = good_strip_to_b.get(good_id, good_id)
        good_line = _endpoints(good_strips[good_id])
        corridor_distance = _finite_segment_distance(aligned_defective, good_line)
        local = surface_distance[corridor_distance <= float(args.corridor_mm)]
        if len(local):
            percentiles = np.quantile(local, [0.5, 0.8, 0.95, 0.99])
            local_metrics = {
                "point_count": int(len(local)),
                "p50_mm": float(percentiles[0]),
                "p80_mm": float(percentiles[1]),
                "p95_mm": float(percentiles[2]),
                "p99_mm": float(percentiles[3]),
                "fraction_above_1mm": float(np.mean(local > 1.0)),
                "fraction_above_1_5mm": float(np.mean(local > 1.5)),
            }
        else:
            local_metrics = {"point_count": 0}
        f1_reasons = []
        if float(match["segment_distance_mm"]) >= 1.0:
            f1_reasons.append("axis_offset_at_least_1mm")
        if abs(float(match["direction_delta_deg"])) >= 1.0:
            f1_reasons.append("axis_direction_delta_at_least_1deg")
        if abs(float(match["angle_delta_deg"])) >= 0.75:
            f1_reasons.append("incident_plane_angle_delta_at_least_0_75deg")
        heatmap_hot = float(local_metrics.get("p95_mm") or 0.0) >= 1.5
        rows.append(
            {
                "bend_id": bend_id,
                "good_strip_id": good_id,
                "defective_strip_id": defective_id,
                "f1": {
                    "axis_offset_mm": float(match["segment_distance_mm"]),
                    "axis_direction_delta_deg": float(match["direction_delta_deg"]),
                    "span_delta_mm": float(match["span_delta_mm"]),
                    "incident_plane_angle_delta_deg": float(match["angle_delta_deg"]),
                    "screening_deviation": bool(f1_reasons),
                    "screening_reasons": f1_reasons,
                },
                "surface_heatmap_corridor": local_metrics,
                "heatmap_hot_at_p95_1_5mm": heatmap_hot,
                "f1_heatmap_agreement": bool(f1_reasons) == heatmap_hot,
                "concordant_deviation": bool(f1_reasons) and heatmap_hot,
            }
        )
    rows.sort(key=lambda item: int("".join(ch for ch in item["bend_id"] if ch.isdigit()) or 10**9))

    center = np.median(good_points, axis=0)
    sample = good_points[:: max(1, len(good_points) // 200_000)] - center
    _, _, basis = np.linalg.svd(sample, full_matrices=False)
    sample_step = max(1, len(aligned_defective) // 180_000)
    render_points = (aligned_defective[::sample_step] - center) @ basis.T
    render_distance = surface_distance[::sample_step]
    row_by_b = {item["bend_id"]: item for item in rows}
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=180, constrained_layout=True)
    views = ((0, 1, "top / plan"), (0, 2, "long-side profile"), (1, 2, "short-side profile"))
    for ax, (first, second, title) in zip(axes, views):
        scatter = ax.scatter(
            render_points[:, first],
            render_points[:, second],
            c=np.clip(render_distance, 0.0, 2.0),
            cmap="turbo",
            vmin=0.0,
            vmax=2.0,
            s=0.35,
            linewidths=0,
            alpha=0.72,
            rasterized=True,
        )
        for good_id, bend_id in good_strip_to_b.items():
            line = (_endpoints(good_strips[good_id]) - center) @ basis.T
            row = row_by_b[bend_id]
            f1_flagged = bool(row["f1"]["screening_deviation"])
            heatmap_hot = bool(row["heatmap_hot_at_p95_1_5mm"])
            if f1_flagged and heatmap_hot:
                color = "#e5173f"
            elif f1_flagged:
                color = "#f59e0b"
            elif heatmap_hot:
                color = "#7c3aed"
            else:
                color = "#20c875"
            ax.plot(line[:, first], line[:, second], color="black", linewidth=5.0, alpha=0.7)
            ax.plot(line[:, first], line[:, second], color=color, linewidth=2.8)
            midpoint = line.mean(axis=0)
            ax.text(
                midpoint[first],
                midpoint[second],
                bend_id,
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": color, "edgecolor": "black", "alpha": 0.95},
            )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.14)
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
    fig.colorbar(scatter, ax=axes, label="X-to-V nearest-surface deviation (mm), clipped at 2")
    fig.suptitle(
        "38172000 — corrected F1 over independent X-vs-V surface heatmap\n"
        "red = corroborated, amber = F1-only, purple = heatmap-only, green = neither"
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "f1_bends_over_surface_heatmap.png"
    fig.savefig(image_path, facecolor="white")
    plt.close(fig)

    payload = {
        "schema_version": 1,
        "part_id": "38172000_01",
        "good_label": "V",
        "defective_label": "X",
        "screening_thresholds_not_production_tolerances": {
            "axis_offset_mm": 1.0,
            "axis_direction_delta_deg": 1.0,
            "incident_plane_angle_delta_deg": 0.75,
            "local_surface_p95_mm": 1.5,
            "corridor_radius_mm": float(args.corridor_mm),
        },
        "global_surface_reference": registration.get("defective_to_good"),
        "bends": rows,
        "f1_screening_deviations": [item["bend_id"] for item in rows if item["f1"]["screening_deviation"]],
        "heatmap_hot_bends": [item["bend_id"] for item in rows if item["heatmap_hot_at_p95_1_5mm"]],
        "concordant_deviations": [item["bend_id"] for item in rows if item["concordant_deviation"]],
        "render_path": str(image_path),
    }
    (output_dir / "f1_heatmap_correlation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Corrected F1 versus independent surface heatmap",
        "",
        "Screening thresholds below are diagnostic and are not production tolerances.",
        "",
        "| Bend | axis offset mm | direction Δ° | plane-angle Δ° | local surface P95 mm | >1 mm | F1 flag | heatmap hot | corroborated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in rows:
        f1 = item["f1"]
        local = item["surface_heatmap_corridor"]
        lines.append(
            f"| {item['bend_id']} | {f1['axis_offset_mm']:.3f} | {f1['axis_direction_delta_deg']:.3f} | "
            f"{f1['incident_plane_angle_delta_deg']:+.3f} | {float(local.get('p95_mm') or 0.0):.3f} | "
            f"{100.0 * float(local.get('fraction_above_1mm') or 0.0):.1f}% | "
            f"{'yes' if f1['screening_deviation'] else 'no'} | {'yes' if item['heatmap_hot_at_p95_1_5mm'] else 'no'} | "
            f"{'yes' if item['concordant_deviation'] else 'no'} |"
        )
    (output_dir / "F1_HEATMAP_CORRELATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "f1_screening_deviations": payload["f1_screening_deviations"],
                "heatmap_hot_bends": payload["heatmap_hot_bends"],
                "concordant_deviations": payload["concordant_deviations"],
                "render_path": str(image_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
