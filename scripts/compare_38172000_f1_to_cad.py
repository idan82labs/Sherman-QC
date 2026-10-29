#!/usr/bin/env python3
"""Compare corrected 38172000 F1 bends with STEP nominal plus good-scan control.

The known-good V scan supplies the control frame for the repetitive prismatic
part.  Defective X is first registered rigidly to V, then both are carried into
the CAD frame by the same V-to-CAD transform.  This prevents X-only auto-scale
decisions from masquerading as manufacturing deviation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bend_inspection_pipeline import load_cad_geometry
from cad_bend_extractor import CADBendExtractor


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _points(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        points = np.asarray(mesh.vertices, dtype=np.float64)
    return points[np.isfinite(points).all(axis=1)]


def _recover_row_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if len(source) != len(target):
        raise ValueError("source and target clouds must preserve point order")
    indices = np.linspace(0, len(source) - 1, min(10_000, len(source)), dtype=np.int64)
    design = np.column_stack((source[indices], np.ones(len(indices))))
    affine = np.linalg.lstsq(design, target[indices], rcond=None)[0]
    residual = np.linalg.norm(design @ affine - target[indices], axis=1)
    singular_values = np.linalg.svd(affine[:3, :].T, compute_uv=False)
    return affine, {
        "sample_count": int(len(indices)),
        "residual_p50_mm": float(np.quantile(residual, 0.5)),
        "residual_max_mm": float(np.max(residual)),
        "linear_singular_values": [float(value) for value in singular_values],
    }


def _apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return np.column_stack((points, np.ones(len(points)))) @ affine


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _prediction_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for record in payload.get("owned_bend_regions") or ():
        prediction_id = str(record["bend_id"])
        records[prediction_id] = {
            "prediction_id": prediction_id,
            "line": np.asarray([record["line_start"], record["line_end"]], dtype=np.float64),
            "angle_deg": float((record.get("metadata") or {}).get("dominant_angle_deg") or 0.0),
            "confidence": float((record.get("metadata") or {}).get("confidence") or 0.0),
            "family": str((record.get("metadata") or {}).get("strip_family") or "unknown"),
        }
    return records


def _good_physical_lines(payload: dict[str, Any], manual_score: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prediction_to_gold = {
        str(match["prediction_id"]): str(match["gold_id"])
        for match in manual_score.get("matches") or ()
    }
    records = _prediction_records(payload)
    lines = {}
    for prediction_id, record in records.items():
        bend_id = prediction_to_gold.get(prediction_id)
        if bend_id is None:
            continue
        lines[bend_id] = dict(record)
    return lines


def _defective_physical_lines(
    payload: dict[str, Any],
    good_lines: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records = _prediction_records(payload)
    good_strip_to_bend = {
        str(record["prediction_id"]): bend_id for bend_id, record in good_lines.items()
    }
    lines = {}
    for match in comparison.get("matched_bends") or ():
        good_strip = str(match["good_strip_id"])
        defective_strip = str(match["defective_strip_id"])
        bend_id = good_strip_to_bend.get(good_strip)
        if bend_id is not None and defective_strip in records:
            lines[bend_id] = dict(records[defective_strip])
    return lines


def _axis(line: np.ndarray) -> np.ndarray:
    direction = line[1] - line[0]
    return direction / max(float(np.linalg.norm(direction)), 1e-12)


def _axis_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(abs(float(np.dot(_axis(left), _axis(right)))), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _transverse_offset_mm(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.mean(right, axis=0) - np.mean(left, axis=0)
    left_axis = _axis(left)
    right_axis = _axis(right)
    return max(
        float(np.linalg.norm(delta - left_axis * np.dot(delta, left_axis))),
        float(np.linalg.norm(delta - right_axis * np.dot(delta, right_axis))),
    )


def _segment_distance_mm(left: np.ndarray, right: np.ndarray) -> float:
    parameters = np.linspace(0.0, 1.0, 401)[:, None]
    left_points = left[0] + parameters * (left[1] - left[0])
    right_points = right[0] + parameters * (right[1] - right[0])
    return float(
        min(
            np.min(cKDTree(right_points).query(left_points, workers=1)[0]),
            np.min(cKDTree(left_points).query(right_points, workers=1)[0]),
        )
    )


def _line_length(line: np.ndarray) -> float:
    return float(np.linalg.norm(line[1] - line[0]))


def _finite_line_distance(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    delta = line[1] - line[0]
    parameter = np.clip(
        (points - line[0]) @ delta / max(float(np.dot(delta, delta)), 1e-12), 0.0, 1.0
    )
    closest = line[0] + parameter[:, None] * delta
    return np.linalg.norm(points - closest, axis=1)


def _mesh_distances(vertices: np.ndarray, triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    chunks = []
    for start in range(0, len(points), 200_000):
        chunks.append(
            scene.compute_distance(
                o3d.core.Tensor(points[start : start + 200_000].astype(np.float32))
            ).numpy()
        )
    return np.concatenate(chunks).astype(np.float64)


def _local_stats(points: np.ndarray, distances: np.ndarray, line: np.ndarray, corridor_mm: float) -> dict[str, Any]:
    mask = _finite_line_distance(points, line) <= corridor_mm
    local = distances[mask]
    if not len(local):
        return {"point_count": 0}
    return {
        "point_count": int(len(local)),
        "p50_mm": float(np.quantile(local, 0.50)),
        "p80_mm": float(np.quantile(local, 0.80)),
        "p95_mm": float(np.quantile(local, 0.95)),
        "p99_mm": float(np.quantile(local, 0.99)),
        "fraction_above_1mm": float(np.mean(local > 1.0)),
        "fraction_above_1_5mm": float(np.mean(local > 1.5)),
    }


def _cad_bends(vertices: np.ndarray, triangles: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    extraction = CADBendExtractor().extract_from_mesh(mesh)
    bends = []
    for bend in extraction.bends:
        bends.append(
            {
                "cad_bend_id": f"CAD_B{bend.bend_id}",
                "line": np.asarray([bend.bend_line_start, bend.bend_line_end], dtype=np.float64),
                "bend_angle_deg": float(bend.bend_angle),
                "confidence": float(bend.confidence),
                "surface_ids": [int(bend.surface1_id), int(bend.surface2_id)],
            }
        )
    return bends, {
        "raw_extracted_bend_count": int(extraction.total_bends),
        "surface_count": int(extraction.total_surfaces),
        "major_surface_count": int(extraction.total_major_surfaces),
        "warnings": list(extraction.warnings),
    }


def _match_control_to_cad(
    control: dict[str, dict[str, Any]], cad_bends: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, float]]:
    bend_ids = sorted(control, key=lambda value: int(value[1:]))
    costs = np.zeros((len(bend_ids), len(cad_bends)), dtype=np.float64)
    for row, bend_id in enumerate(bend_ids):
        line = control[bend_id]["cad_line"]
        for column, cad in enumerate(cad_bends):
            cad_line = cad["line"]
            costs[row, column] = (
                _segment_distance_mm(line, cad_line)
                + 0.35 * _axis_angle_deg(line, cad_line)
                + 0.02 * float(np.linalg.norm(np.mean(line, axis=0) - np.mean(cad_line, axis=0)))
            )
    rows, columns = linear_sum_assignment(costs)
    assignment = {bend_ids[row]: int(column) for row, column in zip(rows, columns)}
    assigned_costs = {bend_ids[row]: float(costs[row, column]) for row, column in zip(rows, columns)}
    return assignment, assigned_costs


def _stable_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    sample = points - center
    covariance = sample.T @ sample / max(len(sample), 1)
    _values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, ::-1]
    for column in range(3):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return center, basis


def _render_overlay(
    output: Path,
    cad_vertices: np.ndarray,
    defective_points: np.ndarray,
    defective_distance: np.ndarray,
    rows: list[dict[str, Any]],
) -> None:
    center, basis = _stable_basis(cad_vertices)
    cad_step = max(1, len(cad_vertices) // 80_000)
    scan_step = max(1, len(defective_points) // 180_000)
    cad_view = (cad_vertices[::cad_step] - center) @ basis
    scan_view = (defective_points[::scan_step] - center) @ basis
    scan_distance = defective_distance[::scan_step]
    views = ((0, 1, "top / plan"), (0, 2, "long-side profile"), (1, 2, "short-side profile"))
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=180, constrained_layout=True)
    for axis, (first, second, title) in zip(axes, views):
        axis.scatter(
            cad_view[:, first], cad_view[:, second], s=0.35, color="#94a3b8", alpha=0.18,
            linewidths=0, rasterized=True,
        )
        scatter = axis.scatter(
            scan_view[:, first], scan_view[:, second], c=np.clip(scan_distance, 0.0, 3.0),
            cmap="turbo", vmin=0.0, vmax=3.0, s=0.42, alpha=0.58, linewidths=0,
            rasterized=True,
        )
        for row in rows:
            line = (np.asarray(row["defective_f1_line_cad_frame"]) - center) @ basis
            if row["primary_defect_location"]:
                color = "#dc2626"
            elif row["cad_only_secondary_screen"]:
                color = "#f59e0b"
            else:
                color = "#16a34a"
            axis.plot(line[:, first], line[:, second], color="black", linewidth=5.5, alpha=0.75)
            axis.plot(line[:, first], line[:, second], color=color, linewidth=3.0)
            midpoint = np.mean(line, axis=0)
            axis.text(
                midpoint[first], midpoint[second], row["bend_id"], color="white", fontsize=8,
                fontweight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": color, "edgecolor": "black", "alpha": 0.96},
            )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.grid(alpha=0.12)
        axis.set_xlabel("mm")
        axis.set_ylabel("mm")
    fig.colorbar(scatter, ax=axes, label="defective X to nominal CAD surface distance (mm), clipped at 3")
    fig.suptitle(
        "38172000 defective X — corrected F1 axes in the good-control CAD frame\n"
        "red = F1 + CAD + independent X/V; amber = CAD-only secondary; green = not corroborated"
    )
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def _render_bars(output: Path, rows: list[dict[str, Any]]) -> None:
    ids = [row["bend_id"] for row in rows]
    x_values = [row["cad_surface_corridor"]["defective_x"]["p95_mm"] for row in rows]
    v_values = [row["cad_surface_corridor"]["good_v_control"]["p95_mm"] for row in rows]
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(14, 5.5), dpi=180)
    axis.bar(positions - 0.19, v_values, width=0.38, color="#0891b2", label="good V → CAD P95")
    axis.bar(positions + 0.19, x_values, width=0.38, color="#e11d48", label="defective X → CAD P95")
    axis.set_xticks(positions, ids)
    axis.set_ylabel("local CAD surface distance P95 (mm)")
    axis.set_xlabel("F1 physical bend identity")
    axis.set_title("CAD-relative surface deviation in 12 mm F1 bend corridors")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cad", type=Path, required=True)
    parser.add_argument("--good-scan", type=Path, required=True)
    parser.add_argument("--defective-scan", type=Path, required=True)
    parser.add_argument("--good-aligned-cad", type=Path, required=True)
    parser.add_argument("--x-to-v-registration", type=Path, required=True)
    parser.add_argument("--good-f1", type=Path, required=True)
    parser.add_argument("--defective-f1", type=Path, required=True)
    parser.add_argument("--good-manual-score", type=Path, required=True)
    parser.add_argument("--x-v-bend-comparison", type=Path, required=True)
    parser.add_argument("--x-v-correlation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corridor-mm", type=float, default=12.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cad_vertices, cad_triangles, cad_source = load_cad_geometry(str(args.cad.resolve()))
    cad_bends, cad_extraction = _cad_bends(cad_vertices, cad_triangles)

    good_raw = _points(args.good_scan.resolve())
    defective_raw = _points(args.defective_scan.resolve())
    good_aligned = _points(args.good_aligned_cad.resolve())
    good_to_cad_affine, affine_diagnostics = _recover_row_affine(good_raw, good_aligned)
    registration = _read(args.x_to_v_registration.resolve())
    x_to_v_transform = np.asarray(registration["transform_defective_to_good"], dtype=np.float64)
    defective_in_good = _apply_transform(defective_raw, x_to_v_transform)
    defective_in_cad = _apply_affine(defective_in_good, good_to_cad_affine)

    good_f1_payload = _read(args.good_f1.resolve())
    defective_f1_payload = _read(args.defective_f1.resolve())
    good_f1 = _good_physical_lines(
        good_f1_payload, _read(args.good_manual_score.resolve())
    )
    defective_f1 = _defective_physical_lines(
        defective_f1_payload,
        good_f1,
        _read(args.x_v_bend_comparison.resolve()),
    )
    if sorted(good_f1) != sorted(defective_f1):
        raise ValueError("good and defective F1 physical bend identities differ")

    for bend_id in good_f1:
        good_f1[bend_id]["cad_line"] = _apply_affine(
            good_f1[bend_id]["line"], good_to_cad_affine
        )
        defective_in_good_line = _apply_transform(
            defective_f1[bend_id]["line"], x_to_v_transform
        )
        defective_f1[bend_id]["cad_line"] = _apply_affine(
            defective_in_good_line, good_to_cad_affine
        )

    cad_assignment, assignment_costs = _match_control_to_cad(good_f1, cad_bends)
    x_v_correlation = _read(args.x_v_correlation.resolve())
    independent_by_bend = {
        str(row["bend_id"]): row for row in x_v_correlation.get("bends") or ()
    }
    good_cad_distance = _mesh_distances(cad_vertices, cad_triangles, good_aligned)
    defective_cad_distance = _mesh_distances(cad_vertices, cad_triangles, defective_in_cad)

    rows = []
    for bend_id in sorted(good_f1, key=lambda value: int(value[1:])):
        good = good_f1[bend_id]
        defective = defective_f1[bend_id]
        cad_index = cad_assignment[bend_id]
        cad = cad_bends[cad_index]
        good_line = good["cad_line"]
        defective_line = defective["cad_line"]
        cad_line = cad["line"]
        good_local = _local_stats(
            good_aligned, good_cad_distance, good_line, float(args.corridor_mm)
        )
        defective_local = _local_stats(
            defective_in_cad, defective_cad_distance, good_line, float(args.corridor_mm)
        )
        x_v_offset = _transverse_offset_mm(good_line, defective_line)
        x_v_direction = _axis_angle_deg(good_line, defective_line)
        x_v_angle_delta = defective["angle_deg"] - good["angle_deg"]
        f1_differential = (
            x_v_offset >= 1.0
            or x_v_direction >= 1.0
            or abs(x_v_angle_delta) >= 0.75
        )
        surface_excess = float(defective_local["p95_mm"] - good_local["p95_mm"])
        cad_surface_excess = defective_local["p95_mm"] >= 1.5 and surface_excess >= 0.35
        independent = independent_by_bend.get(bend_id) or {}
        independent_local = dict(independent.get("surface_heatmap_corridor") or {})
        independent_hot = bool(independent.get("heatmap_hot_at_p95_1_5mm"))
        primary_location = bool(f1_differential and cad_surface_excess and independent_hot)
        cad_only_secondary = bool(f1_differential and cad_surface_excess and not independent_hot)
        cad_axis_angle = _axis_angle_deg(good_line, cad_line)
        cad_axis_distance = _segment_distance_mm(good_line, cad_line)
        cad_axis_match_quality = (
            "usable" if cad_axis_angle <= 6.0 and cad_axis_distance <= 15.0 else "low"
        )
        rows.append(
            {
                "bend_id": bend_id,
                "cad_bend_id": cad["cad_bend_id"],
                "cad_axis_match_quality": cad_axis_match_quality,
                "cad_assignment_cost": assignment_costs[bend_id],
                "nominal_cad": {
                    "bend_angle_deg": cad["bend_angle_deg"],
                    "axis_length_mm": _line_length(cad_line),
                    "confidence": cad["confidence"],
                    "surface_ids": cad["surface_ids"],
                },
                "good_v_control": {
                    "f1_strip_id": good["prediction_id"],
                    "f1_angle_deg": good["angle_deg"],
                    "angle_error_vs_cad_deg": good["angle_deg"] - cad["bend_angle_deg"],
                    "axis_direction_error_vs_cad_deg": _axis_angle_deg(good_line, cad_line),
                    "axis_segment_distance_to_cad_mm": _segment_distance_mm(good_line, cad_line),
                    "axis_transverse_offset_to_cad_mm": _transverse_offset_mm(cad_line, good_line),
                },
                "defective_x": {
                    "f1_strip_id": defective["prediction_id"],
                    "f1_angle_deg": defective["angle_deg"],
                    "angle_error_vs_cad_deg": defective["angle_deg"] - cad["bend_angle_deg"],
                    "axis_direction_error_vs_cad_deg": _axis_angle_deg(defective_line, cad_line),
                    "axis_segment_distance_to_cad_mm": _segment_distance_mm(defective_line, cad_line),
                    "axis_transverse_offset_to_cad_mm": _transverse_offset_mm(cad_line, defective_line),
                },
                "x_minus_v_differential": {
                    "incident_plane_angle_delta_deg": x_v_angle_delta,
                    "axis_direction_delta_deg": x_v_direction,
                    "axis_transverse_offset_mm": x_v_offset,
                    "span_delta_mm": _line_length(defective_line) - _line_length(good_line),
                    "f1_screening_deviation": f1_differential,
                },
                "cad_surface_corridor": {
                    "corridor_radius_mm": float(args.corridor_mm),
                    "good_v_control": good_local,
                    "defective_x": defective_local,
                    "defective_minus_good_p95_mm": surface_excess,
                    "cad_surface_excess": cad_surface_excess,
                },
                "independent_x_to_v_surface_corridor": {
                    "p95_mm": independent_local.get("p95_mm"),
                    "heatmap_hot_at_p95_1_5mm": independent_hot,
                    "source": str(args.x_v_correlation.resolve()),
                },
                "corroborated_defect_screen": bool(f1_differential and cad_surface_excess),
                "primary_defect_location": primary_location,
                "cad_only_secondary_screen": cad_only_secondary,
                "good_f1_line_cad_frame": good_line.tolist(),
                "defective_f1_line_cad_frame": defective_line.tolist(),
                "cad_line": cad_line.tolist(),
            }
        )

    assigned_indices = set(cad_assignment.values())
    unmatched_cad = [
        {"cad_bend_id": cad["cad_bend_id"], "bend_angle_deg": cad["bend_angle_deg"]}
        for index, cad in enumerate(cad_bends)
        if index not in assigned_indices
    ]
    corroborated = [row["bend_id"] for row in rows if row["corroborated_defect_screen"]]
    primary_locations = [row["bend_id"] for row in rows if row["primary_defect_location"]]
    cad_only_secondary = [row["bend_id"] for row in rows if row["cad_only_secondary_screen"]]
    payload = {
        "schema_version": 1,
        "kind": "f1_bend_by_bend_cad_control_comparison",
        "part_id": "38172000_01",
        "labels": {"good_control": "V", "defective": "X"},
        "cad_source": str(args.cad.resolve()),
        "cad_import_source": cad_source,
        "method": {
            "frame": "X rigidly registered to good V, then both transformed by V-to-CAD alignment",
            "reason": "Avoid X-only auto-scale and repetitive-axis assignment ambiguity.",
            "f1_physical_identity_source": "corrected 10-bend mesh_topology_profile_v1",
            "screening_thresholds_not_production_tolerances": {
                "f1_axis_offset_mm": 1.0,
                "f1_axis_direction_delta_deg": 1.0,
                "f1_incident_plane_angle_delta_deg": 0.75,
                "cad_surface_p95_mm": 1.5,
                "cad_surface_p95_excess_over_good_mm": 0.35,
            },
        },
        "alignment": {
            "x_to_v_registration": registration.get("registration"),
            "x_to_v_global_deviation": registration.get("defective_to_good"),
            "good_v_to_cad_affine_diagnostics": affine_diagnostics,
            "good_v_to_cad_affine_row_matrix": good_to_cad_affine.tolist(),
            "x_to_v_transform": x_to_v_transform.tolist(),
        },
        "cad_extraction": cad_extraction,
        "unmatched_cad_extractor_axes": unmatched_cad,
        "bends": rows,
        "corroborated_defect_screens": corroborated,
        "primary_defect_locations": primary_locations,
        "cad_only_secondary_screens": cad_only_secondary,
        "limitations": [
            "The STEP mesh extractor returns 12 axes while drawing-backed physical truth contains 10 bends.",
            "CAD-axis matches marked low quality are not used as reliable positional metrology.",
            "Good V also departs systematically from the STEP surface, so X-specific interpretation uses X-minus-V differential evidence.",
            "Thresholds are diagnostic screens, not drawing tolerances or a production disposition.",
        ],
    }
    json_path = output_dir / "f1_defective_X_vs_CAD_bend_by_bend.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    overlay_path = output_dir / "f1_defective_X_vs_CAD_overlay.png"
    bars_path = output_dir / "f1_defective_X_vs_CAD_surface_p95.png"
    _render_overlay(overlay_path, cad_vertices, defective_in_cad, defective_cad_distance, rows)
    _render_bars(bars_path, rows)

    lines = [
        "# 38172000 defective X versus nominal CAD — F1 bend-by-bend",
        "",
        "X is registered rigidly to known-good V, then both are transformed into the same CAD frame. "
        "The good control is shown beside X because direct X-only CAD alignment attempted an unsafe scale correction.",
        "",
        "| Bend | CAD axis | match | CAD angle ° | V F1 ° | X F1 ° | X−V angle Δ° | X−V axis offset mm | V→CAD P95 mm | X→CAD P95 mm | excess mm | X→V P95 mm | final |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['bend_id']} | {row['cad_bend_id']} | {row['cad_axis_match_quality']} | "
            f"{row['nominal_cad']['bend_angle_deg']:.2f} | {row['good_v_control']['f1_angle_deg']:.2f} | "
            f"{row['defective_x']['f1_angle_deg']:.2f} | "
            f"{row['x_minus_v_differential']['incident_plane_angle_delta_deg']:+.2f} | "
            f"{row['x_minus_v_differential']['axis_transverse_offset_mm']:.2f} | "
            f"{row['cad_surface_corridor']['good_v_control']['p95_mm']:.2f} | "
            f"{row['cad_surface_corridor']['defective_x']['p95_mm']:.2f} | "
            f"{row['cad_surface_corridor']['defective_minus_good_p95_mm']:+.2f} | "
            f"{float(row['independent_x_to_v_surface_corridor'].get('p95_mm') or 0.0):.2f} | "
            f"{'PRIMARY' if row['primary_defect_location'] else 'CAD-ONLY' if row['cad_only_secondary_screen'] else 'not confirmed'} |"
        )
    lines.extend(
        [
            "",
            f"Primary defect locations (all three evidence channels): **{', '.join(primary_locations) if primary_locations else 'none'}**.",
            "",
            f"CAD-only secondary screen: **{', '.join(cad_only_secondary) if cad_only_secondary else 'none'}**.",
            "",
            "This is a diagnostic comparison, not a production pass/fail decision. The STEP extractor returns 12 candidate axes while the drawing-backed physical model has 10; low-quality CAD-axis matches are explicitly identified.",
            "",
            f"- Overlay: `{overlay_path}`",
            f"- CAD corridor chart: `{bars_path}`",
            f"- Machine-readable data: `{json_path}`",
        ]
    )
    report_path = output_dir / "F1_DEFECTIVE_X_VS_CAD_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "primary_defect_locations": primary_locations,
                "cad_only_secondary_screens": cad_only_secondary,
                "corroborated_defect_screens": corroborated,
                "unmatched_cad_extractor_axes": unmatched_cad,
                "json_path": str(json_path),
                "report_path": str(report_path),
                "overlay_path": str(overlay_path),
                "bars_path": str(bars_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
