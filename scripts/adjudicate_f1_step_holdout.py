#!/usr/bin/env python3
"""Post-lock STEP-backed adjudication for a scan-only F1 prediction.

The script never participates in inference.  It is intentionally run only
after a blind prediction has been written and hashed.  It derives conventional
bend axes from paired STEP cylindrical faces, rigidly registers the CAD mesh
to the raw scan frame, and scores the frozen prediction with the repository's
line-level contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import gmsh
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.bend_inspection_pipeline import load_cad_geometry  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    Thresholds,
    match_lines,
    pair_metrics,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bbox_corners(bbox: list[float]) -> np.ndarray:
    lo = np.asarray(bbox[:3], dtype=np.float64)
    hi = np.asarray(bbox[3:], dtype=np.float64)
    return np.asarray(list(itertools.product(*zip(lo, hi))), dtype=np.float64)


def _axis_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))))


def _step_cylinders(step_path: Path, minimum_area_mm2: float) -> list[dict[str, Any]]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("step_holdout")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        cylinders: list[dict[str, Any]] = []
        for _dim, tag in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, tag) != "Cylinder":
                continue
            area = float(gmsh.model.occ.getMass(2, tag))
            if area < minimum_area_mm2:
                continue
            bounds = gmsh.model.getParametrizationBounds(2, tag)
            uv = [
                0.5 * (float(bounds[0][0]) + float(bounds[1][0])),
                0.5 * (float(bounds[0][1]) + float(bounds[1][1])),
            ]
            kmax, kmin, dmax, dmin = gmsh.model.getPrincipalCurvatures(tag, uv)
            curvatures = [float(kmax[0]), float(kmin[0])]
            directions = [np.asarray(dmax[:3], dtype=np.float64), np.asarray(dmin[:3], dtype=np.float64)]
            zero_index = int(np.argmin(np.abs(curvatures)))
            axis = directions[zero_index]
            axis /= np.linalg.norm(axis)
            nonzero = curvatures[1 - zero_index]
            radius = 1.0 / max(abs(nonzero), 1e-12)
            bbox = [float(value) for value in gmsh.model.getBoundingBox(2, tag)]
            corners = _bbox_corners(bbox)
            interval = [float(np.min(corners @ axis)), float(np.max(corners @ axis))]
            cylinders.append(
                {
                    "surface_tag": int(tag),
                    "area_mm2": area,
                    "radius_mm": radius,
                    "axis": axis,
                    "center_of_mass": np.asarray(gmsh.model.occ.getCenterOfMass(2, tag), dtype=np.float64),
                    "bbox": bbox,
                    "axis_interval": interval,
                }
            )
        return cylinders
    finally:
        gmsh.finalize()


def _pair_cylinders(cylinders: list[dict[str, Any]], thickness_mm: float) -> list[dict[str, Any]]:
    possible: list[tuple[float, int, int, dict[str, float]]] = []
    for left_index, left in enumerate(cylinders):
        for right_index in range(left_index + 1, len(cylinders)):
            right = cylinders[right_index]
            angle = _axis_angle_deg(left["axis"], right["axis"])
            if angle > 1.0:
                continue
            radius_delta = abs(float(left["radius_mm"]) - float(right["radius_mm"]))
            if abs(radius_delta - thickness_mm) > max(0.35, 0.20 * thickness_mm):
                continue
            axis = left["axis"]
            if np.dot(axis, right["axis"]) < 0:
                axis = -axis
            delta = right["center_of_mass"] - left["center_of_mass"]
            transverse = float(np.linalg.norm(delta - left["axis"] * np.dot(delta, left["axis"])))
            if transverse > 4.5 * thickness_mm:
                continue
            left_interval = left["axis_interval"]
            right_corners = _bbox_corners(right["bbox"])
            right_interval = [float(np.min(right_corners @ left["axis"])), float(np.max(right_corners @ left["axis"]))]
            overlap = max(0.0, min(left_interval[1], right_interval[1]) - max(left_interval[0], right_interval[0]))
            shorter = min(left_interval[1] - left_interval[0], right_interval[1] - right_interval[0])
            overlap_fraction = overlap / max(shorter, 1e-9)
            if overlap_fraction < 0.90:
                continue
            metrics = {
                "axis_angle_deg": angle,
                "radius_delta_mm": radius_delta,
                "transverse_center_distance_mm": transverse,
                "shorter_axis_overlap": overlap_fraction,
            }
            cost = transverse + 5.0 * angle + 10.0 * abs(radius_delta - thickness_mm)
            possible.append((cost, left_index, right_index, metrics))

    used: set[int] = set()
    paired: list[dict[str, Any]] = []
    for _cost, left_index, right_index, metrics in sorted(possible):
        if left_index in used or right_index in used:
            continue
        used.update((left_index, right_index))
        left = cylinders[left_index]
        right = cylinders[right_index]
        axis = np.asarray(left["axis"], dtype=np.float64)
        if np.dot(axis, right["axis"]) < 0:
            axis = -axis
        center = 0.5 * (left["center_of_mass"] + right["center_of_mass"])
        left_projection = _bbox_corners(left["bbox"]) @ axis
        right_projection = _bbox_corners(right["bbox"]) @ axis
        lo = max(float(np.min(left_projection)), float(np.min(right_projection)))
        hi = min(float(np.max(left_projection)), float(np.max(right_projection)))
        center_projection = float(np.dot(center, axis))
        endpoints = np.asarray(
            [center + (lo - center_projection) * axis, center + (hi - center_projection) * axis]
        )
        paired.append(
            {
                "surface_tags": [left["surface_tag"], right["surface_tag"]],
                "surface_radii_mm": sorted([left["radius_mm"], right["radius_mm"]]),
                "surface_areas_mm2": sorted([left["area_mm2"], right["area_mm2"]]),
                "line_cad": endpoints,
                "pair_metrics": metrics,
            }
        )
    paired.sort(key=lambda item: tuple(np.mean(item["line_cad"], axis=0).tolist()))
    return paired


def _geometry_points(scan_path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(scan_path))
    if len(cloud.points):
        return np.asarray(cloud.points, dtype=np.float64)
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    if not len(mesh.vertices):
        raise ValueError(f"no geometry in {scan_path}")
    return np.asarray(mesh.vertices, dtype=np.float64)


def _basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    vectors = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1
    return center, vectors


def _signed_permutations() -> list[np.ndarray]:
    matrices = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return matrices


def _cloud(points: np.ndarray, voxel_mm: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud.voxel_down_sample(voxel_mm)


def _align_cad_to_scan(cad_points: np.ndarray, scan_points: np.ndarray, voxel_mm: float) -> dict[str, Any]:
    cad = _cloud(cad_points, voxel_mm)
    scan = _cloud(scan_points, voxel_mm)
    cad_array = np.asarray(cad.points)
    scan_array = np.asarray(scan.points)
    cad_center, cad_basis = _basis(cad_array)
    scan_center, scan_basis = _basis(scan_array)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120)
    trials = []
    for hypothesis_index, signed_permutation in enumerate(_signed_permutations()):
        rotation = scan_basis @ signed_permutation @ cad_basis.T
        translation = scan_center - rotation @ cad_center
        initial = np.eye(4)
        initial[:3, :3] = rotation
        initial[:3, 3] = translation
        result = o3d.pipelines.registration.registration_icp(
            cad,
            scan,
            max(6.0, 4.0 * voxel_mm),
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            criteria,
        )
        trials.append((float(result.fitness), float(result.inlier_rmse), hypothesis_index, result.transformation))
    fitness, rmse, hypothesis_index, transform = max(trials, key=lambda item: (item[0], -item[1], -item[2]))
    for threshold in (3.0, 1.5):
        refined = o3d.pipelines.registration.registration_icp(
            cad,
            scan,
            threshold,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            criteria,
        )
        transform = refined.transformation
        fitness = float(refined.fitness)
        rmse = float(refined.inlier_rmse)
    return {
        "cad_to_scan_transform": transform,
        "fitness": fitness,
        "rmse_mm": rmse,
        "pca_hypothesis_index": int(hypothesis_index),
        "cad_downsampled_point_count": int(len(cad.points)),
        "scan_downsampled_point_count": int(len(scan.points)),
    }


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _prediction_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("owned_bend_regions")
    if not isinstance(rows, list):
        raise ValueError("prediction payload has no owned_bend_regions")
    return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, type=Path)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sheet-thickness-mm", type=float, default=2.0)
    parser.add_argument("--minimum-cylinder-area-mm2", type=float, default=50.0)
    parser.add_argument("--alignment-voxel-mm", type=float, default=1.5)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cad_vertices, cad_triangles, cad_source = load_cad_geometry(str(args.step.resolve()), cad_import_deflection=0.05)
    if cad_triangles is None or not len(cad_triangles):
        raise ValueError("STEP import did not provide a triangle mesh")
    cylinders = _step_cylinders(args.step.resolve(), float(args.minimum_cylinder_area_mm2))
    pairs = _pair_cylinders(cylinders, float(args.sheet_thickness_mm))
    scan_points = _geometry_points(args.scan.resolve())
    alignment = _align_cad_to_scan(cad_vertices, scan_points, float(args.alignment_voxel_mm))
    transform = alignment["cad_to_scan_transform"]

    aligned_vertices = _transform_points(cad_vertices, transform)
    aligned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(aligned_vertices),
        o3d.utility.Vector3iVector(np.asarray(cad_triangles, dtype=np.int32)),
    )
    aligned_mesh.compute_vertex_normals()
    aligned_mesh_path = output_dir / "cad_mesh_in_scan_raw.ply"
    o3d.io.write_triangle_mesh(str(aligned_mesh_path), aligned_mesh, write_ascii=False)

    gold = []
    pair_diagnostics = []
    for index, pair in enumerate(pairs, start=1):
        transformed_line = _transform_points(pair["line_cad"], transform)
        gold.append(
            {
                "bend_id": f"CAD_B{index}",
                "line_start": transformed_line[0].tolist(),
                "line_end": transformed_line[1].tolist(),
                "span_endpoints": transformed_line.tolist(),
                "truth_source": "paired_step_cylindrical_faces_postlock",
                "surface_tags": pair["surface_tags"],
            }
        )
        pair_diagnostics.append(
            {
                "bend_id": f"CAD_B{index}",
                "surface_tags": pair["surface_tags"],
                "surface_radii_mm": pair["surface_radii_mm"],
                "surface_areas_mm2": pair["surface_areas_mm2"],
                "pair_metrics": pair["pair_metrics"],
                "line_cad": pair["line_cad"].tolist(),
                "line_scan_raw": transformed_line.tolist(),
            }
        )

    prediction_payload = _read(args.prediction.resolve())
    predictions = _prediction_rows(prediction_payload)
    thresholds = Thresholds()
    evaluation = match_lines(gold, predictions, thresholds)
    nearest = []
    for truth in gold:
        truth_line = np.asarray(truth["span_endpoints"], dtype=np.float64)
        options = []
        for prediction in predictions:
            endpoints = prediction.get("span_endpoints") or [prediction["line_start"], prediction["line_end"]]
            metrics = pair_metrics(truth_line, np.asarray(endpoints, dtype=np.float64))
            cost = metrics["transverse_offset_mm"] + metrics["axis_angle_deg"]
            options.append((cost, str(prediction.get("bend_id") or prediction.get("candidate_id")), metrics))
        if options:
            _cost, prediction_id, metrics = min(options, key=lambda item: (item[0], item[1]))
            nearest.append({"gold_id": truth["bend_id"], "nearest_prediction_id": prediction_id, **metrics})

    labels_payload = {
        "schema_version": 1,
        "kind": "cad_drawing_backed_postlock_bend_line_adjudication",
        "authoritative_human_truth": False,
        "inference_exposure": "created_after_prediction_lock",
        "cad_source": cad_source,
        "step_path": str(args.step.resolve()),
        "scan_path": str(args.scan.resolve()),
        "sheet_thickness_mm": float(args.sheet_thickness_mm),
        "step_cylinder_surface_count_after_area_gate": len(cylinders),
        "paired_conventional_bend_count": len(gold),
        "alignment": {
            **{key: value for key, value in alignment.items() if key != "cad_to_scan_transform"},
            "cad_to_scan_transform": transform.tolist(),
            "coordinate_convention": "p_scan_raw = p_cad @ R.T + t",
        },
        "owned_bend_regions": gold,
        "pair_diagnostics": pair_diagnostics,
    }
    _write(output_dir / "cad_drawing_backed_bend_labels.json", labels_payload)
    _write(
        output_dir / "line_evaluation.json",
        {
            "schema_version": 1,
            "prediction_path": str(args.prediction.resolve()),
            "prediction_count": len(predictions),
            "gold_count": len(gold),
            "thresholds": asdict(thresholds),
            "evaluation": evaluation,
            "nearest_prediction_per_gold": nearest,
        },
    )
    print(
        json.dumps(
            {
                "paired_conventional_bend_count": len(gold),
                "prediction_count": len(predictions),
                "alignment_fitness": alignment["fitness"],
                "alignment_rmse_mm": alignment["rmse_mm"],
                "true_positive": evaluation["true_positive"],
                "false_positive": evaluation["false_positive"],
                "false_negative": evaluation["false_negative"],
                "exact": evaluation["exact"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
