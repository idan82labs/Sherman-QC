#!/usr/bin/env python3
"""Register a bad scan to a good scan and render source-only deviation maps."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _cloud(path: Path, voxel: float) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(str(path))
    cloud = o3d.geometry.PointCloud()
    if len(mesh.vertices):
        cloud.points = mesh.vertices
    else:
        cloud = o3d.io.read_point_cloud(str(path))
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(3 * voxel, 3.0), max_nn=40))
    return cloud


def _pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return center, basis


def _initial_transforms(source: np.ndarray, target: np.ndarray):
    source_center, source_basis = _pca(source)
    target_center, target_basis = _pca(target)
    for permutation in itertools.permutations(range(3)):
        permuted = source_basis[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = permuted @ np.diag(signs)
            rotation = target_basis @ candidate.T
            if np.linalg.det(rotation) < 0.999:
                continue
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = target_center - rotation @ source_center
            yield transform


def _register(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> tuple[np.ndarray, dict]:
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    best = None
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    coarse = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    for initial in _initial_transforms(source_points, target_points):
        result = o3d.pipelines.registration.registration_icp(
            source, target, 12.0, initial, estimation, coarse
        )
        key = (float(result.fitness), -float(result.inlier_rmse))
        if best is None or key > best[0]:
            best = (key, result.transformation)
    if best is None:
        raise RuntimeError("no proper PCA registration seed")
    fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        3.0,
        best[1],
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return fine.transformation, {
        "fitness": float(fine.fitness),
        "inlier_rmse_mm": float(fine.inlier_rmse),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", type=Path, required=True)
    parser.add_argument("--bad", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--voxel-mm", type=float, default=1.2)
    args = parser.parse_args()

    good = _cloud(args.good, args.voxel_mm)
    bad = _cloud(args.bad, args.voxel_mm)
    transform, registration = _register(bad, good)
    aligned_bad = o3d.geometry.PointCloud(bad)
    aligned_bad.transform(transform)
    distances = np.asarray(aligned_bad.compute_point_cloud_distance(good), dtype=float)
    bad_points = np.asarray(aligned_bad.points)
    good_points = np.asarray(good.points)
    center, basis = _pca(good_points)
    bad_local = (bad_points - center) @ basis
    good_local = (good_points - center) @ basis
    clip = max(2.0, float(np.quantile(distances, 0.99)))

    views = (
        ("PC1 / PC2", (0, 1)),
        ("PC1 / PC3", (0, 2)),
        ("PC2 / PC3", (1, 2)),
        ("PC2 / PC1", (1, 0)),
        ("PC3 / PC1", (2, 0)),
        ("PC3 / PC2", (2, 1)),
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=220, constrained_layout=True)
    figure_scatter = None
    for ax, (title, dimensions) in zip(axes.flat, views):
        ax.scatter(
            good_local[:, dimensions[0]], good_local[:, dimensions[1]],
            s=0.12, c="#334155", alpha=0.18, linewidths=0, rasterized=True,
        )
        figure_scatter = ax.scatter(
            bad_local[:, dimensions[0]], bad_local[:, dimensions[1]],
            s=0.22, c=np.clip(distances, 0.0, clip), cmap="turbo", vmin=0.0, vmax=clip,
            alpha=0.78, linewidths=0, rasterized=True,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    fig.colorbar(figure_scatter, ax=axes, shrink=0.78, label="bad-to-good nearest-surface distance (mm)")
    fig.suptitle("47251002_02 scan-pair deviation audit — good reference in gray, aligned defective scan colored by distance")
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, facecolor="white")
    plt.close(fig)

    full_good = o3d.io.read_point_cloud(str(args.good))
    full_bad = o3d.io.read_point_cloud(str(args.bad))
    full_bad.transform(transform)
    full_distances = np.asarray(full_bad.compute_point_cloud_distance(full_good), dtype=float)

    report = {
        "schema_version": 1,
        "kind": "sherman_qc_source_only_scan_pair_deviation_audit",
        "voxel_mm": float(args.voxel_mm),
        "registration": registration,
        "distance_mm": {
            "median": float(np.median(distances)),
            "p90": float(np.quantile(distances, 0.90)),
            "p95": float(np.quantile(distances, 0.95)),
            "p99": float(np.quantile(distances, 0.99)),
            "max": float(np.max(distances)),
            "fraction_gt_1mm": float(np.mean(distances > 1.0)),
            "fraction_gt_2mm": float(np.mean(distances > 2.0)),
            "fraction_gt_3mm": float(np.mean(distances > 3.0)),
        },
        "full_resolution_distance_mm": {
            "point_count": int(len(full_distances)),
            "median": float(np.median(full_distances)),
            "p90": float(np.quantile(full_distances, 0.90)),
            "p95": float(np.quantile(full_distances, 0.95)),
            "p99": float(np.quantile(full_distances, 0.99)),
            "max": float(np.max(full_distances)),
            "fraction_gt_1mm": float(np.mean(full_distances > 1.0)),
            "fraction_gt_2mm": float(np.mean(full_distances > 2.0)),
        },
        "transform_bad_to_good": transform.tolist(),
        "source_only": True,
        "cad_used": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report["distance_mm"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
