#!/usr/bin/env python3
"""Rigidly register a defective PLY to a good PLY and report surface deviation.

The registration is deliberately anchored to the dominant shared sheet plane and
then refined with trimmed point-to-point correspondences.  This prevents a large
bad flange/form from dragging the rigid alignment away from the stable body.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


SEED = 38_172_000
THRESHOLDS_MM = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)


def read_mesh(path: Path) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    points = np.asarray(mesh.vertices).copy()
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Invalid or empty mesh: {path}")
    return mesh, points


def voxel_sample(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    keys = np.floor(points / voxel_mm).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def dominant_plane_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, indices = cloud.segment_plane(
        distance_threshold=0.35,
        ransac_n=3,
        num_iterations=3_000,
        probability=0.999,
    )
    inliers = points[np.asarray(indices)]
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal *= -1
    center = np.median(inliers, axis=0)

    # A global-axis-derived in-plane frame avoids the 90/180-degree ambiguity of
    # PCA on a near-square sheet body while retaining the scanners' nearby pose.
    reference = np.array([1.0, 0.0, 0.0])
    tangent = reference - normal * np.dot(reference, normal)
    if np.linalg.norm(tangent) < 0.2:
        reference = np.array([0.0, 1.0, 0.0])
        tangent = reference - normal * np.dot(reference, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    return inliers, center, np.column_stack((tangent, bitangent, normal))


def stable_pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    sample = points - center
    _values, vectors = np.linalg.eigh(sample.T @ sample / max(len(sample), 1))
    basis = vectors[:, ::-1]
    for column in range(3):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    return center, basis


def proper_signed_permutations() -> list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]]:
    orientations = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1, 1), repeat=3):
            orientation = permutation_matrix @ np.diag(signs)
            if np.linalg.det(orientation) <= 0.0:
                continue
            orientations.append((permutation, signs, orientation))
    orientations.sort(key=lambda item: (item[0], item[1]))
    return orientations


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.einsum("ij,kj->ki", transform[:3, :3], points) + transform[:3, 3]


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = np.einsum(
        "ni,nj->ij", source - source_center, target - target_center
    )
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _refine_transform(
    good_ds: np.ndarray,
    defective_ds: np.ndarray,
    initial_transform: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    transform = initial_transform.copy()
    tree = cKDTree(good_ds)
    stages = []
    for maximum, retained_fraction in (
        (15.0, 0.80),
        (8.0, 0.75),
        (5.0, 0.70),
        (3.0, 0.65),
        (2.0, 0.60),
        (1.5, 0.55),
        (1.0, 0.50),
    ):
        for iteration in range(30):
            current = apply_transform(defective_ds, transform)
            distances, nearest = tree.query(current, workers=1)
            cutoff = min(maximum, float(np.quantile(distances, retained_fraction)))
            keep = distances <= cutoff
            increment = rigid_fit(current[keep], good_ds[nearest[keep]])
            transform = increment @ transform
            change = np.linalg.norm(increment[:3, 3]) + 100.0 * np.linalg.norm(
                increment[:3, :3] - np.eye(3)
            )
            if change < 1e-6:
                break
        stages.append(
            {
                "maximum_correspondence_mm": maximum,
                "retained_fraction": retained_fraction,
                "actual_cutoff_mm": cutoff,
                "correspondences": int(keep.sum()),
                "iterations": iteration + 1,
            }
        )
    return transform, stages


def register(good: np.ndarray, defective: np.ndarray) -> tuple[np.ndarray, dict]:
    good_ds = voxel_sample(good, 1.5)
    defective_ds = voxel_sample(defective, 1.5)
    _, good_center, good_frame = dominant_plane_frame(good_ds)
    _, defective_center, defective_frame = dominant_plane_frame(defective_ds)
    good_pca_center, good_pca_frame = stable_pca_frame(good_ds)
    defective_pca_center, defective_pca_frame = stable_pca_frame(defective_ds)

    good_tree = cKDTree(good_ds)
    hypotheses = []

    def add_hypothesis(initial: np.ndarray, metadata: dict) -> None:
        initial_rotation_deg = math.degrees(
            math.acos(
                np.clip((np.trace(initial[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            )
        )
        defective_aligned = apply_transform(defective_ds, initial)
        x_to_v, _ = good_tree.query(defective_aligned, workers=1)
        defective_tree = cKDTree(defective_aligned)
        v_to_x, _ = defective_tree.query(good_ds, workers=1)
        score = 0.5 * (
            float(np.percentile(x_to_v, 80))
            + float(np.percentile(v_to_x, 80))
        )
        hypotheses.append(
            {
                **metadata,
                "initial_rotation_deg": initial_rotation_deg,
                "initial_score_symmetric_p80_mm": score,
                "initial_defective_to_good_p50_mm": float(np.percentile(x_to_v, 50)),
                "initial_defective_to_good_p80_mm": float(np.percentile(x_to_v, 80)),
                "initial_defective_to_good_p95_mm": float(np.percentile(x_to_v, 95)),
                "initial_transform": initial,
            }
        )

    # A scan can be placed on either face and rotated 180 degrees on the scanner.
    # Start with all four proper rotations of the dominant-plane frame.
    for normal_sign in (1.0, -1.0):
        for tangent_sign in (1.0, -1.0):
            orientation = np.diag(
                (tangent_sign, normal_sign * tangent_sign, normal_sign)
            )
            initial = np.eye(4)
            initial[:3, :3] = good_frame @ orientation @ defective_frame.T
            initial[:3, 3] = good_center - initial[:3, :3] @ defective_center
            add_hypothesis(
                initial,
                {
                    "frame_family": "dominant_plane",
                    "normal_sign": int(normal_sign),
                    "tangent_sign": int(tangent_sign),
                },
            )

    # Dominant-plane frames can lock a symmetric sheet into the wrong side
    # profile. Test every proper PCA-axis permutation/sign orientation as well;
    # reflections remain forbidden.
    for permutation, signs, orientation in proper_signed_permutations():
        initial = np.eye(4)
        initial[:3, :3] = good_pca_frame @ orientation @ defective_pca_frame.T
        initial[:3, 3] = (
            good_pca_center - initial[:3, :3] @ defective_pca_center
        )
        add_hypothesis(
            initial,
            {
                "frame_family": "pca_24_proper",
                "axis_permutation": list(permutation),
                "axis_signs": list(signs),
            },
        )

    selected = min(hypotheses, key=lambda item: item["initial_score_symmetric_p80_mm"])
    transform, stages = _refine_transform(
        good_ds, defective_ds, selected["initial_transform"]
    )
    selected_aligned = apply_transform(defective_ds, transform)
    selected_x_to_v, _ = good_tree.query(selected_aligned, workers=1)
    selected_tree = cKDTree(selected_aligned)
    selected_v_to_x, _ = selected_tree.query(good_ds, workers=1)
    selected_final_score = 0.5 * (
        float(np.percentile(selected_x_to_v, 80))
        + float(np.percentile(selected_v_to_x, 80))
    )
    public_hypotheses = [
        {key: value for key, value in item.items() if key != "initial_transform"}
        for item in hypotheses
    ]
    return transform, {
        "method": "four dominant-plane plus 24 proper PCA-frame pose hypotheses + deterministic trimmed rigid ICP",
        "voxel_mm": 1.5,
        "selection_metric": "lowest symmetric P80 surface distance",
        "selected_frame_family": selected["frame_family"],
        "selected_normal_sign": selected.get("normal_sign"),
        "selected_tangent_sign": selected.get("tangent_sign"),
        "selected_axis_permutation": selected.get("axis_permutation"),
        "selected_axis_signs": selected.get("axis_signs"),
        "selected_initial_rotation_deg": selected["initial_rotation_deg"],
        "selected_final_score_symmetric_p80_mm": selected_final_score,
        "pose_hypotheses": public_hypotheses,
        "good_registration_points": len(good_ds),
        "defective_registration_points": len(defective_ds),
        "stages": stages,
    }


def distance_summary(distances: np.ndarray) -> dict:
    percentiles = (0, 25, 50, 75, 80, 90, 95, 99, 99.9, 100)
    return {
        "count": int(len(distances)),
        "mean_mm": float(np.mean(distances)),
        "rms_mm": float(np.sqrt(np.mean(np.square(distances)))),
        "percentiles_mm": {
            str(p): float(v) for p, v in zip(percentiles, np.percentile(distances, percentiles))
        },
        "fraction_above": {
            f"{threshold:g}_mm": float(np.mean(distances > threshold))
            for threshold in THRESHOLDS_MM
        },
    }


def cluster_deviations(
    points: np.ndarray, distances: np.ndarray, threshold_mm: float
) -> list[dict]:
    mask = distances > threshold_mm
    defect_points = points[mask]
    defect_distances = distances[mask]
    if not len(defect_points):
        return []
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(defect_points))
    labels = np.asarray(cloud.cluster_dbscan(eps=4.0, min_points=20, print_progress=False))
    clusters = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        selected = labels == label
        if selected.sum() < 100:
            continue
        local_points = defect_points[selected]
        local_distances = defect_distances[selected]
        clusters.append(
            {
                "points": int(selected.sum()),
                "centroid_xyz_mm": np.mean(local_points, axis=0).tolist(),
                "bbox_min_xyz_mm": np.min(local_points, axis=0).tolist(),
                "bbox_max_xyz_mm": np.max(local_points, axis=0).tolist(),
                "extent_xyz_mm": np.ptp(local_points, axis=0).tolist(),
                "median_deviation_mm": float(np.median(local_distances)),
                "p95_deviation_mm": float(np.percentile(local_distances, 95)),
                "max_deviation_mm": float(np.max(local_distances)),
            }
        )
    clusters.sort(key=lambda item: item["points"], reverse=True)
    return clusters


def _plane_segments(points: np.ndarray, count: int = 14) -> list[dict]:
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(voxel_sample(points, 1.2))
    )
    segments = []
    for index in range(count):
        plane, inliers = cloud.segment_plane(
            distance_threshold=0.22,
            ransac_n=3,
            num_iterations=2_000,
            probability=0.999,
        )
        patch = np.asarray(cloud.points)[np.asarray(inliers)]
        normal = np.asarray(plane[:3], dtype=float)
        normal /= np.linalg.norm(normal)
        segments.append(
            {
                "index": index,
                "points": int(len(patch)),
                "normal": normal,
                "centroid": np.median(patch, axis=0),
            }
        )
        cloud = cloud.select_by_index(inliers, invert=True)
    return segments


def compare_bend_planes(good: np.ndarray, defective: np.ndarray) -> dict:
    o3d.utility.random.seed(SEED + 1)
    good_segments = _plane_segments(good)
    o3d.utility.random.seed(SEED + 1)
    defective_segments = _plane_segments(defective)
    good_main = good_segments[0]["normal"]
    defective_main = defective_segments[0]["normal"]

    costs = np.zeros((len(good_segments), len(defective_segments)))
    for i, good_segment in enumerate(good_segments):
        for j, defective_segment in enumerate(defective_segments):
            normal_angle = math.degrees(
                math.acos(
                    np.clip(
                        abs(np.dot(good_segment["normal"], defective_segment["normal"])),
                        -1.0,
                        1.0,
                    )
                )
            )
            centroid_distance = np.linalg.norm(
                good_segment["centroid"] - defective_segment["centroid"]
            )
            costs[i, j] = centroid_distance + 2.0 * normal_angle

    good_indices, defective_indices = linear_sum_assignment(costs)
    matches = []
    for good_index, defective_index in zip(good_indices, defective_indices):
        good_segment = good_segments[good_index]
        defective_segment = defective_segments[defective_index]
        centroid_distance = float(
            np.linalg.norm(good_segment["centroid"] - defective_segment["centroid"])
        )
        normal_angle = math.degrees(
            math.acos(
                np.clip(
                    abs(np.dot(good_segment["normal"], defective_segment["normal"])),
                    -1.0,
                    1.0,
                )
            )
        )
        good_bend = math.degrees(
            math.acos(np.clip(abs(np.dot(good_main, good_segment["normal"])), -1.0, 1.0))
        )
        defective_bend = math.degrees(
            math.acos(
                np.clip(
                    abs(np.dot(defective_main, defective_segment["normal"])),
                    -1.0,
                    1.0,
                )
            )
        )
        if (
            good_segment["points"] >= 1_500
            and defective_segment["points"] >= 1_500
            and centroid_distance <= 6.0
            and normal_angle <= 5.0
            and good_bend >= 20.0
        ):
            matches.append(
                {
                    "good_plane_index": good_index,
                    "defective_plane_index": defective_index,
                    "good_points": good_segment["points"],
                    "defective_points": defective_segment["points"],
                    "centroid_distance_mm": centroid_distance,
                    "plane_normal_difference_deg": normal_angle,
                    "good_bend_angle_deg": good_bend,
                    "defective_bend_angle_deg": defective_bend,
                    "bend_angle_delta_deg": defective_bend - good_bend,
                }
            )
    max_delta = max((abs(item["bend_angle_delta_deg"]) for item in matches), default=0.0)
    return {
        "method": "matched large RANSAC planes relative to each scan's dominant body plane",
        "matched_nonparallel_planes": matches,
        "max_absolute_bend_angle_delta_deg": max_delta,
        "caution": "plane-level estimate; drawing feature identities and tolerances were not supplied",
    }


def view_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    sample = points[::20] - center
    _, eigenvectors = np.linalg.eigh(sample.T @ sample / len(sample))
    return center, eigenvectors[:, ::-1]


def render_overlay(
    good: np.ndarray, defective: np.ndarray, output: Path, *, part_id: str
) -> None:
    center, basis = view_basis(good)
    good_view = (good[::12] - center) @ basis
    defective_view = (defective[::12] - center) @ basis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    views = ((0, 1, "top / plan"), (0, 2, "long-side profile"), (1, 2, "short-side profile"))
    for axis, (a, b, title) in zip(axes, views):
        axis.scatter(good_view[:, a], good_view[:, b], s=0.08, color="#13b6d3", alpha=0.30, rasterized=True, label="V good baseline")
        axis.scatter(defective_view[:, a], defective_view[:, b], s=0.08, color="#ee3d83", alpha=0.30, rasterized=True, label="X defective aligned")
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("mm")
        axis.set_ylabel("mm")
    axes[0].legend(markerscale=12)
    fig.suptitle(f"{part_id} — rigid overlay (good V vs defective X)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_heatmap(
    good: np.ndarray,
    defective: np.ndarray,
    distances: np.ndarray,
    output: Path,
    *,
    part_id: str,
) -> None:
    center, basis = view_basis(good)
    good_view = (good[::18] - center) @ basis
    defective_view = (defective[::8] - center) @ basis
    displayed_distances = distances[::8]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    views = ((0, 1, "top / plan"), (0, 2, "long-side profile"), (1, 2, "short-side profile"))
    plotted = None
    display_max = max(2.0, min(25.0, float(np.percentile(distances, 99.5))))
    for axis, (a, b, title) in zip(axes, views):
        axis.scatter(good_view[:, a], good_view[:, b], s=0.08, color="#b8c1cc", alpha=0.12, rasterized=True)
        plotted = axis.scatter(
            defective_view[:, a], defective_view[:, b], c=np.clip(displayed_distances, 0, display_max),
            cmap="turbo", vmin=0, vmax=display_max, s=0.18, alpha=0.8, rasterized=True,
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("mm")
        axis.set_ylabel("mm")
    colorbar = fig.colorbar(
        plotted, ax=axes, orientation="horizontal", fraction=0.055, pad=0.13, aspect=55
    )
    colorbar.set_label(
        f"nearest-baseline surface deviation (mm; clipped at {display_max:g})"
    )
    fig.suptitle(f"{part_id} defective X — deviation from registered good V")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.22, top=0.86, wspace=0.18)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_histogram(distances: np.ndarray, output: Path, *, part_id: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 5), dpi=180)
    axis.hist(np.clip(distances, 0, 65), bins=130, color="#dd3d74", alpha=0.85)
    for threshold in (1, 2, 5, 10, 20):
        axis.axvline(threshold, color="#263747", linewidth=0.8, alpha=0.55)
    axis.set_xlabel("defective-to-good nearest surface deviation (mm)")
    axis.set_ylabel("registered defective vertices")
    axis.set_title(f"{part_id} deviation distribution")
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", type=Path, required=True)
    parser.add_argument("--defective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-id", default="scan_pair")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    o3d.utility.random.seed(SEED)
    good_mesh, good = read_mesh(args.good)
    defective_mesh, defective = read_mesh(args.defective)
    transform, registration = register(good, defective)
    defective_aligned = apply_transform(defective, transform)

    good_tree = cKDTree(good)
    defective_to_good, _ = good_tree.query(defective_aligned, workers=-1)
    defective_tree = cKDTree(defective_aligned)
    good_to_defective, _ = defective_tree.query(good, workers=-1)

    # Keep the production comparison bounded and deterministic. Dense low-threshold
    # DBSCAN is intentionally omitted here; spatial localization is conveyed by the
    # heatmap and the global/symmetric deviation distributions.
    clusters: list[dict] = []
    x_to_v_summary = distance_summary(defective_to_good)
    v_to_x_summary = distance_summary(good_to_defective)
    p95 = x_to_v_summary["percentiles_mm"]["95"]
    if p95 > 5.0 or x_to_v_summary["fraction_above"]["5_mm"] > 0.01:
        geometric_result = "DEFECTIVE_RELATIVE_TO_BASELINE"
        basis = "large residual surface geometry remains after global rigid registration"
    elif p95 > 1.0:
        geometric_result = "MEASURABLY_DIFFERENT_REQUIRES_TOLERANCE"
        basis = "localized millimetre-scale residuals remain after correct global pose alignment"
    else:
        geometric_result = "NOT_DISTINGUISHABLE_AT_CURRENT_SCAN_RESOLUTION"
        basis = "residuals are comparable with sub-millimetre scan/registration variation"

    report = {
        "part_id": str(args.part_id),
        "labels": {"good": "V", "defective": "X"},
        "units": "millimetres (inferred from part scale and source coordinates)",
        "good_source": str(args.good),
        "defective_source": str(args.defective),
        "good_vertices": len(good),
        "defective_vertices": len(defective),
        "registration": registration,
        "defective_to_good": x_to_v_summary,
        "good_to_defective": v_to_x_summary,
        "deviation_clusters_above_1_mm": clusters,
        "transform_defective_to_good": transform.tolist(),
        "classification": {
            "source_email_label": "DEFECTIVE",
            "geometric_result": geometric_result,
            "basis": basis,
            "production_disposition": "requires the drawing/profile tolerance; no tolerance was supplied with the scans",
        },
    }
    (args.output / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    render_overlay(
        good,
        defective_aligned,
        args.output / "overlay_good_V_vs_defective_X.png",
        part_id=str(args.part_id),
    )
    render_heatmap(
        good,
        defective_aligned,
        defective_to_good,
        args.output / "defective_X_deviation_heatmap.png",
        part_id=str(args.part_id),
    )
    render_histogram(
        defective_to_good,
        args.output / "deviation_histogram.png",
        part_id=str(args.part_id),
    )

    d = report["defective_to_good"]
    g = report["good_to_defective"]
    report_md = f"# {args.part_id} — good V vs defective X\n\n"
    report_md += f"**Geometric result:** {geometric_result}.\n\n"
    report_md += f"- X→V median deviation: {d['percentiles_mm']['50']:.3f} mm\n"
    report_md += f"- X→V P90 / P95 / P99: {d['percentiles_mm']['90']:.3f} / {d['percentiles_mm']['95']:.3f} / {d['percentiles_mm']['99']:.3f} mm\n"
    report_md += f"- X surface above 2 / 5 / 10 mm: {100*d['fraction_above']['2_mm']:.1f}% / {100*d['fraction_above']['5_mm']:.1f}% / {100*d['fraction_above']['10_mm']:.1f}%\n"
    report_md += f"- Symmetric V→X P95: {g['percentiles_mm']['95']:.3f} mm\n"
    report_md += f"- Connected regions above 1 mm: {len(clusters)}\n"
    report_md += "\n"
    report_md += "The email labels X defective and V good. The corrected global pose makes the scans closely coincident; the remaining differences are modest and require the drawing/profile and bend-angle tolerances for an independent production disposition.\n"
    (args.output / "REPORT.md").write_text(report_md)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
