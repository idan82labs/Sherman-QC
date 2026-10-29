#!/usr/bin/env python3
"""Run truth-denying deterministic registration on authenticated new CAD/scan pairs.

This is an intake and visual-actionability gate.  It does not open evaluator
truth and does not score bend predictions.  Every ready scan is aligned to the
face-owned folded-body mesh derived from its authenticated SLDPRT, then rendered
as a native-resolution three-view distance overlay for manual review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp_cad_0014_analytic_plane_registration import AnalyticPlaneCaptureEngine


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"invalid reference mesh: {path}")
    return vertices, triangles


def _set_equal_axes(axis, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def exact_point_to_mesh_distances(
    points: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    chunk_size: int = 50000,
) -> np.ndarray:
    """Compute exact unsigned point-to-triangle distances in bounded chunks."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    distances = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        _closest, chunk_distances, _triangle_ids = mesh.nearest.on_surface(points[start:stop])
        distances[start:stop] = chunk_distances
    if not np.isfinite(distances).all():
        raise ValueError("exact point-to-mesh query produced non-finite distances")
    return distances


def render_overlay(
    output: Path,
    *,
    case_token: str,
    reference_vertices: np.ndarray,
    reference_triangles: np.ndarray,
    aligned_scan_points: np.ndarray,
    scan_distances: np.ndarray,
    fitness: float,
    rmse: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scan_step = max(1, len(aligned_scan_points) // 90000)
    scan = aligned_scan_points[::scan_step]
    distances = scan_distances[::scan_step]
    triangle_step = max(1, len(reference_triangles) // 18000)
    triangles = reference_triangles[::triangle_step]
    combined = np.vstack([reference_vertices, scan])
    views = ((20, -65, "oblique"), (90, -90, "top"), (5, 0, "side"))
    figure = plt.figure(figsize=(30, 18), dpi=100, facecolor="#0b1016")
    scatter = None
    for index, (elevation, azimuth, label) in enumerate(views, 1):
        axis = figure.add_subplot(1, 3, index, projection="3d", facecolor="#0b1016")
        axis.plot_trisurf(
            reference_vertices[:, 0],
            reference_vertices[:, 1],
            reference_vertices[:, 2],
            triangles=triangles,
            color="#d7dce2",
            alpha=0.22,
            linewidth=0.15,
            edgecolor="#aeb7c2",
            shade=True,
        )
        scatter = axis.scatter(
            scan[:, 0],
            scan[:, 1],
            scan[:, 2],
            c=np.clip(distances, 0.0, 8.0),
            cmap="turbo",
            vmin=0.0,
            vmax=8.0,
            s=1.1,
            alpha=0.88,
            linewidths=0,
        )
        axis.view_init(elev=elevation, azim=azimuth)
        _set_equal_axes(axis, combined)
        axis.set_title(label, color="white", fontsize=18, pad=14)
        axis.set_axis_off()
    if scatter is not None:
        colorbar = figure.colorbar(scatter, ax=figure.axes, fraction=0.018, pad=0.015, shrink=0.72)
        colorbar.set_label("exact scan-to-CAD triangle distance (mm; clipped at 8)", color="white", fontsize=15)
        colorbar.ax.tick_params(colors="white")
    figure.suptitle(
        f"{case_token} — authenticated scan over face-owned folded CAD\n"
        f"registration fitness={fitness:.3f}, ICP RMSE={rmse:.3f} mm",
        color="white",
        fontsize=23,
        y=0.96,
    )
    figure.savefig(output, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def execute_case(row: dict, mesh_root: Path, output_root: Path, seed: int) -> dict:
    scan_path = Path(row["scan_path"]).resolve()
    source_cad_path = Path(row["cad_path"]).resolve()
    mesh_stem = source_cad_path.stem
    mesh_path = (mesh_root / f"{mesh_stem}.ply").resolve()
    mesh_report_path = (mesh_root / f"{mesh_stem}.report.json").resolve()
    for path, expected in (
        (scan_path, row["scan_sha256"]),
        (source_cad_path, row["cad_sha256"]),
    ):
        if not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"authenticated source mismatch: {path}")
    if not mesh_path.is_file() or not mesh_report_path.is_file():
        raise FileNotFoundError(f"engine mesh unavailable for {mesh_stem}")
    mesh_report = json.loads(mesh_report_path.read_text(encoding="utf-8"))
    if mesh_report.get("source", {}).get("sha256") != row["cad_sha256"]:
        raise ValueError("engine mesh source binding mismatch")
    if mesh_report.get("mesh", {}).get("sha256") != sha256_path(mesh_path):
        raise ValueError("engine mesh hash mismatch")

    engine = AnalyticPlaneCaptureEngine()
    engine.load_reference(str(mesh_path))
    engine.load_scan(str(scan_path))
    raw_scan_count = int(len(engine.scan_pcd.points))
    voxel = 2.0 if raw_scan_count >= 650000 else 1.5 if raw_scan_count >= 450000 else 1.0
    engine.preprocess(voxel_size=voxel)
    fitness, rmse = engine.align(auto_scale=True, tolerance=1.0, random_seed=seed)
    aligned = engine.aligned_scan_raw if engine.aligned_scan_raw is not None else engine.aligned_scan
    aligned_points = np.asarray(aligned.points, dtype=np.float64)
    if len(aligned_points) == 0 or not np.isfinite(aligned_points).all():
        raise ValueError("alignment produced invalid scan points")

    vertices, triangles = _load_mesh(mesh_path)
    distances = exact_point_to_mesh_distances(aligned_points, vertices, triangles)
    finite_distances = distances[np.isfinite(distances)]
    if len(finite_distances) != len(distances):
        raise ValueError("non-finite registration distances")
    case_root = output_root / "cases" / row["opaque_id"]
    aligned_path = case_root / "aligned_scan.ply"
    case_root.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(aligned_path), aligned, write_ascii=False, compressed=False)
    overlay_path = case_root / "native_three_view_overlay.png"
    render_overlay(
        overlay_path,
        case_token=row["opaque_id"],
        reference_vertices=vertices,
        reference_triangles=triangles,
        aligned_scan_points=aligned_points,
        scan_distances=distances,
        fitness=float(fitness),
        rmse=float(rmse),
    )

    metrics = {
        "median_distance_mm": float(np.median(finite_distances)),
        "p90_distance_mm": float(np.percentile(finite_distances, 90)),
        "p95_distance_mm": float(np.percentile(finite_distances, 95)),
        "within_1mm_fraction": float(np.mean(finite_distances <= 1.0)),
        "within_2mm_fraction": float(np.mean(finite_distances <= 2.0)),
        "within_5mm_fraction": float(np.mean(finite_distances <= 5.0)),
    }
    payload = {
        "schema": "EXP_CAD_0015_NEW_PHYSICAL_REGISTRATION_CASE_V2",
        "experiment_id": "EXP_CAD_0015",
        "case_token": row["opaque_id"],
        "scan_sha256": row["scan_sha256"],
        "source_cad_sha256": row["cad_sha256"],
        "engine_mesh_sha256": sha256_path(mesh_path),
        "engine_mesh_report_sha256": sha256_path(mesh_report_path),
        "raw_scan_point_count": raw_scan_count,
        "aligned_scan_point_count": len(aligned_points),
        "voxel_size_mm": voxel,
        "fitness": float(fitness),
        "icp_rmse_mm": float(rmse),
        "scale_factor": float(engine.scale_factor),
        "alignment_source": engine.alignment_metadata.source,
        "alignment_abstained": bool(engine.alignment_metadata.abstained),
        "alignment_conditioning": engine.alignment_metadata.conditioning,
        "raw_scan_to_reference_transform": np.asarray(engine.raw_scan_to_reference_transform).tolist(),
        "distance_metrics": metrics,
        "distance_method": "EXACT_UNSIGNED_POINT_TO_TRIANGLE_TRIMESH_CHUNKED_V1",
        "artifacts": {
            "aligned_scan": {"path": str(aligned_path.resolve()), "sha256": sha256_path(aligned_path)},
            "overlay": {"path": str(overlay_path.resolve()), "sha256": sha256_path(overlay_path)},
        },
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "gold_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "generalization_evidence": False,
    }
    atomic_json(case_root / "semantic_result.json", payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    records: list[dict] = []
    source_manifests: list[dict] = []
    for manifest_path in args.source_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("evaluator_only_manifest_opened") is not False:
            raise ValueError("truth-denying source manifest required")
        if manifest.get("gold_or_expected_counts_present") is not False:
            raise ValueError("source manifest must not contain truth")
        records.extend(manifest["records"])
        source_manifests.append({"path": str(manifest_path.resolve()), "sha256": sha256_path(manifest_path)})

    started = perf_counter()
    completed: list[dict] = []
    blocked: list[dict] = []
    for index, row in enumerate(records, 1):
        print(f"[{index}/{len(records)}] {row['opaque_id']}", flush=True)
        try:
            payload = execute_case(row, args.mesh_root, args.output_root, args.seed)
            completed.append(
                {
                    "case_token": payload["case_token"],
                    "semantic_result_sha256": sha256_path(
                        args.output_root / "cases" / payload["case_token"] / "semantic_result.json"
                    ),
                    "fitness": payload["fitness"],
                    "icp_rmse_mm": payload["icp_rmse_mm"],
                    "distance_metrics": payload["distance_metrics"],
                }
            )
        except Exception as exc:
            blocked.append(
                {
                    "case_token": row["opaque_id"],
                    "cad_source_sha256": row["cad_sha256"],
                    "reason_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
    summary = {
        "schema": "EXP_CAD_0015_NEW_PHYSICAL_REGISTRATION_SUMMARY_V2",
        "experiment_id": "EXP_CAD_0015",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_ONLY_REGISTRATION_VISUALS_READY" if completed else "BLOCKED_NO_READY_CASES",
        "source_manifests": source_manifests,
        "declared_case_count": len(records),
        "completed_case_count": len(completed),
        "blocked_case_count": len(blocked),
        "elapsed_seconds": perf_counter() - started,
        "completed": completed,
        "blocked": blocked,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "gold_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "generalization_evidence": False,
    }
    atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps({key: summary[key] for key in ("declared_case_count", "completed_case_count", "blocked_case_count", "elapsed_seconds")}, indent=2))
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
