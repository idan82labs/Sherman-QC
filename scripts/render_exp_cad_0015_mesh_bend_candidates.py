#!/usr/bin/env python3
"""Render source-only mesh-derived CAD bend candidates on actionable EXP15 scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.cad_bend_extractor import CADBendExtractor


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


def _equal_axes(axis, points: np.ndarray) -> None:
    mins, maxs = points.min(0), points.max(0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def render(path: Path, case_token: str, mesh, scan_points: np.ndarray, events: list[dict]) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    scan = scan_points[:: max(1, len(scan_points) // 80000)]
    combined = np.vstack([vertices, scan])
    views = ((20, -65, "oblique"), (90, -90, "top"), (5, 0, "side"))
    figure = plt.figure(figsize=(30, 18), dpi=100, facecolor="#0b1016")
    for index, (elevation, azimuth, label) in enumerate(views, 1):
        axis = figure.add_subplot(1, 3, index, projection="3d", facecolor="#0b1016")
        axis.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles,
            color="#bec6cf", edgecolor="#7f8b98", linewidth=0.12, alpha=0.28, shade=True,
        )
        axis.scatter(scan[:, 0], scan[:, 1], scan[:, 2], color="#46d9ff", s=0.7, alpha=0.32, linewidths=0)
        for event in events:
            start = np.asarray(event["line_start_mm"], dtype=float)
            end = np.asarray(event["line_end_mm"], dtype=float)
            axis.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color="#ff2da8", linewidth=5.0, solid_capstyle="round",
            )
            midpoint = (start + end) / 2.0
            axis.text(midpoint[0], midpoint[1], midpoint[2], str(event["event_id"]), color="#ffd8ef", fontsize=8)
        axis.view_init(elev=elevation, azim=azimuth)
        _equal_axes(axis, combined)
        axis.set_title(label, color="white", fontsize=18, pad=14)
        axis.set_axis_off()
    figure.suptitle(
        f"{case_token} — mesh-derived CAD bend candidates (magenta)\n"
        "cyan: aligned authenticated scan; grey: folded CAD",
        color="white", fontsize=23, y=0.96,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-review", required=True, type=Path)
    parser.add_argument("--registration-root", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    review = json.loads(args.visual_review.read_text(encoding="utf-8"))
    pass_cases = [row["case_token"] for row in review["decisions"] if row["decision"].startswith("PASS_")]
    records = []
    extractor = CADBendExtractor(
        coplanar_threshold=15.0,
        min_area_pct=0.2,
        max_bend_gap=20.0,
        min_bend_angle=8.0,
        dedup_distance=8.0,
    )
    for case_token in pass_cases:
        semantic_path = args.registration_root / "cases" / case_token / "semantic_result.json"
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        aligned_scan_path = Path(semantic["artifacts"]["aligned_scan"]["path"])
        mesh_hash = semantic["engine_mesh_sha256"]
        matches = [path for path in args.mesh_root.glob("*.ply") if sha256_path(path) == mesh_hash]
        if len(matches) != 1:
            raise ValueError(f"expected one mesh for {case_token}, got {len(matches)}")
        mesh_path = matches[0]
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        result = extractor.extract_from_mesh(mesh)
        events = []
        for index, bend in enumerate(result.bends, 1):
            events.append(
                {
                    "event_id": index,
                    "line_start_mm": np.asarray(bend.bend_line_start, dtype=float).tolist(),
                    "line_end_mm": np.asarray(bend.bend_line_end, dtype=float).tolist(),
                    "span_mm": float(np.linalg.norm(bend.bend_line_end - bend.bend_line_start)),
                    "normal_angle_deg": float(bend.angle_between_normals),
                    "surface_region_ids": [int(bend.surface1_id), int(bend.surface2_id)],
                }
            )
        scan_points = np.asarray(o3d.io.read_point_cloud(str(aligned_scan_path)).points, dtype=np.float64)
        case_root = args.output_root / "cases" / case_token
        render_path = case_root / "native_three_view_candidate_overlay.png"
        render(render_path, case_token, mesh, scan_points, events)
        payload = {
            "schema": "EXP_CAD_0015_MESH_BEND_CANDIDATE_CASE_V1",
            "case_token": case_token,
            "registration_semantic_sha256": sha256_path(semantic_path),
            "mesh_path": str(mesh_path.resolve()),
            "mesh_sha256": mesh_hash,
            "candidate_count": len(events),
            "events": events,
            "render": {"path": str(render_path.resolve()), "sha256": sha256_path(render_path)},
            "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
            "gold_or_expected_counts_used": False,
            "evaluator_data_opened": False,
            "generalization_evidence": False,
        }
        atomic_json(case_root / "candidate.json", payload)
        records.append(
            {
                "case_token": case_token,
                "candidate_count": len(events),
                "candidate_sha256": sha256_path(case_root / "candidate.json"),
                "render_sha256": sha256_path(render_path),
            }
        )
    summary = {
        "schema": "EXP_CAD_0015_MESH_BEND_CANDIDATE_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0015",
        "visual_review_sha256": sha256_path(args.visual_review),
        "case_count": len(records),
        "records": records,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "gold_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "generalization_evidence": False,
    }
    atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps({"case_count": len(records), "candidate_counts": {row["case_token"]: row["candidate_count"] for row in records}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
