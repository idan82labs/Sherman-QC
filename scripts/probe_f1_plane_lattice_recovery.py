#!/usr/bin/env python3
"""Probe scan-only plane-lattice bend candidates without truth/count input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.plane_lattice_bend_recovery import (
    build_supported_plane_intersections,
    extract_deterministic_planes,
)


def _load_points(scan_path: Path, voxel_mm: float) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    cloud = o3d.geometry.PointCloud()
    if len(mesh.vertices):
        cloud.points = mesh.vertices
    else:
        cloud = o3d.io.read_point_cloud(str(scan_path))
    cloud = cloud.voxel_down_sample(voxel_mm)
    return np.asarray(cloud.points, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-mm", type=float, default=1.8)
    parser.add_argument("--plane-distance-mm", type=float, default=0.8)
    parser.add_argument("--max-planes", type=int, default=12)
    parser.add_argument("--min-plane-points", type=int, default=75)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    scan_path = args.scan.resolve()
    points = _load_points(scan_path, args.voxel_mm)
    planes = extract_deterministic_planes(
        points,
        distance_threshold_mm=args.plane_distance_mm,
        max_planes=args.max_planes,
        min_plane_points=args.min_plane_points,
        seed=args.seed,
    )
    result = build_supported_plane_intersections(points, planes)
    payload = {
        "scan_path": str(scan_path),
        "owned_bend_regions": [
            {
                "bend_id": candidate["candidate_id"],
                "span_endpoints": candidate["span_endpoints"],
            }
            for candidate in result["candidates"]
        ],
        "plane_lattice_recovery": result,
        "probe_config": {
            "voxel_mm": args.voxel_mm,
            "plane_distance_mm": args.plane_distance_mm,
            "max_planes": args.max_planes,
            "min_plane_points": args.min_plane_points,
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
