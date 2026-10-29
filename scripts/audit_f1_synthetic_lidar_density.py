#!/usr/bin/env python3
"""Compare synthetic nearest-neighbor spacing with geometry-only LiDAR scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


REFERENCE_PATHS = (
    "live_scans/47266000_F_LIDAR_DEMO_20260209_155400_715790.ply",
    "live_scans/11979003_C_LIDAR_DEMO_20260209_154630_251479.ply",
    "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191038_02.ply",
    "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/48333018_00.ply",
    "data/bend_corpus/incoming/2026-07-28-shaked-good-defective-38172000/38172000_01 (V).ply",
    "data/bend_corpus/incoming/2026-07-28-shaked-good-defective-38172000/38172000_01 (X).ply",
    "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/50556000_04/scans/50556000_04_V.ply",
    "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/50556000_04/scans/50556000_04_X.ply",
    "data/bend_corpus/incoming/2026-08-04_shaked_scan_quality_parts/parts/45796000_A/scans/45796000_A_SCAN_1.ply",
    "data/bend_corpus/incoming/2026-08-04_shaked_scan_quality_parts/parts/45796000_A/scans/45796000_A_SCAN_2.ply",
)


def _points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
    return np.unique(points[np.isfinite(points).all(axis=1)], axis=0)


def _row(path: Path, case_id: str | None = None) -> dict:
    points = _points(path)
    sample = points[np.linspace(0, len(points) - 1, min(30_000, len(points)), dtype=np.int64)]
    distances, _indices = cKDTree(points).query(sample, k=2, workers=-1)
    nearest = distances[:, 1]
    return {
        "case_id": case_id,
        "path": str(path),
        "point_count": len(points),
        "bbox_mm": np.ptp(points, axis=0).round(6).tolist(),
        "nearest_neighbor_mm": {str(value): round(float(np.quantile(nearest, value)), 6) for value in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    corpus = args.corpus_root.resolve()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    references = [_row(repo / relative) for relative in REFERENCE_PATHS]
    synthetic = [_row(Path(row["scan_path"]), str(row["case_id"])) for row in manifest["cases"]]
    reference_medians = [row["nearest_neighbor_mm"]["0.5"] for row in references]
    synthetic_medians = [row["nearest_neighbor_mm"]["0.5"] for row in synthetic]
    lower, upper = min(reference_medians), max(reference_medians)
    report = {
        "schema_version": 1,
        "kind": "synthetic_lidar_density_calibration",
        "confirmatory_eligible": False,
        "reference": references,
        "synthetic": synthetic,
        "summary": {
            "reference_scan_count": len(references),
            "synthetic_scan_count": len(synthetic),
            "reference_median_nn_range_mm": [lower, upper],
            "reference_median_of_medians_mm": float(np.median(reference_medians)),
            "synthetic_median_nn_range_mm": [min(synthetic_medians), max(synthetic_medians)],
            "synthetic_median_of_medians_mm": float(np.median(synthetic_medians)),
            "all_synthetic_medians_inside_observed_reference_range": all(lower <= value <= upper for value in synthetic_medians),
            "all_synthetic_points_at_or_below_200k": all(row["point_count"] <= 200_000 for row in synthetic),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if all((report["summary"]["all_synthetic_medians_inside_observed_reference_range"], report["summary"]["all_synthetic_points_at_or_below_200k"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
