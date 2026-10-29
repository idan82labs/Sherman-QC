#!/usr/bin/env python3
"""Create deterministic scan-only jobs for manual bend-axis adjudication.

The generated jobs intentionally contain no detector markers.  Their only
purpose is to expose verified-corpus geometry through Bend Label Studio while
keeping clicked coordinates in the source scan's raw coordinate frame.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
ASSET_ROOT = PROJECT_ROOT.parent / "source-assets"
SOURCE_ROOT = ASSET_ROOT / "02_CAD_QC_3D"
MAX_POINTS = 200_000

CASES = (
    {
        "job_id": "bend_label_12_49125000",
        "part_id": "49125000",
        "part_name": "49125000 — manual bend truth (2 expected)",
        "source": SOURCE_ROOT / "49125000_A00.ply",
        "expected_bends": 2,
        "case_id": "12_49125000",
    },
    {
        "job_id": "bend_label_04_15651004",
        "part_id": "15651004",
        "part_name": "15651004 — manual bend truth (1 expected)",
        "source": SOURCE_ROOT / "15651004_01_SECOND_SCAN.stl",
        "expected_bends": 1,
        "case_id": "04_15651004",
    },
    {
        "job_id": "bend_label_09_47959001",
        "part_id": "47959001",
        "part_name": "47959001 — manual bend truth (5 expected)",
        "source": SOURCE_ROOT / "47959001_01PREL_scan1.stl",
        "expected_bends": 5,
        "case_id": "09_47959001",
    },
    {
        "job_id": "bend_label_03_10839000",
        "part_id": "10839000",
        "part_name": "10839000 scan 2 — manual bend truth (6 expected)",
        "source": SOURCE_ROOT / "10839000" / "10839000_01_scan_2.ply",
        "expected_bends": 6,
        "case_id": "03_10839000",
    },
    {
        "job_id": "bend_label_01_10839000",
        "part_id": "10839000",
        "part_name": "10839000 full scan — manual bend truth (8 expected)",
        "source": SOURCE_ROOT / "10839000_01_full_scan.ply",
        "expected_bends": 8,
        "case_id": "01_10839000",
    },
    {
        "job_id": "bend_label_02_10839000",
        "part_id": "10839000",
        "part_name": "10839000 scan 1 — manual bend truth (4 expected)",
        "source": SOURCE_ROOT / "10839000" / "10839000_01_scan_1.ply",
        "expected_bends": 4,
        "case_id": "02_10839000",
    },
    {
        "job_id": "bend_label_05_37361005",
        "part_id": "37361005",
        "part_name": "37361005 — manual bend truth (4 expected)",
        "source": SOURCE_ROOT / "37361005_01.ply",
        "expected_bends": 4,
        "case_id": "05_37361005",
    },
    {
        "job_id": "bend_label_06_38172000",
        "part_id": "38172000",
        "part_name": "38172000 scan 1 — manual bend truth (6 expected)",
        "source": SOURCE_ROOT / "38172000_01_SCAN_1.ply",
        "expected_bends": 6,
        "case_id": "06_38172000",
    },
    {
        "job_id": "bend_label_07_38172000",
        "part_id": "38172000",
        "part_name": "38172000 scan 2 — manual bend truth (8 expected)",
        "source": SOURCE_ROOT / "38172000_01_SCAN_2 1.ply",
        "expected_bends": 8,
        "case_id": "07_38172000",
    },
    {
        "job_id": "bend_label_08_40500000",
        "part_id": "40500000",
        "part_name": "40500000 scan 2 — manual bend truth (7 expected)",
        "source": SOURCE_ROOT / "40500000_0_SCAN2 1.ply",
        "expected_bends": 7,
        "case_id": "08_40500000",
    },
    {
        "job_id": "bend_label_10_49001000",
        "part_id": "49001000",
        "part_name": "49001000 — manual bend truth (5 expected)",
        "source": SOURCE_ROOT / "49001000_A-PRELIM.ply",
        "expected_bends": 5,
        "case_id": "10_49001000",
    },
    {
        "job_id": "bend_label_11_49024000",
        "part_id": "49024000",
        "part_name": "49024000 — manual bend truth (3 expected)",
        "source": SOURCE_ROOT / "49024000_A.ply",
        "expected_bends": 3,
        "case_id": "11_49024000",
    },
    {
        "job_id": "bend_label_13_ysherman",
        "part_id": "ysherman",
        "part_name": "ysherman — manual bend truth (12 expected)",
        "source": ASSET_ROOT / "wetransfer_ysherman-asc_2026-04-20_0802" / "ysherman.ply",
        "expected_bends": 12,
        "case_id": "13_ysherman",
    },
)


def _source_vertices(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError(f"no finite vertices in {path}")
    if len(points) > MAX_POINTS:
        # An index grid is deterministic and preserves the scan's source frame.
        indices = np.linspace(0, len(points) - 1, MAX_POINTS, dtype=np.int64)
        points = points[indices]
    return points


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_jobs() -> None:
    # Persisted legacy jobs use local, timezone-naive ISO strings.  Match that
    # convention so the existing list sorter never compares aware and naive
    # datetime objects.
    base_time = datetime(2026, 7, 15, 12, 0)
    for offset, case in enumerate(CASES):
        source = Path(case["source"])
        if not source.is_file():
            raise FileNotFoundError(source)
        output_dir = OUTPUT_ROOT / str(case["job_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        aligned_scan = output_dir / "aligned_scan.ply"
        points = _source_vertices(source)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if not o3d.io.write_point_cloud(str(aligned_scan), cloud, write_ascii=False):
            raise OSError(f"could not write {aligned_scan}")

        created_at = (base_time + timedelta(seconds=offset)).isoformat()
        report = {
            "part_id": case["part_id"],
            "matches": [],
            "unmatched_detections": [],
            "operator_actions": [],
            "process_feature_actions": [],
            "summary": {
                "total_bends": 0,
                "countable_total_bends": 0,
                "detected": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "progress_pct": 0,
                "completed_bends": 0,
                "remaining_bends": case["expected_bends"],
                "expected_bends": case["expected_bends"],
                "scan_quality_status": "MANUAL_REVIEW",
            },
        }
        meta = {
            "status": "completed",
            "progress": 100,
            "part_id": case["part_id"],
            "part_name": case["part_name"],
            "created_at": created_at,
            "completed_at": created_at,
            "seed_case": {
                "state": "manual_label_review",
                "labeling_mode": "scan_only",
                "case_id": case["case_id"],
                "source_scan": str(source),
                "source_sha256": _sha256(source),
                "display_point_count": int(len(points)),
                "coordinate_frame": "reference_mesh_raw",
                "coordinate_frame_provenance": "unaligned_source_scan_raw",
                "expected_bends": case["expected_bends"],
            },
        }
        _write_json(output_dir / "bend_inspection_report.json", report)
        _write_json(output_dir / "bend_inspection_meta.json", meta)
        print(f"{case['job_id']}: {len(points)} points from {source.name}")


if __name__ == "__main__":
    create_jobs()
