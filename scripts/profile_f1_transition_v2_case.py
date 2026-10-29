#!/usr/bin/env python3
"""Measure one bounded TRANSITION_ATOM_V2 candidate-pool augmentation job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_scan_only_candidate_pool import (  # noqa: E402
    _load_geometry_views,
    augment_case_pool_with_transition_v2,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_case(
    *,
    scan_path: Path,
    source_pool_path: Path,
    output_path: Path,
    case_token: str,
    plane_voxel_mm: float = 1.8,
) -> dict[str, Any]:
    started = perf_counter()
    source_pool = _read(source_pool_path)
    points, normals, plane_points, plane_normals = _load_geometry_views(
        scan_path,
        plane_voxel_mm=float(plane_voxel_mm),
    )
    loaded_seconds = perf_counter() - started
    build_started = perf_counter()
    payload = augment_case_pool_with_transition_v2(
        source_pool,
        points=points,
        normals=normals,
        plane_points=plane_points,
        plane_normals=plane_normals,
    )
    build_seconds = perf_counter() - build_started
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    if sys.platform.startswith("linux"):
        peak_rss_bytes *= 1024
    result = {
        "schema_version": 1,
        "kind": "f1_transition_atom_v2_case_memory_profile",
        "case_token": str(case_token),
        "scan_sha256": _sha256(scan_path),
        "source_pool_sha256": _sha256(source_pool_path),
        "working_point_count": int(payload["transient_sampled_point_count"]),
        "plane_point_count": int(payload["transient_plane_point_count"]),
        "transition_parent_count": len(payload["support_parent_table"]),
        "transition_mode_count": len(payload["support_mode_table"]),
        "transition_mode_edge_count": len(payload["support_mode_edges"]),
        "geometry_load_seconds": round(loaded_seconds, 6),
        "v2_augmentation_seconds": round(build_seconds, 6),
        "wall_seconds": round(perf_counter() - started, 6),
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gib": round(peak_rss_bytes / float(1024**3), 6),
        "peak_rss_below_16_gib": peak_rss_bytes < 16 * 1024**3,
        "source_point_serialization": False,
        "candidate_geometry_frozen": True,
        "transition_atom_caps": payload["transition_atom_caps"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--source-pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-token", required=True)
    parser.add_argument("--plane-voxel-mm", type=float, default=1.8)
    args = parser.parse_args()
    result = profile_case(
        scan_path=args.scan.resolve(),
        source_pool_path=args.source_pool.resolve(),
        output_path=args.output.resolve(),
        case_token=args.case_token,
        plane_voxel_mm=args.plane_voxel_mm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
