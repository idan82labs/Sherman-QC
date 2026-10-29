#!/usr/bin/env python3
"""Metadata-only inventory for authentic organized observation availability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RAW_EXTENSIONS = {".bag", ".mcap", ".raw", ".exr", ".tif", ".tiff", ".npy", ".npz"}
MANIFEST_NAME = "organized_frame_bundle_v1.json"


def audit(root: Path) -> dict:
    root = root.resolve()
    raw_candidates = []
    manifests = []
    fused_cloud_count = 0
    skipped_protected = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        lowered_parts = {part.lower() for part in relative.parts}
        if "evaluator_only" in lowered_parts or path.name.lower() == "evaluator_only_manifest.json":
            skipped_protected += 1
            continue
        if path.name == MANIFEST_NAME:
            manifests.append(str(relative))
        if path.suffix.lower() == ".ply":
            fused_cloud_count += 1
        if path.suffix.lower() in RAW_EXTENSIONS:
            raw_candidates.append({
                "relative_path": str(relative),
                "byte_count": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    status = "READY_FOR_ORGANIZED_FRAME_VALIDATION" if manifests and raw_candidates else "CAPABILITY_NOT_PRESENT"
    return {
        "schema": "F1_ORGANIZED_FRAME_CAPACITY_AUDIT_V1",
        "status": status,
        "root": str(root),
        "fused_ply_count": fused_cloud_count,
        "organized_bundle_manifest_count": len(manifests),
        "raw_sensor_candidate_count": len(raw_candidates),
        "organized_bundle_manifests": manifests,
        "raw_sensor_candidates": raw_candidates,
        "protected_paths_skipped": skipped_protected,
        "synthetic_or_render_cache_accepted": False,
        "required_next_input": "AUTHENTIC_PRE_FUSION_DEPTH_OR_RANGE_FRAMES_PLUS_CALIBRATION_AND_VALIDITY_MASKS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print(json.dumps({key: result[key] for key in ("status", "fused_ply_count", "organized_bundle_manifest_count", "raw_sensor_candidate_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
