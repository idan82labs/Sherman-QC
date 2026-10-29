#!/usr/bin/env python3
"""Generate a fresh prospective calibrated raw-range replication corpus.

V2 uses the same frozen acquisition model and 15 near-neighbour profile
families as V1, but creates capture variants 3 and 4.  Those variants have
independent geometry noise, holes, occlusion, scan dropout, and rigid poses.
The evaluator tree remains private until predictions and visual rulings freeze.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v1 as v1


CORPUS_KIND = "sherman_qc_f1_calibrated_raw_range_synthetic_capture_replication_v2"
CAPTURE_INDICES = (2, 3)


def generate(
    output_root: Path,
    *,
    width: int = v1.DEFAULT_WIDTH,
    height: int = v1.DEFAULT_HEIGHT,
    view_count: int = v1.DEFAULT_VIEW_COUNT,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    public_rows = []
    private_rows = []
    metadata_rows = []
    for family_index, family in enumerate(v1.FAMILIES):
        for slot, capture_index in enumerate(CAPTURE_INDICES):
            opaque_case_id = f"O{family_index * 2 + slot + 1:03d}"
            public_row, private_row, metadata = v1._write_case(
                output_root=output_root,
                opaque_case_id=opaque_case_id,
                family=family,
                capture=capture_index,
                width=width,
                height=height,
                view_count=view_count,
            )
            metadata["replication_capture_slot"] = slot + 1
            metadata["source_capture_index"] = capture_index
            public_rows.append(public_row)
            private_rows.append(private_row)
            metadata_rows.append(metadata)
    public_manifest = {
        "schema_version": 1,
        "kind": CORPUS_KIND,
        "token": v1.TOKEN,
        "confirmatory_eligible": False,
        "prospective_replication": True,
        "case_count": len(public_rows),
        "family_count": len(v1.FAMILIES),
        "cases": public_rows,
    }
    private_manifest = {
        "schema_version": 1,
        "token": v1.TOKEN,
        "confirmatory_eligible": False,
        "prospective_replication": True,
        "cases": private_rows,
    }
    renderer_manifest = {
        "schema_version": 1,
        "token": v1.TOKEN,
        "algorithm": "open3d_raycast_hidden_scene_to_organized_depth_then_fuse",
        "replication_capture_indices": list(CAPTURE_INDICES),
        "width": width,
        "height": height,
        "view_count": view_count,
        "horizontal_fov_degrees": v1.HORIZONTAL_FOV_DEGREES,
        "depth_noise_sigma_mm": v1.DEPTH_NOISE_SIGMA_MM,
        "random_dropout_probability": v1.RANDOM_DROPOUT_PROBABILITY,
        "initial_fusion_voxel_mm": v1.INITIAL_FUSION_VOXEL_MM,
        "max_source_points": v1.MAX_SOURCE_POINTS,
        "generator_sha256": v1._sha256(Path(__file__).resolve()),
        "v1_acquisition_generator_sha256": v1._sha256(Path(v1.__file__).resolve()),
        "base_scene_generator_sha256": v1._sha256(Path(v1.__file__).with_name("generate_f1_lidar_like_synthetic_corpus.py").resolve()),
    }
    v1._atomic_json(output_root / "public_manifest.json", public_manifest)
    v1._atomic_json(output_root / "evaluator_only" / "private_manifest.json", private_manifest)
    v1._atomic_json(output_root / "evaluator_only" / "renderer_manifest.json", renderer_manifest)
    v1._atomic_json(output_root / "evaluator_only" / "all_generation_metadata.json", {"schema_version": 1, "token": v1.TOKEN, "cases": metadata_rows})
    return public_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=v1.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=v1.DEFAULT_HEIGHT)
    parser.add_argument("--view-count", type=int, default=v1.DEFAULT_VIEW_COUNT)
    args = parser.parse_args()
    report = generate(args.output_root.resolve(), width=args.width, height=args.height, view_count=args.view_count)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "case_count": report["case_count"], "family_count": report["family_count"], "capture_indices": list(CAPTURE_INDICES), "confirmatory_eligible": False, "token": v1.TOKEN}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
