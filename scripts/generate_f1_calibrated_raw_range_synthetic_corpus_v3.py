#!/usr/bin/env python3
"""Generate the harder calibrated raw-range synthetic geometry corpus V3.

V3 changes geometry, not merely capture noise.  It contains compound profiles,
close-spaced shallow creases, mixed small and large bends, tight fillets, and
broad rounded facets.  All public clouds are fused from organized calibrated
depth frames rendered from evaluator-only meshes.  This remains synthetic and
non-confirmatory for physical scanner performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v1 as capture
from scripts.generate_f1_lidar_like_synthetic_corpus import build_case


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V3_COMPLEX_GEOMETRY_NON_CONFIRMATORY"
CORPUS_KIND = "sherman_qc_f1_calibrated_raw_range_complex_geometry_v3"

# Two unseen stochastic capture variants per family.  Geometry family identity
# and acquisition settings are frozen before any inference result is produced.
CAPTURE_INDICES = (6, 7)

# Profile coordinates are (cross-section horizontal mm, cross-section vertical
# mm).  Interior vertices are evaluator-only bend targets.  Radius is the
# nominal cross-section fillet radius, bounded by neighbouring facet lengths by
# the base mesh builder.
FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "id": "HF01",
        "name": "micro_ribbed_hat",
        "profile": [(0, 0), (18, 0), (21, 3), (26, 3), (29, 0), (54, 0), (59, 7), (68, 7), (73, 0), (98, 0)],
        "length": 182,
        "radius": 0.8,
        "challenge": "two close 3 mm micro creases plus a shallow 7 mm rib",
    },
    {
        "id": "HF02",
        "name": "rolled_omega_return",
        "profile": [(0, 0), (12, 0), (20, 10), (20, 44), (31, 56), (67, 56), (78, 44), (78, 10), (86, 0), (110, 0)],
        "length": 215,
        "radius": 7.5,
        "challenge": "broad rounded shoulders mixed with close vertical returns",
    },
    {
        "id": "HF03",
        "name": "mixed_pitch_corrugation",
        "profile": [(0, 0), (14, 0), (18, 4), (23, 4), (27, 0), (45, 0), (54, 16), (67, 16), (73, 7), (80, 7), (85, 0), (108, 0)],
        "length": 196,
        "radius": 1.2,
        "challenge": "alternating 4, 7, and 16 mm bends with unequal pitch",
    },
    {
        "id": "HF04",
        "name": "nested_return_channel",
        "profile": [(9, 50), (0, 50), (0, 8), (6, 2), (72, 2), (78, 8), (78, 50), (69, 50), (64, 45)],
        "length": 228,
        "radius": 2.2,
        "challenge": "deep channel with nested inward returns and short terminal facets",
    },
    {
        "id": "HF05",
        "name": "shallow_crowned_panel",
        "profile": [(0, 0), (24, 0), (31, 2.5), (42, 4.5), (55, 5.5), (68, 4.5), (79, 2.5), (86, 0), (112, 0)],
        "length": 205,
        "radius": 4.8,
        "challenge": "subtle rounded crown distributed across six small angle changes",
    },
    {
        "id": "HF06",
        "name": "compound_s_joggle",
        "profile": [(0, 0), (20, 0), (25, 5), (25, 14), (32, 21), (52, 21), (58, 15), (58, 8), (66, 0), (94, 0)],
        "length": 174,
        "radius": 1.5,
        "challenge": "compact S-joggle with close bends and unequal facet widths",
    },
    {
        "id": "HF07",
        "name": "acute_to_rolled_transition",
        "profile": [(0, 0), (30, 0), (38, 25), (47, 36), (62, 42), (79, 42), (87, 37), (93, 25), (118, 25)],
        "length": 188,
        "radius": 6.0,
        "challenge": "acute rise transitioning into a broad multi-facet rolled shoulder",
    },
    {
        "id": "HF08",
        "name": "micro_double_step",
        "profile": [(0, 0), (16, 0), (16, 2.2), (24, 2.2), (24, 5.0), (34, 5.0), (34, 9.0), (52, 9.0), (58, 15), (86, 15)],
        "length": 162,
        "radius": 0.55,
        "challenge": "2.2 to 4 mm close-spaced steps near sampling limits",
    },
    {
        "id": "HF09",
        "name": "asymmetric_compound_hat",
        "profile": [(0, 0), (15, 0), (19, 5), (24, 22), (34, 35), (69, 35), (76, 30), (82, 12), (91, 4), (118, 4)],
        "length": 238,
        "radius": 3.2,
        "challenge": "different left and right shoulder structures with short facets",
    },
    {
        "id": "HF10",
        "name": "rolled_c_return",
        "profile": [(14, 58), (4, 58), (0, 52), (0, 12), (7, 2), (72, 2), (82, 12), (82, 52), (77, 58), (66, 58)],
        "length": 246,
        "radius": 8.0,
        "challenge": "large rolled corners terminating in short inward return facets",
    },
    {
        "id": "HF11",
        "name": "close_parallel_crease_shelf",
        "profile": [(0, 0), (22, 0), (25, 2.6), (30, 2.6), (33, 0), (47, 0), (51, 3.4), (57, 3.4), (61, 0), (88, 0)],
        "length": 154,
        "radius": 0.7,
        "challenge": "two pairs of nearly parallel low-amplitude crease lines",
    },
    {
        "id": "HF12",
        "name": "rounded_wide_pan",
        "profile": [(0, 24), (4, 10), (13, 2), (28, 0), (84, 0), (99, 2), (108, 10), (112, 24)],
        "length": 224,
        "radius": 9.0,
        "challenge": "wide pan with multi-facet rounded sides instead of a single sharp fold",
    },
    {
        "id": "HF13",
        "name": "miniature_joggle_train",
        "profile": [(0, 0), (13, 0), (16, 3), (21, 3), (24, 7), (30, 7), (33, 3), (39, 3), (42, 0), (58, 0), (64, 8), (86, 8)],
        "length": 168,
        "radius": 0.9,
        "challenge": "dense sequence of short 3 to 4 mm joggles followed by a larger offset",
    },
    {
        "id": "HF14",
        "name": "deep_hat_micro_flange",
        "profile": [(0, 0), (11, 0), (14, 2.5), (18, 2.5), (25, 48), (34, 58), (70, 58), (79, 48), (86, 2.5), (90, 2.5), (93, 0), (108, 0)],
        "length": 208,
        "radius": 2.5,
        "challenge": "deep hat with micro-flange joggles beside the main walls",
    },
    {
        "id": "HF15",
        "name": "asymmetric_multiangle_labyrinth",
        "profile": [(0, 0), (14, 0), (19, 4), (22, 14), (31, 22), (45, 22), (52, 31), (67, 35), (81, 35), (88, 27), (92, 12), (104, 5), (122, 5)],
        "length": 252,
        "radius": 2.0,
        "challenge": "twelve-vertex asymmetric compound profile with mixed bend magnitudes",
    },
)


def _write_case(
    *,
    output_root: Path,
    opaque_case_id: str,
    family: dict[str, Any],
    capture_index: int,
    width: int,
    height: int,
    view_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    public = output_root / "public" / opaque_case_id
    frames_dir = public / "frames"
    evaluator = output_root / "evaluator_only" / opaque_case_id
    frames_dir.mkdir(parents=True)
    evaluator.mkdir(parents=True)

    hidden_mesh, truth, base_metadata = build_case(family, capture_index)
    hidden_path = evaluator / "hidden_scene_mesh.ply"
    if not o3d.io.write_triangle_mesh(
        str(hidden_path),
        hidden_mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
    ):
        raise RuntimeError(f"failed to write {hidden_path}")
    vertices = np.asarray(hidden_mesh.vertices, dtype=np.float64)
    target = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    bbox_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if bbox_diagonal <= 0.0 or not np.isfinite(bbox_diagonal):
        raise ValueError("hidden scene has invalid scale")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hidden_mesh))
    intrinsics = capture._intrinsics(width, height)
    family_number = int(family["id"][-2:])
    rng = np.random.default_rng(170_000 + 211 * family_number + 19 * capture_index)
    frame_rows = []
    fusion_rows = []
    view_metadata = []
    for ordinal in range(view_count):
        pose = capture._camera_pose(target, bbox_diagonal, ordinal, view_count)
        depth, validity, incidence = capture._render_frame(
            scene,
            width=width,
            height=height,
            K=intrinsics,
            T=pose,
            rng=rng,
        )
        depth_path = frames_dir / f"f{ordinal:03d}_depth.npy"
        validity_path = frames_dir / f"f{ordinal:03d}_validity.npy"
        np.save(depth_path, depth, allow_pickle=False)
        np.save(validity_path, validity, allow_pickle=False)
        fusion_rows.append(capture._backproject(depth, validity, intrinsics, pose))
        frame_rows.append(
            {
                "ordinal": ordinal,
                "depth_file": depth_path.relative_to(public).as_posix(),
                "depth_sha256": capture._sha256(depth_path),
                "validity_file": validity_path.relative_to(public).as_posix(),
                "validity_sha256": capture._sha256(validity_path),
                "K": intrinsics.tolist(),
                "T_world_from_camera": pose.tolist(),
            }
        )
        valid_incidence = incidence[validity]
        view_metadata.append(
            {
                "ordinal": ordinal,
                "valid_pixel_count": int(validity.sum()),
                "valid_fraction": float(validity.mean()),
                "depth_min_mm": float(depth[validity].min()) if validity.any() else None,
                "depth_max_mm": float(depth[validity].max()) if validity.any() else None,
                "median_abs_incidence": float(np.median(valid_incidence)) if len(valid_incidence) else None,
            }
        )

    source_points, fusion_voxel = capture._fuse_points(fusion_rows)
    source_path = public / "source.ply"
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    if not o3d.io.write_point_cloud(str(source_path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write {source_path}")
    public_manifest = {
        "schema_version": 1,
        # Individual bundles keep the established capture schema identifier so
        # the frozen validator and calibrated route can consume them.  The top-
        # level public manifest carries the V3 corpus identity.
        "kind": capture.KIND,
        "opaque_case_id": opaque_case_id,
        "source_geometry": {"file": "source.ply", "sha256": capture._sha256(source_path)},
        "frames": frame_rows,
    }
    capture._atomic_json(public / "manifest.json", public_manifest)
    capture._atomic_json(
        evaluator / "truth.json",
        {
            "schema_version": 1,
            "token": TOKEN,
            "opaque_case_id": opaque_case_id,
            "manual_bends": truth,
        },
    )
    metadata = {
        "opaque_case_id": opaque_case_id,
        "family_id": family["id"],
        "family_name": family["name"],
        "challenge": family["challenge"],
        "nominal_radius_mm": family["radius"],
        "capture_index": capture_index,
        "hidden_scene_sha256": capture._sha256(hidden_path),
        "source_sha256": capture._sha256(source_path),
        "source_point_count": len(source_points),
        "fusion_voxel_mm": fusion_voxel,
        "bbox_diagonal_mm": bbox_diagonal,
        "views": view_metadata,
        "base_scene": base_metadata,
    }
    capture._atomic_json(evaluator / "generation_metadata.json", metadata)
    public_row = {
        "opaque_case_id": opaque_case_id,
        "bundle_path": str(public.resolve()),
        "manifest_sha256": capture._sha256(public / "manifest.json"),
        "source_sha256": capture._sha256(source_path),
        "source_point_count": len(source_points),
        "frame_count": view_count,
    }
    private_row = {
        "opaque_case_id": opaque_case_id,
        "family_id": family["id"],
        "capture_index": capture_index,
        "truth_sha256": capture._sha256(evaluator / "truth.json"),
        "generation_metadata_sha256": capture._sha256(evaluator / "generation_metadata.json"),
    }
    return public_row, private_row, metadata


def generate(
    output_root: Path,
    *,
    width: int = capture.DEFAULT_WIDTH,
    height: int = capture.DEFAULT_HEIGHT,
    view_count: int = capture.DEFAULT_VIEW_COUNT,
    family_limit: int | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if not 3 <= view_count <= 64:
        raise ValueError("view_count must be in 3..64")
    families = FAMILIES if family_limit is None else FAMILIES[:family_limit]
    if not families:
        raise ValueError("family_limit selected no families")
    public_rows = []
    private_rows = []
    metadata_rows = []
    for family_index, family in enumerate(families):
        for slot, capture_index in enumerate(CAPTURE_INDICES):
            case_id = f"O{family_index * 2 + slot + 1:03d}"
            public_row, private_row, metadata = _write_case(
                output_root=output_root,
                opaque_case_id=case_id,
                family=family,
                capture_index=capture_index,
                width=width,
                height=height,
                view_count=view_count,
            )
            public_rows.append(public_row)
            private_rows.append(private_row)
            metadata_rows.append(metadata)
    public_manifest = {
        "schema_version": 1,
        "kind": CORPUS_KIND,
        "token": TOKEN,
        "confirmatory_eligible": False,
        "geometry_stress": True,
        "case_count": len(public_rows),
        "family_count": len(families),
        "cases": public_rows,
    }
    capture._atomic_json(output_root / "public_manifest.json", public_manifest)
    capture._atomic_json(
        output_root / "evaluator_only" / "private_manifest.json",
        {
            "schema_version": 1,
            "token": TOKEN,
            "confirmatory_eligible": False,
            "geometry_stress": True,
            "cases": private_rows,
        },
    )
    capture._atomic_json(
        output_root / "evaluator_only" / "renderer_manifest.json",
        {
            "schema_version": 1,
            "token": TOKEN,
            "algorithm": "open3d_raycast_hidden_scene_to_organized_depth_then_fuse",
            "width": width,
            "height": height,
            "view_count": view_count,
            "horizontal_fov_degrees": capture.HORIZONTAL_FOV_DEGREES,
            "depth_noise_sigma_mm": capture.DEPTH_NOISE_SIGMA_MM,
            "random_dropout_probability": capture.RANDOM_DROPOUT_PROBABILITY,
            "initial_fusion_voxel_mm": capture.INITIAL_FUSION_VOXEL_MM,
            "max_source_points": capture.MAX_SOURCE_POINTS,
            "capture_indices": list(CAPTURE_INDICES),
            "generator_sha256": capture._sha256(Path(__file__).resolve()),
            "frozen_capture_generator_sha256": capture._sha256(Path(capture.__file__).resolve()),
        },
    )
    capture._atomic_json(
        output_root / "evaluator_only" / "all_generation_metadata.json",
        {"schema_version": 1, "token": TOKEN, "cases": metadata_rows},
    )
    return public_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=capture.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=capture.DEFAULT_HEIGHT)
    parser.add_argument("--view-count", type=int, default=capture.DEFAULT_VIEW_COUNT)
    parser.add_argument("--family-limit", type=int)
    args = parser.parse_args()
    report = generate(
        args.output_root.resolve(),
        width=args.width,
        height=args.height,
        view_count=args.view_count,
        family_limit=args.family_limit,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "case_count": report["case_count"],
                "family_count": report["family_count"],
                "confirmatory_eligible": False,
                "token": TOKEN,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
