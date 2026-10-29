#!/usr/bin/env python3
"""Generate a prospective calibrated raw-range synthetic F1 corpus.

Unlike the older mesh-first stress corpus, each public source point cloud here
is fused from simulator-originated organized depth frames.  The hidden scene
mesh and truth are written only to the evaluator area.  This lane is synthetic,
non-confirmatory, and may not be represented as recovered sensor provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_f1_lidar_like_synthetic_corpus import FAMILIES, build_case


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY"
KIND = "sherman_qc_calibrated_raw_range_capture_bundle_v1"
DEFAULT_WIDTH = 384
DEFAULT_HEIGHT = 288
DEFAULT_VIEW_COUNT = 8
HORIZONTAL_FOV_DEGREES = 50.0
DEPTH_NOISE_SIGMA_MM = 0.075
RANDOM_DROPOUT_PROBABILITY = 0.0025
# Frozen from the pilot-only density calibration: 0.62 mm produced a median
# nearest-neighbor spacing of 0.451 mm versus 0.463 mm across the ten physical
# LiDAR references.  The pilot was not scored and no bend output informed it.
INITIAL_FUSION_VOXEL_MM = 0.62
MAX_SOURCE_POINTS = 195_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("cannot normalize camera vector")
    return vector / norm


def _camera_pose(target: np.ndarray, B: float, view_index: int, view_count: int) -> np.ndarray:
    # Deterministic golden-angle coverage with alternating elevations.  The
    # first camera is not privileged by family identity or output feedback.
    golden = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = view_index * golden
    elevation_pattern = (-0.48, 0.34, 0.10, -0.22, 0.52, -0.36, 0.24, 0.0)
    elevation = elevation_pattern[view_index % len(elevation_pattern)]
    direction = np.asarray([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ])
    center = target + direction * (1.20 * B)
    forward = _unit(target - center)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up))) > 0.90:
        up = np.asarray([0.0, 1.0, 0.0])
    right = _unit(np.cross(forward, up))
    down = _unit(np.cross(forward, right))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack((right, down, forward))
    T[:3, 3] = center
    return T


def _intrinsics(width: int, height: int) -> np.ndarray:
    fx = 0.5 * width / math.tan(math.radians(HORIZONTAL_FOV_DEGREES) / 2.0)
    fy = fx
    return np.asarray([
        [fx, 0.0, (width - 1.0) / 2.0],
        [0.0, fy, (height - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _rays(width: int, height: int, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.indices((height, width), dtype=np.float64)
    camera = np.stack(
        ((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)),
        axis=-1,
    )
    world = camera @ T[:3, :3].T
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    origins = np.broadcast_to(T[:3, 3], (height, width, 3))
    return np.concatenate((origins, world), axis=-1).astype(np.float32), world


def _render_frame(
    scene: o3d.t.geometry.RaycastingScene,
    *,
    width: int,
    height: int,
    K: np.ndarray,
    T: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays, world_directions = _rays(width, height, K, T)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = result["t_hit"].numpy().astype(np.float64)
    validity = np.isfinite(t_hit)
    hit_world = T[:3, 3] + world_directions * t_hit[..., None]
    forward = T[:3, 2]
    depth = np.einsum("ijk,k->ij", hit_world - T[:3, 3], forward)
    validity &= np.isfinite(depth) & (depth > 0.0)

    # Incidence-sensitive loss approximates grazing-angle scanner dropout while
    # remaining frozen and independent of family identity or truth.
    primitive_normals = result["primitive_normals"].numpy().astype(np.float64)
    incidence = np.abs(np.einsum("ijk,ijk->ij", primitive_normals, -world_directions))
    grazing_loss = np.clip((0.18 - incidence) / 0.18, 0.0, 1.0) * 0.70
    validity &= rng.random((height, width)) >= (RANDOM_DROPOUT_PROBABILITY + grazing_loss)

    noisy = np.zeros((height, width), dtype=np.float64)
    noisy[validity] = depth[validity] + rng.normal(0.0, DEPTH_NOISE_SIGMA_MM, int(validity.sum()))
    validity &= noisy > 0.0
    noisy[~validity] = 0.0
    return noisy, validity.astype(np.bool_), incidence


def _backproject(depth: np.ndarray, validity: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    flat = np.flatnonzero(validity.ravel())
    v = flat // depth.shape[1]
    u = flat % depth.shape[1]
    pixels = np.column_stack((u, v, np.ones(len(flat), dtype=np.float64)))
    camera = (np.linalg.inv(K) @ pixels.T).T * depth.ravel()[flat, None]
    return camera @ T[:3, :3].T + T[:3, 3]


def _fuse_points(frame_points: Iterable[np.ndarray]) -> tuple[np.ndarray, float]:
    rows = [row for row in frame_points if len(row)]
    if not rows:
        raise ValueError("all simulated views are empty")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(rows)))
    voxel = INITIAL_FUSION_VOXEL_MM
    fused = cloud.voxel_down_sample(voxel)
    while len(fused.points) > MAX_SOURCE_POINTS:
        voxel *= 1.08
        fused = cloud.voxel_down_sample(voxel)
    points = np.asarray(fused.points, dtype=np.float64)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return np.ascontiguousarray(points[order]), voxel


def _write_case(
    *,
    output_root: Path,
    opaque_case_id: str,
    family: dict[str, Any],
    capture: int,
    width: int,
    height: int,
    view_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    public = output_root / "public" / opaque_case_id
    frames_dir = public / "frames"
    evaluator = output_root / "evaluator_only" / opaque_case_id
    frames_dir.mkdir(parents=True)
    evaluator.mkdir(parents=True)

    hidden_mesh, truth, base_metadata = build_case(family, capture)
    hidden_path = evaluator / "hidden_scene_mesh.ply"
    if not o3d.io.write_triangle_mesh(
        str(hidden_path), hidden_mesh, write_ascii=False, compressed=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"failed to write {hidden_path}")
    vertices = np.asarray(hidden_mesh.vertices, dtype=np.float64)
    target = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    B = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if B <= 0.0 or not np.isfinite(B):
        raise ValueError("hidden scene has invalid scale")

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(hidden_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    K = _intrinsics(width, height)
    rng = np.random.default_rng(70_000 + 131 * int(family["id"][-2:]) + 17 * capture)
    frame_rows = []
    fusion_rows = []
    view_metadata = []
    for ordinal in range(view_count):
        T = _camera_pose(target, B, ordinal, view_count)
        depth, validity, incidence = _render_frame(
            scene, width=width, height=height, K=K, T=T, rng=rng
        )
        depth_path = frames_dir / f"f{ordinal:03d}_depth.npy"
        validity_path = frames_dir / f"f{ordinal:03d}_validity.npy"
        np.save(depth_path, depth, allow_pickle=False)
        np.save(validity_path, validity, allow_pickle=False)
        fusion_rows.append(_backproject(depth, validity, K, T))
        frame_rows.append({
            "ordinal": ordinal,
            "depth_file": depth_path.relative_to(public).as_posix(),
            "depth_sha256": _sha256(depth_path),
            "validity_file": validity_path.relative_to(public).as_posix(),
            "validity_sha256": _sha256(validity_path),
            "K": K.tolist(),
            "T_world_from_camera": T.tolist(),
        })
        valid_incidence = incidence[validity]
        view_metadata.append({
            "ordinal": ordinal,
            "valid_pixel_count": int(validity.sum()),
            "valid_fraction": float(validity.mean()),
            "depth_min_mm": float(depth[validity].min()) if validity.any() else None,
            "depth_max_mm": float(depth[validity].max()) if validity.any() else None,
            "median_abs_incidence": float(np.median(valid_incidence)) if len(valid_incidence) else None,
        })

    source_points, fusion_voxel = _fuse_points(fusion_rows)
    source_path = public / "source.ply"
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    if not o3d.io.write_point_cloud(str(source_path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write {source_path}")
    manifest = {
        "schema_version": 1,
        "kind": KIND,
        "opaque_case_id": opaque_case_id,
        "source_geometry": {"file": "source.ply", "sha256": _sha256(source_path)},
        "frames": frame_rows,
    }
    _atomic_json(public / "manifest.json", manifest)
    truth_payload = {
        "schema_version": 1,
        "token": TOKEN,
        "opaque_case_id": opaque_case_id,
        "manual_bends": truth,
    }
    _atomic_json(evaluator / "truth.json", truth_payload)
    metadata = {
        "opaque_case_id": opaque_case_id,
        "family_id": family["id"],
        "family_name": family["name"],
        "capture": capture + 1,
        "hidden_scene_sha256": _sha256(hidden_path),
        "source_sha256": _sha256(source_path),
        "source_point_count": len(source_points),
        "fusion_voxel_mm": fusion_voxel,
        "bbox_diagonal_mm": B,
        "views": view_metadata,
        "base_scene": base_metadata,
    }
    _atomic_json(evaluator / "generation_metadata.json", metadata)
    public_row = {
        "opaque_case_id": opaque_case_id,
        "bundle_path": str(public.resolve()),
        "manifest_sha256": _sha256(public / "manifest.json"),
        "source_sha256": _sha256(source_path),
        "source_point_count": len(source_points),
        "frame_count": view_count,
    }
    private_row = {
        "opaque_case_id": opaque_case_id,
        "family_id": family["id"],
        "capture": capture + 1,
        "truth_sha256": _sha256(evaluator / "truth.json"),
        "generation_metadata_sha256": _sha256(evaluator / "generation_metadata.json"),
    }
    return public_row, private_row, metadata


def generate(
    output_root: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    view_count: int = DEFAULT_VIEW_COUNT,
    family_limit: int | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if not 3 <= view_count <= 64:
        raise ValueError("view_count must be in 3..64")
    if width < 1 or height < 1 or max(width, height) > 8192:
        raise ValueError("invalid image dimensions")
    families = FAMILIES if family_limit is None else FAMILIES[:family_limit]
    if not families:
        raise ValueError("family_limit selected no families")
    public_rows = []
    private_rows = []
    metadata_rows = []
    for family_index, family in enumerate(families):
        for capture in range(2):
            opaque_case_id = f"O{family_index * 2 + capture + 1:03d}"
            public_row, private_row, metadata = _write_case(
                output_root=output_root,
                opaque_case_id=opaque_case_id,
                family=family,
                capture=capture,
                width=width,
                height=height,
                view_count=view_count,
            )
            public_rows.append(public_row)
            private_rows.append(private_row)
            metadata_rows.append(metadata)
    public_manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_calibrated_raw_range_synthetic_corpus_v1",
        "token": TOKEN,
        "confirmatory_eligible": False,
        "case_count": len(public_rows),
        "family_count": len(families),
        "cases": public_rows,
    }
    private_manifest = {
        "schema_version": 1,
        "token": TOKEN,
        "confirmatory_eligible": False,
        "cases": private_rows,
    }
    renderer_manifest = {
        "schema_version": 1,
        "token": TOKEN,
        "algorithm": "open3d_raycast_hidden_scene_to_organized_depth_then_fuse",
        "width": width,
        "height": height,
        "view_count": view_count,
        "horizontal_fov_degrees": HORIZONTAL_FOV_DEGREES,
        "depth_noise_sigma_mm": DEPTH_NOISE_SIGMA_MM,
        "random_dropout_probability": RANDOM_DROPOUT_PROBABILITY,
        "initial_fusion_voxel_mm": INITIAL_FUSION_VOXEL_MM,
        "max_source_points": MAX_SOURCE_POINTS,
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "base_scene_generator_sha256": _sha256(
            Path(__file__).with_name("generate_f1_lidar_like_synthetic_corpus.py").resolve()
        ),
    }
    _atomic_json(output_root / "public_manifest.json", public_manifest)
    _atomic_json(output_root / "evaluator_only" / "private_manifest.json", private_manifest)
    _atomic_json(output_root / "evaluator_only" / "renderer_manifest.json", renderer_manifest)
    _atomic_json(
        output_root / "evaluator_only" / "all_generation_metadata.json",
        {"schema_version": 1, "token": TOKEN, "cases": metadata_rows},
    )
    return public_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--view-count", type=int, default=DEFAULT_VIEW_COUNT)
    parser.add_argument("--family-limit", type=int)
    args = parser.parse_args()
    report = generate(
        args.output_root.resolve(),
        width=args.width,
        height=args.height,
        view_count=args.view_count,
        family_limit=args.family_limit,
    )
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "case_count": report["case_count"],
        "family_count": report["family_count"],
        "confirmatory_eligible": False,
        "token": TOKEN,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
