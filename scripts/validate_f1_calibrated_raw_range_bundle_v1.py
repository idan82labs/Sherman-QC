#!/usr/bin/env python3
"""Validate a prospective calibrated raw-range capture bundle.

This is a custody/schema preflight, not a bend detector.  It emits only
dimensions, counts, hashes, finite-value checks, calibration residuals, and
cap margins.  It never reads truth, CAD, labels, expected counts, identities,
or any V53-V56 artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


SCHEMA_VERSION = 1
KIND = "sherman_qc_calibrated_raw_range_capture_bundle_v1"
MIN_FRAMES = 3
MAX_FRAMES = 64
MAX_DIMENSION = 8192
MAX_RAW_PIXELS = 100_000_000
MAX_SOURCE_POINTS = 200_000
SAMPLE_PIXELS_PER_VIEW = 4_096

TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "opaque_case_id",
    "source_geometry",
    "frames",
}
SOURCE_KEYS = {"file", "sha256"}
FRAME_KEYS = {
    "ordinal",
    "depth_file",
    "depth_sha256",
    "validity_file",
    "validity_sha256",
    "K",
    "T_world_from_camera",
}


class BundleValidationError(ValueError):
    """Raised when a capture bundle fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise BundleValidationError(
            f"{where} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _resolve_member(root: Path, relative: Any, expected_name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BundleValidationError(f"{expected_name} must be a non-empty relative filename")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BundleValidationError(f"{expected_name} must stay inside the bundle")
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise BundleValidationError(f"{expected_name} may not be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise BundleValidationError(f"{expected_name} escapes the bundle") from error
    if not resolved.is_file():
        raise BundleValidationError(f"{expected_name} is not a regular bundle file")
    return resolved


def _require_hash(path: Path, expected: Any, where: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise BundleValidationError(f"{where} sha256 is malformed")
    actual = _sha256(path)
    if actual != expected.lower():
        raise BundleValidationError(f"{where} sha256 mismatch")
    return actual


def _load_source_points(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=False, remove_infinite_points=False)
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        points = np.asarray(mesh.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise BundleValidationError("source geometry has no 3-D points")
    if len(points) > MAX_SOURCE_POINTS:
        raise BundleValidationError(f"source point cap exceeded: {len(points)} > {MAX_SOURCE_POINTS}")
    if not np.isfinite(points).all():
        raise BundleValidationError("source geometry contains nonfinite points")
    return np.ascontiguousarray(points, dtype=np.float64)


def _matrix(value: Any, shape: tuple[int, int], where: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise BundleValidationError(f"{where} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def _load_frame(root: Path, row: dict[str, Any], expected_ordinal: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise BundleValidationError(f"frame {expected_ordinal} is not an object")
    _require_exact_keys(row, FRAME_KEYS, f"frame {expected_ordinal}")
    if row["ordinal"] != expected_ordinal:
        raise BundleValidationError("frame ordinals must be contiguous and zero-based")

    depth_path = _resolve_member(root, row["depth_file"], "depth_file")
    validity_path = _resolve_member(root, row["validity_file"], "validity_file")
    if depth_path.suffix.lower() != ".npy" or validity_path.suffix.lower() != ".npy":
        raise BundleValidationError("normalized depth and validity arrays must use .npy")
    depth_file_hash = _require_hash(depth_path, row["depth_sha256"], "depth")
    validity_file_hash = _require_hash(validity_path, row["validity_sha256"], "validity")

    depth = np.load(depth_path, allow_pickle=False)
    validity_raw = np.load(validity_path, allow_pickle=False)
    if depth.ndim != 2 or validity_raw.shape != depth.shape:
        raise BundleValidationError("depth and validity must be same-shaped organized 2-D arrays")
    height, width = map(int, depth.shape)
    if height < 1 or width < 1 or max(height, width) > MAX_DIMENSION:
        raise BundleValidationError("frame dimensions are outside the 1..8192 bound")
    if validity_raw.dtype != np.bool_:
        if not np.issubdtype(validity_raw.dtype, np.integer) or not np.isin(validity_raw, (0, 1)).all():
            raise BundleValidationError("validity must be bool or a binary integer array")
    validity = np.asarray(validity_raw, dtype=np.bool_)
    depth = np.asarray(depth, dtype=np.float64)
    if not np.isfinite(depth[validity]).all() or not np.all(depth[validity] > 0.0):
        raise BundleValidationError("every valid depth must be finite and positive")
    if not np.all(depth[~validity] == 0.0):
        raise BundleValidationError("every invalid depth must be canonical zero")

    K = _matrix(row["K"], (3, 3), "K")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0 or abs(np.linalg.det(K)) <= 1e-15:
        raise BundleValidationError("K must be invertible with positive focal lengths")
    T = _matrix(row["T_world_from_camera"], (4, 4), "T_world_from_camera")
    if not np.array_equal(T[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise BundleValidationError("T bottom row must be exactly [0,0,0,1]")
    R = T[:3, :3]
    orthogonality_residual = float(np.max(np.abs(R.T @ R - np.eye(3))))
    determinant_residual = float(abs(np.linalg.det(R) - 1.0))
    if orthogonality_residual > 1e-9 or determinant_residual > 1e-9:
        raise BundleValidationError("T rotation is not rigid to 1e-9")

    semantic_hash = _canonical_hash(
        np.asarray([height, width], dtype=">u8").tobytes(),
        np.ascontiguousarray(depth, dtype=">f8").tobytes(),
        np.ascontiguousarray(validity, dtype=np.uint8).tobytes(),
        np.ascontiguousarray(K, dtype=">f8").tobytes(),
        np.ascontiguousarray(T, dtype=">f8").tobytes(),
    )
    return {
        "ordinal": expected_ordinal,
        "height": height,
        "width": width,
        "pixel_count": height * width,
        "valid_count": int(np.count_nonzero(validity)),
        "depth": depth,
        "validity": validity,
        "K": K,
        "T": T,
        "semantic_hash": semantic_hash,
        "depth_file_sha256": depth_file_hash,
        "validity_file_sha256": validity_file_hash,
        "orthogonality_residual": orthogonality_residual,
        "determinant_residual": determinant_residual,
    }


def _sample_valid_pixels(frame: dict[str, Any]) -> np.ndarray:
    indices = np.flatnonzero(frame["validity"].ravel())
    if len(indices) == 0:
        raise BundleValidationError("a frame has no valid pixels")
    count = min(SAMPLE_PIXELS_PER_VIEW, len(indices))
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    offset = int(frame["semantic_hash"][:16], 16) % len(indices)
    return indices[(positions + offset) % len(indices)]


def _backproject(frame: dict[str, Any], flat_indices: np.ndarray) -> np.ndarray:
    height, width = frame["height"], frame["width"]
    del height
    v = flat_indices // width
    u = flat_indices % width
    depth = frame["depth"].ravel()[flat_indices]
    pixels = np.column_stack((u, v, np.ones(len(u), dtype=np.float64)))
    camera = (np.linalg.inv(frame["K"]) @ pixels.T).T * depth[:, None]
    homogeneous = np.column_stack((camera, np.ones(len(camera), dtype=np.float64)))
    return (frame["T"] @ homogeneous.T).T[:, :3]


def validate_bundle(bundle_root: Path) -> dict[str, Any]:
    root = bundle_root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise BundleValidationError("manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise BundleValidationError("manifest must be an object")
    _require_exact_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["kind"] != KIND:
        raise BundleValidationError("manifest schema or kind is unsupported")
    opaque_case_id = manifest["opaque_case_id"]
    if not isinstance(opaque_case_id, str) or not opaque_case_id.startswith("O") or not opaque_case_id[1:].isdigit():
        raise BundleValidationError("opaque_case_id must match O followed by digits")

    source_row = manifest["source_geometry"]
    if not isinstance(source_row, dict):
        raise BundleValidationError("source_geometry must be an object")
    _require_exact_keys(source_row, SOURCE_KEYS, "source_geometry")
    source_path = _resolve_member(root, source_row["file"], "source_geometry.file")
    source_hash = _require_hash(source_path, source_row["sha256"], "source geometry")
    source_points = _load_source_points(source_path)
    extent = np.ptp(source_points, axis=0)
    B = float(np.linalg.norm(extent))
    if not np.isfinite(B) or B <= 0.0:
        raise BundleValidationError("source bounding-box diagonal is not finite and positive")

    frame_rows = manifest["frames"]
    if not isinstance(frame_rows, list) or not MIN_FRAMES <= len(frame_rows) <= MAX_FRAMES:
        raise BundleValidationError("frame count must be in 3..64")
    frames = [_load_frame(root, row, ordinal) for ordinal, row in enumerate(frame_rows)]
    semantic_hashes = [frame["semantic_hash"] for frame in frames]
    if len(set(semantic_hashes)) != len(semantic_hashes):
        raise BundleValidationError("duplicate semantic frame hashes are forbidden")
    raw_pixels = sum(frame["pixel_count"] for frame in frames)
    if raw_pixels > MAX_RAW_PIXELS:
        raise BundleValidationError("raw pixel cap exceeded")

    tree = cKDTree(source_points)
    all_distances: list[np.ndarray] = []
    per_frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        sample_indices = _sample_valid_pixels(frame)
        world_points = _backproject(frame, sample_indices)
        distances, _ = tree.query(world_points, k=1, workers=1)
        distances = np.asarray(distances, dtype=np.float64)
        all_distances.append(distances)
        per_frame_rows.append({
            "ordinal": frame["ordinal"],
            "height": frame["height"],
            "width": frame["width"],
            "pixel_count": frame["pixel_count"],
            "valid_count": frame["valid_count"],
            "sample_count": int(len(distances)),
            "semantic_hash": frame["semantic_hash"],
            "depth_file_sha256": frame["depth_file_sha256"],
            "validity_file_sha256": frame["validity_file_sha256"],
            "rotation_orthogonality_residual": frame["orthogonality_residual"],
            "rotation_determinant_residual": frame["determinant_residual"],
        })
    residuals = np.concatenate(all_distances)
    median_over_B = float(np.median(residuals) / B)
    p95_over_B = float(np.percentile(residuals, 95) / B)
    fraction_le_002B = float(np.mean(residuals <= 0.020 * B))
    if median_over_B > 0.003 or p95_over_B > 0.015 or fraction_le_002B < 0.90:
        raise BundleValidationError(
            "source/frame/calibration coordinate agreement failed: "
            f"median/B={median_over_B:.9g}, p95/B={p95_over_B:.9g}, fraction<=.020B={fraction_le_002B:.9g}"
        )

    return {
        "schema_version": 1,
        "kind": "sherman_qc_calibrated_raw_range_capture_preflight_v1",
        "status": "PASS",
        "opaque_case_id": opaque_case_id,
        "manifest_sha256": _sha256(manifest_path),
        "source_geometry_sha256": source_hash,
        "source_point_count": int(len(source_points)),
        "source_point_cap_margin": MAX_SOURCE_POINTS - int(len(source_points)),
        "bbox_diagonal": B,
        "frame_count": len(frames),
        "frame_cap_margin": MAX_FRAMES - len(frames),
        "raw_pixel_count": raw_pixels,
        "raw_pixel_cap_margin": MAX_RAW_PIXELS - raw_pixels,
        "all_valid_depths_finite_positive": True,
        "all_invalid_depths_canonical_zero": True,
        "all_intrinsics_invertible_positive_focal": True,
        "all_transforms_rigid": True,
        "semantic_frame_hashes_unique": True,
        "coordinate_agreement": {
            "sample_count": int(len(residuals)),
            "median_over_B": median_over_B,
            "p95_over_B": p95_over_B,
            "fraction_le_002B": fraction_le_002B,
            "median_gate_over_B": 0.003,
            "p95_gate_over_B": 0.015,
            "fraction_gate_le_002B": 0.90,
        },
        "frames": per_frame_rows,
        "detector_invocation_count": 0,
        "truth_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_bundle(args.bundle)
    except (BundleValidationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, sort_keys=True))
        return 2
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        _atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
