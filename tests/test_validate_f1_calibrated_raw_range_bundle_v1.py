import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from scripts.validate_f1_calibrated_raw_range_bundle_v1 import (
    BundleValidationError,
    KIND,
    validate_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(root: Path) -> Path:
    root.mkdir()
    frames_dir = root / "frames"
    frames_dir.mkdir()
    height = width = 12
    K = np.asarray([[100.0, 0.0, 5.5], [0.0, 100.0, 5.5], [0.0, 0.0, 1.0]])
    depth = np.full((height, width), 100.0, dtype=np.float64)
    validity = np.ones((height, width), dtype=np.bool_)
    source_rows = []
    frame_rows = []
    for ordinal, translation in enumerate((0.0, 2.0, 4.0)):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = translation
        depth_path = frames_dir / f"f{ordinal:03d}_depth.npy"
        validity_path = frames_dir / f"f{ordinal:03d}_validity.npy"
        np.save(depth_path, depth, allow_pickle=False)
        np.save(validity_path, validity, allow_pickle=False)
        v, u = np.indices(depth.shape)
        pixels = np.column_stack((u.ravel(), v.ravel(), np.ones(depth.size)))
        camera = (np.linalg.inv(K) @ pixels.T).T * depth.ravel()[:, None]
        world = camera + np.asarray([translation, 0.0, 0.0])
        source_rows.append(world)
        frame_rows.append({
            "ordinal": ordinal,
            "depth_file": depth_path.relative_to(root).as_posix(),
            "depth_sha256": _sha256(depth_path),
            "validity_file": validity_path.relative_to(root).as_posix(),
            "validity_sha256": _sha256(validity_path),
            "K": K.tolist(),
            "T_world_from_camera": T.tolist(),
        })
    source_path = root / "source.ply"
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(source_rows)))
    assert o3d.io.write_point_cloud(str(source_path), cloud, write_ascii=False, compressed=False)
    manifest = {
        "schema_version": 1,
        "kind": KIND,
        "opaque_case_id": "O001",
        "source_geometry": {"file": "source.ply", "sha256": _sha256(source_path)},
        "frames": frame_rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root


def test_valid_bundle_passes_without_detector_or_truth(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "bundle")
    report = validate_bundle(bundle)
    assert report["status"] == "PASS"
    assert report["frame_count"] == 3
    assert report["semantic_frame_hashes_unique"] is True
    assert report["coordinate_agreement"]["median_over_B"] < 1e-12
    assert report["detector_invocation_count"] == 0
    assert report["truth_opened"] is False


def test_nonrigid_transform_fails_closed(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["frames"][1]["T_world_from_camera"][0][0] = 1.01
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="not rigid"):
        validate_bundle(bundle)


def test_invalid_depth_must_be_canonical_zero(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "bundle")
    depth_path = bundle / "frames/f000_depth.npy"
    validity_path = bundle / "frames/f000_validity.npy"
    depth = np.load(depth_path, allow_pickle=False)
    validity = np.load(validity_path, allow_pickle=False)
    validity[0, 0] = False
    depth[0, 0] = np.nan
    np.save(depth_path, depth, allow_pickle=False)
    np.save(validity_path, validity, allow_pickle=False)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["depth_sha256"] = _sha256(depth_path)
    manifest["frames"][0]["validity_sha256"] = _sha256(validity_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="canonical zero"):
        validate_bundle(bundle)


def test_duplicate_semantic_frames_fail_closed(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("depth_file", "depth_sha256", "validity_file", "validity_sha256", "K", "T_world_from_camera"):
        manifest["frames"][1][key] = manifest["frames"][0][key]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="duplicate semantic"):
        validate_bundle(bundle)


def test_extra_identity_key_fails_exact_schema(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["part_number"] = "FORBIDDEN"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="extra=.*part_number"):
        validate_bundle(bundle)


def test_member_path_escape_fails_closed(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["depth_file"] = "../outside.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="inside the bundle"):
        validate_bundle(bundle)


def test_coordinate_disagreement_fails_closed(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        frame["T_world_from_camera"][2][3] = 1000.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="coordinate agreement failed"):
        validate_bundle(bundle)
