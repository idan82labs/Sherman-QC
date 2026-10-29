from __future__ import annotations

import json

import numpy as np
import pytest

from research.f1_organized_frame_intake_v1 import OrganizedFrameIntakeError, validate_bundle


def _write_bundle(tmp_path):
    (tmp_path / "scan.ply").write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
    )
    calibration = {
        "schema": "F1_PINHOLE_DEPTH_CALIBRATION_V1",
        "width": 3, "height": 2, "fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 0.5,
        "depth_unit_mm": 1.0, "distortion_model": "NONE", "distortion_coefficients": [],
        "sensor_serial_sha256": "c" * 64,
    }
    (tmp_path / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    frames = []
    for index in range(2):
        np.save(tmp_path / f"depth_{index}.npy", np.full((2, 3), 100.0 + index, dtype=np.float32))
        np.save(tmp_path / f"validity_{index}.npy", np.ones((2, 3), dtype=np.uint8))
        frames.append({
            "depth_relative_path": f"depth_{index}.npy",
            "validity_relative_path": f"validity_{index}.npy",
            "timestamp_ns": index + 1,
            "world_from_sensor": [[1, 0, 0, index], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        })
    return {
        "schema": "F1_ORGANIZED_FRAME_BUNDLE_V1",
        "acquisition_protocol_sha256": "a" * 64,
        "cases": [{
            "case_token": "b" * 64,
            "fused_scan_relative_path": "scan.ply",
            "calibration_relative_path": "calibration.json",
            "frames": frames,
        }],
    }


def test_valid_bundle_preserves_organized_observation_receipt(tmp_path):
    result = validate_bundle(_write_bundle(tmp_path), intake_root=tmp_path)
    assert result["status"] == "ORGANIZED_OBSERVATIONS_VALIDATED"
    assert result["case_count"] == 1
    assert result["frame_count"] == 2
    assert result["cases"][0]["valid_sample_count"] == 12
    assert result["cases"][0]["minimum_valid_depth_mm"] == 100.0
    assert len(result["receipt_sha256"]) == 64


def test_rejects_truth_identity_and_cad_fields_recursively(tmp_path):
    for key in ("expected_count", "part_id", "cad_path"):
        payload = _write_bundle(tmp_path)
        payload["cases"][0][key] = "forbidden"
        with pytest.raises(OrganizedFrameIntakeError, match="forbidden"):
            validate_bundle(payload, intake_root=tmp_path)


def test_rejects_nonmonotonic_time_bad_pose_and_shape(tmp_path):
    payload = _write_bundle(tmp_path)
    payload["cases"][0]["frames"][1]["timestamp_ns"] = 1
    with pytest.raises(OrganizedFrameIntakeError, match="strictly increasing"):
        validate_bundle(payload, intake_root=tmp_path)
    payload = _write_bundle(tmp_path)
    payload["cases"][0]["frames"][0]["world_from_sensor"][0][0] = 2
    with pytest.raises(OrganizedFrameIntakeError, match="orthonormal"):
        validate_bundle(payload, intake_root=tmp_path)
    payload = _write_bundle(tmp_path)
    np.save(tmp_path / "depth_0.npy", np.ones((3, 2), dtype=np.float32))
    with pytest.raises(OrganizedFrameIntakeError, match="shape or dtype"):
        validate_bundle(payload, intake_root=tmp_path)


def test_rejects_invalid_depth_duplicate_content_and_path_escape(tmp_path):
    payload = _write_bundle(tmp_path)
    depth = np.full((2, 3), 100.0, dtype=np.float32)
    depth[0, 0] = 0.0
    np.save(tmp_path / "depth_0.npy", depth)
    with pytest.raises(OrganizedFrameIntakeError, match="positive and finite"):
        validate_bundle(payload, intake_root=tmp_path)
    payload = _write_bundle(tmp_path)
    np.save(tmp_path / "depth_1.npy", np.full((2, 3), 100.0, dtype=np.float32))
    with pytest.raises(OrganizedFrameIntakeError, match="duplicate depth"):
        validate_bundle(payload, intake_root=tmp_path)
    payload = _write_bundle(tmp_path)
    payload["cases"][0]["frames"][0]["depth_relative_path"] = "../escape.npy"
    with pytest.raises(OrganizedFrameIntakeError, match="escapes"):
        validate_bundle(payload, intake_root=tmp_path)


def test_rejects_nonbinary_validity_and_duplicate_fused_scans(tmp_path):
    payload = _write_bundle(tmp_path)
    validity = np.ones((2, 3), dtype=np.uint8)
    validity[0, 0] = 2
    np.save(tmp_path / "validity_0.npy", validity)
    with pytest.raises(OrganizedFrameIntakeError, match="binary"):
        validate_bundle(payload, intake_root=tmp_path)

    payload = _write_bundle(tmp_path)
    second = dict(payload["cases"][0])
    second["case_token"] = "d" * 64
    # Give the second case distinct frames so the fused-scan duplicate is the
    # first duplicate invariant reached.
    frames = []
    for index in range(2):
        np.save(tmp_path / f"depth_second_{index}.npy", np.full((2, 3), 200.0 + index, dtype=np.float32))
        np.save(tmp_path / f"validity_second_{index}.npy", np.ones((2, 3), dtype=np.uint8))
        frames.append({
            "depth_relative_path": f"depth_second_{index}.npy",
            "validity_relative_path": f"validity_second_{index}.npy",
            "timestamp_ns": index + 11,
            "world_from_sensor": [[1, 0, 0, index], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        })
    second["frames"] = frames
    payload["cases"].append(second)
    with pytest.raises(OrganizedFrameIntakeError, match="duplicate fused-scan"):
        validate_bundle(payload, intake_root=tmp_path)
