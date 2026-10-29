"""Fail-closed intake for authentic organized depth/range observations.

This is acquisition infrastructure, not a bend detector.  It preserves the
pixel/ray neighbourhood, occlusion and validity information that is destroyed
when scanner observations are fused into an unordered PLY.  Labels, expected
counts, CAD and identity are forbidden from the model-lane receipt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.f1_physical_intake_v2 import _ply_header


SCHEMA = "F1_ORGANIZED_FRAME_BUNDLE_V1"
CALIBRATION_SCHEMA = "F1_PINHOLE_DEPTH_CALIBRATION_V1"
HEX = frozenset("0123456789abcdef")
FORBIDDEN_KEYS = frozenset({
    "gold", "label", "labels", "expected_count", "bend_count", "truth",
    "prediction", "quality_class", "part_id", "part_identity", "filename",
    "cad", "cad_path", "drawing", "good_bad", "defect_class",
})
CASE_FIELDS = frozenset({"case_token", "fused_scan_relative_path", "calibration_relative_path", "frames"})
FRAME_FIELDS = frozenset({"depth_relative_path", "validity_relative_path", "timestamp_ns", "world_from_sensor"})
CALIBRATION_FIELDS = frozenset({
    "schema", "width", "height", "fx", "fy", "cx", "cy", "depth_unit_mm",
    "distortion_model", "distortion_coefficients", "sensor_serial_sha256",
})


class OrganizedFrameIntakeError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _hex64(value: Any, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise OrganizedFrameIntakeError(f"{path} must be lowercase hex64")
    return value


def _safe_path(root: Path, relative: Any, path: str, suffix: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise OrganizedFrameIntakeError(f"{path} must be a non-empty relative path")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise OrganizedFrameIntakeError(f"{path} escapes intake root") from exc
    if source.suffix.lower() != suffix or not source.is_file():
        raise OrganizedFrameIntakeError(f"{path} must reference an existing {suffix} file")
    return source


def _reject_forbidden(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_KEYS:
                raise OrganizedFrameIntakeError(f"{path}.{key} is forbidden")
            _reject_forbidden(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden(item, f"{path}[{index}]")


def _positive_finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrganizedFrameIntakeError(f"{path} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise OrganizedFrameIntakeError(f"{path} must be positive and finite")
    return result


def _calibration(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise OrganizedFrameIntakeError("calibration must be valid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != CALIBRATION_FIELDS:
        raise OrganizedFrameIntakeError("calibration contains missing or forbidden fields")
    if value.get("schema") != CALIBRATION_SCHEMA:
        raise OrganizedFrameIntakeError(f"calibration schema must be {CALIBRATION_SCHEMA}")
    width, height = value.get("width"), value.get("height")
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 16384:
        raise OrganizedFrameIntakeError("calibration.width outside bounded range")
    if isinstance(height, bool) or not isinstance(height, int) or not 1 <= height <= 16384:
        raise OrganizedFrameIntakeError("calibration.height outside bounded range")
    for name in ("fx", "fy"):
        _positive_finite(value.get(name), f"calibration.{name}")
    for name in ("cx", "cy"):
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(float(number)):
            raise OrganizedFrameIntakeError(f"calibration.{name} must be finite")
    _positive_finite(value.get("depth_unit_mm"), "calibration.depth_unit_mm")
    if value.get("distortion_model") not in {"NONE", "BROWN_CONRADY"}:
        raise OrganizedFrameIntakeError("unsupported calibration.distortion_model")
    coefficients = value.get("distortion_coefficients")
    if not isinstance(coefficients, list) or len(coefficients) not in {0, 5}:
        raise OrganizedFrameIntakeError("distortion_coefficients must contain zero or five values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not np.isfinite(float(item)) for item in coefficients):
        raise OrganizedFrameIntakeError("distortion_coefficients must be finite")
    if value.get("distortion_model") == "NONE" and coefficients:
        raise OrganizedFrameIntakeError("NONE distortion requires zero coefficients")
    if value.get("distortion_model") == "BROWN_CONRADY" and len(coefficients) != 5:
        raise OrganizedFrameIntakeError("BROWN_CONRADY requires five coefficients")
    _hex64(value.get("sensor_serial_sha256"), "calibration.sensor_serial_sha256")
    canonical = _canonical_bytes(value)
    return dict(value), hashlib.sha256(canonical).hexdigest()


def _pose(value: Any, path: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise OrganizedFrameIntakeError(f"{path} must be a finite 4x4 matrix")
    if not np.allclose(array[3], np.asarray((0.0, 0.0, 0.0, 1.0)), atol=1e-9, rtol=0.0):
        raise OrganizedFrameIntakeError(f"{path} must be homogeneous")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise OrganizedFrameIntakeError(f"{path} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
        raise OrganizedFrameIntakeError(f"{path} rotation must be right-handed")
    return array


def validate_bundle(payload: Any, *, intake_root: Path) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise OrganizedFrameIntakeError(f"schema must be {SCHEMA}")
    _reject_forbidden(payload)
    if set(payload) != {"schema", "acquisition_protocol_sha256", "cases"}:
        raise OrganizedFrameIntakeError("bundle contains missing or forbidden top-level fields")
    protocol = _hex64(payload.get("acquisition_protocol_sha256"), "acquisition_protocol_sha256")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise OrganizedFrameIntakeError("cases must be a non-empty list")
    root = intake_root.resolve()
    receipts: list[dict[str, Any]] = []
    seen_case_tokens: set[str] = set()
    seen_scan_hashes: set[str] = set()
    seen_frame_hashes: set[str] = set()
    for case_index, row in enumerate(cases):
        if not isinstance(row, Mapping) or set(row) != CASE_FIELDS:
            raise OrganizedFrameIntakeError(f"cases[{case_index}] contains missing or forbidden fields")
        token = _hex64(row.get("case_token"), f"cases[{case_index}].case_token")
        if token in seen_case_tokens:
            raise OrganizedFrameIntakeError("case_token must be unique")
        seen_case_tokens.add(token)
        scan = _safe_path(root, row.get("fused_scan_relative_path"), f"cases[{case_index}].fused_scan_relative_path", ".ply")
        scan_header = _ply_header(scan)
        scan_sha = hashlib.sha256(scan.read_bytes()).hexdigest()
        if scan_sha in seen_scan_hashes:
            raise OrganizedFrameIntakeError("duplicate fused-scan content")
        seen_scan_hashes.add(scan_sha)
        calibration_path = _safe_path(root, row.get("calibration_relative_path"), f"cases[{case_index}].calibration_relative_path", ".json")
        calibration, calibration_sha = _calibration(calibration_path)
        frames = row.get("frames")
        if not isinstance(frames, list) or len(frames) < 2 or len(frames) > 4096:
            raise OrganizedFrameIntakeError(f"cases[{case_index}].frames must contain 2..4096 frames")
        timestamps: list[int] = []
        valid_total = 0
        valid_min = None
        valid_max = None
        frame_receipts = []
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping) or set(frame) != FRAME_FIELDS:
                raise OrganizedFrameIntakeError(f"cases[{case_index}].frames[{frame_index}] contains missing or forbidden fields")
            timestamp = frame.get("timestamp_ns")
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                raise OrganizedFrameIntakeError("timestamp_ns must be a nonnegative integer")
            if timestamps and timestamp <= timestamps[-1]:
                raise OrganizedFrameIntakeError("frame timestamps must be strictly increasing")
            timestamps.append(timestamp)
            pose = _pose(frame.get("world_from_sensor"), f"cases[{case_index}].frames[{frame_index}].world_from_sensor")
            depth_path = _safe_path(root, frame.get("depth_relative_path"), "depth_relative_path", ".npy")
            validity_path = _safe_path(root, frame.get("validity_relative_path"), "validity_relative_path", ".npy")
            try:
                depth = np.load(depth_path, allow_pickle=False)
                validity = np.load(validity_path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise OrganizedFrameIntakeError("depth and validity arrays must be safe NPY files") from exc
            expected_shape = (calibration["height"], calibration["width"])
            if depth.shape != expected_shape or depth.dtype not in {np.dtype("float32"), np.dtype("float64"), np.dtype("uint16")}:
                raise OrganizedFrameIntakeError("depth array shape or dtype does not match calibration")
            if validity.shape != expected_shape or validity.dtype not in {np.dtype("bool"), np.dtype("uint8")}:
                raise OrganizedFrameIntakeError("validity array shape or dtype does not match calibration")
            if validity.dtype == np.dtype("uint8") and not np.isin(validity, (0, 1)).all():
                raise OrganizedFrameIntakeError("uint8 validity arrays must be binary 0/1 masks")
            mask = validity.astype(bool, copy=False)
            valid_count = int(mask.sum())
            if valid_count == 0:
                raise OrganizedFrameIntakeError("every frame must contain valid depth samples")
            valid_depth = depth[mask].astype(np.float64, copy=False) * float(calibration["depth_unit_mm"])
            if not np.isfinite(valid_depth).all() or np.any(valid_depth <= 0.0):
                raise OrganizedFrameIntakeError("valid depths must be positive and finite")
            depth_sha = hashlib.sha256(depth_path.read_bytes()).hexdigest()
            validity_sha = hashlib.sha256(validity_path.read_bytes()).hexdigest()
            if depth_sha in seen_frame_hashes:
                raise OrganizedFrameIntakeError("duplicate depth-frame content")
            seen_frame_hashes.add(depth_sha)
            valid_total += valid_count
            current_min, current_max = float(valid_depth.min()), float(valid_depth.max())
            valid_min = current_min if valid_min is None else min(valid_min, current_min)
            valid_max = current_max if valid_max is None else max(valid_max, current_max)
            frame_receipts.append({
                "frame_ordinal": frame_index,
                "timestamp_ns": timestamp,
                "depth_sha256": depth_sha,
                "validity_sha256": validity_sha,
                "valid_sample_count": valid_count,
                "pose_sha256": hashlib.sha256(np.asarray(pose, dtype="<f8").tobytes()).hexdigest(),
            })
        receipt = {
            "case_token": token,
            "fused_scan_sha256": scan_sha,
            "fused_scan_vertex_count": scan_header["vertex_count"],
            "calibration_sha256": calibration_sha,
            "frame_count": len(frame_receipts),
            "valid_sample_count": valid_total,
            "minimum_valid_depth_mm": valid_min,
            "maximum_valid_depth_mm": valid_max,
            "frames": frame_receipts,
        }
        receipt["observation_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
        receipts.append(receipt)
    receipts.sort(key=lambda row: row["case_token"])
    result = {
        "schema": "F1_ORGANIZED_FRAME_INTAKE_RECEIPT_V1",
        "status": "ORGANIZED_OBSERVATIONS_VALIDATED",
        "acquisition_protocol_sha256": protocol,
        "case_count": len(receipts),
        "frame_count": sum(row["frame_count"] for row in receipts),
        "cases": receipts,
        "forbidden_information_absent": sorted(FORBIDDEN_KEYS),
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result
