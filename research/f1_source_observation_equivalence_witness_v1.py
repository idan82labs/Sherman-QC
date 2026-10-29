"""Source-observation equivalence witness for the terminal V20 STOP.

This module is a proof harness, not a detector.  Physical-role and expected-
count records remain outside the worker observation.  Both roles receive the
same immutable byte slice and therefore every deterministic source-only worker
must return the same complete output for both roles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Iterable, Mapping, Sequence

import numpy as np


PROFILE = "SOURCE_OBSERVATION_EQUIVALENCE_WITNESS_V1"
OBSERVATION_MAGIC = b"SOWV1\0\0\0"
WORKER_OUTPUT_MAGIC = b"SOWO1\0\0\0"


@dataclass(frozen=True)
class HarnessRole:
    fixture_id: str
    physical_interpretation: str
    required_axis_count: int


@dataclass(frozen=True)
class Observation:
    name: str
    payload: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


ROLES = (
    HarnessRole("NI01", "MONOLITHIC_BEND", 1),
    HarnessRole("NI02", "GROUND_FULL_PENETRATION_JOINT", 0),
)


def _little_endian_bytes(values: np.ndarray, dtype: str) -> bytes:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return array.tobytes(order="C")


def serialize_observation(
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    span_endpoints: Sequence[Sequence[float]],
    scale: float,
    seed_normals: Sequence[Sequence[float]],
) -> bytes:
    """Serialize the complete permitted worker observation canonically."""

    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int32)
    endpoints = np.asarray(span_endpoints, dtype=np.float64)
    seeds = np.asarray(seed_normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("vertices must be finite Nx3 coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles must be Mx3 indices")
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("span_endpoints must be finite 2x3 coordinates")
    if seeds.shape != (2, 3) or not np.isfinite(seeds).all():
        raise ValueError("seed_normals must be finite 2x3 vectors")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    if len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= len(points)):
        raise ValueError("triangle index outside vertex buffer")

    header = bytearray(64)
    header[:8] = OBSERVATION_MAGIC
    struct.pack_into("<IIII", header, 8, 1, len(points), len(faces), 0)
    struct.pack_into("<d", header, 24, float(scale))
    struct.pack_into("<QQ", header, 32, points.nbytes, faces.nbytes)
    header[48:64] = hashlib.sha256(b"ShermanQC/source-observation-v1").digest()[:16]
    return b"".join(
        (
            bytes(header),
            _little_endian_bytes(points, "<f8"),
            _little_endian_bytes(faces, "<i4"),
            _little_endian_bytes(endpoints, "<f8"),
            _little_endian_bytes(seeds, "<f8"),
        )
    )


def witness_worker_output(
    observation: Observation,
    *,
    frozen_axis_count: int,
    frozen_detector_output_sha256: str,
) -> bytes:
    """Return the complete deterministic proof-worker output.

    The adapter does not implement a new bend detector.  It records the frozen
    detector's already-computed count and binds it to the exact observation
    bytes.  It consumes neither fixture role nor expected count.
    """

    if frozen_axis_count < 0:
        raise ValueError("axis count must be nonnegative")
    detector_digest = bytes.fromhex(frozen_detector_output_sha256)
    if len(detector_digest) != 32:
        raise ValueError("expected a SHA-256 detector digest")
    result = bytearray(96)
    result[:8] = WORKER_OUTPUT_MAGIC
    struct.pack_into("<II", result, 8, 1, int(frozen_axis_count))
    result[16:48] = hashlib.sha256(observation.payload).digest()
    result[48:80] = detector_digest
    result[80:96] = hashlib.sha256(PROFILE.encode("ascii")).digest()[:16]
    return bytes(result)


def role_pair(
    observation: Observation,
    *,
    frozen_axis_count: int,
    frozen_detector_output_sha256: str,
) -> dict:
    """Invoke the same role-blind proof worker independently for both roles."""

    positive_input = observation.payload
    negative_input = observation.payload
    positive_output = witness_worker_output(
        Observation(observation.name, positive_input),
        frozen_axis_count=frozen_axis_count,
        frozen_detector_output_sha256=frozen_detector_output_sha256,
    )
    negative_output = witness_worker_output(
        Observation(observation.name, negative_input),
        frozen_axis_count=frozen_axis_count,
        frozen_detector_output_sha256=frozen_detector_output_sha256,
    )
    semantic_rows = [
        {
            "fixture_id": role.fixture_id,
            "physical_interpretation": role.physical_interpretation,
            "required_axis_count": role.required_axis_count,
            "actual_axis_count": int(frozen_axis_count),
            "semantic_pass": int(frozen_axis_count) == role.required_axis_count,
        }
        for role in ROLES
    ]
    return {
        "observation": observation.name,
        "input_bytes": len(observation.payload),
        "input_sha256": observation.sha256,
        "same_input_object": positive_input is negative_input,
        "input_byte_identical": positive_input == negative_input,
        "output_bytes": len(positive_output),
        "output_sha256": hashlib.sha256(positive_output).hexdigest(),
        "output_byte_identical": positive_output == negative_output,
        "semantic_rows": semantic_rows,
        "semantic_success_count": sum(row["semantic_pass"] for row in semantic_rows),
        "contradiction": positive_output == negative_output
        and semantic_rows[0]["required_axis_count"] != semantic_rows[1]["required_axis_count"],
        "positive_output": positive_output,
        "negative_output": negative_output,
    }


def public_pair_record(pair: Mapping[str, object]) -> dict:
    """Remove raw output buffers from a pair for canonical JSON evidence."""

    return {key: value for key, value in pair.items() if key not in {"positive_output", "negative_output"}}


def canonical_json_bytes(value: Mapping[str, object] | Iterable[object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
