from __future__ import annotations

import hashlib

import numpy as np

from research.f1_source_observation_equivalence_witness_v1 import (
    Observation,
    canonical_json_bytes,
    public_pair_record,
    role_pair,
    serialize_observation,
)


def _observation() -> Observation:
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    triangles = np.asarray(((0, 1, 2),), dtype=np.int32)
    payload = serialize_observation(
        vertices=vertices,
        triangles=triangles,
        span_endpoints=((0, 0, 0), (1, 0, 0)),
        scale=1.0,
        seed_normals=((0, 1, 0), (0, 0, 1)),
    )
    return Observation("tiny", payload)


def test_role_and_expectation_never_enter_worker_observation():
    observation = _observation()
    pair = role_pair(
        observation,
        frozen_axis_count=1,
        frozen_detector_output_sha256=hashlib.sha256(b"frozen").hexdigest(),
    )
    assert pair["same_input_object"]
    assert pair["input_byte_identical"]
    assert pair["output_byte_identical"]
    assert pair["contradiction"]
    assert pair["semantic_success_count"] == 1
    assert [row["required_axis_count"] for row in pair["semantic_rows"]] == [1, 0]


def test_canonical_evidence_replay_is_byte_identical():
    pair = role_pair(
        _observation(),
        frozen_axis_count=0,
        frozen_detector_output_sha256=hashlib.sha256(b"frozen").hexdigest(),
    )
    evidence = public_pair_record(pair)
    assert canonical_json_bytes(evidence) == canonical_json_bytes(evidence)


def test_observation_serialization_is_sensitive_to_source_bytes():
    observation = _observation()
    changed = serialize_observation(
        vertices=np.asarray(((0, 0, 0), (2, 0, 0), (0, 1, 0)), dtype=np.float64),
        triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
        span_endpoints=((0, 0, 0), (1, 0, 0)),
        scale=1.0,
        seed_normals=((0, 1, 0), (0, 0, 1)),
    )
    assert observation.payload != changed
