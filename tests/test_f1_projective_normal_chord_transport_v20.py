from __future__ import annotations

import hashlib
import math

import numpy as np

from backend.f1_projective_normal_chord_transport_v20 import (
    RECORD_BYTES,
    SCAN_HEADER_BYTES,
    analyze_candidate,
    encode_sidecar,
)


def offset_fold(*, angle_deg=90.0, mid_radius=12.0, thickness=8.0, length=40.0, phase_steps=12):
    angle = math.radians(angle_deg)
    profiles = []
    for radius in (mid_radius - thickness / 2.0, mid_radius + thickness / 2.0):
        arc = np.asarray([(radius * math.cos(t), radius * math.sin(t)) for t in np.linspace(0.0, angle, phase_steps + 1)])
        normal0 = np.asarray((math.cos(0.0), math.sin(0.0)))
        normal1 = np.asarray((math.cos(angle), math.sin(angle)))
        tangent0 = np.asarray((-math.sin(0.0), math.cos(0.0)))
        tangent1 = np.asarray((-math.sin(angle), math.cos(angle)))
        profiles.append(np.vstack((arc[0] - 20.0 * tangent0, arc, arc[-1] + 20.0 * tangent1)))
    stations = np.asarray([(index - 5) * length / 12.0 for index in range(11)])
    x_values = np.unique(np.concatenate(([-length / 2.0], stations, [length / 2.0])))
    vertices = []
    triangles = []
    for profile in profiles:
        base = len(vertices)
        for x in x_values:
            vertices.extend([[x, y, z] for y, z in profile])
        width = len(profile)
        for xi in range(len(x_values) - 1):
            for pi in range(width - 1):
                a = base + xi * width + pi
                b = a + width
                triangles.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def row(length=40.0):
    return {"candidate_id": "sf01", "candidate_key": "01" * 16, "span_endpoints": [[-length / 2, 0, 0], [length / 2, 0, 0]]}


def analyze(vertices, triangles, *, angle=90.0, length=40.0, scale=1.0):
    seed_b = (0.0, math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    return analyze_candidate(vertices=vertices, triangles=triangles, row=row(length), scale=scale, seed_normals=((0, 1, 0), seed_b))


def test_standard_fold_retains_and_serialization_is_fixed():
    vertices, triangles = offset_fold()
    record = analyze(vertices, triangles)
    assert record.retain, record.summary()
    assert len(record.encode()) == RECORD_BYTES
    assert len(encode_sidecar((record,))) == SCAN_HEADER_BYTES + RECORD_BYTES


def test_permutation_winding_axis_seed_and_replay_are_byte_identical():
    vertices, triangles = offset_fold()
    base = analyze(vertices, triangles)
    assert base.retain, base.summary()
    rng = np.random.default_rng(2001)
    variants = [triangles[:, ::-1], triangles[rng.permutation(len(triangles))]]
    for variant in variants:
        assert analyze(vertices, variant).encode() == base.encode()
    reversed_row = row()
    reversed_row["span_endpoints"] = list(reversed(reversed_row["span_endpoints"]))
    swapped = analyze_candidate(vertices=vertices, triangles=triangles, row=reversed_row, scale=1.0, seed_normals=((0, 0, 1), (0, 1, 0)))
    assert swapped.encode() == base.encode()
    assert hashlib.sha256(analyze(vertices, triangles).encode()).digest() == hashlib.sha256(base.encode()).digest()


def test_single_skin_is_rejected():
    vertices, triangles = offset_fold()
    half = len(vertices) // 2
    mask = np.all(triangles < half, axis=1)
    record = analyze(vertices[:half], triangles[mask])
    assert not record.retain
