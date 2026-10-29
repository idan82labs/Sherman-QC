from __future__ import annotations

import hashlib
import math

import numpy as np

from backend.f1_persistent_tube_component_parity_v19 import (
    FAIL_STRUCTURAL_CAP,
    RECORD_BYTES,
    SCAN_HEADER_BYTES,
    STATUS_RETAIN,
    analyze_candidate,
    encode_sidecar,
)


def _polyline_sample(points: np.ndarray, distance: float) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    index = min(len(segments) - 1, int(np.searchsorted(cumulative, distance, side="right") - 1))
    fraction = (distance - cumulative[index]) / segments[index] if segments[index] > 0.0 else 0.0
    return points[index] + fraction * (points[index + 1] - points[index])


def _l_path(*, offset: float, radius: float, reach: float, arc_steps: int = 64) -> np.ndarray:
    center = np.asarray((offset, offset), dtype=np.float64)
    arc = np.asarray([
        center + radius * np.asarray((math.cos(angle), math.sin(angle)))
        for angle in np.linspace(0.0, math.pi / 2.0, arc_steps + 1)
    ])
    return np.vstack((
        np.asarray(((reach, offset), (offset + radius, offset))),
        arc[1:-1],
        np.asarray(((offset, offset + radius), (offset, reach))),
    ))


def _fragmented_surface(
    path: np.ndarray,
    *,
    length: float,
    axial_fragments: int = 12,
    cross_fragments: int = 14,
    gap: float = 1.0,
    crop_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    path_segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_length = float(np.sum(path_segments))
    axial_start = 0.5 * crop_fraction * length
    axial_end = length - 0.5 * crop_fraction * length
    x_edges = np.linspace(axial_start, axial_end, axial_fragments + 1)
    u_edges = np.linspace(0.0, path_length, cross_fragments + 1)
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    for xi in range(axial_fragments):
        x0 = x_edges[xi] + (0.5 * gap if xi else 0.0)
        x1 = x_edges[xi + 1] - (0.5 * gap if xi + 1 < axial_fragments else 0.0)
        for ui in range(cross_fragments):
            u0 = u_edges[ui] + (0.5 * gap if ui else 0.0)
            u1 = u_edges[ui + 1] - (0.5 * gap if ui + 1 < cross_fragments else 0.0)
            if x1 <= x0 or u1 <= u0:
                continue
            yz0, yz1 = _polyline_sample(path, u0), _polyline_sample(path, u1)
            base = len(vertices)
            vertices.extend((
                [x0, yz0[0], yz0[1]], [x1, yz0[0], yz0[1]],
                [x1, yz1[0], yz1[1]], [x0, yz1[0], yz1[1]],
            ))
            triangles.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def _combine(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    offset = 0
    for part_vertices, part_triangles in parts:
        vertices.append(part_vertices)
        triangles.append(part_triangles + offset)
        offset += len(part_vertices)
    return np.vstack(vertices), np.vstack(triangles)


def _positive_geometry(*, angle: float = 90.0, thickness: float = 8.0, crop_fraction: float = 0.0):
    # The frozen 60-degree case changes the parent arm angle while preserving
    # a nested thin-sheet pair.  The procedural path is rotated accordingly.
    outer = _l_path(offset=0.0, radius=12.0 if angle == 90.0 else 21.0, reach=96.0)
    inner_radius = 12.0 if angle == 90.0 else 21.0
    inner = _l_path(offset=thickness, radius=inner_radius, reach=96.0)
    if angle != 90.0:
        shear = math.tan(math.radians(90.0 - angle))
        for path in (outer, inner):
            path[:, 0] += shear * path[:, 1]
    return _combine([
        _fragmented_surface(outer, length=320.0, gap=1.0, crop_fraction=crop_fraction),
        _fragmented_surface(inner, length=320.0, gap=1.0, crop_fraction=crop_fraction),
    ])


def _row(length: float = 320.0):
    return {
        "candidate_id": "synthetic-ptcp",
        "span_endpoints": [[0.0, 0.0, 0.0], [length, 0.0, 0.0]],
    }


def _analyze(vertices, triangles, *, angle=90.0, scale=1.0, length=320.0):
    normal_a = np.asarray((0.0, 0.0, 1.0))
    normal_b = np.asarray((0.0, -1.0, 0.0))
    if angle != 90.0:
        radians = math.radians(90.0 - angle)
        normal_b = np.asarray((0.0, -math.cos(radians), math.sin(radians)))
    return analyze_candidate(
        vertices=vertices,
        triangles=triangles,
        row=_row(length),
        scale=scale,
        parent_normals=(normal_a, normal_b),
        parent_centroids=(np.asarray((length / 2.0, 70.0, 0.0)), np.asarray((length / 2.0, 0.0, 70.0))),
    )


def _single_surface(path, *, length=320.0, gap=1.0):
    return _fragmented_surface(path, length=length, gap=gap)


def _wall(y0, z0, y1, z1, *, length=320.0, holes=False):
    path = np.asarray(((y0, z0), (y1, z1)), dtype=np.float64)
    return _fragmented_surface(path, length=length, axial_fragments=12 if holes else 1, cross_fragments=1, gap=0.5 if holes else 0.0)


def _native_bridge(left_part, right_part, *, arm: str, holes: bool = False):
    """Make a wall from exact vertices already present on both skins.

    The earlier draft fixture used arbitrary cross-section coordinates that
    did not touch either skin.  This helper is deliberately geometry-derived:
    it selects exact native endpoints in the requested port and reuses a
    shared native axial tile, so zero-gap connectivity is unambiguous.
    """
    left_vertices, _ = left_part
    right_vertices, _ = right_part
    if arm == "a":
        left_pool = left_vertices[(left_vertices[:, 1] >= 55.0) & (left_vertices[:, 1] <= 72.0) & (np.abs(left_vertices[:, 2]) < 1e-9)]
        right_pool = right_vertices[(right_vertices[:, 1] >= 55.0) & (right_vertices[:, 1] <= 72.0) & (np.abs(right_vertices[:, 2] - 8.0) < 1e-9)]
    else:
        left_pool = left_vertices[(left_vertices[:, 2] >= 55.0) & (left_vertices[:, 2] <= 72.0) & (np.abs(left_vertices[:, 1]) < 1e-9)]
        right_pool = right_vertices[(right_vertices[:, 2] >= 55.0) & (right_vertices[:, 2] <= 72.0) & (np.abs(right_vertices[:, 1] - 8.0) < 1e-9)]
    common_x = sorted(set(left_pool[:, 0].tolist()) & set(right_pool[:, 0].tolist()))
    central = [value for value in common_x if 120.0 <= value <= 200.0]
    x0, x1 = central[0], central[1]
    if holes:
        x0 += 0.25
        x1 -= 0.25
    def choose(pool, x):
        rows = pool[np.isclose(pool[:, 0], x)]
        return sorted(rows.tolist(), key=lambda row: (abs((row[1] + row[2]) - 65.0), row[1], row[2]))[0]
    a0, a1 = np.asarray(choose(left_pool, common_x[common_x.index(x0 - 0.25)] if holes else x0)), np.asarray(choose(left_pool, common_x[common_x.index(x1 + 0.25)] if holes else x1))
    b0, b1 = np.asarray(choose(right_pool, common_x[common_x.index(x0 - 0.25)] if holes else x0)), np.asarray(choose(right_pool, common_x[common_x.index(x1 + 0.25)] if holes else x1))
    a0[0], a1[0], b0[0], b1[0] = x0, x1, x0, x1
    vertices = np.asarray((a0, a1, b1, b0), dtype=np.float64)
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    return vertices, triangles


def test_fragmented_90_degree_double_skin_retains_and_record_is_fixed():
    vertices, triangles = _positive_geometry()
    record = _analyze(vertices, triangles)
    assert record.status_bits & FAIL_STRUCTURAL_CAP == 0, record.summary()
    assert record.retain, record.summary()
    assert record.birth_gap <= record.death_gap / 4.0
    assert int(record.masks[1]).bit_count() >= 3
    assert int(record.masks[2]).bit_count() >= 3
    assert len(record.encode()) == RECORD_BYTES
    assert len(encode_sidecar((record,))) == SCAN_HEADER_BYTES + RECORD_BYTES


def test_fragmented_60_degree_cropped_double_skin_retains():
    vertices, triangles = _positive_geometry(angle=60.0, thickness=6.0, crop_fraction=0.20)
    record = _analyze(vertices, triangles, angle=60.0)
    assert record.retain, record.summary()
    assert record.birth_gap <= record.death_gap / 4.0


def test_mandatory_negatives_emit_zero():
    outer = _l_path(offset=0.0, radius=12.0, reach=96.0)
    inner = _l_path(offset=8.0, radius=12.0, reach=96.0)
    positives = [_fragmented_surface(outer, length=320.0, gap=1.0), _fragmented_surface(inner, length=320.0, gap=1.0)]
    bridge_a = _native_bridge(positives[0], positives[1], arm="a")
    bridge_b = _native_bridge(positives[0], positives[1], arm="b")
    holed_bridge = _native_bridge(positives[0], positives[1], arm="a", holes=True)
    cases = {
        "single_observed_l_skin": [positives[0]],
        "terminal_thickness_wall": [*positives, bridge_a],
        "opening_rim_complete_wall": [*positives, bridge_a, bridge_b],
        "wall_with_holes": [*positives, holed_bridge],
        "two_flat_panels": [
            _single_surface(np.asarray(((96.0, 0.0), (12.0, 0.0))), gap=1.0),
            _single_surface(np.asarray(((96.0, 8.0), (20.0, 8.0))), gap=1.0),
        ],
        "four_through_skins": [
            *positives,
            _fragmented_surface(_l_path(offset=20.0, radius=12.0, reach=96.0), length=320.0, gap=1.0),
            _fragmented_surface(_l_path(offset=28.0, radius=12.0, reach=96.0), length=320.0, gap=1.0),
        ],
        "weld_bead_bridge": [*positives, bridge_a],
        "cycle_closing_corner": [*positives, bridge_a, bridge_b],
    }
    for name, parts in cases.items():
        vertices, triangles = _combine(parts)
        record = _analyze(vertices, triangles)
        assert not record.retain, (name, record.summary())


def _transform(vertices, rotation, translation, scale):
    return scale * (vertices @ rotation.T) + translation


def test_metamorphic_decisions_and_sidecars_are_identical_after_normalization():
    vertices, triangles = _positive_geometry()
    base = _analyze(vertices, triangles)
    assert base.retain, base.summary()
    base_bytes = base.encode()
    configurations = [
        (17.0, np.asarray((11.0, -7.0, 23.0)), 1.0),
        (53.0, np.asarray((-19.0, 31.0, -5.0)), 1.0),
        (101.0, np.asarray((7.0, 13.0, -29.0)), 1.0),
        (0.0, np.zeros(3), 0.5),
        (0.0, np.zeros(3), 2.0),
    ]
    for degrees, translation, factor in configurations:
        angle = math.radians(degrees)
        rotation = np.asarray(((math.cos(angle), -math.sin(angle), 0.0), (math.sin(angle), math.cos(angle), 0.0), (0.0, 0.0, 1.0)))
        transformed = _transform(vertices, rotation, translation, factor)
        row = _row(320.0)
        row["span_endpoints"] = _transform(np.asarray(row["span_endpoints"]), rotation, translation, factor).tolist()
        normal_a = np.asarray((0.0, 0.0, 1.0)) @ rotation.T
        normal_b = np.asarray((0.0, -1.0, 0.0)) @ rotation.T
        record = analyze_candidate(
            vertices=transformed, triangles=triangles, row=row, scale=factor,
            parent_normals=(normal_a, normal_b),
            parent_centroids=(
                _transform(np.asarray([[160.0, 70.0, 0.0]]), rotation, translation, factor)[0],
                _transform(np.asarray([[160.0, 0.0, 70.0]]), rotation, translation, factor)[0],
            ),
        )
        assert record.retain, record.summary()
        assert record.status_bits == base.status_bits
        assert record.masks == base.masks
        assert (record.birth_q, record.death_q) == (base.birth_q, base.death_q)
        assert record.topology_hash == base.topology_hash
        assert record.encode() == base_bytes


def test_orientation_and_index_permutation_preserve_sidecar():
    vertices, triangles = _positive_geometry()
    base = _analyze(vertices, triangles)
    reversed_record = _analyze(vertices, triangles[:, ::-1])
    rng = np.random.default_rng(1701)
    vertex_order = rng.permutation(len(vertices))
    inverse = np.empty_like(vertex_order)
    inverse[vertex_order] = np.arange(len(vertex_order))
    permuted_vertices = vertices[vertex_order]
    permuted_triangles = inverse[triangles]
    permuted_triangles = permuted_triangles[rng.permutation(len(permuted_triangles))]
    permuted_record = _analyze(permuted_vertices, permuted_triangles)
    assert base.retain and reversed_record.retain and permuted_record.retain
    assert reversed_record.encode() == base.encode()
    assert permuted_record.encode() == base.encode()


def test_two_replays_and_worker_scheduling_are_byte_identical():
    vertices, triangles = _positive_geometry()
    first = _analyze(vertices, triangles)
    second = _analyze(vertices, triangles)
    assert hashlib.sha256(first.encode()).digest() == hashlib.sha256(second.encode()).digest()
    assert first.encode() == second.encode()
