from __future__ import annotations

import json

import numpy as np
import pytest

from patch_graph_latent_decomposition import (
    MAX_COMPACT_SUPPORT_RECORDS,
    MAX_MODE_CHILDREN_PER_PARENT,
    MAX_TRANSITION_MODE_CHILDREN,
    MAX_TRANSITION_MODE_EDGES,
    MAX_TRANSITION_PARENT_ATOMS,
    TRANSITION_ATOM_INPUT_POINT_CAP,
    TRANSITION_ATOM_PROFILE,
    _surface_atom_graph_from_dict,
    build_surface_atom_graph,
    build_transition_atom_graph_v2,
    compact_surface_atom_graph,
    compact_transition_graph_v2,
    reject_recursive_compact_keys,
)


def _wedge_points() -> tuple[np.ndarray, np.ndarray]:
    points = []
    normals = []
    for y_value in (0.2, 1.1):
        for z_value in (0.2, 0.3):
            points.append([0.2, y_value, z_value])
            normals.append([1.0, 0.0, 0.0])
    for x_value in (0.2, 1.1):
        for z_value in (0.4, 0.5):
            points.append([x_value, 0.2, z_value])
            normals.append([0.0, 1.0, 0.0])
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _assert_forbidden_absent(value: object) -> None:
    forbidden = {
        "point_indices",
        "source_indices",
        "source_point_indices",
        "vertex_indices",
        "points",
        "source_points",
        "point_coordinates",
        "coordinates",
        "bbox_points",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            _assert_forbidden_absent(child)
    elif isinstance(value, list):
        for child in value:
            _assert_forbidden_absent(child)


def test_transition_atom_v2_preserves_two_projective_modes_in_one_parent() -> None:
    points, normals = _wedge_points()
    graph = build_transition_atom_graph_v2(points, normals, local_spacing_mm=0.2)
    payload = compact_transition_graph_v2(graph)

    assert graph.profile == TRANSITION_ATOM_PROFILE
    assert len(graph.parents) == 1
    assert len(graph.modes) == 2
    assert {edge.edge_kind for edge in graph.mode_edges} == {"CROSS_MODE_LOCAL"}
    parent = graph.parents[0]
    assert parent.normalized_support_mass >= 1
    assert parent.valid_mode_count == 2
    assert parent.mode_overflow is False
    assert parent.transition_certificate_allowed is True
    assert sum(mode.point_count for mode in graph.modes) == parent.point_count
    assert {mode.parent_atom_id for mode in graph.modes} == {parent.parent_atom_id}
    assert payload["transition_atom_caps"] == {
        "input_point_cap": TRANSITION_ATOM_INPUT_POINT_CAP,
        "mode_cap": MAX_TRANSITION_MODE_CHILDREN,
        "mode_children_per_parent_cap": MAX_MODE_CHILDREN_PER_PARENT,
        "mode_edge_cap": MAX_TRANSITION_MODE_EDGES,
        "parent_cap": MAX_TRANSITION_PARENT_ATOMS,
        "parent_plus_mode_cap": MAX_COMPACT_SUPPORT_RECORDS,
    }
    _assert_forbidden_absent(payload)


def test_transition_atom_v2_is_point_order_and_normal_sign_invariant() -> None:
    points, normals = _wedge_points()
    baseline = compact_transition_graph_v2(
        build_transition_atom_graph_v2(points, normals, local_spacing_mm=0.2)
    )
    permutation = np.asarray([7, 1, 5, 3, 0, 6, 2, 4], dtype=np.int64)
    flipped = normals[permutation].copy()
    flipped[[0, 2, 3, 6]] *= -1.0
    replay = compact_transition_graph_v2(
        build_transition_atom_graph_v2(
            points[permutation], flipped, local_spacing_mm=0.2
        )
    )

    assert _canonical_bytes(replay) == _canonical_bytes(baseline)
    assert [mode.mode_atom_id for mode in build_transition_atom_graph_v2(
        points[permutation], flipped, local_spacing_mm=0.2
    ).modes] == [mode["mode_atom_id"] for mode in baseline["support_mode_table"]]


def test_transition_atom_v2_overflow_is_explicit_and_disables_certificate() -> None:
    mode_normals = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points = []
    normals = []
    for mode_normal in mode_normals:
        for x_value in (0.2, 1.1):
            for z_value in (0.2, 0.3):
                points.append([x_value, 0.2, z_value])
                normals.append(mode_normal)
    graph = build_transition_atom_graph_v2(
        np.asarray(points, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        local_spacing_mm=0.2,
    )

    assert len(graph.parents) == 1
    assert len(graph.modes) == MAX_MODE_CHILDREN_PER_PARENT
    parent = graph.parents[0]
    assert parent.mode_overflow is True
    assert parent.transition_certificate_allowed is False
    assert parent.unmodeled_point_count == 4
    assert graph.diagnostics["raw_valid_mode_count"] == 5
    assert graph.diagnostics["overflow_parent_count"] == 1
    assert graph.diagnostics["mode_overflow_disables_hard_transition_certificate"] is True
    compact_transition_graph_v2(graph)


def test_transition_atom_v2_enforces_input_cap_before_building() -> None:
    points = np.zeros((TRANSITION_ATOM_INPUT_POINT_CAP + 1, 3), dtype=np.float64)
    normals = np.zeros_like(points)
    normals[:, 2] = 1.0
    with pytest.raises(ValueError, match="input count"):
        build_transition_atom_graph_v2(points, normals, local_spacing_mm=0.5)


def test_compact_serializers_recursively_reject_raw_point_membership() -> None:
    with pytest.raises(ValueError, match="forbidden field nested.point_indices"):
        reject_recursive_compact_keys({"nested": {"point_indices": [1, 2]}})
    with pytest.raises(ValueError, match="forbidden field nested.0.coordinates"):
        reject_recursive_compact_keys({"nested": [{"coordinates": [[0, 0, 0]]}]})

    points, normals = _wedge_points()
    legacy = build_surface_atom_graph(
        points,
        normals,
        voxel_size_mm=2.5,
        local_spacing_mm=0.2,
    )
    compact = compact_surface_atom_graph(legacy)
    _assert_forbidden_absent(compact)
    restored = _surface_atom_graph_from_dict(compact)
    assert len(restored.atoms) == len(legacy.atoms)
    assert all(atom.point_indices == () for atom in restored.atoms)
