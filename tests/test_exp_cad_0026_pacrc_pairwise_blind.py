from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_exp_cad_0026_pacrc_pairwise_blind.py"
SPEC = importlib.util.spec_from_file_location("exp26_pacrc", PATH)
EXP26 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP26)


def test_weighted_quantile_is_deterministic_with_ties() -> None:
    values = np.array([3.0, 1.0, 1.0, 2.0])
    weights = np.ones(4)
    assert EXP26.weighted_quantile(values, weights, 0.50) == 1.0
    assert EXP26.weighted_quantile(values, weights, 0.75) == 2.0


def test_equal_area_atlas_preserves_area_and_caps_cell_area() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    triangles = np.array([[0, 1, 2]])
    atlas = EXP26.deterministic_equal_area_atlas(
        vertices, triangles, target_cell_area_mm2=0.5
    )
    assert len(atlas["points"]) == 4
    assert np.isclose(np.sum(atlas["area_weights_mm2"]), 2.0)
    assert np.max(atlas["area_weights_mm2"]) <= 0.5
    assert np.all(atlas["parent_faces"] == 0)

    one_cell = EXP26.deterministic_equal_area_atlas(
        vertices, triangles, target_cell_area_mm2=10.0
    )
    assert np.allclose(one_cell["points"][0], np.mean(vertices, axis=0))


def test_component_connectivity_requires_parent_face_topology() -> None:
    first = np.column_stack((np.linspace(0.0, 0.55, 12), np.zeros(12), np.zeros(12)))
    second = np.column_stack((np.linspace(0.70, 1.25, 12), np.zeros(12), np.zeros(12)))
    points = np.vstack((first, second))
    parents = np.array([0] * 12 + [1] * 12)
    weights = np.ones(24)
    excess = np.ones(24)
    active = np.ones(24, dtype=bool)

    disconnected, _ = EXP26.coherent_components(
        points,
        parents,
        weights,
        excess,
        active,
        face_adjacency=set(),
        observable_area_mm2=24.0,
        sign="TEST",
    )
    connected, _ = EXP26.coherent_components(
        points,
        parents,
        weights,
        excess,
        active,
        face_adjacency={(0, 1)},
        observable_area_mm2=24.0,
        sign="TEST",
    )
    assert sum(row["accepted"] for row in disconnected) == 2
    assert sum(row["accepted"] for row in connected) == 1
    assert max(row["cell_count"] for row in connected) == 24


def test_face_adjacency_welds_duplicate_display_list_seam_vertices() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    assert EXP26.face_adjacency_set(vertices, triangles) == {(0, 1)}


def planar_atlas() -> tuple[dict[str, np.ndarray], np.ndarray]:
    grid = np.array(
        [[2.0 * x, 2.0 * y, 0.0] for y in range(20) for x in range(20)],
        dtype=np.float64,
    )
    atlas = {
        "points": grid,
        "area_weights_mm2": np.ones(len(grid), dtype=np.float64),
        "parent_faces": np.zeros(len(grid), dtype=np.int64),
    }
    return atlas, np.array([[0, 1, 2]], dtype=np.int64)


def test_pair_score_finds_a_coherent_localized_first_scan_anomaly() -> None:
    atlas, triangles = planar_atlas()
    second = atlas["points"].copy()
    first = second.copy()
    patch = (
        (first[:, 0] >= 14.0)
        & (first[:, 0] <= 22.0)
        & (first[:, 1] >= 14.0)
        & (first[:, 1] <= 22.0)
    )
    first[patch, 2] = 2.0
    result = EXP26.score_pair_core(
        atlas,
        set(),
        first,
        second,
        sheet_thickness_mm=1.0,
        retain_arrays=False,
    )
    assert result["signed_pair_score"] > 0.99
    assert result["positive_excess_mass_mm3"] > 0.0
    assert result["negative_excess_mass_mm3"] == 0.0

    reversed_result = EXP26.score_pair_core(
        atlas,
        set(),
        second,
        first,
        sheet_thickness_mm=1.0,
        retain_arrays=False,
    )
    assert np.isclose(
        reversed_result["signed_pair_score"], -result["signed_pair_score"]
    )
    assert np.isclose(
        reversed_result["pair_bias_mm"], -result["pair_bias_mm"]
    )


def test_pair_bias_cancels_a_uniform_subthreshold_registration_offset() -> None:
    atlas, triangles = planar_atlas()
    second = atlas["points"].copy()
    first = second + np.array([0.0, 0.0, 0.4])
    result = EXP26.score_pair_core(
        atlas,
        set(),
        first,
        second,
        sheet_thickness_mm=1.0,
        retain_arrays=False,
    )
    assert np.isclose(result["pair_bias_mm"], 0.4)
    assert result["signed_pair_score"] == 0.0
    assert result["positive_excess_mass_mm3"] == 0.0
    assert result["negative_excess_mass_mm3"] == 0.0


def test_one_sided_missing_support_never_becomes_anomaly_mass() -> None:
    atlas, _triangles = planar_atlas()
    first = atlas["points"].copy()
    second = first + np.array([0.0, 0.0, 20.0])
    result = EXP26.score_pair_core(
        atlas,
        set(),
        first,
        second,
        sheet_thickness_mm=1.0,
        retain_arrays=False,
    )
    assert result["one_sided_support_area_fraction"] == 1.0
    assert result["positive_excess_mass_mm3"] == 0.0
    assert result["negative_excess_mass_mm3"] == 0.0


def test_uncertainty_grid_contains_symmetric_translation_and_rotation_checks() -> None:
    points = np.zeros((2, 3), dtype=np.float64)
    variants = EXP26.uncertainty_variants(points, np.zeros(3), 2.0)
    assert len(variants) == 12
    assert len({name for name, _points in variants}) == 12


def test_variant_must_pass_score_magnitude_not_merely_preserve_sign() -> None:
    otherwise_valid = {
        "signed_pair_score": 0.59,
        "anchor_count": 1000,
        "anchor_area_fraction": 0.90,
        "one_sided_support_area_fraction": 0.0,
    }
    assert EXP26.central_rank_gate_reasons(otherwise_valid) == [
        "PAIR_SCORE_MAGNITUDE_GATE"
    ]
    otherwise_valid["signed_pair_score"] = 0.61
    assert EXP26.central_rank_gate_reasons(otherwise_valid) == []
