from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "exp_cad_0029_cfpdg_core.py"
PREFLIGHT_PATH = ROOT / "scripts" / "run_exp_cad_0029_cad_only_preflight.py"
PREREG_PATH = ROOT / "docs" / "research" / "2026-08-30_exp_cad_0029_cfpdg_preregistration.json"
SPEC = importlib.util.spec_from_file_location("exp29_core", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(CORE)


def _scene(vertices: np.ndarray, triangles: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(triangles, dtype=np.int64)),
        )
    )
    return scene


def test_preregistration_inventory_and_zero_component_visual_are_frozen() -> None:
    payload = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    rows = payload["robustness_inventory"]["rows"]
    assert payload["robustness_inventory"]["semantic_row_count"] == 32
    assert sum(rows.values()) == 32
    assert payload["native_visual_contract"]["all_node_diagnostic_required_even_with_zero_retained_components"] is True
    assert payload["execution_lock"]["physical_scan_load_authorized_before_prereg_hash_and_cad_preflight_audit"] is False
    assert payload["execution_lock"]["cad_actionability_does_not_authorize_source_execution"] is True
    assert payload["cad_only_actionability"]["fold_exclusion_halo_mm_strictly_greater_than"] > 6.4


def test_voxel_aggregation_is_byte_exact_under_arbitrary_permutation() -> None:
    points = np.asarray(
        [
            [0.01, 0.02, 0.03],
            [0.19, 0.18, 0.17],
            [0.10, 0.11, 0.12],
            [1.01, 0.02, 0.03],
            [-0.01, -0.02, -0.03],
            [0.01, 0.02, 0.03],
        ],
        dtype=np.float64,
    )
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.995],
            [0.0, -0.1, 0.995],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    baseline = CORE.permutation_invariant_voxel_aggregate(points, normals, 0.25)
    for seed in range(20):
        permutation = np.random.default_rng(seed).permutation(len(points))
        actual = CORE.permutation_invariant_voxel_aggregate(
            points[permutation], normals[permutation], 0.25
        )
        for expected_array, actual_array in zip(baseline, actual, strict=True):
            assert expected_array.tobytes() == actual_array.tobytes()


def test_cad_interior_disk_accepts_center_and_rejects_open_edge() -> None:
    vertices = np.asarray(
        [[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    scene = _scene(vertices, triangles)
    points = np.asarray([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    mask = CORE.cad_interior_mask(
        scene,
        points,
        normals,
        radii_mm=[1.0, 2.0, 3.0],
        azimuth_count=8,
        maximum_distance_mm=0.075,
        maximum_normal_angle_deg=5.0,
    )
    assert mask.tolist() == [True, False]


def test_topology_graph_does_not_bridge_a_coplanar_gap() -> None:
    vertices = np.asarray(
        [
            [-4.0, -2.0, 0.0], [-1.0, -2.0, 0.0], [-1.0, 2.0, 0.0], [-4.0, 2.0, 0.0],
            [1.0, -2.0, 0.0], [4.0, -2.0, 0.0], [4.0, 2.0, 0.0], [1.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    normals = np.tile([0.0, 0.0, 1.0], (len(triangles), 1))
    topology = CORE.welded_mesh_topology(
        vertices,
        triangles,
        normals,
        weld_tolerance_mm=1e-8,
        smooth_angle_deg=5.0,
    )
    points = np.asarray([[-1.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
    parents = np.asarray([0, 2], dtype=np.int64)
    regions = topology["smooth_region_labels"][parents]
    edges, geodesic = CORE.topology_qualified_graph_edges(
        points,
        parents,
        regions,
        topology,
        maximum_length_mm=4.5,
        maximum_geodesic_to_euclidean_ratio=1.1,
        atlas_edge_sample_spacing_mm=0.75,
    )
    assert edges.shape == (0, 2)
    assert geodesic.shape == (0,)


def test_poisson_choice_is_independent_of_candidate_input_order() -> None:
    candidates = {
        "points": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.1, 0.0, 0.0]]),
        "normals": np.tile([0.0, 0.0, 1.0], (3, 1)),
        "parent_faces": np.asarray([0, 0, 0]),
        "ordinals": np.asarray([0, 1, 2]),
        "area_weights_mm2": np.ones(3),
    }
    selected = CORE.deterministic_poisson_nodes(
        candidates, np.ones(3, dtype=bool), np.asarray([0]), minimum_spacing_mm=3.0
    )
    assert selected.tolist() == [0, 2]
    expected_records = sorted(
        (tuple(candidates["points"][index]), int(candidates["ordinals"][index]))
        for index in selected
    )
    for seed in range(10):
        permutation = np.random.default_rng(seed).permutation(3)
        permuted = {name: values[permutation] for name, values in candidates.items()}
        actual = CORE.deterministic_poisson_nodes(
            permuted, np.ones(3, dtype=bool), np.asarray([0]), minimum_spacing_mm=3.0
        )
        records = sorted(
            (tuple(permuted["points"][index]), int(permuted["ordinals"][index]))
            for index in actual
        )
        assert records == expected_records


def test_exact_transitive_weld_and_five_degree_boundary() -> None:
    # v0-v1 and v1-v2 weld, while v0-v2 is outside tolerance: transitive
    # closure must still create one class.
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [0.75e-8, 0.0, 0.0], [1.5e-8, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pairs = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    union = CORE.UnionFind(len(vertices))
    for first, second in pairs:
        if np.linalg.norm(vertices[first] - vertices[second]) <= 1e-8:
            union.union(int(first), int(second))
    assert union.find(0) == union.find(2)

    angle = np.deg2rad(5.0)
    square = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, np.cos(angle), np.sin(angle)]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    face_normals = np.cross(
        square[faces[:, 1]] - square[faces[:, 0]],
        square[faces[:, 2]] - square[faces[:, 0]],
    )
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
    topology = CORE.welded_mesh_topology(
        square, faces, face_normals, weld_tolerance_mm=1e-8, smooth_angle_deg=5.0
    )
    assert len(np.unique(topology["smooth_region_labels"])) == 1


def test_boundary_clearance_and_area_tie_choose_lowest_node() -> None:
    vertices = np.asarray(
        [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], dtype=np.float64
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    topology = CORE.welded_mesh_topology(
        vertices, triangles, normals, weld_tolerance_mm=1e-8, smooth_angle_deg=5.0
    )
    points = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    clearance = CORE.smooth_region_boundary_clearance_mm(
        points,
        np.asarray([0, 0]),
        topology["smooth_region_labels"],
        topology["welded_vertices"],
        topology["boundary_edges_by_smooth_region"],
    )
    assert clearance[0] == 5.0
    assert clearance[1] == 1.0

    candidates = {
        "points": np.asarray([[-1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]]),
        "normals": np.tile([0.0, 0.0, 1.0], (3, 1)),
        "parent_faces": np.zeros(3, dtype=np.int64),
        "ordinals": np.arange(3, dtype=np.int64),
        "area_weights_mm2": np.asarray([1.0, 1.0, 2.0]),
    }
    weights, assigned = CORE.assign_node_area_weights(
        candidates,
        np.ones(3, dtype=bool),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        maximum_distance_mm=3.0,
    )
    assert assigned.all()
    assert weights.tolist() == [3.0, 1.0]


def test_spatial_folds_are_area_bands_and_halo_excludes_neighbors() -> None:
    points = np.column_stack((np.arange(16, dtype=np.float64), np.zeros(16), np.zeros(16)))
    weights = np.ones(16, dtype=np.float64)
    folds, basis = CORE.spatial_pca_fold_ids(points, weights, fold_count=4)
    assert folds.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert basis["leading_to_second_eigenvalue_ratio"] == float("inf")
    edges = np.column_stack((np.arange(15), np.arange(1, 16))).astype(np.int64)
    graph = CORE.weighted_node_graph(16, edges, np.ones(15))
    masks = CORE.fold_training_masks_with_halo(
        points,
        np.zeros(16, dtype=np.int64),
        folds,
        graph,
        fold_count=4,
        halo_mm=1.1,
    )
    assert not masks[0][:5].any()
    assert masks[0][6:].all()


def test_preflight_cli_has_no_physical_or_evaluator_input_interface() -> None:
    source = PREFLIGHT_PATH.read_text(encoding="utf-8")
    assert "read_point_cloud" not in source
    assert 'add_argument("--scan' not in source
    assert 'add_argument("--evaluator' not in source
    assert 'add_argument("--role' not in source
    assert 'add_argument("--label' not in source
    assert 'add_argument("--gold' not in source
