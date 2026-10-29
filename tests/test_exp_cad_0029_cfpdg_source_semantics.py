from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "scripts" / "exp_cad_0029_cfpdg_source.py"
RUNNER_PATH = ROOT / "scripts" / "run_exp_cad_0029_cfpdg_source.py"
PREREG_PATH = ROOT / "docs" / "research" / "2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json"
SPEC = importlib.util.spec_from_file_location("exp29_source", SOURCE_PATH)
SRC = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(SRC)
RUNNER_SPEC = importlib.util.spec_from_file_location("exp29_source_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_SPEC.loader.exec_module(RUNNER)
RENDER_PATH = ROOT / "scripts" / "render_exp_cad_0030_cfpdg_native.py"
RENDER_SPEC = importlib.util.spec_from_file_location("exp29_renderer_for_source", RENDER_PATH)
RENDER = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC and RENDER_SPEC.loader
RENDER_SPEC.loader.exec_module(RENDER)


def node_dtype() -> np.dtype:
    return np.dtype(
        [
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
            ("e1x", "<f8"), ("e1y", "<f8"), ("e1z", "<f8"),
            ("e2x", "<f8"), ("e2y", "<f8"), ("e2z", "<f8"),
            ("parent_face", "<i8"), ("candidate_ordinal", "<i8"),
            ("smooth_region", "<i8"), ("area_mm2", "<f8"),
            ("fold_id", "u1"), ("graph_component", "<i8"),
        ]
    )


def synthetic_nodes(count: int = 80) -> np.ndarray:
    rng = np.random.default_rng(1234)
    nodes = np.zeros(count, dtype=node_dtype())
    points = rng.normal(size=(count, 3)) * np.asarray([20.0, 15.0, 10.0])
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    first, second = SRC.CAD.canonical_tangent_basis(normals)
    for names, values in (
        (("x", "y", "z"), points),
        (("nx", "ny", "nz"), normals),
        (("e1x", "e1y", "e1z"), first),
        (("e2x", "e2y", "e2z"), second),
    ):
        for axis, name in enumerate(names):
            nodes[name] = values[:, axis]
    nodes["area_mm2"] = 1.0
    nodes["fold_id"] = np.arange(count) % 4
    nodes["parent_face"] = np.arange(count)
    nodes["candidate_ordinal"] = 0
    nodes["smooth_region"] = 0
    return nodes


def prereg() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def test_normal_gauge_uses_largest_absolute_axis_and_is_sign_invariant() -> None:
    normals = np.asarray(
        [[1e-10, -2e-10, -1.0], [-0.4, 0.9, 0.1], [0.8, 0.8, -0.1]],
        dtype=np.float64,
    )
    first = SRC.canonical_unoriented_normals(normals)
    second = SRC.canonical_unoriented_normals(-normals)
    assert first.tobytes() == second.tobytes()
    assert first[0, 2] > 0.0
    assert first[1, 1] > 0.0
    assert first[2, 0] > 0.0  # x wins exact |x|==|y| tie


def test_identity_hash_uses_same_normal_gauge_and_survives_member_swap() -> None:
    points = np.asarray([[0.0, -0.0, 1.0], [2.0, 3.0, 4.0]])
    normals = np.asarray([[1e-9, 0.0, -1.0], [-0.8, 0.2, 0.1]])
    assert np.array_equal(
        SRC.canonical_identity_bits(points, normals),
        SRC.canonical_identity_bits(points, -normals),
    )
    for bad_points, bad_normals in (
        (np.asarray([[np.nan, 0.0, 0.0]]), np.asarray([[0.0, 0.0, 1.0]])),
        (np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[0.0, np.inf, 1.0]])),
        (np.asarray([0.0, 0.0, 0.0]), np.asarray([[0.0, 0.0, 1.0]])),
        (np.zeros((2, 3)), np.zeros((1, 3))),
    ):
        try:
            SRC.canonical_identity_bits(bad_points, bad_normals)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid identity arrays were accepted")


def test_spatial_checker_validates_shape_finiteness_and_positive_cell() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.76, 0.01, 0.01]])
    assert SRC.spatial_checker_bits(points, 0.75).tolist() == [0, 1]
    for bad_points, bad_cell in (
        (np.asarray([0.0, 0.0, 0.0]), 0.75),
        (np.asarray([[0.0, np.nan, 0.0]]), 0.75),
        (np.asarray([[0.0, 0.0, np.inf]]), 0.75),
        (np.asarray([[0.0, 0.0, 0.0]]), 0.0),
        (np.asarray([[0.0, 0.0, 0.0]]), -1.0),
        (np.asarray([[0.0, 0.0, 0.0]]), np.nan),
        (np.asarray([[1e308, 0.0, 0.0]]), 1e-300),
    ):
        try:
            SRC.spatial_checker_bits(bad_points, bad_cell)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid spatial-checker input was accepted")


def test_rigid_transform_rejects_nonfinite_shape_and_length_mismatch() -> None:
    for points, normals in (
        (np.asarray([[np.inf, 0.0, 0.0]]), np.asarray([[0.0, 0.0, 1.0]])),
        (np.asarray([0.0, 0.0, 0.0]), np.asarray([[0.0, 0.0, 1.0]])),
        (np.zeros((2, 3)), np.ones((1, 3))),
    ):
        try:
            SRC.apply_rigid_transform(points, normals, np.eye(4))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid transform arrays were accepted")


def test_transform_loader_rejects_missing_malformed_shear_and_reflection(tmp_path: Path) -> None:
    scan_sha = "b" * 64

    def persist(name: str, matrix: object | None) -> Path:
        payload: dict[str, object] = {
            "case_token": "N009",
            "scan_sha256": scan_sha,
        }
        if matrix is not None:
            payload["selected_absolute_transform"] = matrix
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return path

    angle = np.deg2rad(13.0)
    valid = np.eye(4)
    valid[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    valid[:3, 3] = [1.0, -2.0, 3.0]
    valid_path = persist("valid", valid.tolist())
    assert np.array_equal(
        RUNNER.load_transform(
            valid_path, "N009", scan_sha, RUNNER.CAD.sha256_path(valid_path)
        ),
        valid,
    )
    shear = np.eye(4); shear[0, 1] = 0.2
    reflection = np.eye(4); reflection[0, 0] = -1.0
    malformed = np.eye(3)
    for name, matrix in (
        ("missing", None),
        ("malformed", malformed.tolist()),
        ("shear", shear.tolist()),
        ("reflection", reflection.tolist()),
    ):
        path = persist(name, matrix)
        try:
            RUNNER.load_transform(path, "N009", scan_sha, RUNNER.CAD.sha256_path(path))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} transform was accepted")


def test_cad_artifacts_are_loadable_before_scan_and_malformed_edges_fail(tmp_path: Path) -> None:
    nodes = synthetic_nodes(8)
    edges = np.column_stack((np.arange(7), np.arange(1, 8))).astype(np.int64)
    weights = np.ones(7, dtype=np.float64)
    nodes_path, edges_path, weights_path = (
        tmp_path / "nodes.npy", tmp_path / "edges.npy", tmp_path / "weights.npy"
    )
    np.save(nodes_path, nodes, allow_pickle=False)
    np.save(edges_path, edges, allow_pickle=False)
    np.save(weights_path, weights, allow_pickle=False)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        ),
        o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32)),
    )
    mesh_path = tmp_path / "mesh.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)
    loaded = RUNNER.load_cad_source_artifacts_before_scan(
        nodes_path=nodes_path,
        edges_path=edges_path,
        edge_weights_path=weights_path,
        mesh_path=mesh_path,
    )
    assert loaded[0].shape == (8,)
    bad_edges = edges.copy(); bad_edges[-1, 1] = 999
    np.save(edges_path, bad_edges, allow_pickle=False)
    try:
        RUNNER.load_cad_source_artifacts_before_scan(
            nodes_path=nodes_path,
            edges_path=edges_path,
            edge_weights_path=weights_path,
            mesh_path=mesh_path,
        )
    except ValueError as error:
        assert "out-of-range" in str(error)
    else:
        raise AssertionError("malformed CAD edge artifact was accepted")


def test_exact_two_by_sixteen_footprint_boundaries() -> None:
    angles = (np.arange(16) + 0.5) * 2.0 * np.pi / 16.0
    inner = np.column_stack((1.49 * np.cos(angles), 1.49 * np.sin(angles)))
    outer = np.column_stack((2.99 * np.cos(angles), 2.99 * np.sin(angles)))
    count, cells = SRC.footprint_occupancy(np.vstack((inner, outer)), 3.0, 16, 2)
    assert count == 32
    assert cells == list(range(32))
    _count, boundary_cells = SRC.footprint_occupancy(
        np.asarray([[1.5, 0.0], [np.nextafter(1.5, np.inf), 0.0]]), 3.0, 16, 2
    )
    assert boundary_cells == [0, 16]


def clustered_sheet_points(second_height: float) -> tuple[np.ndarray, ...]:
    angles = (np.arange(16) + 0.5) * 2.0 * np.pi / 16.0
    uv = np.vstack(
        (
            np.column_stack((1.0 * np.cos(angles), 1.0 * np.sin(angles))),
            np.column_stack((2.5 * np.cos(angles), 2.5 * np.sin(angles))),
        )
    )
    points = np.vstack(
        (
            np.column_stack((uv, np.zeros(len(uv)))),
            np.column_stack((uv, np.full(len(uv), second_height))),
        )
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    keys = np.arange(len(points) * 3, dtype=np.int64).reshape(-1, 3)
    regions = np.zeros(len(points), dtype=np.int64)
    sheets = np.r_[np.full(len(uv), -1), np.full(len(uv), 1)].astype(np.int8)
    return points, normals, keys, regions, sheets


def test_parallel_skin_runner_ambiguity_and_exact_runner_boundary() -> None:
    config = prereg()["local_surface_fit"]
    arguments = dict(
        node_point=np.zeros(3), node_normal=np.asarray([0.0, 0.0, 1.0]),
        tangent_first=np.asarray([1.0, 0.0, 0.0]),
        tangent_second=np.asarray([0.0, 1.0, 0.0]), node_region=0, config=config,
    )
    values = clustered_sheet_points(0.5)
    ambiguous = SRC.local_surface_fit(
        **arguments,
        voxel_points=values[0], voxel_normals=values[1], voxel_keys=values[2],
        voxel_regions=values[3], voxel_sheets=values[4],
    )
    assert ambiguous["valid"] is False
    assert ambiguous["reason"] == "HEIGHT_CLUSTER_RUNNER_AMBIGUOUS"
    visual = RUNNER._renderer_member_diagnostic(
        slot="FIRST",
        diagnostic=ambiguous,
        feature_pre=np.full(3, np.nan),
        feature_post=np.full(3, np.nan),
    )
    assert visual["valid"] is False
    assert visual["invalid_reason"] == "HEIGHT_CLUSTER_RUNNER_AMBIGUOUS"
    assert sum(sum(row) for row in visual["footprint_occupancy"]) == 32
    assert visual["plane_coefficients"] is not None
    assert visual["plane_p90_mm"] is not None
    values = clustered_sheet_points(0.75)
    exact = SRC.local_surface_fit(
        **arguments,
        voxel_points=values[0], voxel_normals=values[1], voxel_keys=values[2],
        voxel_regions=values[3], voxel_sheets=values[4],
    )
    assert exact["valid"] is True
    assert exact["selected_voxel_count"] == 32


def test_whole_node_lts_is_deterministic_and_checks_retained_rank() -> None:
    rng = np.random.default_rng(9)
    design = rng.normal(size=(40, 3, 6))
    parameter = np.asarray([0.02, -0.01, 0.015, 0.0002, -0.0001, 0.0003])
    response = np.einsum("nij,j->ni", design, parameter)
    response[-6:, 0] += 2.0
    args = dict(
        design=design, response=response, area=np.ones(40),
        node_ids=np.arange(40), retained_area_fraction=0.7, refit_count=5,
        maximum_condition_number=1e4,
    )
    first, mask_first = SRC.least_trimmed_nuisance(**args)
    second, mask_second = SRC.least_trimmed_nuisance(**args)
    assert first.tobytes() == second.tobytes()
    assert mask_first.tobytes() == mask_second.tobytes()
    assert np.allclose(first, parameter, atol=1e-12)
    bad = design.copy()
    bad[:, :, 5] = bad[:, :, 4]
    try:
        SRC.least_trimmed_nuisance(**(args | {"design": bad}))
    except ValueError as error:
        assert "design gate" in str(error)
    else:
        raise AssertionError("rank-deficient retained solve did not fail")


def crossfit_fixture(scale: float = 1.0) -> tuple[dict, np.ndarray, np.ndarray]:
    nodes = synthetic_nodes(80)
    design = SRC.nuisance_design(nodes, np.zeros(3))
    parameter = scale * np.asarray([0.02, -0.015, 0.01, 0.0002, -0.00015, 0.0001])
    first = np.einsum("nij,j->ni", design, parameter)
    second = np.zeros_like(first)
    masks = [nodes["fold_id"] != fold for fold in range(4)]
    config = prereg()["global_crossfit_nuisance"] | {
        "maximum_fold_translation_dispersion_mm": 1.0,
        "maximum_fold_rotation_dispersion_deg": 1.0,
    }
    pair = prereg()["pair_actionability"] | {
        "minimum_common_support_node_count": 10,
        "minimum_common_support_area_fraction_of_cad_interior": 0.1,
        "per_fold_per_row_minimum_post_halo_training_node_count": 10,
        "per_fold_per_row_minimum_post_halo_training_area_fraction_of_that_row_common_support": 0.5,
    }
    result = SRC.crossfit_pair_field(
        nodes=nodes,
        feature_first=first,
        valid_first=np.ones(80, dtype=bool),
        feature_second=second,
        valid_second=np.ones(80, dtype=bool),
        fold_training_masks=masks,
        fixed_centroid=np.zeros(3),
        config=config,
        pair_config=pair,
    )
    return result, first, second


def test_crossfit_swap_antisymmetry_and_bound_abstention() -> None:
    result, first, second = crossfit_fixture()
    assert result["actionable"] is True
    nodes = synthetic_nodes(80)
    masks = [nodes["fold_id"] != fold for fold in range(4)]
    config = prereg()["global_crossfit_nuisance"] | {
        "maximum_fold_translation_dispersion_mm": 1.0,
        "maximum_fold_rotation_dispersion_deg": 1.0,
    }
    pair = prereg()["pair_actionability"] | {
        "minimum_common_support_node_count": 10,
        "minimum_common_support_area_fraction_of_cad_interior": 0.1,
        "per_fold_per_row_minimum_post_halo_training_node_count": 10,
        "per_fold_per_row_minimum_post_halo_training_area_fraction_of_that_row_common_support": 0.5,
    }
    swapped = SRC.crossfit_pair_field(
        nodes=nodes, feature_first=second, valid_first=np.ones(80, bool),
        feature_second=first, valid_second=np.ones(80, bool),
        fold_training_masks=masks, fixed_centroid=np.zeros(3), config=config,
        pair_config=pair,
    )
    assert swapped["actionable"] is True
    assert np.allclose(result["residual"], -swapped["residual"], atol=1e-12)
    bounded, _, _ = crossfit_fixture(scale=30.0)
    assert bounded["actionable"] is False
    assert bounded["reason"] == "NUISANCE_BOUND_GATE"


def test_localized_held_fold_bump_survives_oof_while_broad_rigid_is_removed() -> None:
    nodes = synthetic_nodes(80)
    design = SRC.nuisance_design(nodes, np.zeros(3))
    parameter = np.asarray([0.02, -0.015, 0.01, 0.0002, -0.00015, 0.0001])
    first = np.einsum("nij,j->ni", design, parameter)
    # Keep the defect below the frozen 30% LTS trimming budget in every
    # non-held training split.  Using the whole 25% fold would make it 33.3%
    # of the other folds' training population and would correctly contaminate
    # the nuisance estimate instead of testing localized-defect survival.
    bump = (nodes["fold_id"] == 0) & (np.arange(len(nodes)) < 48)
    first[bump, 0] += 0.30
    masks = [nodes["fold_id"] != fold for fold in range(4)]
    config = prereg()["global_crossfit_nuisance"] | {
        "maximum_fold_translation_dispersion_mm": 1.0,
        "maximum_fold_rotation_dispersion_deg": 1.0,
    }
    pair = prereg()["pair_actionability"] | {
        "minimum_common_support_node_count": 10,
        "minimum_common_support_area_fraction_of_cad_interior": 0.1,
        "per_fold_per_row_minimum_post_halo_training_node_count": 10,
        "per_fold_per_row_minimum_post_halo_training_area_fraction_of_that_row_common_support": 0.5,
    }
    result = SRC.crossfit_pair_field(
        nodes=nodes, feature_first=first, valid_first=np.ones(80, bool),
        feature_second=np.zeros_like(first), valid_second=np.ones(80, bool),
        fold_training_masks=masks, fixed_centroid=np.zeros(3), config=config,
        pair_config=pair,
    )
    assert result["actionable"] is True
    assert np.median(result["residual"][bump, 0]) > 0.25
    assert np.median(np.abs(result["residual"][~bump])) < 1e-8


def test_lts_retained_subset_rank_failure_is_distinct_from_initial_rank() -> None:
    rng = np.random.default_rng(77)
    design = rng.normal(size=(10, 3, 6))
    design[:7, :, 5] = 0.0
    response = np.zeros((10, 3), dtype=np.float64)
    response[7:] = np.asarray([[20.0, -15.0, 9.0], [-17.0, 12.0, 22.0], [14.0, 19.0, -11.0]])
    initial_rank, _ = SRC.normalized_design_rank_condition(design, np.ones(10))
    assert initial_rank == 6
    try:
        SRC.least_trimmed_nuisance(
            design=design, response=response, area=np.ones(10), node_ids=np.arange(10),
            retained_area_fraction=0.7, refit_count=5, maximum_condition_number=1e4,
        )
    except ValueError as error:
        assert "retained LTS design gate" in str(error)
    else:
        raise AssertionError("rank-deficient retained 70% was accepted")


def test_split_noise_exact_and_missing_is_infinite() -> None:
    rows = [np.zeros((4, 3)) for _ in range(4)]
    rows[1][:, 0] = 2.0
    rows[3][:, 1] = 4.0
    masks = [np.ones(4, dtype=bool) for _ in range(4)]
    masks[2][2] = False
    noise, complete = SRC.split_noise_from_rows(*rows, masks)
    assert noise[[0, 1, 3]].tolist() == [2.0, 2.0, 2.0]
    assert complete.tolist() == [True, True, False, True]
    assert np.isinf(noise[2])


def test_empty_partition_and_normal_cancellation_are_shape_safe() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    voxel_points, voxel_normals, voxel_keys = SRC.voxelize_partition(
        points, normals, 0.25, np.zeros(2, dtype=bool)
    )
    assert voxel_points.shape == (0, 3)
    assert voxel_normals.shape == (0, 3)
    assert voxel_keys.shape == (0, 3)
    features, valid, counts, diagnostics, selected = SRC.fit_scan_to_nodes(
        nodes=line_nodes(2), voxel_points=voxel_points, voxel_normals=voxel_normals,
        voxel_keys=voxel_keys, voxel_regions=np.empty(0, dtype=np.int64),
        voxel_sheets=np.empty(0, dtype=np.int8), config=prereg()["local_surface_fit"],
        keep_diagnostics=True,
    )
    assert features.shape == (2, 3)
    assert not valid.any() and not counts.any()
    assert not selected.any()
    assert all(row["reason"] == "EMPTY_VOXEL_PARTITION" for row in diagnostics)


def line_nodes(count: int) -> np.ndarray:
    nodes = synthetic_nodes(count)
    nodes["x"] = np.arange(count, dtype=np.float64)
    nodes["y"] = 0.0
    nodes["z"] = 0.0
    nodes["area_mm2"] = 1.0
    return nodes


def test_active_induced_graph_does_not_cross_inactive_bridge_and_medoid_is_exact() -> None:
    nodes = line_nodes(7)
    edges = np.column_stack((np.arange(6), np.arange(1, 7))).astype(np.int64)
    weights = np.ones(6)
    active = np.asarray([1, 1, 1, 0, 1, 1, 1], dtype=bool)
    components, labels = SRC.induced_components(
        active=active, nodes=nodes, edges=edges, edge_weights=weights,
        minimum_node_count=2, minimum_area_mm2=2.0, minimum_span_mm=1.0,
    )
    assert len(components) == 2
    assert components[0]["node_ids"] == [0, 1, 2]
    assert components[1]["node_ids"] == [4, 5, 6]
    assert components[0]["area_medoid_node_id"] == 1
    assert labels[3] == -1


def test_component_bijection_jaccard_extras_and_over_ten_fail_closed() -> None:
    area = np.ones(30)
    central = [{"node_ids": [0, 1]}, {"node_ids": [10, 11]}]
    variant = [{"node_ids": [10, 11]}, {"node_ids": [0, 1]}]
    assignment, matrix = SRC.canonical_component_bijection(central, variant, area)
    assert assignment == [1, 0]
    assert matrix[0, 1] == 1.0
    assert SRC.canonical_component_bijection(central, variant + [{"node_ids": [20]}], area) is None
    many = [{"node_ids": [index]} for index in range(11)]
    many_assignment, _ = SRC.canonical_component_bijection(many, many, area)
    assert many_assignment == list(range(11))
    tied = [{"node_ids": [0, 1]} for _ in range(4)]
    tied_assignment, tied_matrix = SRC.canonical_component_bijection(tied, tied, area)
    assert tied_matrix.tolist() == [[1.0] * 4] * 4
    assert tied_assignment == [0, 1, 2, 3]


def test_attribution_zero_energy_and_denominator_floor() -> None:
    component = {
        "component_id": 0, "node_ids": [0, 1], "node_ids_sha256": "x",
        "node_count": 2, "area_mm2": 2.0, "two_sweep_span_mm": 1.0,
        "area_medoid_node_id": 0,
    }
    zeros = np.zeros((2, 3))
    result = SRC.attribute_components(
        components=[dict(component)], area=np.ones(2), feature_first=zeros,
        feature_second=zeros, prediction=zeros, first_token="N009", second_token="N010",
        minimum_ratio=2.0, major_fraction=0.25, denominator_floor=1e-12,
    )
    assert result["reason"] == "ZERO_PAIRWISE_ENERGY"
    first = np.asarray([[1.0, 0, 0], [1.0, 0, 0]])
    result = SRC.attribute_components(
        components=[dict(component)], area=np.ones(2), feature_first=first,
        feature_second=zeros, prediction=zeros, first_token="N009", second_token="N010",
        minimum_ratio=2.0, major_fraction=0.25, denominator_floor=1e-12,
    )
    assert result["actionable"] is True
    assert result["attributed_opaque_member"] == "N009"


def test_exact_robustness_inventory_and_frozen_threshold_contract() -> None:
    assert SRC.robustness_row_ids() == [f"R{index:02d}" for index in range(32)]
    payload = prereg()
    assert "reuse the identical T_i" in payload["robustness_inventory"]["frozen_threshold"]
    assert payload["native_visual_contract"]["all_node_diagnostic_required_even_with_zero_retained_components"] is True
    assert list(payload["robustness_inventory"]["exact_row_ids"]) == [
        f"R{index:02d}" for index in range(32)
    ]
    expected = {
        "R00": "CENTRAL", "R01": "VOXEL_0.15", "R02": "VOXEL_0.225",
        "R03": "IDENTITY_0", "R04": "IDENTITY_1",
        "R05": "SPATIAL_0", "R06": "SPATIAL_1",
    }
    row = 7
    for member in ("FIRST", "SECOND"):
        for operation in ("TRANSLATION", "ROTATION"):
            for axis in "XYZ":
                for sign in ("NEG", "POS"):
                    expected[f"R{row:02d}"] = f"{member}_{operation}_{axis}_{sign}"
                    row += 1
    expected["R31"] = "MEMBER_SWAP"
    assert payload["robustness_inventory"]["exact_row_ids"] == expected


def test_split_star_one_sided_area_is_recomputed_and_rejected() -> None:
    nodes = synthetic_nodes(80)
    config = prereg()["pair_actionability"] | {
        "minimum_common_support_node_count": 10,
        "minimum_common_support_area_fraction_of_cad_interior": 0.1,
    }
    passed, metrics = RUNNER._support_gate_for_split_complete(
        first_support=np.ones(80, dtype=bool),
        second_support=np.arange(80) < 60,
        nodes=nodes,
        fixed_centroid=np.zeros(3),
        pair_config=config,
    )
    assert passed is False
    assert metrics["one_sided_support_area_fraction"] == 0.25


def test_runtime_member_swap_contract_detects_corrupted_amplitude() -> None:
    prediction = np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    residual = np.asarray([[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]])
    first_row = {
        "first_token": "N009", "second_token": "N010",
        "pair": {
            "actionable": True, "common": np.ones(2, bool),
            "fold_rows": [{"parameters": [0.1] * 6}] * 4,
            "prediction": prediction, "residual": residual,
        }
    }
    second_row = {
        "first_token": "N010", "second_token": "N009",
        "pair": {
            "actionable": True, "common": np.ones(2, bool),
            "fold_rows": [{"parameters": [-0.1] * 6}] * 4,
            "prediction": -prediction, "residual": -residual,
        }
    }
    component = {
        "node_ids": [0, 1], "area_mm2": 2.0,
        "first_member_energy_mm2": 4.0,
        "second_member_energy_mm2": 1.0,
    }
    swapped_component = dict(component) | {
        "first_member_energy_mm2": 1.0,
        "second_member_energy_mm2": 4.0,
    }
    first_components = {
        "components": [component], "amplitude": np.asarray([0.3, 0.4]),
        "active": np.ones(2, bool), "attributed_opaque_member": "N009",
    }
    second_components = {
        "components": [swapped_component], "amplitude": np.asarray([0.3, 0.4]),
        "active": np.ones(2, bool), "attributed_opaque_member": "N009",
    }
    assert RUNNER._member_swap_contract(
        central_row=first_row, swap_row=second_row,
        central_components=first_components, swap_components=second_components,
    )["passed"] is True
    second_components["amplitude"] = np.asarray([0.3, 0.41])
    assert RUNNER._member_swap_contract(
        central_row=first_row, swap_row=second_row,
        central_components=first_components, swap_components=second_components,
    )["passed"] is False
    second_components["amplitude"] = np.asarray([0.3, 0.4])
    second_components["components"][0]["first_member_energy_mm2"] = 1.01
    energy_failure = RUNNER._member_swap_contract(
        central_row=first_row, swap_row=second_row,
        central_components=first_components, swap_components=second_components,
    )
    assert energy_failure["passed"] is False
    assert energy_failure["checks"]["per_component_member_energies_exchange"] is False
    second_components["components"][0]["first_member_energy_mm2"] = 1.0
    second_row["first_token"], second_row["second_token"] = "N009", "N010"
    slot_failure = RUNNER._member_swap_contract(
        central_row=first_row, swap_row=second_row,
        central_components=first_components, swap_components=second_components,
    )
    assert slot_failure["passed"] is False
    assert slot_failure["checks"]["winning_slot_flips_before_token_remap"] is False
    second_row["first_token"], second_row["second_token"] = "N010", "N009"
    first_components["attributed_opaque_member"] = None
    second_components["attributed_opaque_member"] = None
    null_winner_failure = RUNNER._member_swap_contract(
        central_row=first_row, swap_row=second_row,
        central_components=first_components, swap_components=second_components,
    )
    assert null_winner_failure["passed"] is False
    assert null_winner_failure["checks"]["winning_slot_flips_before_token_remap"] is False
    assert null_winner_failure["checks"]["opaque_token_equal_after_remap"] is False


def test_energy_drift_does_not_create_heavy_visual_obligation_but_geometry_does() -> None:
    component = {
        "component_id": 0,
        "node_ids": [0, 1, 2, 3],
        "node_ids_sha256": "a" * 64,
        "area_medoid_node_id": 1,
        "first_member_energy_mm2": 4.0,
        "second_member_energy_mm2": 1.0,
        "energy_ratio": 4.0,
        "pair_energy_mm4": 12.0,
        "pair_energy_fraction": 1.0,
        "higher_energy_opaque_member": "N009",
    }
    central = {
        "status": "ACTIONABLE",
        "components": [component],
        "attributed_opaque_member": "N009",
    }
    variant_component = dict(component) | {
        "first_member_energy_mm2": 4.000000000000001,
        "energy_ratio": 4.000000000000001,
    }
    variant = {
        "status": "ACTIONABLE",
        "components": [variant_component],
        "attributed_opaque_member": "N009",
    }
    comparison = {
        "passed": True,
        "reason": None,
        "matches": [
            {
                "central_component_id": 0,
                "variant_component_id": 0,
                "area_weighted_jaccard": 1.0,
                "medoid_graph_geodesic_displacement_mm": 0.0,
                "passed": True,
            }
        ],
    }
    row = {"first_token": "N009", "second_token": "N010"}
    unchanged = RUNNER._robustness_row_change(
        central=central, variant=central, comparison=comparison,
        central_row=row, variant_row=row,
    )
    assert unchanged == {"row_changed": False, "row_change_reasons": []}
    energy_only = RUNNER._robustness_row_change(
        central=central, variant=variant, comparison=comparison,
        central_row=row, variant_row=row,
    )
    assert energy_only == {"row_changed": False, "row_change_reasons": []}

    geometry_comparison = dict(comparison) | {
        "matches": [
            dict(comparison["matches"][0])
            | {
                "area_weighted_jaccard": 0.82,
                "medoid_graph_geodesic_displacement_mm": 0.4,
            }
        ]
    }
    geometry_changed = RUNNER._robustness_row_change(
        central=central, variant=variant, comparison=geometry_comparison,
        central_row=row, variant_row=row,
    )
    assert geometry_changed["row_changed"] is True
    assert geometry_changed["row_change_reasons"] == [
        "COMPONENT_JACCARD_CHANGED", "COMPONENT_MEDOID_CHANGED"
    ]


def test_r31_valid_member_swap_is_not_row_changed_until_opaque_decision_changes() -> None:
    component = {
        "component_id": 0,
        "node_ids": [0, 1, 2],
        "node_ids_sha256": "b" * 64,
        "area_medoid_node_id": 1,
        "higher_energy_opaque_member": "OPAQUE_A",
    }
    central = {
        "status": "ACTIONABLE", "components": [component],
        "attributed_opaque_member": "OPAQUE_A",
    }
    swap = {
        "status": "ACTIONABLE", "components": [dict(component)],
        "attributed_opaque_member": "OPAQUE_A",
    }
    comparison = {
        "passed": True,
        "reason": None,
        "matches": [{
            "central_component_id": 0,
            "variant_component_id": 0,
            "area_weighted_jaccard": 1.0,
            "medoid_graph_geodesic_displacement_mm": 0.0,
            "passed": True,
        }],
    }
    central_row = {"first_token": "OPAQUE_A", "second_token": "OPAQUE_B"}
    swap_row = {"first_token": "OPAQUE_B", "second_token": "OPAQUE_A"}
    unchanged = RUNNER._robustness_row_change(
        central=central, variant=swap, comparison=comparison,
        central_row=central_row, variant_row=swap_row,
    )
    assert unchanged == {"row_changed": False, "row_change_reasons": []}

    wrong_component = dict(component) | {"higher_energy_opaque_member": "OPAQUE_B"}
    wrong_swap = {
        "status": "ACTIONABLE", "components": [wrong_component],
        "attributed_opaque_member": "OPAQUE_B",
    }
    changed = RUNNER._robustness_row_change(
        central=central, variant=wrong_swap, comparison=comparison,
        central_row=central_row, variant_row=swap_row,
    )
    assert changed["row_changed"] is True
    assert changed["row_change_reasons"] == [
        "COMPONENT_ATTRIBUTED_TOKEN_CHANGED", "ROW_ATTRIBUTED_TOKEN_CHANGED"
    ]


def test_renderer_diagnostic_invalid_no_winner_preserves_null_fit_values() -> None:
    row = RUNNER._renderer_member_diagnostic(
        slot="FIRST",
        diagnostic={
            "valid": False,
            "reason": "NO_ELIGIBLE_HEIGHT_CLUSTER",
            "clusters": [],
        },
        feature_pre=np.full(3, np.nan),
        feature_post=np.full(3, np.nan),
    )
    assert row["valid"] is False
    assert row["invalid_reason"] == "NO_ELIGIBLE_HEIGHT_CLUSTER"
    assert row["winner_cluster_index"] is None
    assert row["plane_coefficients"] is None
    assert row["plane_p90_mm"] is None
    assert row["feature_pre_nuisance"] == [None, None, None]
    assert row["feature_post_nuisance"] == [None, None, None]


def test_visual_sheet_ownership_is_nullable_not_zero_sentinel() -> None:
    assert RUNNER.nullable_sheet_ids(
        np.asarray([-1, 0, 1], dtype=np.int8),
        np.asarray([True, False, True]),
    ) == [-1, None, 1]
    for sheets, valid in (
        (np.asarray([0]), np.asarray([True])),
        (np.asarray([2]), np.asarray([True])),
        (np.asarray([1, -1]), np.asarray([True])),
    ):
        try:
            RUNNER.nullable_sheet_ids(sheets, valid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sheet ownership was serialized")


def test_source_runner_has_no_forbidden_cli_and_approval_precedes_scan_hash() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in ("--evaluator", "--role", "--label", "--gold", "--expected", "--quality"):
        assert forbidden not in source
    approval_position = source.index("verify_pre_scan_approvals(", source.index("def main"))
    transform_position = source.index("transform_first = load_transform(", source.index("def main"))
    cad_load_position = source.index(
        "nodes, edges, edge_weights, mesh = load_cad_source_artifacts_before_scan(",
        source.index("def main"),
    )
    scan_hash_position = source.index("_require_hash(args.scan_first", source.index("def main"))
    assert approval_position < transform_position < cad_load_position < scan_hash_position
    assert '"physical_scan_hash_call_count": 2' in source
    assert '"physical_scan_point_cloud_load_call_count": 2' in source
    assert '"physical_scan_byte_open_operation_count": 4' in source
    assert '"physical_scan_open_call_count"' not in source


def test_approval_headers_reject_substring_and_wrong_physical_authority() -> None:
    valid_source = {
        "schema": "EXP_CAD_0030_V3_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1",
        "experiment_id": "EXP_CAD_0030",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
        "decision": "APPROVE_ONE_PHYSICAL_PRIMARY_REPLAY_EXECUTION",
        "physical_source_execution_authorized": True,
    }
    RUNNER.validate_source_approval_header(valid_source)
    for mutation in (
        {"decision": "NOT_APPROVED_ONE_PHYSICAL_PRIMARY_REPLAY_EXECUTION"},
        {"physical_source_execution_authorized": False},
        {"schema": "EXP_CAD_0029_SOURCE_APPROVAL"},
        {"experiment_id": "EXP_CAD_0029"},
        {"architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V2_HARDENED"},
    ):
        try:
            RUNNER.validate_source_approval_header(valid_source | mutation)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid source approval header was accepted")
    valid_cad = {
        "schema": "EXP_CAD_0030_V3_INDEPENDENT_CAD_ONLY_APPROVAL_V1",
        "experiment_id": "EXP_CAD_0030",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
        "decision": "PASS_CAD_ONLY_FREEZE_FOR_SOURCE_RUNNER_IMPLEMENTATION",
        "physical_source_execution_authorized": False,
    }
    RUNNER.validate_cad_approval_header(valid_cad)
    for mutation in (
        {"decision": "NOT_APPROVED_CAD"},
        {"physical_source_execution_authorized": True},
        {"experiment_id": "EXP_CAD_0029"},
    ):
        try:
            RUNNER.validate_cad_approval_header(valid_cad | mutation)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid CAD approval header was accepted")


def test_cad_preflight_identity_rejects_exp29_shaped_pass_and_stale_prereg() -> None:
    prereg_sha = "c" * 64
    valid = {
        "schema": "EXP_CAD_0030_CFPDG_V3_CAD_ONLY_PREFLIGHT_V1",
        "experiment_id": "EXP_CAD_0030",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
        "preregistration_sha256": prereg_sha,
        "status": "PASS_CAD_ONLY_ACTIONABLE",
        "cad_actionable": True,
        "source_execution_authorized": False,
    }
    RUNNER.validate_cad_preflight_identity(valid, prereg_sha)
    for mutation in (
        {"schema": "EXP_CAD_0029_CAD_ONLY_PREFLIGHT_V2"},
        {"experiment_id": "EXP_CAD_0029"},
        {"architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V2_HARDENED"},
        {"preregistration_sha256": "d" * 64},
        {"status": "PASS_EXP29_SHAPED"},
        {"cad_actionable": False},
        {"source_execution_authorized": True},
    ):
        try:
            RUNNER.validate_cad_preflight_identity(valid | mutation, prereg_sha)
        except ValueError:
            pass
        else:
            raise AssertionError("stale or wrong-identity CAD preflight was accepted")


def test_approval_hash_lookup_requires_exact_named_keys_and_shapes() -> None:
    exact = "a" * 64
    assert RUNNER._bound_file_sha(
        {"bound_files": {"renderer": {"sha256": exact}}}, "renderer"
    ) == exact
    assert RUNNER._bound_artifact_sha(
        {"bound_cad_artifacts": {"CAD_GRAPH_NODES.npy": exact}},
        "CAD_GRAPH_NODES.npy",
    ) == exact
    for payload in (
        {"renderer_sha256": exact},
        {"bound_files": {"some_recursive_renderer": {"sha256": exact}}},
        {"bound_files": {"renderer": exact}},
        {"bound_files": {"renderer": {"sha256": "NOT_APPROVED_" + exact}}},
    ):
        try:
            RUNNER._bound_file_sha(payload, "renderer")
        except ValueError:
            pass
        else:
            raise AssertionError("non-exact renderer hash binding was accepted")
    for payload in (
        {"CAD_GRAPH_NODES.npy": exact},
        {"bound_cad_artifacts": {"nested": {"CAD_GRAPH_NODES.npy": exact}}},
        {"bound_cad_artifacts": {"CAD_GRAPH_NODES.npy": {"sha256": exact}}},
    ):
        try:
            RUNNER._bound_artifact_sha(payload, "CAD_GRAPH_NODES.npy")
        except ValueError:
            pass
        else:
            raise AssertionError("non-exact CAD artifact binding was accepted")


def test_approval_namespaces_require_exact_frozen_key_sets() -> None:
    digest = "d" * 64
    cad = {
        "bound_files": {
            key: {"sha256": digest}
            for key in RUNNER.CAD_APPROVAL_BOUND_FILE_KEYS
        },
        "bound_cad_artifacts": {
            key: digest for key in RUNNER.CAD_APPROVAL_ARTIFACT_KEYS
        },
    }
    RUNNER._require_exact_approval_key_sets(
        cad,
        bound_file_keys=RUNNER.CAD_APPROVAL_BOUND_FILE_KEYS,
        require_cad_artifacts=True,
    )
    source = {
        "bound_files": {
            key: {"sha256": digest}
            for key in RUNNER.SOURCE_APPROVAL_BOUND_FILE_KEYS
        }
    }
    RUNNER._require_exact_approval_key_sets(
        source,
        bound_file_keys=RUNNER.SOURCE_APPROVAL_BOUND_FILE_KEYS,
        require_cad_artifacts=False,
    )
    cad_mutations = [
        cad | {
            "bound_files": dict(cad["bound_files"])
            | {"surprise": {"sha256": digest}}
        },
        cad | {
            "bound_files": {
                key: value
                for key, value in cad["bound_files"].items()
                if key != "mesh"
            }
        },
        cad | {
            "bound_cad_artifacts": dict(cad["bound_cad_artifacts"])
            | {"EXTRA.npy": digest}
        },
    ]
    for mutation in cad_mutations:
        with np.testing.assert_raises(ValueError):
            RUNNER._require_exact_approval_key_sets(
                mutation,
                bound_file_keys=RUNNER.CAD_APPROVAL_BOUND_FILE_KEYS,
                require_cad_artifacts=True,
            )
    with np.testing.assert_raises(ValueError):
        RUNNER._require_exact_approval_key_sets(
            source | {"bound_cad_artifacts": {}},
            bound_file_keys=RUNNER.SOURCE_APPROVAL_BOUND_FILE_KEYS,
            require_cad_artifacts=False,
        )


def test_source_free_producer_payload_is_renderer_valid_and_deterministic() -> None:
    nodes = line_nodes(8)
    nodes["x"] = np.tile(np.asarray([2.0, 6.0, 10.0, 14.0]), 2)
    nodes["y"] = np.repeat(np.asarray([2.0, 6.0]), 4)
    nodes["z"] = 0.0
    nodes["nx"] = 0.0; nodes["ny"] = 0.0; nodes["nz"] = 1.0
    nodes["e1x"] = 1.0; nodes["e1y"] = 0.0; nodes["e1z"] = 0.0
    nodes["e2x"] = 0.0; nodes["e2y"] = 1.0; nodes["e2z"] = 0.0
    nodes["fold_id"] = np.arange(8) % 4
    nodes["smooth_region"] = 0
    edges = np.column_stack((np.arange(7), np.arange(1, 8))).astype(np.int64)
    edge_weights = np.linalg.norm(
        np.column_stack((nodes["x"], nodes["y"], nodes["z"]))[edges[:, 0]]
        - np.column_stack((nodes["x"], nodes["y"], nodes["z"]))[edges[:, 1]],
        axis=1,
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.asarray([[0, 0, 0], [16, 0, 0], [16, 8, 0], [0, 8, 0]], dtype=float)
        ),
        o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
    )
    grid = np.asarray(
        [[x, y, 0.02 * np.sin(x + y)] for x in np.linspace(0.2, 15.8, 65) for y in np.linspace(0.2, 7.8, 33)],
        dtype=np.float64,
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(grid), 1))
    config = prereg()
    config["local_surface_fit"]["minimum_points"] = 4
    config["local_surface_fit"]["minimum_occupied_footprint_cells"] = 4
    config["pair_actionability"]["minimum_common_support_node_count"] = 4
    config["pair_actionability"]["minimum_common_support_area_fraction_of_cad_interior"] = 0.1
    config["pair_actionability"]["per_fold_per_row_minimum_post_halo_training_node_count"] = 1
    config["pair_actionability"]["per_fold_per_row_minimum_post_halo_training_area_fraction_of_that_row_common_support"] = 0.0
    cad_preflight = {
        "cad_metrics": {"cad_interior_area_weighted_centroid_mm": [8.0, 4.0, 0.0]}
    }
    arguments = dict(
        prereg=config, cad_preflight=cad_preflight, nodes=nodes, edges=edges,
        edge_weights=edge_weights, mesh=mesh, raw_first=grid, normals_first=normals,
        raw_second=grid + np.asarray([0.0, 0.0, 0.03]), normals_second=normals,
        transform_first=np.eye(4), transform_second=np.eye(4),
        first_token="N009", second_token="N010",
    )
    semantic_first, payload_first = RUNNER.execute_source(**arguments)
    semantic_second, payload_second = RUNNER.execute_source(**arguments)
    assert json.dumps(semantic_first, sort_keys=True, default=str) == json.dumps(semantic_second, sort_keys=True, default=str)
    assert json.dumps(payload_first, sort_keys=True, default=str) == json.dumps(payload_second, sort_keys=True, default=str)
    assert semantic_first["schema"] == "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1"
    assert payload_first["schema"] == "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1"
    visual_only = {
        "variant_components", "member_support",
        "changed_component_local_diagnostics", "changed_visual_components",
    }
    assert all(not (visual_only & set(row)) for row in semantic_first["robustness"])
    assert all(
        row["visual_only_fields_omitted_from_semantic_result"] == sorted(visual_only)
        for row in semantic_first["robustness"]
    )
    assert all(visual_only <= set(row) for row in payload_first["robustness"]["rows"])
    semantic_bytes = json.dumps(semantic_first, sort_keys=True, default=str).encode()
    visual_bytes = json.dumps(payload_first, sort_keys=True, default=str).encode()
    assert len(semantic_bytes) < len(visual_bytes)
    assert len(semantic_bytes) < 1_000_000
    normalized = RENDER.validate_payload(payload_first)
    assert normalized["pair_id"] == "PAIR_N009_N010"
    assert payload_first["robustness"]["semantic_row_count"] == 32
    rows = payload_first["robustness"]["rows"]
    assert [row["row_id"] for row in rows] == [f"R{index:02d}" for index in range(32)]
    assert [row["row_descriptor"] for row in rows] == [
        config["robustness_inventory"]["exact_row_ids"][f"R{index:02d}"]
        for index in range(32)
    ]
    if semantic_first["central_component_count"] == 0:
        assert all(row["passed"] is False for row in rows)
        assert payload_first["global"]["robustness_pass"] is False
    for node in payload_first["nodes"]:
        for key in ("amplitude_mm", "split_noise_mm", "threshold_mm"):
            value = node[key]
            assert value is None or (np.isfinite(value) and abs(value) < 1e8)
        if node["valid"] is False:
            assert node["invalid_reason"]
            assert any(node[key] is None for key in ("amplitude_mm", "split_noise_mm", "threshold_mm"))
        for diagnostic in node["member_diagnostics"]:
            if diagnostic["valid"] is False:
                assert diagnostic["invalid_reason"]
            if diagnostic["winner_cluster_index"] is None:
                assert diagnostic["plane_coefficients"] is None
                assert diagnostic["plane_p90_mm"] is None


def test_real_cad_scale_semantic_robustness_omits_all_visual_component_payloads() -> None:
    node_ids = list(range(8077))
    rows = []
    for index in range(32):
        rows.append(
            {
                "row_id": f"R{index:02d}",
                "status": "ACTIONABLE",
                "passed": True,
                "reason": None,
                "component_count": 1,
                "component_hashes": ["a" * 64],
                "matches": [{
                    "central_component_id": 0,
                    "variant_component_id": 0,
                    "area_weighted_jaccard": 1.0,
                    "medoid_graph_geodesic_displacement_mm": 0.0,
                    "passed": True,
                }],
                "attributed_opaque_member": "OPAQUE_A",
                "row_descriptor": "CENTRAL" if index == 0 else "ROBUSTNESS_VARIANT",
                "row_changed": index == 7,
                "row_change_reasons": ["COMPONENT_JACCARD_CHANGED"] if index == 7 else [],
                "variant_components": [{
                    "component_id": 0,
                    "node_ids": node_ids,
                    "node_ids_sha256": "a" * 64,
                    "component_labels": node_ids,
                }],
                "member_support": [{"voxel_xyz_mm": [[float(n), 0.0, 0.0] for n in node_ids]}],
                "changed_component_local_diagnostics": [{"node_id": n} for n in node_ids],
                "changed_visual_components": [{"component_id": "V", "node_ids": node_ids}],
            }
        )
    semantic = RUNNER.semantic_robustness_rows(rows)
    omitted = sorted(RUNNER.VISUAL_ONLY_ROBUSTNESS_FIELDS)
    assert all(not (set(row) & set(omitted)) for row in semantic)
    assert all(
        row["visual_only_fields_omitted_from_semantic_result"] == omitted
        for row in semantic
    )
    assert semantic[7]["row_changed"] is True
    assert semantic[7]["row_change_reasons"] == ["COMPONENT_JACCARD_CHANGED"]
    semantic_size = len(json.dumps(semantic, sort_keys=True).encode())
    visual_size = len(json.dumps(rows, sort_keys=True).encode())
    assert semantic_size < 100_000
    assert semantic_size * 50 < visual_size
