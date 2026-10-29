from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from scripts import exp_cad_0029_cfpdg_source as PARENT
from scripts import exp_cad_0031_source_interface as PARENT_INTAKE
from scripts import run_exp_cad_0029_cfpdg_source as PARENT_RUNNER
from scripts import run_exp_cad_0031_cfpdg_source as V4_RUNNER
from scripts import exp_cad_0032_cfpdg_v5_engine as V5
from scripts import run_exp_cad_0032_cfpdg_source_free as V5_RUNNER


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs/research/2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json"
V5_PREREG = ROOT / "docs/research/2026-08-30_exp_cad_0032_cfpdg_v5_execution_equivalent_preregistration.json"


def _config() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _node_dtype() -> np.dtype:
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


def _line_nodes(count: int) -> np.ndarray:
    nodes = np.zeros(count, dtype=_node_dtype())
    nodes["x"] = np.arange(count, dtype=np.float64)
    nodes["nz"] = 1.0
    nodes["e1x"] = 1.0
    nodes["e2y"] = 1.0
    nodes["area_mm2"] = 1.0
    nodes["fold_id"] = np.arange(count) % 4
    nodes["smooth_region"] = 0
    nodes["parent_face"] = np.arange(count)
    return nodes


def _canonical(value):
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clustered_sheet_points(second_height: float) -> tuple[np.ndarray, ...]:
    angles = (np.arange(16) + 0.5) * 2.0 * np.pi / 16.0
    uv = np.vstack(
        (
            np.column_stack((np.cos(angles), np.sin(angles))),
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


def test_preregistration_binds_terminal_parent_and_production_envelope() -> None:
    payload = json.loads(V5_PREREG.read_text(encoding="utf-8"))
    expected_science = {
        "cad_node_count": 8077,
        "cad_edge_count": 19267,
        "unique_member_fit_count": 38,
        "robustness_row_ids": "R00-R31 exactly",
    }
    assert {
        key: payload["immutable_science"][key] for key in expected_science
    } == expected_science
    envelope = payload["source_independent_production_scale_envelope"]
    assert envelope["retained_rows_per_member"] == 1_000_000
    assert envelope["fresh_source_child_run_count"] == 3
    assert envelope["maximum_wall_seconds_per_source_child"] == 150
    assert envelope["maximum_peak_rss_bytes_per_source_child"] == 4 * 1024**3
    lineage = payload["lineage"]
    for key in (
        "terminal_parent_result_sha256",
        "terminal_parent_orchestration_sha256",
        "terminal_parent_durable_journal_sha256",
        "terminal_parent_report_sha256",
    ):
        assert len(lineage[key]) == 64


def test_vector_normal_canonicalization_is_byte_exact_for_ties_and_signed_zero() -> None:
    rng = np.random.default_rng(20260830)
    random = rng.normal(size=(4096, 3))
    special = np.asarray(
        [
            [1.0, 1.0, -0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -2.0, 2.0],
            [-0.0, 0.0, -1.0],
            [3.0, -3.0, 3.0],
            [-3.0, -3.0, -3.0],
        ]
    )
    normals = np.vstack((special, random))
    expected = PARENT.canonical_unoriented_normals(normals)
    actual = V5.canonical_unoriented_normals_v5(normals)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert not np.any(np.signbit(actual[actual == 0.0]))


def test_uint8_intake_is_report_and_array_exact_for_arange_and_carried_ids() -> None:
    count = 512
    index = np.arange(count, dtype=np.float64)
    points = np.column_stack((index, np.sin(index), np.cos(index)))
    normals = np.column_stack((np.ones(count), 0.01 * index, np.ones(count)))
    points[7, 0] = np.nan
    normals[31, 1] = np.inf
    normals[119] = 0.0
    for ids in (None, np.arange(count, dtype=np.uint64)[::-1]):
        expected = PARENT_INTAKE.filter_source_rows(
            points, normals, member_token="SYNTHETIC", original_row_ids=ids
        )
        actual = V5.filter_source_rows_v5(
            points, normals, member_token="SYNTHETIC", original_row_ids=ids
        )
        for key in ("points", "normals", "original_row_ids"):
            assert actual[key].dtype == expected[key].dtype
            assert actual[key].tobytes() == expected[key].tobytes()
        assert actual["report"] == expected["report"]


def test_full_source_free_nonactionable_interface_matches_v4_without_science_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 102_000
    index = np.arange(count, dtype=np.float64)
    points = np.column_stack((0.001 * index, np.sin(index), np.cos(index)))
    normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    normals[-1021:] = 0.0  # >1% invalid while retaining more than 100k rows.
    clean_normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    parent_prereg = _config()
    v4_prereg = json.loads(
        (ROOT / "docs/research/2026-08-30_exp_cad_0031_cfpdg_v4_preregistration.json").read_text()
    )
    v5_prereg = json.loads(V5_PREREG.read_text())
    common = dict(
        parent_prereg=parent_prereg,
        cad_preflight={},
        nodes=np.empty(0, dtype=_node_dtype()),
        edges=np.empty((0, 2), dtype=np.int64),
        edge_weights=np.empty(0),
        mesh=None,
        raw_first=points,
        normals_first=normals,
        raw_second=points,
        normals_second=clean_normals,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="N009",
        second_token="N010",
    )
    expected_semantic, expected_visual, expected_filtered = V4_RUNNER.execute_source_v4(
        v4_prereg=v4_prereg, **common
    )

    def forbidden_science(**_kwargs):
        raise AssertionError("nonactionable interface reached parent science")

    monkeypatch.setattr(V5_RUNNER, "execute_parent_science_v5", forbidden_science)
    semantic, visual, filtered, metrics = V5_RUNNER.execute_source_v5(
        v5_prereg=v5_prereg, **common
    )
    assert _canonical(semantic) == _canonical(expected_semantic)
    assert visual is expected_visual is None
    assert filtered["first"]["report"] == PARENT_INTAKE.filter_source_rows(
        points,
        normals,
        member_token="N009",
        original_row_ids=np.arange(count, dtype=np.uint64),
    )["report"]
    assert filtered["first"]["original_row_ids"].tobytes() == expected_filtered[
        "first_original_row_ids"
    ].tobytes()
    assert filtered["second"]["original_row_ids"].tobytes() == expected_filtered[
        "second_original_row_ids"
    ].tobytes()
    assert metrics["scientific_execute_source_reached"] is False
    assert metrics["operation_counters"]["unique_member_fit_count"] == 0


def test_batched_kd_and_compact_ambiguous_local_path_match_parent() -> None:
    config = _config()["local_surface_fit"]
    values = _clustered_sheet_points(0.5)
    nodes = _line_nodes(1)
    nodes["x"] = 0.0
    arguments = dict(
        nodes=nodes,
        voxel_points=values[0],
        voxel_normals=values[1],
        voxel_keys=values[2],
        voxel_regions=values[3],
        voxel_sheets=values[4],
        config=config,
    )
    expected = PARENT.fit_scan_to_nodes(**arguments, keep_diagnostics=False)
    engine = V5.V5Engine(expected_node_count=1)
    engine._member_fit_depth = 1
    actual = V5.fit_scan_to_nodes_v5(
        engine=engine, **arguments, keep_diagnostics=False
    )
    for first, second in zip(expected[:3], actual[:3], strict=True):
        assert first.dtype == second.dtype
        assert first.tobytes() == second.tobytes()
    assert expected[3] == actual[3]
    assert expected[4].tobytes() == actual[4].tobytes()
    assert not actual[1][0]
    assert engine.counters["batched_kd_fit_count"] == 1
    assert engine.counters["compact_local_fit_call_count"] == 1

    diagnostic_expected = PARENT.fit_scan_to_nodes(
        **arguments, keep_diagnostics=True
    )
    diagnostic_actual = V5.fit_scan_to_nodes_v5(
        engine=V5.V5Engine(expected_node_count=1),
        **arguments,
        keep_diagnostics=True,
    )
    assert _canonical(diagnostic_actual) == _canonical(diagnostic_expected)
    assert diagnostic_actual[3][0]["reason"] == "HEIGHT_CLUSTER_RUNNER_AMBIGUOUS"


def test_exact_boundary_duplicate_kd_neighbor_ids_match_scalar_parent_query() -> None:
    assert V5._kd_tie_self_check() is True
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    scalar = np.asarray(sorted(tree.query_ball_point(np.zeros(3), 1.0)))
    batch = np.sort(
        np.asarray(tree.query_ball_point(np.zeros((1, 3)), 1.0, return_sorted=True)[0]),
        kind="mergesort",
    )
    assert scalar.tobytes() == batch.tobytes()


def test_nuisance_cache_is_byte_exact_and_does_not_change_design() -> None:
    nodes = _line_nodes(24)
    centroid = np.asarray([3.25, -1.5, 0.75])
    engine = V5.V5Engine(expected_node_count=len(nodes))
    expected = PARENT.nuisance_design(nodes, centroid)
    first = V5.nuisance_design_v5(engine, nodes, centroid)
    second = V5.nuisance_design_v5(engine, nodes, centroid)
    assert first.tobytes() == expected.tobytes()
    assert second.tobytes() == expected.tobytes()
    assert engine.counters["nuisance_design_build_count"] == 1
    assert engine.counters["nuisance_design_cache_hit_count"] == 1


def test_streamed_component_medoid_is_exact_on_equal_cost_tie_and_cache_copy() -> None:
    nodes = _line_nodes(4)
    nodes["area_mm2"] = 1.0
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    weights = np.ones(4, dtype=np.float64)
    arguments = dict(
        active=np.ones(4, dtype=bool),
        nodes=nodes,
        edges=edges,
        edge_weights=weights,
        minimum_node_count=1,
        minimum_area_mm2=0.0,
        minimum_span_mm=0.0,
    )
    expected = PARENT.induced_components(**arguments)
    engine = V5.V5Engine(expected_node_count=4)
    actual = V5.induced_components_v5(engine=engine, **arguments)
    assert _canonical(actual) == _canonical(expected)
    assert actual[0][0]["area_medoid_node_id"] == 0
    actual[0][0]["area_medoid_node_id"] = 999
    replay = V5.induced_components_v5(engine=engine, **arguments)
    assert replay[0][0]["area_medoid_node_id"] == 0
    assert engine.counters["induced_components_cache_hit_count"] == 1
    assert engine.counters["component_streamed_dijkstra_row_count"] == 4


def test_streamed_component_medoid_crosses_two_batch_boundaries_exactly() -> None:
    count = 257
    nodes = _line_nodes(count)
    edges = np.column_stack((np.arange(count - 1), np.arange(1, count))).astype(np.int64)
    weights = np.ones(count - 1, dtype=np.float64)
    arguments = dict(
        active=np.ones(count, dtype=bool),
        nodes=nodes,
        edges=edges,
        edge_weights=weights,
        minimum_node_count=1,
        minimum_area_mm2=0.0,
        minimum_span_mm=0.0,
    )
    expected = PARENT.induced_components(**arguments)
    engine = V5.V5Engine(expected_node_count=count)
    actual = V5.induced_components_v5(engine=engine, **arguments)
    assert _canonical(actual) == _canonical(expected)
    assert actual[0][0]["area_medoid_node_id"] == 128
    assert engine.counters["component_streamed_dijkstra_row_count"] == 257
    assert engine.counters["component_dijkstra_batch_call_count"] == 3


def test_missing_runner_dijkstra_fails_unbound_and_scoped_binding_restores_namespace() -> None:
    nodes = _line_nodes(2)
    edges = np.asarray([[0, 1]], dtype=np.int64)
    graph = PARENT.CAD.weighted_node_graph(2, edges, np.ones(1))
    component = {
        "component_id": 0,
        "node_ids": [0, 1],
        "area_medoid_node_id": 0,
        "energy_ratio": 3.0,
    }
    row = {"components": [component], "attributed_opaque_member": "S000"}
    if hasattr(PARENT_RUNNER, "dijkstra"):
        delattr(PARENT_RUNNER, "dijkstra")
    with pytest.raises(NameError, match="dijkstra"):
        PARENT_RUNNER._robustness_compare(
            central=row,
            variant=row,
            nodes=nodes,
            full_graph=graph,
            prereg=_config(),
        )
    assert not hasattr(PARENT_RUNNER, "dijkstra")
    with V5.optimized_science_context(V5.V5Engine(expected_node_count=2)):
        result = PARENT_RUNNER._robustness_compare(
            central=row,
            variant=row,
            nodes=nodes,
            full_graph=graph,
            prereg=_config(),
        )
        assert result["passed"] is True
        assert hasattr(PARENT_RUNNER, "dijkstra")
    assert not hasattr(PARENT_RUNNER, "dijkstra")


def test_raycast_path_is_identical_on_shared_edge_tie() -> None:
    vertices = np.asarray(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float64
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    tensor = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.int64)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor)
    points = np.asarray([[1.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1.5, 1.5, 0.0]])
    regions = np.asarray([11, 29], dtype=np.int64)
    first = PARENT_RUNNER.raycast_regions_and_sheets(scene, points, regions)
    second = PARENT_RUNNER.raycast_regions_and_sheets(scene, points, regions)
    assert _canonical(first) == _canonical(second)
    assert V5.V5Engine().metrics()["equivalence_checks"][
        "raycast_tie_differential_passed"
    ] is True


def _full_branch_arguments(*, two_sheet: bool, changed_component: bool) -> dict:
    count = 80
    nodes = _line_nodes(count)
    nodes["x"] = (np.arange(count) % 10) * 8.0 + 4.0
    nodes["y"] = (np.arange(count) // 10) * 8.0 + 4.0
    rng = np.random.default_rng(9917)
    node_normals = rng.normal(size=(count, 3))
    node_normals[:, 2] = np.abs(node_normals[:, 2]) + 0.4
    node_normals /= np.linalg.norm(node_normals, axis=1)[:, None]
    first_tangent, second_tangent = PARENT.CAD.canonical_tangent_basis(node_normals)
    for names, values in (
        (("nx", "ny", "nz"), node_normals),
        (("e1x", "e1y", "e1z"), first_tangent),
        (("e2x", "e2y", "e2z"), second_tangent),
    ):
        for axis, name in enumerate(names):
            nodes[name] = values[:, axis]
    nodes["area_mm2"] = 3.0
    edges = np.column_stack((np.arange(count - 1), np.arange(1, count))).astype(np.int64)
    xyz = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    edge_weights = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)

    angles = (np.arange(16) + 0.5) * 2.0 * np.pi / 16.0
    base_uv = np.vstack(
        (
            np.column_stack((1.0 * np.cos(angles), 1.0 * np.sin(angles))),
            np.column_stack((2.5 * np.cos(angles), 2.5 * np.sin(angles))),
        )
    )
    radial_scales = np.asarray([0.91, 0.97, 1.03, 1.09])
    uv = np.vstack([base_uv * scale for scale in radial_scales])
    member_a = []
    member_b = []
    scan_normals = []
    for node_id in range(count):
        patch = (
            xyz[node_id]
            + uv[:, :1] * first_tangent[node_id]
            + uv[:, 1:] * second_tangent[node_id]
        )
        member_a.append(patch)
        offset = 0.45 if changed_component and 30 <= node_id < 40 else 0.0
        member_b.append(patch + offset * node_normals[node_id])
        scan_normals.append(np.tile(node_normals[node_id], (len(uv), 1)))
        if two_sheet:
            member_a.append(patch + 0.5 * node_normals[node_id])
            member_b.append(patch + (0.5 + offset) * node_normals[node_id])
            scan_normals.append(np.tile(node_normals[node_id], (len(uv), 1)))
    raw_first = np.vstack(member_a)
    raw_second = np.vstack(member_b)
    normals = np.vstack(scan_normals)
    vertices = np.asarray(
        [[-10, -10, 0], [100, -10, 0], [100, 80, 0], [-10, 80, 0]],
        dtype=np.float64,
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
    )
    config = _config()
    config["local_surface_fit"]["minimum_points"] = 8
    config["local_surface_fit"]["minimum_occupied_footprint_cells"] = 8
    pair = config["pair_actionability"]
    pair["minimum_common_support_node_count"] = 16
    pair["minimum_common_support_area_fraction_of_cad_interior"] = 0.1
    pair["per_fold_per_row_minimum_post_halo_training_node_count"] = 8
    pair["per_fold_per_row_minimum_post_halo_training_area_fraction_of_that_row_common_support"] = 0.1
    config["global_crossfit_nuisance"]["maximum_fold_translation_dispersion_mm"] = 1.0
    config["global_crossfit_nuisance"]["maximum_fold_rotation_dispersion_deg"] = 1.0
    if changed_component:
        # A deliberately strong source-free metamorphic pose row forces the
        # frozen changed-component diagnostic branch; both engines receive the
        # identical mutated fixture contract.
        config["robustness_inventory"]["pose_translation_mm"] = 0.8
        config["robustness_inventory"]["pose_rotation_deg"] = 2.0
    return dict(
        prereg=config,
        cad_preflight={
            "cad_metrics": {
                "cad_interior_area_weighted_centroid_mm": np.average(
                    xyz, axis=0, weights=nodes["area_mm2"]
                ).tolist()
            }
        },
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        mesh=mesh,
        raw_first=raw_first,
        normals_first=normals,
        raw_second=raw_second,
        normals_second=normals,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="S000",
        second_token="S001",
    )


def _assert_full_execution_equivalent(arguments: dict) -> tuple[dict, dict, dict]:
    from scipy.sparse.csgraph import dijkstra as reference_dijkstra

    had_dijkstra = hasattr(PARENT_RUNNER, "dijkstra")
    old_dijkstra = getattr(PARENT_RUNNER, "dijkstra", None)
    PARENT_RUNNER.dijkstra = reference_dijkstra
    try:
        expected_semantic, expected_visual = PARENT_RUNNER.execute_source(**arguments)
    finally:
        if had_dijkstra:
            PARENT_RUNNER.dijkstra = old_dijkstra
        else:
            delattr(PARENT_RUNNER, "dijkstra")
    actual_semantic, actual_visual, metrics = V5.execute_parent_science_v5(**arguments)
    assert _canonical(actual_semantic) == _canonical(expected_semantic)
    assert _canonical(actual_visual) == _canonical(expected_visual)
    assert actual_semantic["robustness"][31]["row_id"] == "R31"
    assert metrics["operation_counters"]["unique_member_fit_count"] == 38
    assert metrics["operation_counters"]["robustness_row_count"] == 32
    return actual_semantic, actual_visual, metrics


def test_full_execute_ambiguous_local_cluster_payload_is_exact() -> None:
    semantic, visual, _metrics = _assert_full_execution_equivalent(
        _full_branch_arguments(two_sheet=True, changed_component=False)
    )
    invalid_reasons = {
        member["invalid_reason"]
        for node in visual["nodes"]
        for member in node["member_diagnostics"]
    }
    assert "HEIGHT_CLUSTER_RUNNER_AMBIGUOUS" in invalid_reasons
    assert semantic["status"] == "ABSTAIN"


def test_full_execute_changed_component_diagnostics_and_r31_are_exact() -> None:
    semantic, visual, _metrics = _assert_full_execution_equivalent(
        _full_branch_arguments(two_sheet=False, changed_component=True)
    )
    changed = [
        row
        for row in visual["robustness"]["rows"]
        if row.get("row_changed") or not row.get("passed", False)
    ]
    assert changed
    assert any(
        row.get("changed_component_local_diagnostics")
        for row in changed
    )
    assert visual["global"]["central_pair"]["actionable"] is True
    assert semantic["central_component_count"] > 0
    assert semantic["robustness"][31]["row_id"] == "R31"
    assert semantic["robustness"][31]["member_swap_contract"]["passed"] is True


def test_full_execute_source_semantic_visual_and_r31_payload_are_exact() -> None:
    nodes = _line_nodes(8)
    nodes["x"] = np.tile(np.asarray([2.0, 6.0, 10.0, 14.0]), 2)
    nodes["y"] = np.repeat(np.asarray([2.0, 6.0]), 4)
    edges = np.column_stack((np.arange(7), np.arange(1, 8))).astype(np.int64)
    xyz = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    weights = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.asarray([[0, 0, 0], [16, 0, 0], [16, 8, 0], [0, 8, 0]], dtype=float)
        ),
        o3d.utility.Vector3iVector(
            np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        ),
    )
    grid = np.asarray(
        [
            [x, y, 0.02 * np.sin(x + y)]
            for x in np.linspace(0.2, 15.8, 65)
            for y in np.linspace(0.2, 7.8, 33)
        ],
        dtype=np.float64,
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(grid), 1))
    config = _config()
    config["local_surface_fit"]["minimum_points"] = 4
    config["local_surface_fit"]["minimum_occupied_footprint_cells"] = 4
    pair = config["pair_actionability"]
    pair["minimum_common_support_node_count"] = 4
    pair["minimum_common_support_area_fraction_of_cad_interior"] = 0.1
    pair["per_fold_per_row_minimum_post_halo_training_node_count"] = 1
    pair["per_fold_per_row_minimum_post_halo_training_area_fraction_of_that_row_common_support"] = 0.0
    arguments = dict(
        prereg=config,
        cad_preflight={
            "cad_metrics": {"cad_interior_area_weighted_centroid_mm": [8.0, 4.0, 0.0]}
        },
        nodes=nodes,
        edges=edges,
        edge_weights=weights,
        mesh=mesh,
        raw_first=grid,
        normals_first=normals,
        raw_second=grid + np.asarray([0.0, 0.0, 0.03]),
        normals_second=normals,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="S000",
        second_token="S001",
    )
    expected_semantic, expected_visual = PARENT_RUNNER.execute_source(**arguments)
    actual_semantic, actual_visual, metrics = V5.execute_parent_science_v5(**arguments)
    assert _canonical(actual_semantic) == _canonical(expected_semantic)
    assert _canonical(actual_visual) == _canonical(expected_visual)
    assert len(actual_semantic["robustness"]) == 32
    assert actual_semantic["robustness"][31]["row_id"] == "R31"
    assert metrics["operation_counters"]["unique_member_fit_count"] == 38
    assert metrics["operation_counters"]["batched_kd_fit_count"] == 38
    assert metrics["operation_counters"]["kd_node_query_count"] == 38 * len(nodes)
    assert metrics["operation_counters"]["robustness_row_count"] == 32
    assert metrics["full_fit_inventory_gate_passed"] is True
    assert metrics["full_fit_batched_kd_gate_passed"] is True
    assert metrics["full_fit_kd_query_gate_passed"] is True
