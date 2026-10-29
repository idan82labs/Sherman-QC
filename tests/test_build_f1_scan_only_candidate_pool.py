import numpy as np
import pytest

from scripts import build_f1_scan_only_candidate_pool as builder


def test_prediction_discovery_is_generic_and_truth_registry_free(tmp_path):
    first = tmp_path / "isolated_cases" / "01_new_part" / "result" / "part"
    second = tmp_path / "isolated_cases" / "02_other_part" / "result" / "part"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "decomposition_result.json").write_text("{}")
    (second / "decomposition_result.json").write_text("{}")

    discovered = builder._discover_isolated_case_map(tmp_path)

    assert list(discovered) == ["01_new_part", "02_other_part"]


def test_prediction_discovery_rejects_ambiguous_decomposition(tmp_path):
    case = tmp_path / "isolated_cases" / "01_bad" / "result"
    (case / "a").mkdir(parents=True)
    (case / "b").mkdir(parents=True)
    (case / "a" / "decomposition_result.json").write_text("{}")
    (case / "b" / "decomposition_result.json").write_text("{}")

    with pytest.raises(ValueError, match="expected one decomposition"):
        builder._discover_isolated_case_map(tmp_path)


def test_candidate_pool_is_identity_free_and_does_not_serialize_points(monkeypatch):
    line = {
        "bend_id": "B1",
        "anchor": [0.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 20.0,
    }
    monkeypatch.setattr(builder, "merge_collinear_fragments", lambda values: [])
    monkeypatch.setattr(builder, "complete_parallel_endpoint_connectors", lambda values: [])
    monkeypatch.setattr(
        builder,
        "complete_cylindrical_bands",
        lambda *args, **kwargs: {"algorithm": "cyl", "candidates": []},
    )
    monkeypatch.setattr(builder, "extract_deterministic_planes", lambda points: [])
    monkeypatch.setattr(
        builder,
        "build_supported_plane_intersections",
        lambda points, planes: {"semantic": "plane", "candidates": []},
    )

    payload = builder.build_case_pool(
        decomposition={
            "owned_bend_regions": [line],
            "bend_hypotheses": [],
            "current_zero_shot_result": {"bend_support_strips": [], "bend_evidence_atoms": []},
        },
        points=np.zeros((4, 3), dtype=np.float64),
        normals=np.ones((4, 3), dtype=np.float64),
    )

    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["candidate_source"] == "strict_blind_owned"
    assert payload["source_point_serialization"] is False
    assert "points" not in payload
    assert payload["inference_contract"] == {
        "part_identifier_exposed": False,
        "expected_count_exposed": False,
        "manual_truth_exposed": False,
        "cad_exposed": False,
    }


def test_global_working_cloud_is_bounded_and_permutation_invariant():
    count = builder.GLOBAL_POINT_BUDGET + 137
    indices = np.arange(count, dtype=np.float64)
    points = np.column_stack(
        (
            indices * 0.001,
            np.mod(indices, 101.0) * 0.01,
            np.mod(indices, 37.0) * 0.02,
        )
    )
    normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    permutation = np.arange(count - 1, -1, -1)

    bounded_points, bounded_normals = builder.deterministic_geometry_sample(
        points,
        normals,
    )
    replay_points, replay_normals = builder.deterministic_geometry_sample(
        points[permutation],
        normals[permutation],
    )

    assert len(bounded_points) == builder.GLOBAL_POINT_BUDGET
    assert len(bounded_normals) == builder.GLOBAL_POINT_BUDGET
    np.testing.assert_array_equal(bounded_points, replay_points)
    np.testing.assert_array_equal(bounded_normals, replay_normals)


def test_ply_vertex_header_gate_prevents_unbounded_mesh_lane(tmp_path):
    path = tmp_path / "oversized.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {builder.GLOBAL_POINT_BUDGET + 1}\n".encode()
        + b"property float x\nproperty float y\nproperty float z\n"
        + b"element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    )

    assert builder._declared_ply_vertex_count(path) == builder.GLOBAL_POINT_BUDGET + 1


def test_local_support_schema_is_explicit_when_no_corridor_points_exist():
    candidate = {
        "span_endpoints": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
    }

    builder._attach_local_bend_support(
        [candidate],
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
    )

    assert candidate["normal_second_eigenvalue"] == 0.0
    assert candidate["two_sided_transition_score"] == 0.0
    assert candidate["two_sided_min_support_fraction"] == 0.0


def test_candidate_ids_are_namespaced_by_generator_source():
    standard = builder._tag([{"candidate_id": "CB2_301"}], "cylindrical_standard")
    wide = builder._tag([{"candidate_id": "CB2_301"}], "cylindrical_wide")

    assert standard[0]["source_candidate_id"] == "CB2_301"
    assert wide[0]["source_candidate_id"] == "CB2_301"
    assert standard[0]["candidate_id"] != wide[0]["candidate_id"]


def test_candidate_pool_retains_only_bounded_support_identifiers():
    tagged = builder._tag(
        [
            {
                "candidate_id": "H1",
                "owned_atom_ids": ["A3", "A1", "A1"],
                "atom_ids": ["A2"],
                "source_atom_ids": ["A4"],
                "support_patch_ids": ["P2", "P1"],
                "merge_provenance": {"source_points": [[1.0, 2.0, 3.0]]},
            }
        ],
        "bend_hypothesis",
    )[0]

    assert tagged["support_atom_ids"] == ["A1", "A2", "A3", "A4"]
    assert tagged["support_atom_ids_truncated"] is False
    assert tagged["support_patch_ids"] == ["P1", "P2"]
    assert tagged["support_patch_ids_truncated"] is False
    assert tagged["support_count"] == 4
    assert "owned_atom_ids" not in tagged
    assert "atom_ids" not in tagged
    assert "source_atom_ids" not in tagged
    assert "merge_provenance" not in tagged


def test_compact_support_graph_omits_point_indices_and_is_deterministic():
    decomposition = {
        "atom_graph": {
            "atoms": [
                {
                    "atom_id": "A2",
                    "point_indices": [100, 101],
                    "centroid": [2.0, 0.0, 0.0],
                    "normal": [0.0, 1.0, 0.0],
                    "local_spacing_mm": 1.2,
                    "curvature_proxy": 0.3,
                    "axis_direction": [1.0, 0.0, 0.0],
                    "weight": 2.0,
                    "area_proxy_mm2": 4.0,
                },
                {
                    "atom_id": "A1",
                    "point_indices": [1, 2, 3],
                    "centroid": [1.0, 0.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    "local_spacing_mm": 1.0,
                    "curvature_proxy": 0.2,
                    "axis_direction": [1.0, 0.0, 0.0],
                    "weight": 3.0,
                    "area_proxy_mm2": 5.0,
                },
            ],
            "adjacency": {"A2": ["A1"], "A1": ["A2"]},
            "edge_weights": {"A1::A2": 0.75},
        }
    }

    atoms, edges = builder._compact_support_graph(decomposition)

    assert [atom["atom_id"] for atom in atoms] == ["A1", "A2"]
    assert all("point_indices" not in atom for atom in atoms)
    assert edges == [{"atom_ids": ["A1", "A2"], "weight": 0.75}]


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "point_indices",
        "source_points",
        "points",
        "bbox_points",
        "filename",
        "part_id",
        "expected_count",
        "manual_truth",
        "cad_metadata",
        "split_group_id",
    ],
)
def test_inference_payload_rejects_forbidden_keys_recursively(forbidden_key):
    with pytest.raises(ValueError, match=forbidden_key):
        builder._reject_recursive_forbidden_keys(
            {"safe": [{"nested": {forbidden_key: "must-not-serialize"}}]}
        )


def test_transition_plane_table_uses_bounded_mode_memberships():
    plane_points = np.asarray(
        [[x, y, 0.0] for x in (-2.0, 0.0, 2.0) for y in (-2.0, 0.0, 2.0)],
        dtype=np.float64,
    )
    transition_payload = {
        "local_spacing_mm": 1.0,
        "support_mode_table": [
            {
                "mode_atom_id": "M1",
                "parent_atom_id": "P1",
                "centroid": [0.0, 0.0, 0.1],
                "normal_projective": [0.0, 0.0, -1.0],
            },
            {
                "mode_atom_id": "M2",
                "parent_atom_id": "P1",
                "centroid": [0.0, 0.0, 0.0],
                "normal_projective": [1.0, 0.0, 0.0],
            },
        ],
    }

    planes, memberships = builder._enrich_transition_plane_table(
        [
            {
                "plane_id": "PLANE:P1",
                "normal": [0.0, 0.0, 1.0],
                "offset": 0.0,
                "support_count": len(plane_points),
                "merged_plane_ids": ["RP1"],
            }
        ],
        plane_points,
        transition_payload,
    )

    assert planes[0]["support_mode_ids"] == ["M1"]
    assert planes[0]["support_parent_ids"] == ["P1"]
    assert planes[0]["support_component_ids"]
    by_mode = {row["mode_atom_id"]: row["plane_side_memberships"] for row in memberships}
    assert by_mode["M1"][0]["canonical_plane_id"] == "PLANE:P1"
    assert by_mode["M1"][0]["eligible"] is True
    assert by_mode["M2"] == []


def test_transition_augmentation_preserves_frozen_candidate_geometries():
    x = np.linspace(0.1, 8.9, 24)
    first = np.column_stack((x, np.full_like(x, 0.2), np.zeros_like(x)))
    second = np.column_stack((x, np.zeros_like(x), np.full_like(x, 0.2)))
    points = np.vstack((first, second))
    normals = np.vstack(
        (
            np.tile([0.0, 1.0, 0.0], (len(first), 1)),
            np.tile([0.0, 0.0, 1.0], (len(second), 1)),
        )
    )
    frozen_candidates = [
        {
            "candidate_id": "LP_FROZEN",
            "span_endpoints": [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        }
    ]
    source = {
        "schema_version": 2,
        "algorithm": "f1_uniform_scan_only_candidate_pool_v2",
        "source_point_serialization": False,
        "support_atom_table": [],
        "support_atom_edges": [],
        "plane_table": [
            {
                "plane_id": "PLANE:P1",
                "normal": [0.0, 1.0, 0.0],
                "offset": 0.0,
                "support_count": len(first),
                "merged_plane_ids": ["PLANE:P1"],
            },
            {
                "plane_id": "PLANE:P2",
                "normal": [0.0, 0.0, 1.0],
                "offset": 0.0,
                "support_count": len(second),
                "merged_plane_ids": ["PLANE:P2"],
            },
        ],
        "candidates": frozen_candidates,
        "line_objects": [{"latent_path_id": "LP_FROZEN"}],
    }

    augmented = builder.augment_case_pool_with_transition_v2(
        source,
        points=points,
        normals=normals,
    )

    assert augmented["schema_version"] == 3
    assert augmented["candidates"] == frozen_candidates
    assert augmented["line_objects"] == source["line_objects"]
    assert augmented["support_representation_profile"] == "TRANSITION_ATOM_V2"
    assert augmented["support_parent_table"]
    assert augmented["support_mode_table"]


def test_zero_normal_repair_is_deterministic_and_bounded():
    grid = np.asarray(
        [[float(x), float(y), 0.0] for x in range(4) for y in range(4)],
        dtype=np.float64,
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(grid), 1))
    normals[[0, 5, 10]] = 0.0

    first = builder._repair_zero_normals(grid, normals, local_spacing_mm=1.0)
    second = builder._repair_zero_normals(grid, normals, local_spacing_mm=1.0)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    np.testing.assert_allclose(np.abs(first[[0, 5, 10], 2]), 1.0)


def test_tight_corridor_normal_mode_detects_two_immediate_surfaces():
    candidate = {
        "span_endpoints": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
    }
    x = np.linspace(2.0, 18.0, 30)
    first_points = np.column_stack((x, np.ones_like(x), np.zeros_like(x)))
    second_points = np.column_stack((x, -np.ones_like(x), np.zeros_like(x)))
    points = np.vstack((first_points, second_points))
    normals = np.vstack(
        (
            np.tile([0.0, 1.0, 0.0], (len(x), 1)),
            np.tile([0.0, 0.0, 1.0], (len(x), 1)),
        )
    )

    builder._attach_local_bend_support([candidate], points, normals)

    assert candidate["tight_normal_sample_count_r2mm"] == 60
    assert candidate["tight_normal_second_eigenvalue_r2mm"] == pytest.approx(0.5)
    assert candidate["tight_to_wide_normal_second_ratio"] == pytest.approx(1.0)


def test_scan_only_pool_includes_unselected_hypotheses_and_control_strips(monkeypatch):
    monkeypatch.setattr(builder, "merge_collinear_fragments", lambda values: [])
    monkeypatch.setattr(builder, "complete_parallel_endpoint_connectors", lambda values: [])
    monkeypatch.setattr(
        builder,
        "complete_cylindrical_bands",
        lambda *args, **kwargs: {"algorithm": "cyl", "candidates": []},
    )
    monkeypatch.setattr(builder, "extract_deterministic_planes", lambda points: [])
    monkeypatch.setattr(
        builder,
        "build_supported_plane_intersections",
        lambda points, planes: {"semantic": "plane", "candidates": []},
    )
    decomposition = {
        "owned_bend_regions": [],
        "bend_hypotheses": [
            {
                "bend_id": "H1",
                "anchor": [0.0, 0.0, 0.0],
                "axis_direction": [1.0, 0.0, 0.0],
                "span_mm": 25.0,
            }
        ],
        "current_zero_shot_result": {
            "bend_support_strips": [
                {
                    "strip_id": "S1",
                    "axis_direction": [0.0, 1.0, 0.0],
                    "axis_line_hint": {"midpoint": [1.0, 2.0, 3.0]},
                    "along_axis_span_mm": 30.0,
                    "support_point_count": 20,
                }
            ],
            "bend_evidence_atoms": [],
        },
    }

    payload = builder.build_case_pool(
        decomposition=decomposition,
        points=np.zeros((4, 3), dtype=np.float64),
        normals=np.ones((4, 3), dtype=np.float64),
    )

    assert payload["candidate_source_counts"] == {
        "bend_hypothesis": 1,
        "control_support_strip": 1,
    }


def test_scan_only_pool_serializes_compact_plane_table_not_points(monkeypatch):
    monkeypatch.setattr(builder, "merge_collinear_fragments", lambda values: [])
    monkeypatch.setattr(builder, "complete_parallel_endpoint_connectors", lambda values: [])
    monkeypatch.setattr(
        builder,
        "complete_cylindrical_bands",
        lambda *args, **kwargs: {"algorithm": "cyl", "candidates": []},
    )
    monkeypatch.setattr(
        builder,
        "extract_deterministic_planes",
        lambda points: [
            {
                "plane_id": "RP1",
                "normal": [0.0, 0.0, 1.0],
                "offset": -2.0,
                "support_count": 12,
            }
        ],
    )
    monkeypatch.setattr(
        builder,
        "build_supported_plane_intersections",
        lambda points, planes: {
            "semantic": "plane",
            "candidates": [],
            "merged_planes": [
                {
                    **planes[0],
                    "merged_plane_ids": ["RP1", "RP2"],
                }
            ],
        },
    )

    payload = builder.build_case_pool(
        decomposition={
            "owned_bend_regions": [],
            "bend_hypotheses": [],
            "current_zero_shot_result": {
                "bend_support_strips": [],
                "bend_evidence_atoms": [],
            },
        },
        points=np.zeros((4, 3), dtype=np.float64),
        normals=np.ones((4, 3), dtype=np.float64),
    )

    assert payload["schema_version"] == 3
    assert payload["source_point_serialization"] is False
    assert payload["support_atom_serialization"] == "bounded_identifiers_only"
    assert payload["support_representation_profile"] == "TRANSITION_ATOM_V2"
    assert payload["support_parent_table"]
    assert payload["transition_atom_caps"]["input_point_cap"] == 200_000
    assert len(payload["plane_table"]) == 1
    assert payload["plane_table"][0]["canonical_plane_id"] == "RP1"
    assert payload["plane_table"][0]["normal"] == [0.0, 0.0, 1.0]
    assert payload["plane_table"][0]["offset"] == -2.0
    assert payload["plane_table"][0]["support_count"] == 12
    assert payload["plane_table"][0]["merged_plane_ids"] == ["RP1", "RP2"]
    assert "tangent_basis" in payload["plane_table"][0]
    assert "extent_uv" in payload["plane_table"][0]
    assert "fit_p90_mm" in payload["plane_table"][0]
    assert "support_parent_ids" in payload["plane_table"][0]
    assert "support_mode_ids" in payload["plane_table"][0]


def test_scan_mesh_adjacency_is_identity_free_candidate_source(monkeypatch):
    monkeypatch.setattr(builder, "merge_collinear_fragments", lambda values: [])
    monkeypatch.setattr(builder, "complete_parallel_endpoint_connectors", lambda values: [])
    monkeypatch.setattr(
        builder,
        "complete_cylindrical_bands",
        lambda *args, **kwargs: {"algorithm": "cyl", "candidates": []},
    )
    monkeypatch.setattr(builder, "extract_deterministic_planes", lambda points: [])
    monkeypatch.setattr(
        builder,
        "build_supported_plane_intersections",
        lambda points, planes: {"semantic": "plane", "candidates": []},
    )
    mesh_candidate = {
        "bend_id": "SMA1",
        "line_start": [0.0, 0.0, 0.0],
        "line_end": [20.0, 0.0, 0.0],
        "bend_angle": 90.0,
    }

    payload = builder.build_case_pool(
        decomposition={
            "owned_bend_regions": [],
            "bend_hypotheses": [],
            "current_zero_shot_result": {"bend_support_strips": [], "bend_evidence_atoms": []},
        },
        points=np.column_stack((np.linspace(0.0, 20.0, 40), np.zeros(40), np.zeros(40))),
        normals=np.tile([0.0, 0.0, 1.0], (40, 1)),
        scan_mesh_adjacency=[mesh_candidate],
    )

    assert payload["candidate_source_counts"] == {"scan_mesh_adjacency": 1}
    assert payload["candidates"][0]["candidate_source"] == "scan_mesh_adjacency"
    assert payload["line_object_count"] == 1
    assert payload["inference_contract"]["cad_exposed"] is False
    assert payload["source_point_serialization"] is False


def test_finite_scan_support_penalizes_floating_extensions():
    supported = {
        "anchor": [5.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 10.0,
    }
    floating = {
        "anchor": [50.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 10.0,
    }
    points = np.column_stack((np.linspace(0.0, 10.0, 101), np.zeros(101), np.zeros(101)))

    builder._attach_finite_scan_support([supported, floating], points, distance_mm=0.5)

    assert supported["finite_support_fraction"] == 1.0
    assert supported["endpoint_support_fraction"] == 1.0
    assert floating["finite_support_fraction"] == 0.0


def test_scan_support_crop_is_deterministic_and_removes_floating_extensions():
    candidate = {
        "bend_id": "L1",
        "span_endpoints": [[-10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
    }
    points = np.column_stack((np.linspace(0.0, 10.0, 101), np.zeros(101), np.zeros(101)))

    first = builder._trim_to_longest_scan_supported_run(
        [candidate], points, distance_mm=0.25, min_supported_span_mm=5.0
    )
    second = builder._trim_to_longest_scan_supported_run(
        [candidate], points, distance_mm=0.25, min_supported_span_mm=5.0
    )

    assert first == second
    assert len(first) == 1
    assert first[0]["span_mm"] < 11.0
    assert first[0]["span_mm"] > 9.0
    assert first[0]["line_start"][0] >= -0.25
    assert first[0]["line_end"][0] <= 10.25


def test_case_relative_features_are_deterministic_percentiles():
    objects = [
        {"support_count": 10, "score": 1.0},
        {"support_count": 30, "score": 2.0},
        {"support_count": 20, "score": 3.0},
    ]

    builder._attach_case_relative_features(objects)

    assert [item["case_percentile_support_count"] for item in objects] == [1 / 3, 1.0, 2 / 3]


def test_two_sided_transition_support_separates_dihedral_from_flat_outline():
    axis = np.asarray([1.0, 0.0, 0.0])
    first, second = builder._basis(axis)
    longitudinal = np.linspace(-10.0, 10.0, 16)
    offsets = np.linspace(1.5, 8.0, 8)
    first_flange = np.asarray([[x, y, 0.0] for x in longitudinal for y in offsets])
    second_flange = np.asarray([[x, 0.0, z] for x in longitudinal for z in offsets])
    radial_dihedral = np.vstack((first_flange, second_flange))
    radial_dihedral[:, 0] = 0.0
    dihedral_normals = np.vstack(
        (
            np.tile([0.0, 0.0, 1.0], (len(first_flange), 1)),
            np.tile([0.0, 1.0, 0.0], (len(second_flange), 1)),
        )
    )
    flat_points = np.asarray([[0.0, y, 0.0] for y in np.r_[-offsets, offsets] for _ in longitudinal])
    flat_normals = np.tile([0.0, 0.0, 1.0], (len(flat_points), 1))

    dihedral = builder._two_sided_transition_features(
        radial_dihedral,
        dihedral_normals,
        first,
        second,
    )
    flat = builder._two_sided_transition_features(flat_points, flat_normals, first, second)

    assert dihedral["two_sided_transition_score"] > 0.8
    assert dihedral["two_sided_normal_angle_deg"] > 80.0
    assert flat["two_sided_transition_score"] < 0.05
